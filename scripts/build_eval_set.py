#!/usr/bin/env python3
"""Сборка golden set для eval_retrieval.py — офлайн, без внешних API.

Режимы (--mode):
  1) llm-gen        — ЧЕРНОВОЙ gold: сэмплируем чанки, локальная Ollama генерит по
                      одному вопросу на чанк; relevant = сам чанк. Быстро, но «слабый gold».
  2) from-questions — вопросы «из головы»: txt (один на строку) или jsonl {"query": ...}.
                      Скрипт достаёт top-N кандидатов (dense по Qdrant) и пишет review-файл;
                      с --auto-judge локальная LLM выбирает лучший.
  3) finalize       — из review-файла собирает финальный qrels.jsonl (валидирует id).
  4) validate       — проверяет готовый qrels: все relevant существуют в корпусе.

Запуск:
  venv/bin/python scripts/build_eval_set.py --mode llm-gen --n 100 --out eval/qrels.jsonl
  venv/bin/python scripts/build_eval_set.py --mode from-questions --questions eval/questions.txt --auto-judge --review-out eval/qrels_review.jsonl
  venv/bin/python scripts/build_eval_set.py --mode finalize --review-out eval/qrels_review.jsonl --out eval/qrels.jsonl
  venv/bin/python scripts/build_eval_set.py --mode validate --out eval/qrels.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_EDITORIAL_RE = re.compile(r"^\s*\((?:В редакции|В ред\.|С учетом|С учётом)", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Корпус и эмбеддер
# ---------------------------------------------------------------------------
def load_corpus_qdrant(host: str, port: int, collection: str) -> list[dict]:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=host, port=port)
    docs, seen, offset = [], set(), None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            pay = p.payload or {}
            cid = str(pay.get("id") or p.id)
            if cid in seen:
                continue
            seen.add(cid)
            docs.append({"id": cid, "title": pay.get("title", ""), "text": pay.get("text", "")})
        if offset is None:
            break
    client.close()
    return docs


class FridaEncoder:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer, models

        word = models.Transformer(str(model_path))
        pooling = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        self.model = SentenceTransformer(modules=[word, pooling], device=device)

    def encode_query(self, q: str) -> list[float]:
        return self.model.encode("search_query: " + q).tolist()


def dense_candidates(encoder: FridaEncoder, client, collection: str,
                     query: str, limit: int) -> list[str]:
    vec = encoder.encode_query(query)
    try:
        pts = client.query_points(collection_name=collection, query=vec,
                                  limit=limit, with_payload=True).points
    except Exception:
        pts = client.search(collection_name=collection, query_vector=vec,
                            limit=limit, with_payload=True)
    return [str((p.payload or {}).get("id") or p.id) for p in pts]


# ---------------------------------------------------------------------------
# Ollama (локально, offline)
# ---------------------------------------------------------------------------
def ollama_generate(prompt: str, model: str, host: str,
                    options: dict | None = None, timeout: int = 180) -> str:
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "options": options or {"temperature": 0.3, "num_ctx": 2048, "num_predict": 120},
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


_QGEN_PROMPT = """Ты — методист, готовящий тестовые вопросы по нормативным актам.
Ниже фрагмент нормативного текста. Сформулируй РОВНО ОДИН естественный вопрос
пользователя (на русском), ответ на который содержится ТОЛЬКО в этом фрагменте.

Требования:
- переформулируй смысл своими словами, не копируй длинные куски дословно;
- НЕ упоминай прямым текстом номер статьи/пункта;
- не добавляй пояснений и кавычек — верни только сам вопрос, заканчивающийся «?».

ФРАГМЕНТ:
{fragment}

ВОПРОС:"""

_JUDGE_PROMPT = """Вопрос пользователя: {question}

Ниже пронумерованные фрагменты-кандидаты. Выбери ОДИН, который наиболее полно
отвечает на вопрос. Верни ТОЛЬКО его номер (целое число), без пояснений.

{candidates}

НОМЕР:"""


def clean_question(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^(Вопрос|ВОПРОС|Ответ|Q)\s*[:\-—]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip().strip('"').strip("«»").strip()
    t = t.split("\n")[0].strip()
    return t


def sample_candidates(corpus: list[dict], n: int, seed: int,
                      min_chars: int, max_chars: int) -> list[dict]:
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = {}
    for d in corpus:
        text = d.get("text", "")
        if not (min_chars <= len(text) <= max_chars):
            continue
        if not d.get("title"):
            continue
        body = text.split("\n", 1)[-1]
        if _EDITORIAL_RE.match(body):
            continue
        by_doc.setdefault(d["id"].split("_")[0], []).append(d)
    for v in by_doc.values():
        rng.shuffle(v)
    docs = sorted(by_doc.keys())
    rng.shuffle(docs)
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(by_doc.values()):
        doc = docs[i % len(docs)]
        if by_doc[doc]:
            picked.append(by_doc[doc].pop())
        i += 1
        if i > 100000:
            break
    return picked


# ---------------------------------------------------------------------------
# Запись файлов
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Режимы
# ---------------------------------------------------------------------------
def mode_llm_gen(args) -> None:
    corpus = load_corpus_qdrant(args.qdrant_host, args.qdrant_port, args.collection)
    log(f"✅ Корпус: {len(corpus)} чанков")
    sample = sample_candidates(corpus, args.n, args.seed, args.min_chars, args.max_chars)
    log(f"🎯 Отобрано чанков для генерации вопросов: {len(sample)}")

    items: list[dict] = []
    seen_q: set[str] = set()
    for i, doc in enumerate(sample, 1):
        prompt = _QGEN_PROMPT.format(fragment=doc["text"][:1500])
        try:
            q = clean_question(ollama_generate(prompt, args.model, args.ollama_host))
        except Exception as e:  # noqa: BLE001
            log(f"   ⚠️ [{i}/{len(sample)}] Ollama error: {e}")
            continue
        if not (15 <= len(q) <= 300) or not q.endswith("?"):
            log(f"   ⚠️ [{i}/{len(sample)}] отклонён вопрос: {q[:80]!r}")
            continue
        key = q.lower()
        if key in seen_q:
            continue
        seen_q.add(key)
        items.append({
            "query": q,
            "relevant": [doc["id"]],
            "source_id": doc["id"],
            "meta": {"title": doc["title"], "generator": f"llm:{args.model}"},
        })
        log(f"   ✅ [{i}/{len(sample)}] {q[:70]}")
        time.sleep(args.sleep)

    _write_jsonl(Path(args.out), items)
    log(f"\n💾 Черновой gold: {args.out} ({len(items)} запросов)")
    log("⚠️  СЛАБЫЙ gold (вопрос сгенерирован из текста чанка). Для честных цифр: "
        "from-questions + ручное подтверждение.")


def _read_questions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".jsonl", ".json"):
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    for k in ("query", "question", "вопрос", "q"):
                        if obj.get(k):
                            out.append(str(obj[k]).strip())
                            break
                elif isinstance(obj, str):
                    out.append(obj.strip())
            except json.JSONDecodeError:
                out.append(line)
        return [q for q in out if q]
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def mode_from_questions(args) -> None:
    from qdrant_client import QdrantClient

    questions = _read_questions(Path(args.questions))
    log(f"📥 Вопросов из файла: {len(questions)}")

    encoder = FridaEncoder(Path(args.model_path), device=args.device)
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    corpus_ids = {d["id"] for d in load_corpus_qdrant(args.qdrant_host, args.qdrant_port, args.collection)}
    log(f"✅ Корпус: {len(corpus_ids)} чанков")

    review: list[dict] = []
    for i, q in enumerate(questions, 1):
        cands = [c for c in dense_candidates(encoder, client, args.collection, q, args.candidates)
                 if c in corpus_ids]
        if not cands:
            log(f"   ⚠️ [{i}] нет кандидатов для: {q[:60]}")
            continue
        chosen = cands[0]
        source = "dense-top1"
        if args.auto_judge and len(cands) > 1:
            listing = "\n".join(f"{j}) {c}" for j, c in enumerate(cands, 1))
            try:
                raw = ollama_generate(_JUDGE_PROMPT.format(question=q, candidates=listing),
                                      args.model, args.ollama_host,
                                      options={"temperature": 0, "num_ctx": 2048, "num_predict": 8})
                m = re.search(r"\d+", raw)
                if m and 1 <= int(m.group()) <= len(cands):
                    chosen = cands[int(m.group()) - 1]
                    source = f"judge:{args.model}"
            except Exception as e:  # noqa: BLE001
                log(f"   ⚠️ [{i}] judge error: {e}")
        review.append({
            "query": q,
            "relevant": [chosen],
            "candidates": cands,
            "needs_review": True,
            "meta": {"labeler": source},
        })
        log(f"   [{i}/{len(questions)}] {q[:55]} -> {chosen} ({source})")

    client.close()
    out = Path(args.review_out)
    _write_jsonl(out, review)
    log(f"\n💾 Review-файл: {out} ({len(review)} записей)")
    log("👉 Проверь/поправь поле 'relevant' (и поставь \"needs_review\": false), затем:")
    log(f"   venv/bin/python scripts/build_eval_set.py --mode finalize "
        f"--review-out {out} --out {args.out}")


def mode_finalize(args) -> None:
    review_path = Path(args.review_out)
    records = [json.loads(ln) for ln in review_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    corpus_ids = {d["id"] for d in load_corpus_qdrant(args.qdrant_host, args.qdrant_port, args.collection)}
    items, dropped = [], 0
    for rec in records:
        q = str(rec.get("query", "")).strip()
        rel = [str(r) for r in (rec.get("relevant") or []) if str(r) in corpus_ids]
        if not q or not rel:
            dropped += 1
            continue
        items.append({"query": q, "relevant": rel,
                      "meta": {"title": rec.get("title", ""), "source": "manual-review"}})
    out = Path(args.out)
    if args.append and out.exists():
        _append_jsonl(out, items)
    else:
        _write_jsonl(out, items)
    log(f"💾 Финальный gold: {out} ({len(items)} запросов, отброшено {dropped})")


def mode_validate(args) -> None:
    path = Path(args.out)
    records = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    corpus_ids = {d["id"] for d in load_corpus_qdrant(args.qdrant_host, args.qdrant_port, args.collection)}
    missing, total = {}, 0
    for rec in records:
        for rid in rec.get("relevant", []):
            total += 1
            if rid not in corpus_ids:
                missing[rid] = missing.get(rid, 0) + 1
    log(f"📄 Записей: {len(records)} | ссылок на чанки: {total} | отсутствуют в корпусе: {len(missing)}")
    for rid, cnt in sorted(missing.items(), key=lambda kv: -kv[1])[:20]:
        log(f"   ❌ {rid} (x{cnt})")
    if not missing:
        log("✅ Все relevant_ids найдены в корпусе — qrels валиден.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сборка golden set для eval retrieval (offline)")
    p.add_argument("--mode", required=True,
                   choices=["llm-gen", "from-questions", "finalize", "validate"])
    p.add_argument("--out", default=str(PROJECT_DIR / "eval" / "qrels.jsonl"))
    p.add_argument("--review-out", default=str(PROJECT_DIR / "eval" / "qrels_review.jsonl"))
    p.add_argument("--questions", default=str(PROJECT_DIR / "eval" / "questions.txt"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--auto-judge", action="store_true")
    p.add_argument("--append", action="store_true")
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--model", default="yagpt5_fns:latest")
    p.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    p.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "fns_collection"))
    p.add_argument("--model-path", default=str(PROJECT_DIR / "hf_cache" / "FRIDA"))
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    handlers = {
        "llm-gen": mode_llm_gen,
        "from-questions": mode_from_questions,
        "finalize": mode_finalize,
        "validate": mode_validate,
    }
    handlers[args.mode](args)


if __name__ == "__main__":
    main()
