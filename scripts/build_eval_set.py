#!/usr/bin/env python3
"""Сборка golden set для eval_retrieval.py.

Retrieval (Qdrant/BM25/эмбеддинги) — офлайн. LLM-разметка (генерация вопросов и
судья релевантности) идёт через GigaChat API (Sber) — только на этапе сборки gold;
сам eval_retrieval.py остаётся офлайн. Ключ: API_KEY_GIGACHAT в .env или --gigachat-key.

Режимы (--mode):
  1) llm-gen        — ЧЕРНОВОЙ gold: сэмплируем чанки, GigaChat генерит по
                      одному вопросу на чанк; relevant = сам чанк. Быстро, но «слабый gold».
  2) from-questions — вопросы «из головы»: txt (один на строку) или jsonl {"query": ...}.
                      Скрипт достаёт top-N кандидатов (dense+bm25 по Qdrant) и пишет review-файл;
                      с --auto-judge GigaChat выбирает лучший.
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
import time
from contextlib import nullcontext
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat as GigaChatSDK

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")
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


def hybrid_candidates(encoder: FridaEncoder, client, collection: str,
                      query: str, limit: int, corpus: list[dict]) -> list[str]:
    """Гибридный поиск: dense (Qdrant) + BM25 + weighted RRF (α=0.7, β=0.3)."""
    from collections import defaultdict

    from rank_bm25 import BM25Okapi

    vec = encoder.encode_query(query)
    # Dense: top_k = limit*2
    try:
        dense_pts = client.query_points(collection_name=collection, query=vec,
                                        limit=limit * 2, with_payload=True).points
    except Exception:
        dense_pts = client.search(collection_name=collection, query_vector=vec,
                                  limit=limit * 2, with_payload=True)
    dense_ids = [str((p.payload or {}).get("id") or p.id) for p in dense_pts]

    # BM25
    def _tokenize(t: str) -> list[str]:
        return re.findall(r"\w+", t.lower())

    tokenized_corpus = [_tokenize(d["text"] + " " + d["title"]) for d in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))
    id_to_idx = {d["id"]: idx for idx, d in enumerate(corpus)}
    bm25_ranked = sorted(
        [d["id"] for d in corpus if d["id"] in id_to_idx],
        key=lambda cid: scores[id_to_idx[cid]],
        reverse=True,
    )[:limit * 2]

    # Weighted RRF fusion
    rank = defaultdict(float)
    alpha, beta = 0.7, 0.3
    for i, cid in enumerate(dense_ids):
        rank[cid] += alpha * 1.0 / (i + 1)
    for i, cid in enumerate(bm25_ranked):
        rank[cid] += beta * 1.0 / (i + 1)
    return [cid for cid, _ in sorted(rank.items(), key=lambda kv: -kv[1])[:limit]]


# ---------------------------------------------------------------------------
# GigaChat (Sber API)
# ---------------------------------------------------------------------------
GIGACHAT_MODEL_DEFAULT = "GigaChat-2-Max"
GIGACHAT_TIMEOUT = 60.0
GIGACHAT_RETRYABLE_STATUS = (429, 500, 502, 503, 504)
GIGACHAT_RETRY_PAUSES = (1.0, 2.0, 4.0)  # 1 базовый вызов + 3 повтора


def open_gigachat(auth_data: str | None = None) -> GigaChatSDK:
    """Открывает один клиент GigaChat на прогон (переиспользуем TLS/OAuth-токен)."""
    auth = auth_data or os.getenv("API_KEY_GIGACHAT", "")
    if not auth:
        raise ValueError("❌ API_KEY_GIGACHAT не задан")
    return GigaChatSDK(credentials=auth, verify_ssl_certs=False, timeout=GIGACHAT_TIMEOUT)


def _gigachat_retryable(exc: Exception) -> bool:
    """429/5xx — повторяем; статуса нет (сеть/таймаут) — тоже повторяем."""
    code = getattr(exc, "status_code", None)
    if code is None:
        return True
    return code in GIGACHAT_RETRYABLE_STATUS


def gigachat_generate(
    prompt: str,
    model: str = GIGACHAT_MODEL_DEFAULT,
    auth_data: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    client=None,
) -> str:
    """Генерация через GigaChat API (Sber).

    client — уже открытый GigaChatSDK: тогда переиспользуем его (без повторного
    TLS-хендшейка и OAuth). Если не передан — открываем разовый клиент на вызов.
    """
    auth = auth_data or os.getenv("API_KEY_GIGACHAT", "")
    if not auth and client is None:
        raise ValueError("❌ API_KEY_GIGACHAT не задан")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt, pause in enumerate((0.0, *GIGACHAT_RETRY_PAUSES)):
        if pause:
            time.sleep(pause)
        try:
            if client is not None:
                response = client.chat(payload)
            else:
                with GigaChatSDK(credentials=auth, verify_ssl_certs=False,
                                 timeout=GIGACHAT_TIMEOUT) as giga:
                    response = giga.chat(payload)
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            if attempt >= len(GIGACHAT_RETRY_PAUSES) or not _gigachat_retryable(exc):
                raise
            log(f"   ⚠️ GigaChat retry {attempt + 1}/{len(GIGACHAT_RETRY_PAUSES)}: {exc}")
    raise RuntimeError("GigaChat: исчерпаны попытки запроса")


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
    errors = 0
    # Один клиент GigaChat на весь прогон: без повторных TLS-хендшейков/OAuth.
    with open_gigachat(args.gigachat_key) as giga:
        for i, doc in enumerate(sample, 1):
            prompt = _QGEN_PROMPT.format(fragment=doc["text"][:1500])
            try:
                q = clean_question(gigachat_generate(prompt, model=args.model,
                                                     auth_data=args.gigachat_key, client=giga))
            except Exception as e:  # noqa: BLE001
                errors += 1
                log(f"   ⚠️ [{i}/{len(sample)}] GigaChat error: {e}")
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
    log(f"📊 GigaChat ({args.model}): ошибок после retry: {errors}")
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

    if args.auto_judge and not args.gigachat_key:
        raise SystemExit("❌ API_KEY_GIGACHAT не задан: судья GigaChat недоступен "
                         "(проверь .env или передай --gigachat-key)")

    questions = _read_questions(Path(args.questions))
    log(f"📥 Вопросов из файла: {len(questions)}")

    encoder = FridaEncoder(Path(args.model_path), device=args.device)
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    corpus = load_corpus_qdrant(args.qdrant_host, args.qdrant_port, args.collection)
    corpus_ids = {d["id"] for d in corpus}
    id_to_text = {d["id"]: d["text"] for d in corpus}
    log(f"✅ Корпус: {len(corpus_ids)} чанков")

    review: list[dict] = []
    judged = fallback = judge_errors = no_candidates = 0
    # Один клиент GigaChat на весь прогон (nullcontext — если судья не запрошен).
    with (open_gigachat(args.gigachat_key) if args.auto_judge else nullcontext()) as giga:
        for i, q in enumerate(questions, 1):
            cands = [c for c in hybrid_candidates(encoder, client, args.collection, q, args.candidates, corpus)
                     if c in corpus_ids]
            if not cands:
                no_candidates += 1
                log(f"   ⚠️ [{i}] нет кандидатов для: {q[:60]}")
                continue
            chosen = cands[0]
            source = "dense-top1"
            if args.auto_judge and len(cands) > 1:
                listing = "\n".join(
                    f"{j}) [{c}] {id_to_text.get(c, '')[:300]}…"
                    for j, c in enumerate(cands, 1)
                )
                try:
                    raw = gigachat_generate(_JUDGE_PROMPT.format(question=q, candidates=listing),
                                            model=args.model, auth_data=args.gigachat_key,
                                            temperature=0.0, max_tokens=8, client=giga)
                    m = re.search(r"\d+", raw)
                    if m and 1 <= int(m.group()) <= len(cands):
                        chosen = cands[int(m.group()) - 1]
                        source = f"judge:{args.model}"
                        judged += 1
                    else:
                        fallback += 1
                        log(f"   ⚠️ [{i}] судья вернул нераспознанный ответ: {raw[:40]!r}")
                except Exception as e:  # noqa: BLE001
                    judge_errors += 1
                    fallback += 1
                    log(f"   ⚠️ [{i}] judge error: {e}")
                time.sleep(args.sleep)  # пауза между запросами к GigaChat (лимиты API)
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
    dense_only = len(review) - judged - fallback - no_candidates
    log(f"\n💾 Review-файл: {out} ({len(review)} записей)")
    log(f"📊 Судья сработал: {judged}/{len(questions)}")
    log(f"📊 Fallback на dense-top1: {fallback + dense_only} "
        f"(в т.ч. один кандидат без вызова судьи: {dense_only})")
    log(f"📊 Ошибок после retry: {judge_errors}")
    log(f"📊 Без кандидатов (запрос пропущен): {no_candidates}")
    if judge_errors > 5:
        log(f"⚠️ Gold может быть неточным: ошибок судьи после retry — {judge_errors}")
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
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--model", default=GIGACHAT_MODEL_DEFAULT)
    p.add_argument("--gigachat-key", default=os.getenv("API_KEY_GIGACHAT", ""))
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
