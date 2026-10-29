#!/usr/bin/env python3
"""Extract failed queries (hit@5=0) from dense eval and map to chunk texts.

Usage:
    venv/bin/python scripts/extract_failed_queries.py

Output:
    eval/manual/failed_retrieval_analysis.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_NLTK_DATA = PROJECT_DIR / "nltk_data"
if _NLTK_DATA.exists():
    os.environ["NLTK_DATA"] = str(_NLTK_DATA) + os.pathsep + os.environ.get("NLTK_DATA", "")

from qdrant_client import QdrantClient

# ── Config ──────────────────────────────────────────────────────────
EVAL_REPORT = Path("/tmp/eval_user_v2/eval_report.json")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fns_collection")
OUTPUT = PROJECT_DIR / "eval" / "manual" / "failed_retrieval_analysis.md"


def load_corpus_qdrant(host: str, port: int, collection: str) -> dict[str, dict]:
    """Загрузить все чанки в память: {chunk_id: payload}."""
    client = QdrantClient(host=host, port=port)
    corpus: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            # ID в Qdrant — строка (uuid) или число, payload содержит node_id
            payload = p.payload or {}
            chunk_id = payload.get("node_id") or payload.get("id") or str(p.id)
            corpus[chunk_id] = payload
        if offset is None or offset == "":
            break
    return corpus


def main() -> None:
    # 1) Загружаем eval report
    report = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
    results = report["results"]
    # Берём первый конфиг (hybrid@1.0/0.0 — production dense)
    dense_result = next(r for r in results if r["config"] == "hybrid@1.0/0.0")
    per_query = dense_result["per_query"]
    total = len(per_query)
    print(f"📊 Всего запросов в отчёте: {total}")

    # 2) Фильтруем провальные: first_hit_rank == 0 (hit@5 = 0)
    failed = [q for q in per_query if q.get("first_hit_rank", 0) == 0]
    print(f"❌ Провалов (hit@5=0): {len(failed)}")

    if not failed:
        print("✅ Нет провалов. Создаю пустой отчёт.")
        OUTPUT.write_text("# ❌ Провалов нет\n\nВ данном прогоне все запросы имеют hit@5 > 0.\n", encoding="utf-8")
        return

    # 3) Загружаем корпус
    print(f"🔎 Загрузка корпуса из {QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION}...")
    corpus = load_corpus_qdrant(QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION)
    print(f"✅ Загружено чанков: {len(corpus)}")

    # 4) Формируем markdown
    lines = [
        "# ❌ Анализ провальных запросов (hit@5 = 0)\n",
        f"**Конфиг:** hybrid@1.0/0.0 (чистый dense)  \n",
        f"**Всего запросов:** {total}  \n",
        f"**Провалов:** {len(failed)} / {total} ({100*len(failed)/total:.1f}%)\n",
        "---\n",
    ]

    for idx, q in enumerate(failed, 1):
        query_text = q["query"]
        relevant_ids = q.get("relevant", [])
        top5_ids = q.get("top5", [])

        lines.append(f"## ❌ ПРОМАХ #{idx}")
        lines.append(f"**Вопрос:** {query_text}\n")

        # GT chunk
        lines.append("### Ожидаемые чанки (Ground Truth)")
        for rid in relevant_ids:
            chunk_payload = corpus.get(rid, {})
            chunk_text = chunk_payload.get("text") or chunk_payload.get("content") or "(текст не найден в корпусе)"
            lines.append(f"- **`{rid}`**")
            lines.append(f"  ```\n  {chunk_text[:500]}\n  ```")
        lines.append("")

        # Фактические топ-5
        lines.append("### Фактический ТОП-5 от ретривера")
        if not top5_ids:
            lines.append("_(пусто — ретривер не вернул результатов)_")
        else:
            for rank, cid in enumerate(top5_ids, 1):
                chunk_payload = corpus.get(cid, {})
                chunk_text = chunk_payload.get("text") or chunk_payload.get("content") or "(текст не найден)"
                source = chunk_payload.get("source") or chunk_payload.get("doc_name") or "?"
                lines.append(f"**{rank}. `{cid}`** (source: {source})")
                lines.append(f"  ```\n  {chunk_text[:500]}\n  ```")
        lines.append("---\n")

        # Ход выполнения
        if idx % 10 == 0:
            print(f"   … {idx}/{len(failed)}")

    # 5) Сохраняем
    OUTDIR = OUTPUT.parent
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Отчёт сохранён: {OUTPUT}")


if __name__ == "__main__":
    main()