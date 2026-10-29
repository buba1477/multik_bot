#!/usr/bin/env python3
"""
Экспорт qrels с текстом чанков для ручного аудита.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)-7s] %(message)s")
logger = logging.getLogger("export_qrels_text")

QRELS_PATH = "eval/manual/qrels_user_clean.jsonl"
OUT_PATH = "eval/manual/qrels_visual_audit.txt"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "fns_collection"


def load_qrels(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qdrant_chunks() -> dict[str, dict[str, Any]]:
    """Загрузить все чанки из Qdrant, вернуть {chunk_id: {metadata}}."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=QDRANT_URL, timeout=120)
    chunks: dict[str, dict[str, Any]] = {}
    offset = None
    total = 0

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for pt in points:
            payload = pt.payload or {}
            cid = payload.get("id", "")
            if not cid:
                continue

            # Извлекаем текст из _node_content (JSON) или напрямую
            raw_content = payload.get("_node_content", "")
            text = ""
            if isinstance(raw_content, str) and raw_content.startswith("{"):
                try:
                    text = json.loads(raw_content).get("text", "")
                except Exception:
                    text = raw_content
            if not text:
                text = payload.get("text", "")

            title = payload.get("title", "Без названия")
            source_url = payload.get("source_url", "")
            parent_title = payload.get("parent_title", "")

            chunks[cid] = {
                "id": cid,
                "title": title,
                "parent_title": parent_title,
                "text": text,
                "source_url": source_url,
            }
            total += 1

        if offset is None:
            break

    logger.info(f"📦 Загружено {total} чанков из Qdrant ({len(chunks)} уникальных ID)")
    return chunks


def fmt_block(
    query: str,
    qrels_line: str,
    chunk_id: str,
    chunk: dict[str, Any] | None,
) -> str:
    """Форматировать блок вопрос-чанк."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"❓ ВОПРОС: {query}")
    lines.append(f"📦 СТРОКА QRELS: {qrels_line}")
    lines.append("-" * 80)
    lines.append(f"🆔 ЧАНК ID: {chunk_id}")

    if chunk is None:
        lines.append("⚠️  ЧАНК НЕ НАЙДЕН В QDRANT")
        lines.append("=" * 80)
        lines.append("")
        return "\n".join(lines)

    lines.append(f"📌 НАЗВАНИЕ: {chunk.get('title', '?')}")
    parent = chunk.get("parent_title", "")
    if parent:
        lines.append(f"📂 РОДИТЕЛЬСКИЙ ЗАГОЛОВОК: {parent}")
    lines.append("📄 ТЕКСТ ЧАНКА:")
    text = chunk.get("text", "")
    if text:
        lines.append(text)
    else:
        lines.append("⚠️  (пустой текст)")
    lines.append(f"🔗 URL: {chunk.get('source_url', '')}")
    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    logger.info(f"📂 Загрузка qrels: {QRELS_PATH}")
    qrels = load_qrels(QRELS_PATH)
    logger.info(f"   Загружено {len(qrels)} записей")

    logger.info(f"🧠 Подключение к Qdrant...")
    chunks = load_qdrant_chunks()

    missing_ids: set[str] = set()
    total_blocks = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"🔬 ВИЗУАЛЬНЫЙ АУДИТ ДАТАСЕТА QRELS\n")
        f.write(f"   Файл: {QRELS_PATH}\n")
        f.write(f"   Всего записей: {len(qrels)}\n")
        f.write(f"   Чанков в Qdrant: {len(chunks)}\n")
        f.write(f"   Дата генерации: см. timestamp\n")
        f.write("=" * 80 + "\n\n")

        for idx, item in enumerate(qrels):
            query = item["query"]
            qrels_line = json.dumps(item, ensure_ascii=False)
            relevant = item.get("relevant", [])

            for chunk_id in relevant:
                chunk = chunks.get(chunk_id)
                if chunk is None:
                    missing_ids.add(chunk_id)

                block = fmt_block(query, qrels_line, chunk_id, chunk)
                f.write(block)
                total_blocks += 1

            # Прогресс каждые 50 записей
            if (idx + 1) % 50 == 0:
                logger.info(f"   Обработано {idx + 1}/{len(qrels)} записей...")

    logger.info(f"\n✅ Файл создан: {OUT_PATH}")
    logger.info(f"   Записей в qrels: {len(qrels)}")
    logger.info(f"   Выведено блоков (вопрос-чанк): {total_blocks}")
    if missing_ids:
        logger.warning(f"   ⚠️  Чанков не найдено в Qdrant: {len(missing_ids)}")
        for mid in sorted(missing_ids):
            logger.warning(f"      - {mid}")
    else:
        logger.info(f"   ✅ Все чанки найдены в Qdrant!")

    # Покажем размер файла
    size = Path(OUT_PATH).stat().st_size
    logger.info(f"   Размер файла: {size:,} байт")


if __name__ == "__main__":
    main()
