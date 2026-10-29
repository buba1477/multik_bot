import json
import logging
import os
import re
from collections import Counter
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)-7s] %(message)s')
logger = logging.getLogger("qrels_audit")

QRELS_PATH = "eval/manual/qrels_user_clean.jsonl"
QDRANT_URL = os.getenv("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "fns_collection"

def get_jaccard_sim(str1: str, str2: str) -> float:
    words1 = set(re.findall(r'\w+', str1.lower()))
    words2 = set(re.findall(r'\w+', str2.lower()))
    if not words1 or not words2: return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))

def fetch_qdrant_chunk_ids() -> set[str]:
    logger.info(f"🧠 Подключение к Qdrant ({QDRANT_URL})...")
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
        chunk_ids = set()
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=["id"],
                with_vectors=False,
                limit=1000,
                offset=offset
            )
            for record in records:
                if record.payload and "id" in record.payload:
                    chunk_ids.add(record.payload["id"])
            if offset is None: break
        logger.info(f"✅ Загружено {len(chunk_ids)} chunk ID из Qdrant.")
        return chunk_ids
    except Exception as e:
        logger.error(f"❌ Ошибка Qdrant: {e}")
        return set()

def run_audit():
    if not os.path.exists(QRELS_PATH):
        logger.error(f"❌ Файл не найден: {QRELS_PATH}")
        return
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    logger.info(f"📊 Всего записей: {len(records)}")

    # 1. Жесткие дубли (In-place фикс)
    seen_queries = {}
    unique_records = []
    duplicates = 0
    for idx, rec in enumerate(records):
        norm_q = rec["query"].strip().lower()
        if norm_q in seen_queries:
            logger.warning(f"🚨 Дубликат (Индекс {idx}): '{rec['query']}'")
            duplicates += 1
        else:
            seen_queries[norm_q] = idx
            unique_records.append(rec)

    if duplicates > 0:
        with open(QRELS_PATH, "w", encoding="utf-8") as f:
            for rec in unique_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records = unique_records
        logger.info(f"✅ Дубли удалены. Осталось: {len(records)}")
    else:
        logger.info("✅ Жестких дубликатов не найдено.")

    # 2. Семантические дубли
    semantic_found = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            sim = get_jaccard_sim(records[i]["query"], records[j]["query"])
            if sim > 0.85:
                semantic_found += 1
                logger.warning(f"⚠️ Перефраз (Jaccard={sim:.2f}):\n Q1: '{records[i]['query']}'\n Q2: '{records[j]['query']}'")
    if semantic_found == 0:
        logger.info("✅ Семантических дублей не найдено.")

    # 3. Верификация чанков
    qdrant_chunks = fetch_qdrant_chunk_ids()
    if qdrant_chunks:
        broken = 0
        for rec in records:
            for chunk_id in rec.get("relevant", []):
                if chunk_id not in qdrant_chunks:
                    logger.error(f"🚨 БИТЫЙ ЧАНК: '{chunk_id}' в запросе '{rec['query']}'")
                    broken += 1
        if broken == 0:
            logger.info(f"✅ Все {sum(len(r['relevant']) for r in records)} chunk ID верифицированы в Qdrant.")

    # 4. Валидация триггеров EXACT/USER
    exact_triggers = ["статья", "ст.", "фз", "указ", "постановление", "%", "процент"]
    exact_errors = 0
    user_errors = 0
    for idx, rec in enumerate(records):
        source = rec.get("meta", {}).get("source", "")
        q_low = rec["query"].lower()
        has_num = any(char.isdigit() for char in q_low)
        has_kw = any(trig in q_low for trig in exact_triggers)

        if source == "dataset-expansion-exact" and not (has_num or has_kw):
            logger.warning(f"🚨 Ошибка EXACT (Индекс {idx}): нет триггеров в '{rec['query']}'")
            exact_errors += 1
        elif source == "dataset-expansion-user" and (has_num or has_kw):
            logger.warning(f"🚨 Ошибка USER (Индекс {idx}): есть триггеры в '{rec['query']}'")
            user_errors += 1

    if exact_errors == 0 and user_errors == 0:
        logger.info("✅ Контуры классификации корректны.")
    else:
        logger.info(f"   EXACT-нарушений: {exact_errors}, USER-нарушений: {user_errors}")

    logger.info("🏁 Аудит завершен.")

if __name__ == "__main__":
    run_audit()
