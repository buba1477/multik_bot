"""
create_index_qdrant_chunks.py

Индексатор готовых chunks (*.jsonl) в Qdrant fns_collection.

Берёт chunks/*.jsonl, валидирует, вычисляет FRIDA embedding на GPU,
пересоздаёт коллекцию fns_collection и загружает векторы.

Порядок операций (безопасный):
  1. Прочитать и провалидировать все JSONL
  2. Проверить дубликаты ID
  3. Проверить лимит токенов
  4. Загрузить FRIDA, определить размерность вектора
  5. Вычислить все embeddings (batch)
  6. Подключиться к Qdrant
  7. Удалить/пересоздать коллекцию
  8. Загрузить векторы
  9. Проверить количество точек
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer, models


# ─── Строгий offline ──────────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ─── Пути ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # multik_bot/
CHUNKS_DIR = BASE_DIR / "chunks"
MODEL_PATH = BASE_DIR / "hf_cache" / "FRIDA"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "fns_collection"
EMBED_BATCH_SIZE = 32
MAX_TOKENS = 400

# ─── Вспомогательные функции ──────────────────────────────────────────


def _load_tokenizer() -> Any:
    """Загрузить FRIDA-токенизатор (offline, кэшированный)."""
    model_dir = MODEL_PATH
    if not model_dir.exists():
        print(f"❌ Модель не найдена: {model_dir}")
        sys.exit(1)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        return tok
    except Exception as e:
        print(f"❌ Ошибка загрузки токенизатора: {e}")
        sys.exit(1)


def count_tokens(text: str, tokenizer: Any) -> int:
    """Число токенов FRIDA для текста."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))



# ─── Класс embedding (Sber RoSBERTa, совместимый с LlamaIndex) ────────

class SberRoSBERTaEmbedding:
    """FRIDA embedding — Sber RoSBERTa через SentenceTransformer.

    Сохраняет интерфейс, совместимый с LlamaIndex BaseEmbedding,
    но не требует llama_index для работы (используется напрямую).
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"  Загрузка модели FRIDA из: {model_path}")
        print(f"  Device: {device}")
        word_embedding_model = models.Transformer(model_path)
        self._dim = word_embedding_model.get_word_embedding_dimension()
        print(f"  Размерность вектора: {self._dim}")
        pooling_model = models.Pooling(self._dim, pooling_mode="cls")
        self._model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model], device=device
        )

    @property
    def vector_dimension(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Batch-encoding с префиксом search_document:."""
        prefixed = [f"search_document: {t}" for t in texts]
        embeddings = self._model.encode(prefixed, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]



# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 76)
    print("QDRANT CHUNKS INDEXER")
    print("=" * 76)

    # ─── 1. Сбор и сортировка JSONL ──────────────────────────────────
    jsonl_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    print(f"\nChunks directory: {CHUNKS_DIR}")
    print(f"JSONL files: {len(jsonl_files)}")
    print()

    if not jsonl_files:
        print("❌ Нет JSONL-файлов в chunks/")
        sys.exit(1)

    # ─── 2. Валидация и чтение ───────────────────────────────────────
    print("=" * 76)
    print("INPUT VALIDATION")
    print("=" * 76)

    tokenizer = _load_tokenizer()
    all_records: list[dict] = []
    all_ids: dict[str, list[tuple[str, int]]] = {}

    for fpath in jsonl_files:
        fname = fpath.name
        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as e:
                    print(f"❌ {fname}:{line_no} — JSON error: {e}")
                    sys.exit(1)

                # Проверка ровно 5 полей
                required_keys = {"id", "title", "text", "local_img", "url"}
                if set(data.keys()) != required_keys:
                    missing = required_keys - set(data.keys())
                    extra = set(data.keys()) - required_keys
                    msg_parts = []
                    if missing:
                        msg_parts.append(f"отсутствуют: {missing}")
                    if extra:
                        msg_parts.append(f"лишние: {extra}")
                    print(f"❌ {fname}:{line_no} — поля не совпадают; {'; '.join(msg_parts)}")
                    print(f"   Полученные ключи: {set(data.keys())}")
                    sys.exit(1)


                # Проверка непустых полей
                if not isinstance(data["id"], str) or not data["id"].strip():
                    print(f"❌ {fname}:{line_no} — id пустой или не строка")
                    sys.exit(1)
                if not isinstance(data["text"], str) or not data["text"].strip():
                    print(f"❌ {fname}:{line_no} — text пустой или не строка")
                    sys.exit(1)
                for key in ("title", "local_img", "url"):
                    if key not in data or not isinstance(data[key], str):
                        print(f"❌ {fname}:{line_no} — {key} отсутствует или не строка")
                        sys.exit(1)

                # ID должен быть уникальным
                chunk_id = data["id"]
                if chunk_id in all_ids:
                    all_ids[chunk_id].append((fname, line_no))
                else:
                    all_ids[chunk_id] = [(fname, line_no)]

                # Проверка токенов
                tok_count = count_tokens(data["text"], tokenizer)
                if tok_count > MAX_TOKENS:
                    print(f"❌ {fname}:{line_no} — chunk {chunk_id} превышает {MAX_TOKENS} токенов")
                    print(f"   Токенов: {tok_count}")
                    print("   Коллекция НЕ удалена.")
                    sys.exit(1)

                all_records.append({
                    "file": fname,
                    "line_no": line_no,
                    "data": data,
                    "tokens": tok_count,
                })


    total_chunks = len(all_records)
    unique_ids = len(all_ids)
    duplicate_ids_list = {k: v for k, v in all_ids.items() if len(v) > 1}
    duplicate_count = sum(len(v) - 1 for v in duplicate_ids_list.values())
    token_counts = [r["tokens"] for r in all_records]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    over_400 = sum(1 for t in token_counts if t > MAX_TOKENS)

    # Вывод статистики по файлам
    print(f"\n{'Файл':<35} {'Chunks':>8}")
    print("-" * 45)
    for fpath in jsonl_files:
        cnt = sum(1 for r in all_records if r["file"] == fpath.name)
        print(f"{fpath.name:<35} {cnt:>8}")

    print(f"\nChunks:             {total_chunks}")
    print(f"Unique chunk IDs:   {unique_ids}")
    print(f"Duplicate IDs:      {duplicate_count}")
    print(f"Min tokens:         {min_tokens}")
    print(f"Max tokens:         {max_tokens}")
    print(f"Avg tokens:         {avg_tokens:.1f}")
    print(f">400:               {over_400}")
    print(f"Validation:         OK")

    if duplicate_count > 0:
        print(f"\n⚠️  Дублирующиеся chunk ID:")
        for cid, locations in sorted(duplicate_ids_list.items()):
            loc_str = "; ".join(f"{f}:{ln}" for f, ln in locations)
            print(f"   {cid:<40} ({len(locations)}x) — {loc_str}")


    # ─── 3. Embedding ────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("EMBEDDING")
    print("=" * 76)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        sys.exit(1)

    embedder = SberRoSBERTaEmbedding(str(MODEL_PATH), device=device)
    vec_dim = embedder.vector_dimension

    print(f"Batch size: {EMBED_BATCH_SIZE}")
    print(f"Vector dimension: {vec_dim}")

    # Подготовка текстов (оригинальный text, без изменений)
    texts = [r["data"]["text"] for r in all_records]

    # Batch embedding
    embeddings: list[list[float]] = []
    batch_size = EMBED_BATCH_SIZE
    total = len(texts)

    print(f"\nВычисление embeddings: {total} chunks")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_emb = embedder.encode(batch_texts)
        embeddings.extend(batch_emb)
        print(f"  Embedding: {end}/{total}")
        if end < total:
            time.sleep(0.1)

    assert len(embeddings) == total, (
        f"Количество embeddings ({len(embeddings)}) не совпадает с числом chunks ({total})"
    )


    # ─── 4. Qdrant ───────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("QDRANT")
    print("=" * 76)

    print(f"Host: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection: {COLLECTION_NAME}")

    q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Проверка доступности
    try:
        collections = q_client.get_collections()
        print(f"  Доступные коллекции: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"❌ Ошибка подключения к Qdrant: {e}")
        sys.exit(1)

    # Пересоздание коллекции
    print(f"\nRecreating collection {COLLECTION_NAME}...")
    q_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest_models.VectorParams(
            size=vec_dim,
            distance=rest_models.Distance.COSINE,
        ),
    )
    print(f"  Коллекция создана: size={vec_dim}, distance=COSINE")

    # Подготовка points
    points: list[rest_models.PointStruct] = []
    for i, rec in enumerate(all_records):
        data = rec["data"]
        payload = {
            "id": data["id"],
            "title": data["title"],
            "text": data["text"],
            "local_img": data["local_img"],
            "source_url": data["url"],
        }
        point = rest_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload=payload,
        )
        points.append(point)

    # Проверка — каждый point должен содержать непустой text в payload
    print(f"  Проверка payload.text у {len(points)} points...")
    empty_text_count = 0
    for pt in points:
        txt = pt.payload.get("text", "")
        if not txt:
            empty_text_count += 1
            print(f"    ⚠️ Пустой text в point id={pt.id}")
    if empty_text_count > 0:
        print(f"❌ {empty_text_count} points с пустым text — загрузка отменена")
        sys.exit(1)
    print(f"  ✅ Все points содержат непустой text")

    # Загрузка
    print(f"Uploading {len(points)} vectors...")
    q_client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=EMBED_BATCH_SIZE,
        wait=True,
    )


    # ─── 5. Проверка ─────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("RESULT")
    print("=" * 76)

    collection_info = q_client.get_collection(COLLECTION_NAME)
    qdrant_points = collection_info.points_count

    print(f"Input chunks:       {total_chunks}")
    print(f"Unique chunk IDs:   {unique_ids}")
    print(f"Duplicate chunk IDs: {duplicate_count}")
    print(f"Qdrant points:      {qdrant_points}")
    print(f"Vector size:        {vec_dim}")
    print(f"Device:             {device}")
    print(f"Status:             {'OK' if total_chunks == qdrant_points else 'MISMATCH'}")
    print("=" * 76)

    if total_chunks != qdrant_points:
        print(f"❌ Количество точек не совпадает: {total_chunks} vs {qdrant_points}")
        sys.exit(1)

    if collection_info.config.params.vectors.size != vec_dim:
        print(f"❌ Размерность вектора не совпадает: "
              f"ожидалось {vec_dim}, получено {collection_info.config.params.vectors.size}")
        sys.exit(1)

    # Проверка случайной точки
    print(f"\n{'=' * 76}")
    print("SPOT CHECK")
    print("=" * 76)
    try:
        scroll_result = q_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if scroll_result[0]:
            sample = scroll_result[0][0]
            sid = sample.id
            payload = sample.payload or {}
            has_text = bool(payload.get("text", ""))
            text_len = len(payload.get("text", ""))
            print(f"  Qdrant point ID: {sid}")
            print(f"  payload.title:   {payload.get('title', 'N/A')[:80]}")
            print(f"  payload.text:    {'✅ присутствует' if has_text else '❌ ОТСУТСТВУЕТ'}")
            print(f"  payload.text len: {text_len}")
            print(f"  payload.source_url: {payload.get('source_url', 'N/A')}")
            print(f"  payload.id:      {payload.get('id', 'N/A')}")
        else:
            print("  ⚠️ Коллекция пуста")
    except Exception as e:
        print(f"  ⚠️ Ошибка проверки точки: {e}")

    q_client.close()
    print("=" * 76)
    print("✅ Индексация завершена успешно.")
    print("=" * 76)


if __name__ == "__main__":
    main()
