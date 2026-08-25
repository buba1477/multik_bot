"""
create_index_qdrant_chunks.py

Индексатор готовых chunks (*.jsonl) в Qdrant fns_collection.

Берёт chunks/*.jsonl, валидирует, вычисляет FRIDA embedding через LlamaIndex
BaseEmbedding + VectorStoreIndex, пересоздаёт коллекцию fns_collection и загружает.

Порядок операций (безопасный):
  1. Прочитать и провалидировать все JSONL
  2. Проверить дубликаты ID
  3. Проверить лимит токенов
  4. Загрузить FRIDA Embedding (BaseEmbedding)
  5. Создать TextNode-ы и построить VectorStoreIndex через QdrantVectorStore
  6. Проверить количество точек
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, List

import torch
from pydantic.v1 import PrivateAttr
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer, models

# LlamaIndex
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore


# Strogij offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Puti
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR = BASE_DIR / "chunks"
MODEL_PATH = BASE_DIR / "hf_cache" / "FRIDA"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "fns_collection"
MAX_TOKENS = 400

# Vspomogatelnye funkcii


def _load_tokenizer() -> Any:
    """Zagruzit FRIDA-tokenizator (offline, kesh.)."""
    model_dir = MODEL_PATH
    if not model_dir.exists():
        print(f"❌ Model ne najdena: {model_dir}")
        sys.exit(1)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        return tok
    except Exception as e:
        print(f"❌ Oshibka zagruzki tokenizatora: {e}")
        sys.exit(1)


def count_tokens(text: str, tokenizer: Any) -> int:
    """Chislo tokenov FRIDA dlja teksta."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


# Klass embedding (FRIDA, BaseEmbedding LlamaIndex)

class FRIDAEmbedding(BaseEmbedding):
    """FRIDA embedding - Sber RoSBERTa cherez SentenceTransformer.

    Polnocennyj naslednik LlamaIndex BaseEmbedding.
    Prefiks search_document: dlja indeksacii, search_query: dlja poiska.
    """

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        print(f"  Zagruzka modeli FRIDA iz: {model_path}")
        print(f"  Device: {device}")
        word_embedding_model = models.Transformer(model_path)
        dim = word_embedding_model.get_word_embedding_dimension()
        print(f"  Razmernost vektora: {dim}")
        pooling_model = models.Pooling(dim, pooling_mode="cls")
        model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device=device,
        )
        self._model = model

    @property
    def vector_dimension(self) -> int:
        """Fakticheskaja razmernost vyhodnogo vektora FRIDA."""
        return self._model.get_sentence_embedding_dimension()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(f"search_query: {query}").tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(f"search_document: {text}").tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


# ===================== MAIN =====================

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 76)
    print("QDRANT CHUNKS INDEXER (LlamaIndex pipeline)")
    print("=" * 76)

    # --- 1. Sbor i sortirovka JSONL
    jsonl_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    print(f"\nChunks directory: {CHUNKS_DIR}")
    print(f"JSONL files: {len(jsonl_files)}")
    print()

    if not jsonl_files:
        print("❌ Net JSONL-fajlov v chunks/")
        sys.exit(1)

    # --- 2. Validacija i chtenie
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
                    print(f"❌ {fname}:{line_no} - JSON error: {e}")
                    sys.exit(1)

                # Определяем формат: новый (расширенный) или старый (5 полей)
                LEGACY_KEYS = {"id", "title", "text", "local_img", "url"}
                NEW_KEYS = {"id", "title", "text", "document_id", "point",
                            "subpoint", "subjects", "categories", "references",
                            "keywords", "context_flat"}

                data_keys = set(data.keys())
                is_new_format = bool(data_keys & NEW_KEYS - LEGACY_KEYS)

                if is_new_format:
                    # Новый формат: проверяем минимальный набор полей
                    required = {"id", "text"}
                    missing = required - data_keys
                    if missing:
                        print(f"❌ {fname}:{line_no} - novyj format: otsutstvujut {missing}")
                        sys.exit(1)
                    if not isinstance(data.get("id"), str) or not data["id"].strip():
                        print(f"❌ {fname}:{line_no} - id pustoj ili ne stroka")
                        sys.exit(1)
                    if not isinstance(data.get("text"), str) or not data["text"].strip():
                        print(f"❌ {fname}:{line_no} - text pustoj ili ne stroka")
                        sys.exit(1)
                else:
                    # Старый формат: строго 5 полей
                    if data_keys != LEGACY_KEYS:
                        missing = LEGACY_KEYS - data_keys
                        extra = data_keys - LEGACY_KEYS
                        msg_parts = []
                        if missing:
                            msg_parts.append(f"otsutstvujut: {missing}")
                        if extra:
                            msg_parts.append(f"lishnie: {extra}")
                        print(f"❌ {fname}:{line_no} - polja ne sovpadajut; {'; '.join(msg_parts)}")
                        print(f"   Poluchennye kljuchi: {sorted(data_keys)}")
                        sys.exit(1)
                    if not isinstance(data["id"], str) or not data["id"].strip():
                        print(f"❌ {fname}:{line_no} - id pustoj ili ne stroka")
                        sys.exit(1)
                    if not isinstance(data["text"], str) or not data["text"].strip():
                        print(f"❌ {fname}:{line_no} - text pustoj ili ne stroka")
                        sys.exit(1)
                    for key in ("title", "local_img", "url"):
                        if key not in data or not isinstance(data[key], str):
                            print(f"❌ {fname}:{line_no} - {key} otsutstvuet ili ne stroka")
                            sys.exit(1)

                # ID dolzhen byt unikalnym
                chunk_id = data["id"]
                if chunk_id in all_ids:
                    all_ids[chunk_id].append((fname, line_no))
                else:
                    all_ids[chunk_id] = [(fname, line_no)]

                # Proverka tokenov
                tok_count = count_tokens(data["text"], tokenizer)
                if tok_count > MAX_TOKENS:
                    print(f"❌ {fname}:{line_no} - chunk {chunk_id} prevyshaet {MAX_TOKENS} tokenov")
                    print(f"   Tokenov: {tok_count}")
                    print("   Kollekcija NE udalena.")
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

    # Vyvod statistiki po fajlam
    print(f"\n{'Fajl':<35} {'Chunks':>8}")
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
        print(f"\n⚠️  Dublirujushhiesja chunk ID:")
        for cid, locations in sorted(duplicate_ids_list.items()):
            loc_str = "; ".join(f"{f}:{ln}" for f, ln in locations)
            print(f"   {cid:<40} ({len(locations)}x) - {loc_str}")

    # --- 3. Embedding cherez LlamaIndex BaseEmbedding + VectorStoreIndex
    print(f"\n{'=' * 76}")
    print("EMBEDDING + INDEX (LlamaIndex)")
    print("=" * 76)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print(f"❌ Model ne najdena: {MODEL_PATH}")
        sys.exit(1)

    # Zagruzhaem FRIDA kak BaseEmbedding
    embed_model = FRIDAEmbedding(str(MODEL_PATH), device=device)
    vec_dim = embed_model.vector_dimension
    print(f"Vector dimension: {vec_dim}")

    # Sozdajom LlamaIndex TextNode-y iz validirovannyh zapisej
    # payload-polja: id, title, text, local_img, source_url (+ novye polja dlja novogo formata)
    # V embedding metadata uchastvuet TOLKO title;
    # id, source_url, local_img iskljucheny iz embedding.
    nodes: list[TextNode] = []
    for rec in all_records:
        data = rec["data"]
        base_meta = {
            "id": data["id"],
            "title": data.get("title", ""),
            "local_img": data.get("local_img", ""),
            "source_url": data.get("url", ""),
            "text": data["text"],
        }
        excluded = ["id", "source_url", "local_img", "text"]

        # Определяем формат по наличию полей
        if "document_id" in data:
            # Новый формат
            meta = {
                **base_meta,
                "document_id": data.get("document_id", ""),
                "point": data.get("point", ""),
                "subpoint": data.get("subpoint", ""),
                "part": data.get("part", 1),
                "total_parts": data.get("total_parts", 1),
                "subjects": data.get("subjects", []),
                "categories": data.get("categories", []),
                "references": data.get("references", []),
                "keywords": data.get("keywords", []),
                "context_flat": data.get("context_flat", ""),
            }

            # Формируем _em_input — embedding-маяки для FRIDA.
            # _em_input НЕ добавляется в excluded_embed_metadata_keys,
            # поэтому он попадает в MetadataMode.EMBED.
            _em_parts: list[str] = []
            if title := data.get("title", ""):
                _em_parts.append(f"Пункт: {title}")
            if point := data.get("point", ""):
                _em_parts.append(f"Пункт документа: {point}")
            if cats := data.get("categories", []):
                _em_parts.append(f"Категории: {', '.join(str(c) for c in cats)}")
            if subs := data.get("subjects", []):
                _em_parts.append(f"Субъекты: {', '.join(str(s) for s in subs)}")
            # keywords — только если выглядят содержательными
            if kws := data.get("keywords", []):
                clean_kws = [str(k) for k in kws if isinstance(k, str) and len(k) > 2]
                if clean_kws and len(clean_kws) >= 2:
                    selected = clean_kws[:5]
                    _em_parts.append(f"Ключевые слова: {', '.join(selected)}")

            if _em_parts:
                meta["_em_input"] = "\n".join(_em_parts)

            excluded = [
                "id", "title", "source_url", "local_img",
                "document_id", "point", "subpoint",
                "part", "total_parts",
                "subjects", "categories",
                "references", "keywords", "context_flat",
                "text",
            ]
            # _em_input НЕ в excluded → попадёт в MetadataMode.EMBED

            node = TextNode(
                text=data["text"],
                metadata=meta,
                excluded_embed_metadata_keys=excluded,
            )
            # Настраиваем шаблоны для EMBED: просто значение _em_input без префикса "ключ: "
            # и разделитель "Текст:" перед оригинальным content.
            node.metadata_template = "{value}"
            node.text_template = "{metadata_str}\n\nТекст:\n{content}"
        else:
            meta = base_meta.copy()
            excluded = ["id", "source_url", "local_img", "text"]

            node = TextNode(
                text=data["text"],
                metadata=meta,
                excluded_embed_metadata_keys=excluded,
            )

        nodes.append(node)

    print(f"\nSozdano TextNode: {len(nodes)}")

    # --- 4. Qdrant
    print(f"\n{'=' * 76}")
    print("QDRANT")
    print("=" * 76)

    print(f"Host: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collection: {COLLECTION_NAME}")

    q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Proverka dostupnosti
    try:
        collections = q_client.get_collections()
        print(f"  Dostupnye kollekcii: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"❌ Oshibka podkljuchenija k Qdrant: {e}")
        sys.exit(1)

    # Peresozdanie kollekcii s FAKTICHESKOJ razmernostju FRIDA
    print(f"\nRecreating collection {COLLECTION_NAME}...")
    q_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest_models.VectorParams(
            size=vec_dim,
            distance=rest_models.Distance.COSINE,
        ),
    )
    print(f"  Kollekcija sozdana: size={vec_dim}, distance=COSINE")

    # Sozdajom QdrantVectorStore (LlamaIndex) na uzhe gotovoj kollekcii
    vector_store = QdrantVectorStore(
        client=q_client,
        collection_name=COLLECTION_NAME,
        text_key="text",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Stavim embed_model v globalnyj Settings + ukazyvaem javno
    Settings.embed_model = embed_model

    print(f"\nBuilding VectorStoreIndex from {len(nodes)} nodes...")
    t0 = time.perf_counter()

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    t1 = time.perf_counter()
    index_time = t1 - t0
    avg_rate = len(nodes) / index_time
    print(f"\n  Index postroen: {len(nodes)} chunks in {index_time:.1f}s ({avg_rate:.1f} ch/s)")

    # --- 5. Proverka
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
        print(f"❌ Kolichestvo tochek ne sovpadaet: {total_chunks} vs {qdrant_points}")
        sys.exit(1)

    if collection_info.config.params.vectors.size != vec_dim:
        print(f"❌ Razmernost vektora ne sovpadaet: "
              f"ozhidalsja {vec_dim}, polucheno {collection_info.config.params.vectors.size}")
        sys.exit(1)

    # Proverka sluchajnoj tochki (payload soderzhit vse polja)
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
            text_val = payload.get("text", "")
            has_text = bool(text_val)
            text_len = len(text_val)
            print(f"  Qdrant point ID: {sid}")
            print(f"  payload.title:   {payload.get('title', 'N/A')[:80]}")
            print(f"  payload.text:    {'✅ prisutstvuet' if has_text else '❌ OTSUTSTVUET'}")
            print(f"  payload.text len: {text_len}")
            print(f"  payload.source_url: {payload.get('source_url', 'N/A')}")
            print(f"  payload.id:      {payload.get('id', 'N/A')}")
            if not has_text:
                print(f"\n❌ SPOT CHECK: payload.text otsutstvuet u tochki {sid}!")
                sys.exit(1)
            # Pokazyvaem metadannye novogo formata, esli est
            if "document_id" in payload:
                print(f"  payload.document_id: {payload.get('document_id', '')}")
                print(f"  payload.point:       {payload.get('point', '')}")
                print(f"  payload.subpoint:    {payload.get('subpoint', '')}")
                print(f"  payload.subjects:    {payload.get('subjects', [])}")
                print(f"  payload.categories:  {payload.get('categories', [])}")
                print(f"  payload.keywords:    {payload.get('keywords', [])}")
            # Struktura payload (kljuchi)
            print(f"  payload keys:     {sorted(payload.keys())}")
        else:
            print("  ⚠️ Kollekcija pusta")
    except Exception as e:
        print(f"  ⚠️ Oshibka proverki tochki: {e}")

    # --- 6. Verifikacija 10 tochek: sravnenie payload.text s ishodnym ---
    print(f"\n{'=' * 76}")
    print("VERIFY 10 POINTS")
    print("=" * 76)

    # Stroit mapping payload.id -> payload.text cherez scroll s paginaciej
    id_to_text: dict[str, str] = {}
    scroll_offset: int | None = None
    scroll_limit = 100

    while True:
        batch = q_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=scroll_limit,
            offset=scroll_offset,
            with_payload=True,
            with_vectors=False,
        )
        batch_points = batch[0]
        if not batch_points:
            break
        for pt in batch_points:
            pid = pt.payload.get("id", "")
            if pid:
                id_to_text[pid] = pt.payload.get("text", "")
        scroll_offset = batch[1]  # next offset
        if scroll_offset is None or scroll_offset == 0:
            break

    print(f"  Vsego prochitano tochek: {len(id_to_text)}")

    # Sravnivaem pervye 10 ishodnyh chunks
    verify_records = all_records[:10]
    match_errors = 0

    for rec in verify_records:
        chunk_id = rec["data"]["id"]
        expected_text = rec["data"]["text"]

        if chunk_id not in id_to_text:
            match_errors += 1
            print(f"  ❌ {chunk_id}: ne najdena v Qdrant")
            continue

        actual_text = id_to_text[chunk_id]
        if actual_text == expected_text:
            print(f"  ✅ {chunk_id}: payload.text sovpadaet ({len(actual_text)} chars)")
        else:
            match_errors += 1
            print(f"  ❌ {chunk_id}: payload.text NE SOVPADAET")
            print(f"     Ozhidalos ({len(expected_text)} chars): {expected_text[:80]}...")
            print(f"     Polucheno ({len(actual_text)} chars): {actual_text[:80]}...")

    if match_errors > 0:
        print(f"\n❌ Verifikacija ne projdena: {match_errors} osibok")
        q_client.close()
        sys.exit(1)
    else:
        print(f"\n✅ Verifikacija 10 tochek projdena uspeshno")

    q_client.close()
    print("=" * 76)
    print("\u2705 Indeksacija zavershena uspeshno.")
    print("=" * 76)


if __name__ == "__main__":
    main()
