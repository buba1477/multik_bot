import os
import json
import logging
import torch
import qdrant_client
from pathlib import Path
from typing import List

# --- СТРОГИЙ ОФФЛАЙН ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer, models
from llama_index.core.node_parser import SentenceSplitter

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Qdrant_Indexer_Final")

# Пути
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "hf_cache" / "ru-en-RoSBERTa"
COLLECTION_NAME = "fns_collection"

if not MODEL_PATH.exists():
    logger.error(f"❌ МОДЕЛЬ НЕ НАЙДЕНА: {MODEL_PATH}")
    exit()

# ========================================
# 1. КЛАСС ЭМБЕДДИНГА (Sber RoSBERTa)
# ========================================
class SberRoSBERTaEmbedding(BaseEmbedding):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        logger.info(f"✨ Инициализация модели на {device}")
        word_embedding_model = models.Transformer(model_path)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), 
            pooling_mode='cls'
        )
        self._model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(f"search_query: {query}").tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(f"search_document: {text}").tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

# Глобальные настройки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Settings.embed_model = SberRoSBERTaEmbedding(model_path=str(MODEL_PATH), device=DEVICE)
Settings.node_parser = SentenceSplitter(chunk_size=10000, chunk_overlap=0)

# ========================================
# 2. ПОДГОТОВКА ДОКУМЕНТОВ
# ========================================
logger.info("🔄 Сборка документов из JSONL...")

# dataset_files = ["graph_nodes.jsonl", "graph_summaries.jsonl"]
dataset_files = ["graph_nodes.jsonl"]
documents = []

for filename in dataset_files:
    file_path = BASE_DIR / filename
    if not file_path.exists():
        logger.warning(f"⚠️ Файл {filename} не найден, пропуск.")
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                if not text: continue
                
                # Собираем метаданные
                entities = data.get('graph_data', {}).get('entities', [])
                graph_info = "СУЩНОСТИ: " + "; ".join([e['name'] for e in entities[:10] if isinstance(e, dict)])

                metadata = {
                    "graph_structure": graph_info,
                    "title": data.get('title') or data.get('id', 'Документ'),
                    "source_url": data.get('url') or data.get('source_url', 'http://kremlin.ru'),
                    "id": data.get('id', f"{filename}_{i}"),
                    "local_img": data.get('local_img', ''),
                }

                # 🔥 Исключаем мусор из промпта LLM
                EXCLUDE_EMBED = ["id", "source_url", "local_img"]

                # Список полей, которые НЕ ДОЛЖНА видеть нейронка (LLM)
                EXCLUDE_LLM = ["id", "local_img", "source_url"]

                doc = Document(
                    text=text,
                    metadata=metadata,
                    excluded_embed_metadata_keys=EXCLUDE_EMBED, # ТЕПЕРЬ ПОИСК ВИДИТ ЗАГОЛОВКИ
                    excluded_llm_metadata_keys=EXCLUDE_LLM 
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"⚠️ Ошибка в {filename} (строка {i}): {e}")

# ========================================
# 3. ЗАЛИВКА В QDRANT
# ========================================
if not documents:
    logger.error("❌ Нечего индексировать.")
    exit()

logger.info(f"🛰 Коннект к Qdrant (localhost:6333)...")
q_client = qdrant_client.QdrantClient(host="localhost", port=6333)

# 🔥 СНОСИМ СТАРУЮ КОЛЛЕКЦИЮ (Зачистка "хуйни")
logger.info(f"💣 Пересоздание коллекции {COLLECTION_NAME}...")
q_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=rest_models.VectorParams(
        size=1024, # СТРОГО ПОД RoSBERTa LARGE
        distance=rest_models.Distance.COSINE
    )
)

vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME, client=q_client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Создаем индекс и пушим данные
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

q_client.close()
print("\n" + "="*50)
print(f"✅ БЕТОН ЗАЛИТ В QDRANT ЗАНОВО!")
print(f"📈 Всего объектов в базе: {len(documents)}")
print("="*50)
