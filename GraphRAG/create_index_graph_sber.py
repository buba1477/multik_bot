import os
import json
import logging
import shutil
import torch
from pathlib import Path
from typing import List

# --- СТРОГИЙ ОФФЛАЙН ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from sentence_transformers import SentenceTransformer, models

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sber_RoSBERTa_Core_Final")

# Пути
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "hf_cache" / "ru-en-RoSBERTa"
PERSIST_DIR = BASE_DIR.parent / "fns_rag_graph_final_sber_new"

if not MODEL_PATH.exists():
    logger.error(f"❌ МОДЕЛЬ НЕ НАЙДЕНА: {MODEL_PATH}")
    exit()

# ========================================
# 1. КАСТОМНЫЙ КЛАСС ЭМБЕДДИНГА (CLS + ПРЕФИКСЫ)
# ========================================
class SberRoSBERTaEmbedding(BaseEmbedding):
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        logger.info(f"✨ Инициализация CORE [CLS Pooling] на {device}")
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

# Настройка глобальных параметров
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Settings.embed_model = SberRoSBERTaEmbedding(model_path=str(MODEL_PATH), device=DEVICE)
Settings.llm = None  # В индексаторе LLM не нужна
Settings.node_parser = None # Отключаем стандартный парсер

# ========================================
# 2. ЗАГРУЗКА И ОБОГАЩЕНИЕ ДАННЫХ
# ========================================
logger.info("🔄 Загрузка и подготовка документов...")

dataset_files = ["graph_nodes.jsonl", "graph_summaries.jsonl"]
documents = []

for filename in dataset_files:
    file_path = BASE_DIR / filename
    if not file_path.exists():
        logger.warning(f"⚠️ Файл {filename} не найден")
        continue
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                doc_text = data.get('text', '')
                if not doc_text: continue
                
                title = data.get('title') or data.get('id', 'Документ')
                
                # Извлечение структуры графа (лимит ТОП-7 для экономии места)
                raw_graph = data.get('graph_data')
                if not raw_graph:
                    nodes = data.get('metadata', {}).get('nodes', [])
                    if nodes:
                        raw_graph = {"entities": [{"name": n, "description": ""} for n in nodes]}
                
                graph_info = ""
                if isinstance(raw_graph, dict):
                    entities = raw_graph.get('entities', [])
                    nodes_list = [f"{e['name']} ({e.get('description', '')[:50]})" if isinstance(e, dict) else str(e) for e in entities[:7]]
                    graph_info = "СУЩНОСТИ: " + "; ".join(nodes_list)

                # Метаданные (с исправленным source_url для API)
                metadata = {
                    "graph_structure": graph_info,
                    "title": title,
                    "source_url": data.get('url') or data.get('source_url', 'http://kremlin.ru'),
                    "id": data.get('id', f"chunk_{i}"),
                    "local_img": data.get('local_img', ''), # 🔥 ВОЗВРАЩАЕМ КАРТИНКИ
                }

                doc = Document(
                    text=doc_text,
                    metadata=metadata,
                    # Исключаем из вектора, чтобы не размывать смысл
                    excluded_embed_metadata_keys=["graph_structure", "source_url", "id", "local_img"]
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"⚠️ Ошибка в {filename} (строка {i}): {e}")

# ========================================
# 3. ИНДЕКСАЦИЯ ЧЕРЕЗ PIPELINE (БЕЗ РЕЗКИ)
# ========================================
if not documents:
    logger.error("❌ Документы не найдены.")
    exit()

logger.info(f"🧠 Запуск Ingestion Pipeline для {len(documents)} объектов...")

# Pipeline гарантирует, что трансформации (эмбеддинги) пройдут,
# а текст при этом НЕ будет разрезан сплиттером
pipeline = IngestionPipeline(transformations=[Settings.embed_model])
nodes = pipeline.run(documents=documents, show_progress=True)

# Создаем индекс из готовых узлов
index = VectorStoreIndex(nodes, show_progress=True)

if PERSIST_DIR.exists():
    logger.info(f"🗑 Очистка {PERSIST_DIR}")
    shutil.rmtree(PERSIST_DIR)

index.storage_context.persist(persist_dir=str(PERSIST_DIR))

print("\n" + "="*50)
print(f"✅ УСПЕХ! Ссылки сохранены, метаданные упакованы.")
print(f"📍 Индекс: {PERSIST_DIR}")
print("="*50)
