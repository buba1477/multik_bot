import os
import json
import logging
import shutil
import torch

# 1. СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import TokenTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphRAG_Indexer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Путь на папку выше к модели
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "././hf_cache/multilingual-e5-large")

if not os.path.exists(MODEL_PATH):
    print(f"❌ МОДЕЛЬ НЕ НАЙДЕНА ПО ПУТИ: {MODEL_PATH}")
    exit()

# ========================================
# 1. КАСТОМНАЯ ЭМБЕДДИНГ-МОДЕЛЬ С ПРЕФИКСАМИ
# ========================================
class PrefixedEmbedding(HuggingFaceEmbedding):
    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding("passage: " + text)
    
    def _get_query_embedding(self, query: str):
        return super()._get_query_embedding("query: " + query)
    
    async def _aget_text_embedding(self, text: str):
        return await super()._aget_text_embedding("passage: " + text)
    
    async def _aget_query_embedding(self, query: str):
        return await super()._aget_query_embedding("query: " + query)

# ========================================
# 2. НАСТРОЙКА RAG
# ========================================
print("\n⚙️ Настройка эмбеддингов (CUDA)...")

Settings.embed_model = PrefixedEmbedding(
    model_name=MODEL_PATH, 
    device="cuda",
    local_files_only=True
)

# Чтобы LlamaIndex не резала наши готовые объекты
Settings.node_parser = TokenTextSplitter(chunk_size=4096, chunk_overlap=0)

# ========================================
# 3. ЗАГРУЗКА ДАННЫХ (СТАТЬИ + САММАРИ)
# ========================================
print("🔄 Читаем файлы GraphRAG...")

# Читаем оба твоих файла
dataset_files = ["graph_nodes.jsonl", "graph_summaries.jsonl"]
documents = []

for filename in dataset_files:
    if not os.path.exists(filename):
        print(f"⚠️ Файл {filename} не найден, пропускаем...")
        continue
    
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue 
            try:
                data = json.loads(line)
                
                # Текст для поиска
                doc_text = data.get('text', '')
                title = data.get('title') or "Обзор раздела (Graph Summary)"
                
                # Собираем всё мясо в метаданные
                # Если это саммари, берем nodes, если статья — graph_data
                graph_info = data.get('graph_data') or data.get('metadata', {}).get('nodes', '')

                metadata = {
                    "id": data.get('id', ''),
                    "source_url": data.get('url', 'URL отсутствует (глобальное знание)'),
                    "title": title,
                    "graph_structure": str(graph_info)
                }

                # Формируем документ
                doc = Document(
                    text=f"НАЗВАНИЕ: {title}\nСОДЕРЖАНИЕ: {doc_text}",
                    metadata=metadata,
                    # ГРАФ ИСКЛЮЧАЕМ ИЗ ЭМБЕДДИНГА, чтобы влезть в 512 токенов
                    excluded_embed_metadata_keys=["graph_structure", "source_url", "id"]
                )
                documents.append(doc)
            except Exception as e:
                print(f"⚠️ Ошибка в {filename} строка {i}: {e}")

# ========================================
# 4. ИНДЕКСАЦИЯ
# ========================================
print(f"🧠 Индексируем {len(documents)} объектов...")
index = VectorStoreIndex.from_documents(documents, show_progress=True)

PERSIST_DIR = ".././fns_rag_graph_final"
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

index.storage_context.persist(persist_dir=PERSIST_DIR)
print(f"✅ ФИНАЛ! Твой GraphRAG индекс собран в {PERSIST_DIR}")
