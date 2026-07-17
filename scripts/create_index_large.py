import os
import json
import logging
import torch

# 1. СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import TokenTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Indexer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hf_cache/multilingual-e5-large")

if not os.path.exists(MODEL_PATH):
    print(f"❌ МОДЕЛЬ НЕ НАЙДЕНА ПО ПУТИ: {MODEL_PATH}")
    exit()

# ========================================
# 1. КАСТОМНАЯ ЭМБЕДДИНГ-МОДЕЛЬ С ПРЕФИКСАМИ
# ========================================
class PrefixedEmbedding(HuggingFaceEmbedding):
    """
    Добавляет префиксы для multilingual-e5:
    - 'query: ' для поисковых запросов
    - 'passage: ' для документов
    """
    def _get_text_embedding(self, text: str):
        """Для индексации документов — добавляем 'passage: '"""
        return super()._get_text_embedding("passage: " + text)
    
    def _get_query_embedding(self, query: str):
        """Для поиска — добавляем 'query: ' (на случай если используется напрямую)"""
        return super()._get_query_embedding("query: " + query)
    
    async def _aget_text_embedding(self, text: str):
        return await super()._aget_text_embedding("passage: " + text)
    
    async def _aget_query_embedding(self, query: str):
        return await super()._aget_query_embedding("query: " + query)

# ========================================
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДОКУМЕНТОВ
# ========================================
print("🔄 Читаем JSONL и делаем 'Тройное усиление' заголовков...")

dataset_file = "fns_100_examples.jsonl"
documents = []

with open(dataset_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line: continue 
        try:
            ex = json.loads(line)
            doc_text = ex.get('text', '')
            title = ex.get("title", "Инструкция ФНС")
            
            # Чистый контент без лишнего шума
            full_content = f"НАЗВАНИЕ: {title}\nСОДЕРЖАНИЕ: {doc_text}"

            metadata = {
                "id": ex.get('id', ''),
                "source_url": ex.get('url', ''),
                "title": title,
                "local_img": ex.get("local_img", ""),
                "type": ex.get("type", "")
            }

            doc = Document(
                text=full_content,
                metadata=metadata
            )
            documents.append(doc)
        except Exception as e:
            print(f"⚠️ Ошибка в строке {i}: {e}")

# ========================================
# 3. НАСТРОЙКА RAG (БЕТОННЫЙ ОФФЛАЙН)
# ========================================
print("\n⚙️ Настройка эмбеддингов и сплиттера...")

# ИСПОЛЬЗУЕМ КАСТОМНУЮ МОДЕЛЬ С ПРЕФИКСАМИ
Settings.embed_model = PrefixedEmbedding(
    model_name=MODEL_PATH, 
    device="cuda",
    local_files_only=True
)

# Settings.llm = Ollama(
#     model="yagpt5:latest", 
#     context_window=2048, 
#     base_url="http://localhost:11434",
#     request_timeout=300.0
# )

# УВЕЛИЧИВАЕМ OVERLAP: чтобы соседние куски лучше "помнили" друг друга
# Settings.node_parser = SentenceSplitter(
#     chunk_size=512,
#     chunk_overlap=50,
#     separator="\n"
# )
Settings.node_parser = TokenTextSplitter(
    chunk_size=4096, 
    chunk_overlap=0
)

# ========================================
# 4. ИНДЕКСАЦИЯ
# ========================================
print("🧠 Индексация пошла...")
index = VectorStoreIndex.from_documents(documents, show_progress=True)

PERSIST_DIR = "./fns_rag_index"
# Удаляем старый индекс перед сохранением, чтобы не было "каши"
if os.path.exists(PERSIST_DIR):
    import shutil
    shutil.rmtree(PERSIST_DIR)

index.storage_context.persist(persist_dir=PERSIST_DIR)
print(f"✅ ГОТОВО! Индекс обновлен с префиксами 'passage:' для документов.")