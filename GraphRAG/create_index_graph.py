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
logger = logging.getLogger("GraphRAG_Indexer_V2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "hf_cache/multilingual-e5-large")

if not os.path.exists(MODEL_PATH):
    print(f"❌ МОДЕЛЬ НЕ НАЙДЕНА ПО ПУТИ: {MODEL_PATH}")
    exit()

# ========================================
# 1. КАСТОМНАЯ ЭМБЕДДИНГ-МОДЕЛЬ
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

Settings.node_parser = TokenTextSplitter(chunk_size=8192, chunk_overlap=0)

# ========================================
# 3. ЗАГРУЗКА ДАННЫХ
# ========================================
print("🔄 Читаем файлы GraphRAG v2...")

dataset_files = ["graph_nodes.jsonl", "graph_summaries.jsonl"]
documents = []

for filename in dataset_files:
    if not os.path.exists(filename):
        print(f"⚠️ Файл {filename} не найден, пропускаем...")
        continue
    
    print(f"📄 Обработка {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue 
            try:
                data = json.loads(line)
                doc_text = data.get('text', '')
                title = data.get('title') or data.get('id', 'Документ ФНС')
                
                # 🔥 ИСПРАВЛЕНИЕ: Обрабатываем оба случая
                raw_graph = data.get('graph_data')
                
                # Если graph_data пустой — пробуем достать метаданные из summary
                if not raw_graph:
                    metadata_field = data.get('metadata', {})
                    nodes = metadata_field.get('nodes', [])
                    if nodes:
                        raw_graph = {"entities": [{"name": n, "description": ""} for n in nodes]}
                
                # Формируем graph_info
                if isinstance(raw_graph, dict):
                    entities = raw_graph.get('entities', [])
                    nodes_list = []
                    for e in entities:
                        if isinstance(e, dict):
                            name = e.get('name', '')
                            desc = e.get('description', '')
                            if desc:
                                nodes_list.append(f"{name} ({desc})")
                            else:
                                nodes_list.append(name)
                        else:
                            nodes_list.append(str(e))
                    graph_info = "СУЩНОСТИ И РОЛИ: " + "; ".join(nodes_list)  # лимит 20 сущностей
                elif isinstance(raw_graph, list):
                    graph_info = "КЛЮЧЕВЫЕ ТЕМЫ РАЗДЕЛА: " + ", ".join([str(n) for n in raw_graph])
                else:
                    graph_info = ""

                metadata = {
                    "graph_structure": graph_info,
                    "title": title,
                    "source_url": data.get('url', ''),
                    "id": data.get('id', ''),
                    "local_img": data.get('local_img', ''),
                }

                # Текст для поиска
                if data.get('type') == 'person' or 'fns_' in data.get('id', ''):
                    text_for_index = f"{title}\n{doc_text}"
                else:
                    text_for_index = doc_text

                doc = Document(
                    text=text_for_index,
                    metadata=metadata,
                    excluded_embed_metadata_keys=["graph_structure", "source_url", "id", "local_img"]
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"⚠️ Ошибка в {filename} строка {i}: {e}")

# ========================================
# 4. ИНДЕКСАЦИЯ
# ========================================
print(f"\n🧠 Индексируем {len(documents)} объектов...")
index = VectorStoreIndex.from_documents(documents, show_progress=True)

PERSIST_DIR = ".././fns_rag_graph_final"
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

index.storage_context.persist(persist_dir=PERSIST_DIR)

print("\n" + "="*40)
print(f"✅ ФИНАЛ! GraphRAG v2 индекс готов.")
print(f"📍 Папка: {PERSIST_DIR}")
print(f"📊 Всего документов: {len(documents)}")
print("="*40)