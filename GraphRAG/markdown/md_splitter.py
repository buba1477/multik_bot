#!/usr/bin/env python3
"""
GraphRAG Ultimate — из папки с MD в векторный индекс за один проход
Поддерживает: законы, указы, положения, приказы, любой формат
"""

import os
import re
import json
import time
import logging
import requests
import shutil
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
import community as community_louvain
from tqdm import tqdm
from threading import Semaphore

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from transformers import AutoTokenizer

# ========== КОНФИГ ==========
DATA_DIR = "../../embendings/"
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/multilingual-e5-large"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "yagpt5_4096:latest"
PERSIST_DIR = ".././fns_rag_graph_final_new"
CHECKPOINT_DIR = "./graphrag_checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

CHUNK_TOKENS = 450                    # Безопасный запас (лимит E5 = 512)
CHUNK_OVERLAP = 50
OLLAMA_TIMEOUT = 120
OLLAMA_RETRIES = 2
OLLAMA_WORKERS = 1                    # Один поток — стабильнее

# ========== ТОКЕНИЗАТОР E5 (ДЛЯ ТОЧНОЙ НАРЕЗКИ) ==========
logger = logging.getLogger("GraphRAG_Ultimate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Загружаем ТОЧНЫЙ токенизатор модели E5
logger.info("🔧 Загрузка токенизатора E5...")
e5_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Создаём сплиттер С ПЕРЕДАЧЕЙ токенизатора E5!
text_splitter = TokenTextSplitter(
    chunk_size=CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP,
    tokenizer=e5_tokenizer.encode  # 👈 ГЛАВНОЕ! Теперь чанки точно влезут в E5
)
logger.info("✅ Токенизатор E5 привязан к сплиттеру")

# ========== КАСТОМНЫЙ ЭМБЕДДЕР С ПРЕФИКСАМИ ==========
class PrefixedEmbedding(HuggingFaceEmbedding):
    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding("passage: " + text)
    
    def _get_query_embedding(self, query: str):
        return super()._get_query_embedding("query: " + query)
    
    async def _aget_text_embedding(self, text: str):
        return await super()._aget_text_embedding("passage: " + text)
    
    async def _aget_query_embedding(self, query: str):
        return await super()._aget_query_embedding("query: " + query)

logger.info("⚙️ Настройка эмбеддингов E5 с префиксами (passage:/query:)...")
Settings.embed_model = PrefixedEmbedding(
    model_name=MODEL_PATH, 
    device="cuda",
    local_files_only=True
)
Settings.llm = None

# ========== УТИЛИТЫ ==========
def clean_json(raw: str) -> Dict:
    raw = raw.strip().replace('```json', '').replace('```', '')
    start, end = raw.find('{'), raw.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end+1])
        except:
            pass
    return {"entities": [], "relationships": []}

def detect_doc_type(content: str) -> str:
    if "Федеральный закон" in content and "№ 79-ФЗ" in content:
        return "law_79fz"
    if "Указ Президента" in content:
        return "ukaz"
    if "ПОЛОЖЕНИЕ" in content and "УТВЕРЖДЕНО" in content:
        return "polozhenie"
    if "ПРИКАЗ" in content[:200]:
        return "prikaz"
    return "document"

def check_ollama_health() -> bool:
    logger.info("🔍 Проверка подключения к Ollama...")
    try:
        resp = requests.post(
            OLLAMA_URL, 
            json={"model": OLLAMA_MODEL, "prompt": "ping", "stream": False}, 
            timeout=10
        )
        if resp.status_code == 200:
            logger.info(f"✅ Ollama OK, модель {OLLAMA_MODEL} доступна")
            return True
    except Exception as e:
        logger.error(f"❌ Ollama недоступна: {e}")
    return False

# ========== СБОР ЧАНКОВ ИЗ ДОКУМЕНТА ==========
def parse_document(file_path: str) -> List[Document]:
    path = Path(file_path)
    content = open(file_path, 'r', encoding='utf-8').read()
    doc_name = path.stem
    doc_type = detect_doc_type(content)
    
    logger.info(f"  📄 {doc_name} [{doc_type}]")
    
    # Универсальный поиск секций (статьи, пункты, разделы)
    sections = re.split(r'\n(?=###\s+Статья|Статья\s+\d+|#\s+Глава|\d+\.\s+[А-Я])', content)
    all_chunks = []
    
    for sec in sections:
        sec = sec.strip()
        if not sec or len(sec) < 50:
            continue
        
        lines = sec.split('\n')
        header = lines[0].replace('#', '').strip()
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        
        if len(header) > 100:
            header = header[:100]
        
        # Используем сплиттер с токенизатором E5
        body_chunks = text_splitter.split_text(body) if body else [body]
        
        for i, chunk in enumerate(body_chunks):
            full_text = f"ИСТОЧНИК: {doc_name}\n{header}\n\n{chunk}"
            doc = Document(
                text=full_text,
                metadata={
                    "id": f"{doc_name}_{header[:30]}_{i}",
                    "doc_source": doc_name,
                    "title": header,
                    "doc_type": doc_type,
                    "chunk_idx": i,
                    "url": "http://kremlin.ru",
                }
            )
            all_chunks.append(doc)
    
    return all_chunks

# ========== ИЗВЛЕЧЕНИЕ ГРАФОВ (С ПРОГРЕСС-БАРОМ) ==========
def extract_graph_for_chunk(chunk: Document, semaphore: Semaphore) -> Document:
    node_id = f"{chunk.metadata['id']}_ch{chunk.metadata.get('chunk_idx', 0)}"
    
    with semaphore:
        cp_file = os.path.join(CHECKPOINT_DIR, f"{node_id}.json")
        if os.path.exists(cp_file):
            with open(cp_file, 'r') as f:
                chunk.metadata["graph_data"] = json.load(f)
            return chunk
        
        prompt = (
            "Ты аналитик ФНС. Извлеки сущности и связи из текста.\n"
            "Верни СТРОГО JSON:\n"
            "{\"entities\": [{\"name\": \"...\", \"description\": \"...\"}],\n"
            " \"relationships\": [{\"source\": \"...\", \"relation\": \"...\", \"target\": \"...\"}]}\n\n"
            f"ТЕКСТ:\n{chunk.text[:2000]}\n\nJSON:"
        )
        
        for attempt in range(OLLAMA_RETRIES):
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_ctx": 4096}},
                    timeout=OLLAMA_TIMEOUT
                )
                result = resp.json().get('response', '')
                if result:
                    data = clean_json(result)
                    chunk.metadata["graph_data"] = data
                    with open(cp_file, 'w') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    return chunk
            except Exception as e:
                if attempt < OLLAMA_RETRIES - 1:
                    time.sleep(2)
                else:
                    logger.error(f"❌ Ошибка чанка {node_id}: {e}")
        
        chunk.metadata["graph_data"] = {"entities": [], "relationships": []}
        return chunk

def enrich_all_chunks(chunks: List[Document]) -> List[Document]:
    semaphore = Semaphore(OLLAMA_WORKERS)
    logger.info(f"🤖 Обработка {len(chunks)} чанков через Ollama ({OLLAMA_WORKERS} потоков)...")
    
    enriched = []
    with ThreadPoolExecutor(max_workers=OLLAMA_WORKERS) as executor:
        futures = [executor.submit(extract_graph_for_chunk, ch, semaphore) for ch in chunks]
        
        with tqdm(total=len(futures), desc="🔄 Ollama", unit="чанк", ncols=80) as pbar:
            for future in as_completed(futures):
                enriched.append(future.result())
                pbar.update(1)
    
    logger.info(f"✅ Обработано {len(enriched)} чанков")
    return enriched

# ========== ПОСТРОЕНИЕ СООБЩЕСТВ И САММАРИ (ЛУВЕН) ==========
def build_communities(enriched_chunks: List[Document]) -> List[Document]:
    if not enriched_chunks:
        logger.warning("Нет данных для построения сообществ")
        return []
    
    logger.info("🔗 Собираем глобальный граф сущностей...")
    G = nx.Graph()
    descriptions = {}
    
    for chunk in enriched_chunks:
        graph_data = chunk.metadata.get("graph_data", {})
        if not graph_data:
            continue
        
        # 👇 ПРОВЕРКА ТИПА: если graph_data — список, пропускаем
        if isinstance(graph_data, list):
            logger.warning(f"Пропущен чанк с graph_data в виде списка")
            continue
        
        # 👇 ПРОВЕРКА: entities и relationships должны быть списками
        entities = graph_data.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        
        relationships = graph_data.get("relationships", [])
        if not isinstance(relationships, list):
            relationships = []
        
        for ent in entities:
            if isinstance(ent, dict) and ent.get("name"):
                name = ent["name"]
                desc = ent.get("description", "")
                if desc is None:
                    desc = ""
                G.add_node(name)
                if len(desc) > len(descriptions.get(name, "")):
                    descriptions[name] = desc
        
        for rel in relationships:
            if isinstance(rel, dict):
                s = rel.get("source")
                t = rel.get("target")
                if s and t and not isinstance(s, list) and not isinstance(t, list):
                    G.add_edge(s, t)
    
    if G.number_of_nodes() == 0:
        logger.warning("Граф пуст — Лувен не выполнен")
        return []
    
    logger.info(f"📊 Узлов: {G.number_of_nodes()}, связей: {G.number_of_edges()}")
    logger.info("🧮 Вычисляем сообщества (Лувен)...")
    partition = community_louvain.best_partition(G)
    
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
    
    logger.info(f"🎯 Найдено {len(communities)} сообществ")
    
    summaries = []
  
    
        # 1. Сортируем сообщества по размеру (самые большие — вперед)
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 2. Берем только те, где больше 3 узлов, и не более 50-70 штук суммарно
    # Этого хватит, чтобы покрыть все основные темы закона
    target_communities = [item for item in sorted_communities if len(item[1]) > 3][:60]
    
    logger.info(f"🎯 Выбрано {len(target_communities)} значимых сообществ для суммаризации (из {len(communities)})")

    # Для саммари лучше держать 1 поток, чтобы карта не ловила тупняка
    summary_semaphore = Semaphore(1) 

    for cid, nodes_list in target_communities:
        # Берем до 20 ключевых узлов для контекста
        context = "\n".join([f"- {n}: {descriptions.get(n, 'нет описания')}" for n in nodes_list[:20]])
        
        prompt = (
            f"Ты аналитик ФНС. Напиши ОДИН связный аналитический обзор (3-5 предложений) для этого блока понятий.\n"
            f"Объясни, как они связаны в системе госслужбы.\n"
            f"СПИСОК ПОНЯТИЙ:\n{context}\n\n"
            f"ОТВЕТ (только текст):"
        )
        
        summary_text = ""
        with summary_semaphore:
            for attempt in range(OLLAMA_RETRIES):
                try:
                    resp = requests.post(
                        OLLAMA_URL, 
                        json={
                            "model": OLLAMA_MODEL, 
                            "prompt": prompt, 
                            "stream": False, 
                            "options": {"temperature": 0.1, "num_ctx": 4096}
                        }, 
                        timeout=OLLAMA_TIMEOUT
                    )
                    summary_text = resp.json().get('response', '').strip()
                    if summary_text:
                        break
                except Exception as e:
                    logger.warning(f"Ошибка саммари сообщества {cid}: {e}")
                    time.sleep(2)
        
        # Если всё-таки пусто - не создаем пустую ноду
        if not summary_text or "Ошибка" in summary_text:
            continue

        doc = Document(
            text=summary_text,
            metadata={
                "id": f"community_{cid}",
                "title": f"Аналитический обзор раздела {cid}",
                "type": "community_summary",
                # Сохраняем список узлов в метаданные для поиска
                "graph_structure": nodes_list, 
                "url": "http://kremlin.ru"
            }
        )
        summaries.append(doc)
        logger.info(f"   ✅ Сообщество {cid} ({len(nodes_list)} узлов) — обзоры готовы.")

    return summaries

# ========== ГЛАВНЫЙ ПАЙПЛАЙН ==========
def main():
    logger.info("="*60)
    logger.info("🚀 ЗАПУСК GRAPHRAG ULTIMATE")
    logger.info("="*60)
    
    # if not check_ollama_health():
    #     logger.error("❌ Ollama недоступна. Запустите: ollama serve")
    #     exit(1)
    
    logger.info("📁 Сканирование документов...")
    all_chunks = []
    md_files = list(Path(DATA_DIR).glob("*.md")) + list(Path(DATA_DIR).glob("*.pdf"))
    
    logger.info(f"📁 Найдено {len(md_files)} файлов")
    for file in md_files:
        all_chunks.extend(parse_document(str(file)))
    
    logger.info(f"✂️ Создано {len(all_chunks)} чанков")
    
    enriched = enrich_all_chunks(all_chunks)

    # 👇 ДОБАВИТЬ ПРОВЕРКУ
    if not enriched:
        logger.error("❌ Нет обогащенных чанков! Что-то пошло не так.")
        exit(1)
    
    summaries = build_communities(enriched)
    logger.info(f"📚 Создано {len(summaries)} саммари сообществ")
    
    final_docs = enriched + summaries
    logger.info(f"🧠 Индексируем {len(final_docs)} объектов...")
    
    index = VectorStoreIndex.from_documents(
        final_docs,
        show_progress=True,
        transformations=[TokenTextSplitter(chunk_size=15000, chunk_overlap=0)], 
        excluded_embed_metadata_keys=["graph_data", "graph_structure", "chunk_idx", "url"]
    )
    
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    
    logger.info("="*60)
    logger.info(f"✅ ГОТОВО! Индекс сохранён в {PERSIST_DIR}")
    logger.info(f"   — {len(enriched)} чанков законов")
    logger.info(f"   — {len(summaries)} аналитических обзоров")
    logger.info("="*60)

if __name__ == "__main__":
    main()