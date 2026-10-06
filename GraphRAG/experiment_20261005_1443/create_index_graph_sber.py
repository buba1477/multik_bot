import os
import json
import logging
import shutil
from pathlib import Path
from typing import List

# --- СТРОГИЙ ОФФЛАЙН ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FRIDA_Graph_Experiment")

# Пути
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / "hf_cache" / "FRIDA"
PERSIST_DIR = BASE_DIR / "fns_rag_graph_frida_experiment_enriched"

if not MODEL_PATH.exists():
    logger.error(f"❌ МОДЕЛЬ НЕ НАЙДЕНА: {MODEL_PATH}")
    exit()

# ========================================
# 1. КАСТОМНЫЙ КЛАСС ЭМБЕДДИНГА (CLS + ПРЕФИКСЫ)
# ========================================
class FRIDAEmbedding(BaseEmbedding):
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        logger.info(f"✨ Инициализация FRIDA [CLS Pooling] на {device}")
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
DEVICE = "cuda"  # Жёстко CPU — GPU занят Ollama
Settings.embed_model = FRIDAEmbedding(model_path=str(MODEL_PATH), device=DEVICE)
Settings.llm = None  # В индексаторе LLM не нужна
Settings.node_parser = None # Отключаем стандартный парсер

# ========================================
# 2. ЗАГРУЗКА И ОБОГАЩЕНИЕ ДАННЫХ
# ========================================
logger.info("🔄 Загрузка и подготовка документов...")

# Заранее загружаем токенизатор FRIDA для контроля длины
logger.info("🔤 Загрузка токенизатора FRIDA...")
frida_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
MAX_FRIDA_TOKENS = 512

dataset_files = ["graph_nodes.jsonl", "graph_summaries.jsonl"]
documents = []

# Счётчики для статистики
total_chunks = 0
empty_graph_count = 0
over_limit_count = 0
total_over_tokens = 0

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

                chunk_id = data.get('id', f'chunk_{i}')
                title = data.get('title') or chunk_id

                # --- Извлечение графа ---
                raw_graph = data.get('graph_data')
                if not raw_graph:
                    nodes = data.get('metadata', {}).get('nodes', [])
                    if nodes:
                        raw_graph = {"entities": [{"name": n, "description": ""} for n in nodes]}

                # graph_info — в metadata (для runtime), как было раньше
                graph_info = ""
                # enrich_suffix — в embedding через enriched_text
                enrich_suffix = ""

                if isinstance(raw_graph, dict):
                    entities = raw_graph.get('entities', [])
                    rels = raw_graph.get('relationships', [])

                    # Для metadata.graph_structure (топ-7, как было)
                    nodes_list = [f"{e['name']} ({e.get('description', '')[:50]})" if isinstance(e, dict) else str(e) for e in entities[:7]]
                    graph_info = "СУЩНОСТИ: " + "; ".join(nodes_list)

                    # Для enriched_text — коротко: топ-5 сущностей + топ-3 связи
                    enrich_parts = []
                    if entities:
                        names = [e.get('name', '') if isinstance(e, dict) else str(e) for e in entities[:5]]
                        enrich_parts.append("Сущности: " + ", ".join(names))
                    if rels:
                        rel_str = "; ".join(
                            f"{r.get('source', '')} {r.get('relation', '')} {r.get('target', '')}"
                            for r in rels[:3] if isinstance(r, dict)
                        )
                        enrich_parts.append(f"Связи: {rel_str}")
                    enrich_suffix = "\n".join(enrich_parts)

                if not enrich_suffix.strip():
                    enriched_text = doc_text
                else:
                    enriched_text = f"{doc_text}\n\n{enrich_suffix}"

                # --- Контроль длины через FRIDA tokenizer ---
                tok_len = len(frida_tokenizer.encode(enriched_text, add_special_tokens=False))
                if tok_len > MAX_FRIDA_TOKENS:
                    over_limit_count += 1
                    total_over_tokens += tok_len - MAX_FRIDA_TOKENS
                    logger.warning(
                        f"⚠️ {chunk_id}: текст={len(doc_text)}ч | граф={len(enrich_suffix)}ч | "
                        f"токенов={tok_len} > {MAX_FRIDA_TOKENS} "
                        f"(превышение на {tok_len - MAX_FRIDA_TOKENS})"
                    )

                # --- Логирование пустого графа ---
                if not enrich_suffix.strip():
                    empty_graph_count += 1

                total_chunks += 1

                # Метаданные (с исправленным source_url для API)
                metadata = {
                    "graph_structure": graph_info,
                    "title": title,
                    "source_url": data.get('url') or data.get('source_url', 'http://kremlin.ru'),
                    "id": chunk_id,
                    "local_img": data.get('local_img', ''), # 🔥 ВОЗВРАЩАЕМ КАРТИНКИ
                }

                doc = Document(
                    text=enriched_text,
                    metadata=metadata,
                    # graph_structure остаётся excluded — граф в embedding через enriched_text
                    excluded_embed_metadata_keys=["graph_structure", "source_url", "id", "local_img"]
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"⚠️ Ошибка в {filename} (строка {i}): {e}")

# --- Статистика ---
logger.info("=" * 50)
logger.info(f"📊 ИТОГО: {total_chunks} чанков")
logger.info(f"📊 Пустой граф: {empty_graph_count} чанков")
if over_limit_count > 0:
    pct_over = over_limit_count / total_chunks * 100 if total_chunks else 0
    avg_over = total_over_tokens / over_limit_count if over_limit_count else 0
    logger.warning(
        f"⚠️ Превышение {MAX_FRIDA_TOKENS} токенов: {over_limit_count} чанков "
        f"({pct_over:.1f}%), среднее превышение: {avg_over:.0f} токенов"
    )
    if pct_over > 50:
        logger.warning("⚠️ Более 50% чанков обрезаются! Уменьшить число сущностей/связей.")
else:
    logger.info("✅ Все чанки укладываются в лимит 512 токенов FRIDA")
logger.info("=" * 50)

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
