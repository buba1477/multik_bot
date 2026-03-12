import os
import glob
import json
import logging
import asyncio
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

# 1. НАСТРОЙКА ЛОГГЕРА (Уровень INFO для ROG терминала)
logger = logging.getLogger(__name__)

# СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. Указываем путь ДО папки со снапшотами (тут всё стабильно)
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "hf_cache", "models--intfloat--multilingual-e5-small", "snapshots")

# 2. ДИНАМИЧЕСКИЙ ПОИСК (Вместо твоей длинной строки)
try:
    # Ищем любую папку внутри snapshots
    found_snapshots = glob.glob(os.path.join(SNAPSHOTS_DIR, "*"))
    if found_snapshots:
        # Берем самый первый (или единственный) найденный хеш-путь
        MODEL_PATH = found_snapshots[0] 
        print(f"🚀 SOTA-загрузка: Нашел модель по пути {MODEL_PATH}")
    else:
        # Если папка пустая, пробуем стандартное имя (вдруг есть интернет)
        MODEL_PATH = "intfloat/multilingual-e5-small"
except Exception as e:
    MODEL_PATH = "intfloat/multilingual-e5-small"

PERSIST_DIR = os.path.join(BASE_DIR, "fns_rag_index")

# 2. ШАБЛОНЫ ПРОМПТОВ (Стиль "Строгий Аналитик")
qa_prompt_str = (
    "Ты — ведущий эксперт-консультант ФНС России. Твоя задача — давать полезные ответы, опираясь на предоставленные ИРМ.\n"
    "---------------------\n"
    "БАЗА ЗНАНИЙ (КОНТЕКСТ):\n"
    "{context_str}\n"
    "---------------------\n"
    "ТВОИ ПРАВИЛА:\n"
    "1. ЛИМИТ: Максимум 3-4 емких пункта или 500 символов.\n"
    "2. Твой ответ должен базироваться на информации из БАЗЫ ЗНАНИЙ. Используй факты, названия и инструкции оттуда.\n"
    "3. Если вопрос касается определений (что это?), а в тексте есть описание процесса или заголовки — СИНТЕЗИРУЙ ответ на их основе.\n"
    "4. 5. Если в БАЗЕ ЗНАНИЙ есть хотя бы ФИО или краткое упоминание — это считается информацией. Отвечай на основе того, что есть. Пиши 'ни слова, бро' только если в контексте ВООБЩЕ нет совпадений по теме.\n"
    "5. Оформляй ответ красиво: списки, жирный шрифт, таблицы.\n"
    "6. Избегай воды и повторных вступлений. Сразу к сути.\n"
    "7. Если есть список — выбери только 3-5 самых важных пункта.\n"
    "8. Не обрывай мысль, всегда дописывай до точки.\n"
    "ВОПРОС: {query_str}\n"
    "ОТВЕТ ЭКСПЕРТА:"
)
qa_prompt = PromptTemplate(qa_prompt_str)

# 3. ЕДИНЫЕ НАСТРОЙКИ
Settings.embed_model = HuggingFaceEmbedding(
    model_name=MODEL_PATH, 
    device="cpu", 
    local_files_only=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

Settings.llm = Ollama(
    model="engine_load:latest", 
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    additional_kwargs={
        "keep_alive": "-1", 
        "num_ctx": 2048, 
        "temperature": 0.1, 
        "num_gpu": 99, 
        "num_predict": 600}
)

# 4. ИНИЦИАЛИЗАЦИЯ (Загрузка в RAM)
logger.info("🧠 Загрузка векторного индекса в 32 ГБ RAM...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(
    similarity_top_k=5,
    streaming=True,
    use_async=True,
    text_qa_template=qa_prompt
)

# 5. ГЛАВНАЯ ФУНКЦИЯ СТРИМИНГА (Без тормозов!)
async def get_ai_streaming_response(query_text: str):
    # Минимальный порог, чтобы вообще считать, что инфа есть в базе
    MIN_SCORE_THRESHOLD = 0.60 
    
    try:
        # ШАГ А: Поиск в 32 ГБ RAM
        response = await query_engine.aquery(query_text)
        nodes = response.source_nodes
        
        sources = []
        seen_urls = set()
        first_img = None
        has_real_context = False

        logger.info(f"🧩 Найдено чанков: {len(nodes)}")
        
        # ШАГ Б: Анализ ТОП-3 чанков (без выноса мозга фильтрами)
        for i, node in enumerate(nodes):
            score = node.score if hasattr(node, 'score') else 0.0
            title = node.node.metadata.get("title", "Инструкция")
            
            logger.info(f"  [Node #{i+1}] {score:.4f} | {title}")

            # Если скор нормальный - берем в оборот
            if score >= MIN_SCORE_THRESHOLD:
                has_real_context = True
                
                # Картинка из самого сочного чанка
                if not first_img:
                    first_img = node.node.metadata.get("local_img")
                
                # Собираем ссылки (LlamaIndex уже отсортировал их по релевантности)
                u = node.node.metadata.get("source_url")
                if u and u not in seen_urls:
                    sources.append({"url": u, "title": title})
                    seen_urls.add(u)

        # ШАГ В: Моментальный выстрел метаданными
        yield json.dumps({
            "type": "metadata", 
            "sources": sources,
            "has_answer": has_real_context,
            "image": first_img
        }) + "\n"

        # ШАГ Г: Стриминг текста из 80W GPU (Прямой проброс токенов)
        if hasattr(response, "async_response_gen"):
            gen_attr = response.async_response_gen
            gen = gen_attr() if callable(gen_attr) else gen_attr
            
            async for text_chunk in gen:
                if text_chunk:
                    yield json.dumps({"type": "text", "content": text_chunk}) + "\n"
        else:
            yield json.dumps({"type": "text", "content": str(response)}) + "\n"
            
    except Exception as e:
        logger.error(f"❌ ОШИБКА ГЕНЕРАТОРА: {str(e)}")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"

#Бот мультик в телеграмм
async def get_ai_response_full(query_text: str):
    """Специально для ТГ-бота: возвращает словарь для красивого вывода"""
    try:
        response = await query_engine.aquery(query_text)
        nodes = response.source_nodes
        
        # Собираем ссылки (Score > 0.65)
        sources = []
        seen_urls = set()
        first_img = None
        
        for node in nodes:
            score = node.score if hasattr(node, 'score') else 0.0
            if score > 0.65:
                u = node.node.metadata.get("source_url")
                t = node.node.metadata.get("title", "Инструкция")
                img = node.node.metadata.get("local_img")
                
                if u and u not in seen_urls:
                    sources.append((u, t)) # Кортеж для InlineKeyboardButton
                    seen_urls.add(u)
                if img and not first_img:
                    first_img = img
        
        return {
            "answer": str(response),
            "sources": sources,
            "image": first_img
        }
    except Exception as e:
        logger.error(f"❌ Ошибка в RAG-движке: {e}")
        return {"answer": "В моих регламентах про это ни слова, бро.", "sources": [], "image": None}
