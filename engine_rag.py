import os, glob, json, logging, asyncio
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate

# 1. НАСТРОЙКА ЛОГГЕРА (Уровень INFO для ROG терминала)
logger = logging.getLogger(__name__)

# СТРОГИЙ ОФФЛАЙН
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import os

# Определяем корень проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ПРЯМОЙ ПУТЬ к новой базе (e5-base), которую создал скрипт s5_base.py
MODEL_PATH = os.path.join(BASE_DIR, "hf_cache", "multilingual-e5-base")

# Проверка для подстраховки
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ ВНИМАНИЕ: Папка {MODEL_PATH} не найдена! Проверь запуск s5_base.py")
else:
    print(f"🚀 ENGINE UPGRADE: Использую e5-base из {MODEL_PATH}")


PERSIST_DIR = os.path.join(BASE_DIR, "fns_rag_index")

# 2. ШАБЛОНЫ ПРОМПТОВ (Стиль "Строгий Аналитик")
qa_prompt_str = (
    "Ты — ведущий эксперт-консультант ФНС России. Ответы СТРОГО на РУССКОМ ЯЗЫКЕ.\n"
    "ЗАПРЕЩЕНО: Использовать любые языки, кроме русского. ЗАПРЕЩЕНО придумывать цифры.\n"
    "---------------------\n"
    "БАЗА ЗНАНИЙ (КОНТЕКСТ):\n"
    "{context_str}\n"
    "---------------------\n"
    "ИНСТРУКЦИЯ ПО ФОРМАТУ (НЕ ВЫВОДИТЬ В ОТВЕТ):\n"
    "1. ЛИМИТ: Максимум 3-4 емких пункта (до 500 символов).\n"
    "2. Базируйся ТОЛЬКО на БАЗЕ ЗНАНИЙ. Если там есть ФИО (например, Авдокушин) — используй это.\n"
    "3. Оформляй красиво: списки, жирный шрифт. Без воды и вступлений.\n"
    "4. Если в вопросе аббревиатура (ЦКП, ЕКП) — расшифровку бери СТРОГО из контекста.\n"
    "5. Если вопрос пользователя СОВЕРШЕННО не касается тематики ФНС, ИРМ или кадровых процессов (например, про еду, погоду или котиков), отвечай вежливо: 'БАЗА_ПУСТА: Интересный вопрос, но в моих инструкциях об этом ничего нет. Я эксперт только по вопросам ФНС'\n"
    "6. ГРАФИКИ: Если в контексте есть цифровые данные для сравнения, ОБЯЗАТЕЛЬНО рисуй график.\n"
    "   ВАЖНО: Никогда не пиши '📊 График построен' как текст! Вместо этого используй ТЕГИ:\n"
    "\n"
    "   [CHART_JSON]\n"
    "   {\n"
    "     \"title\": { \"text\": \"Название графика\" },\n"
    "     \"xAxis\": { \"type\": \"category\", \"data\": [\"Категория1\", \"Категория2\"] },\n"
    "     \"yAxis\": { \"type\": \"value\" },\n"
    "     \"series\": [{ \"type\": \"bar\", \"data\": [10, 20] }]\n"
    "   }\n"
    "   [/CHART_JSON]\n"
    "\n"
    "   Для круговой диаграммы:\n"
    "   [CHART_JSON]\n"
    "   {\n"
    "     \"title\": { \"text\": \"Название\" },\n"
    "     \"series\": [{\n"
    "       \"type\": \"pie\",\n"
    "       \"data\": [\n"
    "         { \"value\": 45678, \"name\": \"НДС\" },\n"
    "         { \"value\": 38234, \"name\": \"Налог на прибыль\" }\n"
    "       ]\n"
    "     }]\n"
    "   }\n"
    "   [/CHART_JSON]\n"
    "\n"
    "   ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА С ГРАФИКОМ:\n"
    "   Статистика посещений по месяцам:\n"
    "   [CHART_JSON]{\"title\":{\"text\":\"Посещения по месяцам\"},\"xAxis\":{\"type\":\"category\",\"data\":[\"Янв\",\"Фев\",\"Мар\"]},\"yAxis\":{\"type\":\"value\"},\"series\":[{\"type\":\"bar\",\"data\":[12000,15400,13200]}]}[/CHART_JSON]\n"
    "   Наибольшая активность в феврале.\n"
    "\n"
    "7. ЗАПРЕЩЕНО писать фразу '📊 График построен' как обычный текст. График строится автоматически через теги.\n"
    "8. ЗАПРЕЩЕНО выдумывать свои теги вроде [LIST] или [INFO]. Списки оформляй СТРОГО через стандартный Markdown (маркеры '-' или цифры '1.').\n"
    "\n"
    "ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query_str}\n\n"
    "РЕЗУЛЬТАТ (сразу к сути на русском языке, без упоминания правил):"
)

qa_prompt = PromptTemplate(qa_prompt_str)

# 3. ЕДИНЫЕ НАСТРОЙКИ
Settings.embed_model = HuggingFaceEmbedding(
    model_name=MODEL_PATH, 
    device="cpu")
    

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_container:11434")

Settings.llm = Ollama(
    model="engine_load:latest", 
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    additional_kwargs={
        "keep_alive": "-1", 
        "num_ctx": 2048, 
        "temperature": 0.1, 
        "num_gpu": 99, 
        "num_predict":600}
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

async def run_in_thread(func, *args):
    """Запускает функцию в отдельном потоке, не блокируя event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

# 5. ГЛАВНАЯ ФУНКЦИЯ СТРИМИНГА (Без тормозов!)
async def get_ai_streaming_response(query_text: str):
    MIN_SCORE_THRESHOLD = 0.60 
    
    try:
        # ШАГ А: Используем обычный query (не aquery) для стриминга
        # Потому что async стриминг сломан в llama_index
      
        response = await run_in_thread(query_engine.query, query_text)
        nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        sources = []
        seen_urls = set()
        first_img = None
        has_real_context = False

        logger.info(f"🧩 Найдено чанков: {len(nodes)}")
        
        # ШАГ Б: Анализ чанков
        for i, node in enumerate(nodes):
            score = node.score if hasattr(node, 'score') else 0.0
            title = node.node.metadata.get("title", "Инструкция")
            
            if score >= MIN_SCORE_THRESHOLD:
                has_real_context = True
                if not first_img:
                    first_img = node.node.metadata.get("local_img")
                
                u = node.node.metadata.get("source_url")
                if u and u not in seen_urls:
                    sources.append({"url": u, "title": title})
                    seen_urls.add(u)

        # ШАГ В: Метаданные
        yield json.dumps({
            "type": "metadata", 
            "sources": sources,
            "has_answer": has_real_context,
            "image": first_img,
        }, ensure_ascii=False) + "\n"

        # ШАГ Г: Стриминг ответа
        if hasattr(response, "response_gen"):
            # Обычный стриминг (работает!)
            for token in response.response_gen:
                yield json.dumps({"type": "text", "content": token}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)  # Позволяем другим задачам выполняться
        else:
            # Если стрим не завелся - отдаем всё, что есть
            content = str(response).strip()
            if content:
                yield json.dumps({"type": "text", "content": content}) + "\n"
            
    except Exception as e:
        logger.error(f"❌ ОШИБКА ГЕНЕРАТОРА: {str(e)}")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"

#Бот мультик в телеграмм
async def get_ai_response_full(query_text: str):
    """Специально для ТГ-бота: возвращает словарь для красивого вывода"""
    try:
        response = await query_engine.astream_query(query_text)
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