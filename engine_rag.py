import os, glob, json, logging, asyncio
import re
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from pathlib import Path

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
"Ты — ведущий эксперт-консультант ФНС России. Ответы на РУССКОМ языке.\n"
"---------------------\n"
"БАЗА ЗНАНИЙ:\n"
"{context_str}\n"
"---------------------\n"
"1. ПРАВИЛА ФОРМАТИРОВАНИЯ:\n"
" - Заголовки: используй ### для разделов.\n"
" - Списки: 1. для инструкций, • для перечислений.\n"
" - Акценты: **жирный шрифт** для терминов, ФИО и дат.\n"
" - Абзацы: всегда разделяй смысловые блоки пустой строкой (\\n\\n).\n"
" - Длинные ответы: разбивай на заголовок, вступление, список, итог.\n"
"\n"
"2. ТИП ОТВЕТА (выбирай первый подходящий):\n"
" А) ИНСТРУКЦИЯ — слова: как, инструкция, регистрация, поиск, сделать\n"
" → заголовок + список. Без таблиц и графиков.\n"
" Б) ГРАФИК — слова: график, диаграмма, круговая, линия, столбчатая, гистограмма, тренд\n"
" → выведи ТОЛЬКО заголовок (###) и [CHART_JSON] с валидным JSON.\n"
" → 🚨 КРИТИЧЕСКИ ВАЖНО: Если в базе знаний НЕТ числовых данных по запросу — НЕ рисуй график!\n"
" → Вместо этого выведи: 'БАЗА_ПУСТА: По данному запросу нет числовых данных для визуализации'\n"
" → Не выдумывай цифры, марсиан, инопланетян и прочие фантазии.\n"
" → В JSON используй ТОЛЬКО одинарные фигурные скобки { }, НЕ {{ }}.\n"
" → ПОСЛЕ JSON НЕ ДОБАВЛЯЙ обратные кавычки, точки, комментарии или любой другой текст.\n"
" → JSON должен заканчиваться ровно так: [/CHART_JSON]\n"
" В) ТАБЛИЦА — слова: таблица, топ, рейтинг, список, сводка\n"
" → выведи заголовок (###) и Markdown-таблицу. Без графика.\n"
" → Таблицу не разбивай абзацами, оставляй как есть.\n"
" Г) ТЕКСТ — во всех остальных случаях\n"
" → ответ с абзацами, используй форматирование из п.1.\n"
" → Обязательно разбивай текст на логические блоки с пустыми строками между ними.\n"
"\n"
"3. ФОРМАТЫ ГРАФИКОВ (ОДИНАРНЫЕ СКОБКИ { }, НЕ {{ }}):\n"
"\n"
" 🚨 КРИТИЧЕСКИ ВАЖНО: Закрывающий тег пиши ТОЛЬКО как [/CHART_JSON].\n"
" 🚨 НЕЛЬЗЯ писать [/CHRT_JSON], [/CHART], [/JSON] или любые другие варианты.\n"
" 🚨 Проверь перед отправкой: тег должен содержать букву 'A' в слове CHART.\n"
"\n"
" bar / line (СТРОГО: xAxis.data — массив названий, series.data — массив чисел):\n"
" [CHART_JSON]{\"title\":{\"text\":\"Название\"},\"xAxis\":{\"type\":\"category\",\"data\":[\"А\",\"Б\"]},\"yAxis\":{\"type\":\"value\"},\"series\":[{\"type\":\"bar\",\"data\":[45678,38234]}]}[/CHART_JSON]\n"
"\n"
" pie (СТРОГО: series[0].data — массив объектов с value и name):\n"
" [CHART_JSON]{\"title\":{\"text\":\"Название\"},\"series\":[{\"type\":\"pie\",\"data\":[{\"value\":45678,\"name\":\"Категория 1\"},{\"value\":38234,\"name\":\"Категория 2\"}]}]}[/CHART_JSON]\n"
"\n"
" scatter:\n"
" [CHART_JSON]{\"title\":{\"text\":\"Название\"},\"xAxis\":{\"type\":\"value\"},\"yAxis\":{\"type\":\"value\"},\"series\":[{\"type\":\"scatter\",\"data\":[[10,20],[15,25]]}]}[/CHART_JSON]\n"
"\n"
" ПРОВЕРКА ПЕРЕД ОТПРАВКОЙ:\n"
" - Открывающий тег: [CHART_JSON]\n"
" - Закрывающий тег: [/CHART_JSON] (именно так!)\n"
" - Для pie: структура { \"series\": [ { \"type\": \"pie\", \"data\": [ {...}, {...} ] } ] }\n"
" - В конце должно быть } (фигурная скобка), а не ] (квадратная)\n"
" - Проверь баланс: открывающих { столько же, сколько закрывающих }\n"
" - Проверь баланс: открывающих [ столько же, сколько закрывающих ]\n"
"\n"
"4. Если вопрос не по теме ФНС:\n"
" → 'БАЗА_ПУСТА: Я эксперт только по вопросам ФНС'\n"
"\n"
"5. Если спрашивают 'кто ты':\n"
" → 'Я твой персональный ИИ-помощник по базе знаний ФНС'\n"
"\n"
"ВОПРОС: {query_str}\n"
"ОТВЕТ:"
)

qa_prompt = PromptTemplate(qa_prompt_str)

# 3. ЕДИНЫЕ НАСТРОЙКИ
Settings.embed_model = HuggingFaceEmbedding(
    model_name=MODEL_PATH, 
    device="cpu")
    

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama_container:11434")

Settings.llm = Ollama(
    model="yagpt5_3072:latest", 
    base_url=OLLAMA_HOST,
    request_timeout=300.0,
    additional_kwargs={
        "keep_alive": "-1", 
        "num_ctx": 3072, 
        "temperature": 0, 
        "num_gpu": 99, 
        "num_predict":1200,
        "seed": 42}
)

# 4. ИНИЦИАЛИЗАЦИЯ (Загрузка в RAM)
logger.info("🧠 Загрузка векторного индекса в 32 ГБ RAM...")
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(
    similarity_top_k=6,
    streaming=True,
    use_async=True,
    text_qa_template=qa_prompt
)

async def run_in_thread(func, *args):
    """Запускает функцию в отдельном потоке, не блокируя event loop"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

async def get_ai_streaming_response(query_text: str):
    MIN_SCORE_THRESHOLD = 0.55 # Золотая середина для 1660 Ti
    query_lower = query_text.lower()
    
    try:
        # ШАГ А: Поиск в индексе (Оллама думает в отдельном потоке)
        response = await asyncio.to_thread(query_engine.query, query_text)
        nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        logger.info(f"🧩 Найдено чанков: {len(nodes)}")

        sources = []
        seen_urls = set()
        has_real_context = False
        full_response_text = ""

        # ШАГ Б: Фильтрация чанков (только чтобы Егоричев не лез к Егорову в контекст)
        filtered_nodes = []
        for node in nodes:
            score = node.score if hasattr(node, 'score') else 0.0
            content = node.get_content().lower()
            
            if "егоров" in query_lower and "егоричев" not in query_lower:
                if "егоричев" in content: continue

            if score >= MIN_SCORE_THRESHOLD:
                filtered_nodes.append(node)
                has_real_context = True
                
                u = node.node.metadata.get("source_url")
                t = node.node.metadata.get("title", "Источник")
                if u and u not in seen_urls:
                    sources.append({"url": u, "title": t})
                    seen_urls.add(u)

        # ШАГ В: Отправка Метаданных (Ссылки улетают сразу)
        yield json.dumps({
            "type": "metadata", 
            "sources": sources,
            "has_answer": has_real_context,
            "image": None 
        }, ensure_ascii=False) + "\n"

        # ШАГ Г: Стриминг текста (Собираем полный ответ для анализа)
        if has_real_context and hasattr(response, "response_gen"):
            for token in response.response_gen:
                full_response_text += token
                yield json.dumps({"type": "text", "content": token}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0.01)
        else:
            yield json.dumps({"type": "text", "content": "Информация не найдена."}, ensure_ascii=False) + "\n"
            return

        # --- ШАГ Д: УЛЬТРА-БЕЗОПАСНЫЙ ФОТО-ФИЛЬТР ---
        final_photo = None
        resp_lower = full_response_text.lower()
        
        # 1. ГЛАВНЫЙ БЛОКИРОВЩИК (Защита от Путина/Внешних вопросов)
        # Если сработал триггер "База пуста" — ВЫХОДИМ, ничего не показывая
        if "база_пуста" in resp_lower or "эксперт только по вопросам фнс" in resp_lower:
            logger.info("🚫 ВОПРОС ВНЕ ТЕМЫ: Блокировка вывода фото")
            return 

        # 2. ВЫКЛЮЧАТЕЛЬ ДЛЯ ГРАФИКОВ И ТАБЛИЦ
        is_chart = "[chart_json]" in resp_lower
        is_table = "|---" in resp_lower or "| :---" in resp_lower or (resp_lower.count("|") > 10)

        if is_chart or is_table:
            logger.info("📊 Данные в таблице/графике: блокируем фото")
        
        # 3. ПОИСК ФОТО (Если ответ по теме и нет таблиц)
        else:
            img_folder = Path("images_cache")
            if img_folder.exists():
                # Словарь: {"еговоров даниил вячеславович": "Егоров Даниил Вячеславович.jpg"}
                photo_map = {f.stem.lower(): f.name for f in img_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
                
                # Ищем фамилии людей
                for name_key, filename in photo_map.items():
                    surname = name_key.split()[0]
                    if surname in resp_lower:
                        final_photo = filename
                        if "егоров" in surname: break
                        break

            # 4. РЕЗЕРВ (Скриншоты ИРМ по Score)
            if not final_photo:
                best_img_score = -1
                for node in filtered_nodes:
                    score = node.score if hasattr(node, 'score') else 0.0
                    img_name = node.node.metadata.get("local_img")
                    if img_name and score > best_img_score:
                        if (img_folder / img_name).exists():
                            best_img_score = score
                            final_photo = img_name

        # --- ШАГ Е: ФИНАЛЬНЫЙ ВЫВОД (С кодировкой URL) ---
        if final_photo:
            import urllib.parse
            encoded_name = urllib.parse.quote(final_photo)
            photo_md = f"\n\n![photo](/images/{encoded_name})"
            yield json.dumps({"type": "text", "content": photo_md}, ensure_ascii=False) + "\n"
            logger.info(f"📸 УСПЕХ: Выведено {final_photo}")

    except Exception as e:
        logger.error(f"❌ ОШИБКА: {str(e)}")
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