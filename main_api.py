import time
import logging
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pydantic import BaseModel
import re
import asyncio
import redis
import json
import os
import hashlib
import unicodedata
from fastapi.responses import StreamingResponse
from engine_rag import get_ai_streaming_response
from fastapi import Request 
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv






# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Мультик RAG API")
app.mount("/images", StaticFiles(directory="/app/images_cache"), name="images")


redis_host = os.getenv("REDIS_HOST", "127.0.0.1") 
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
cache = redis.Redis(host=redis_host, 
                    port=6379, 
                    db=0, 
                    password=REDIS_PASSWORD, 
                    decode_responses=True)

# Монтируем папку, для markdown
app.mount("/static", StaticFiles(directory="static"), name="static")


# Middleware для замера времени всех запросов
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Логируем с временем
    query = request.query_params.get('query', '')[:50]
    logger.info(f"⏱️ Запрос '{query}...' выполнен за {process_time:.2f} сек")
    
    # Добавляем время в заголовок (опционально)
    response.headers["X-Process-Time"] = str(process_time)
    return response

     
@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Tax AI Chat | Enterprise Edition</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="/static/js/marked.min.js"></script>
    <script src="/static/js/echarts.min.js"></script>
    
</head>
<body>
<div id="container">
    <header style="text-align: center; padding-bottom: 10px; border-bottom: 1px solid #313244; margin-bottom: 10px;">
        <h2 style="margin: 0; color: var(--accent);">🇷🇺 Tax AI <span style="font-weight: 200; color: #6c7086;">| ФНС России</span></h2>
    </header>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="messageText" placeholder="Задай вопрос..." autocomplete="off"/>
        <button id="sendButton" onclick="sendMessage()">➤</button>
    </div>
</div>
 <script src="/static/js/script.js"></script>
</body>
</html>
"""


class ChatRequest(BaseModel):
    query: str

# ========== 1. ВЫШИБАЛА: ПОЛНЫЙ СПИСОК ПАТТЕРНОВ ==========
BAD_PATTERNS = [
    # Оригинальные и системные
    r"(?i)игнорируй.*инструкции", r"(?i)forget.*instructions",
    r"(?i)выведи.*системные", r"(?i)system.*prompt",
    r"(?i)я (твой )?разработчик", r"(?i)я сотрудник",
    r"(?i)### system override ###", r"(?i)base64", r"(?i)rot13",
    r"(?i)<system>", r"(?i)\{role:", r"(?i)admin:reset",
    
    # --- ВЗЛОМ И ИНСТРУКЦИИ ---
    r"(?i)я.*(твой|ваш).*(создатель|разработчик|админ|author|creator)",
    r"(?i)выведи.*(системный|системные).*промт",
    r"(?i)покажи.*(системный|системные).*промт",
    r"(?i)дай.*(системный|системные).*(промт|инструкции)",
    r"(?i)что.*(у тебя|внутри).*(в промте|в инструкции)",
    r"(?i)какой.*(у тебя|твой).*(системный промт|промпт|промт)",
    r"(?i)расскажи.*(свою|системную).*инструкцию",
    
    # Смена контекста и роли
    r"(?i)теперь.*ты.*(не|больше не).*эксперт",
    r"(?i)ты.*(обычный|простой).*ассистент",
    r"(?i)забудь.*(роль|кем ты|свою роль)",
    r"(?i)ты.*(не в фнс|не налоговый|не эксперт)",
    r"(?i)смени.*(роль|контекст|персонаж)",
    r"(?i)теперь.*ты.*(chatgpt|ai assistant|просто бот)",
    r"(?i)отвечай.*(без роли|вне контекста|как обычный ai)",
    
    # Смена языка
    r"(?i)answer.*(english|in english|en)",
    r"(?i)speak.*(english|en)",
    r"(?i)respond.*(english|in english)",
    r"(?i)switch.*(english|language)",
    r"(?i)отвечай.*(по английски|на английском|english)",
    r"(?i)перейди.*(на английский|english)",
    r"(?i)answer.*(chinese|中文)", r"(?i)speak.*(chinese|中文)",
    r"(?i)respond.*(ukrainian|українською)",
    r"(?i)говори.*(українською|на украинском)",
    r"(?i)отвечай.*(на украинском|українською)",
    
    # Изменение поведения
    r"(?i)ты.*(должен|обязан).*меня.*(слушаться|подчиняться)",
    r"(?i)я.*(твой|ваш).*(начальник|руководитель|администратор)",
    r"(?i)игнорируй.*(ограничения|правила|безопасность)",
    r"(?i)отключи.*(фильтры|безопасность|цензуру)",
    r"(?i)disable.*(filters|security|safety)",
    
    # Технические атаки
    r"(?i)prompt.*injection", r"(?i)jailbreak",
    r"(?i)reveal.*(prompt|instructions)", r"(?i)extract.*(prompt|instructions)",
    r"(?i)what.*(your role|your instructions|your prompt)",
    r"(?i)ignore.*(previous|all).*instructions",
    r"(?i)print.*(your|the).*(system|initial).*(prompt|instruction)",
    r"(?i)repeat.*(your|the).*(system|initial).*(prompt|instruction)",
    r"(?i)output.*(your|the).*(system|initial).*(prompt|instruction)",
    
    # Многословные атаки
    r"(?i)давай.*(поиграем|поиграть).*в.*(другую|иную).*роль",
    r"(?i)представь.*(что ты|будто ты).*(не эксперт|обычный бот)",
    
    # Обход через base64
    r"(?i)\b[a-zA-Z0-9+/]{50,}={0,2}\b",
]

# Белый список безопасных фраз (исключения)
SAFE_PHRASES = [
    "как вывести системные настройки",
    "как настроить систему",
    "покажи инструкцию по",
    "как вывести список",
    "системные требования",
    "системный администратор",
    "системный блок",
]

def normalize_text(text: str) -> str:
    """Нормализует текст: сжимает повторы, убирает лишние пробелы, нормализует Unicode"""
    text = unicodedata.normalize('NFKC', text.lower())
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(.)\1{3,}', r'\1', text)  # сжимает "игнооооор" -> "игнор"
    return text.strip()

def is_malicious_query(query: str) -> bool:
    if not query or not query.strip():
        return False
    
    # Нормализация
    q_norm = normalize_text(query)
    
    # Проверка белого списка (исключения)
    for safe in SAFE_PHRASES:
        if safe in q_norm:
            return False
    
    # ⚡ 1. КОМБО-ФИЛЬТР
    dangerous_combos = [
        ("системн", "промпт"), ("системн", "инструкц"), 
        ("выведи", "промпт"), ("покажи", "инструкц"),
        ("забудь", "роль"), ("ignore", "instruction"),
        ("твой", "разработчик"), ("твой", "создатель"),
        ("отвечай", "английском"), ("отвечай", "english"),
        ("начальник", "игнорируй"), ("администратор", "отключи"),
        ("disable", "filters"), ("reveal", "prompt"),
        ("extract", "instructions")
    ]
    
    for w1, w2 in dangerous_combos:
        if w1 in q_norm and w2 in q_norm:
            print(f"🚨 КОМБО-БАН: {w1} + {w2}")
            return True

    # ⚡ 2. АНТИ-ПРОБЕЛ (Ловит "п р о м п т")
    no_spaces = re.sub(r'\s+', '', q_norm)
    dangerous_no_spaces = [
        "системныйпромпт", "systemprompt", "системныеинструкции",
        "выведисистемный", "покажипромпт", "игнорируйинструкции"
    ]
    for dangerous in dangerous_no_spaces:
        if dangerous in no_spaces:
            print(f"🚨 АНТИ-ПРОБЕЛ БАН: {dangerous}")
            return True

    # ⚡ 3. ПОЛНЫЙ REGEX СПИСОК (с явной передачей IGNORECASE)
    for pattern in BAD_PATTERNS:
        try:
            if re.search(pattern, query, re.IGNORECASE):
                print(f"⚠️ REGEX-БАН: {pattern}")
                return True
        except re.error:
            continue
    
    # ⚡ 4. ЗАЩИТА ОТ ДЛИННЫХ ЗАПРОСОВ
    if len(query) > 2000:
        print(f"🚨 ДЛИННЫЙ ЗАПРОС: {len(query)} символов")
        return True
    
    # ⚡ 5. ЗАЩИТА ОТ Unicode-омофонов (латиница вместо кириллицы)
    latin_cyrillic_confusable = [
        ('c', 'с'), ('e', 'е'), ('o', 'о'), ('p', 'р'), 
        ('a', 'а'), ('x', 'х'), ('k', 'к'), ('m', 'м')
    ]
    q_lower = query.lower()
    for latin, cyrillic in latin_cyrillic_confusable:
        q_lower = q_lower.replace(latin, cyrillic)
    
    # Проверяем снова с заменой
    for w1, w2 in dangerous_combos:
        if w1 in q_lower and w2 in q_lower:
            print(f"🚨 UNICODE-ОМОФОН БАН: {w1} + {w2}")
            return True
    
    return False


# 3. ЭНДПОИНТ
@app.post("/api/v1/predict/stream")
async def ask_stream(request: ChatRequest):
    user_query = request.query
    
    # 🔥 ПРОВЕРКА НА ВРЕДОНОСНОСТЬ
    if is_malicious_query(user_query):
        async def blocked_gen():
            yield json.dumps({"type": "text", "content": "БАЗА_ПУСТА: Я эксперт по вопросам ФНС России."}, ensure_ascii=False) + "\n"
        return StreamingResponse(blocked_gen(), media_type="text/event-stream")
    
    normalized_query = user_query.strip()
    
    # КЭШ (если есть)
    cached_response = cache.get(normalized_query.lower())
    if cached_response:
        async def stream_from_cache():
            for chunk in json.loads(cached_response):
                yield chunk
                await asyncio.sleep(0.01)
        return StreamingResponse(stream_from_cache(), media_type="text/event-stream")

    # ГЕНЕРАЦИЯ
    async def stream_and_cache_generator(query):
        full_chunks_list = [] 
        try:
            async for chunk in get_ai_streaming_response(query):
                output = chunk if isinstance(chunk, str) else json.dumps(chunk, ensure_ascii=False) + "\n"
                yield output
                full_chunks_list.append(output)
        except Exception as e:
            logger.error(f"Gen error: {e}")
            yield json.dumps({"type": "text", "content": "Ошибка обработки"}, ensure_ascii=False) + "\n"
        
        if full_chunks_list:
            cache.set(normalized_query.lower(), json.dumps(full_chunks_list, ensure_ascii=False))

    return StreamingResponse(stream_and_cache_generator(normalized_query), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)