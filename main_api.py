import time
import logging
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pydantic import BaseModel

from fastapi.responses import StreamingResponse
from engine_rag import get_ai_streaming_response

from fastapi.staticfiles import StaticFiles


# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Мультик RAG API")


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
    <title>TaxAI Chat</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script defer src="/static/js/marked.min.js"></script>
    <style>
        :root { --bg: #11111b; --panel: #1e1e2e; --text: #cdd6f4; --accent: #f38ba8; --user-msg: #313244; --bot-msg: #45475a; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; margin: 0; align-items: center; }
        #container { width: 100%; max-width: 800px; display: flex; flex-direction: column; height: 100%; padding: 20px; box-sizing: border-box; }
        #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
        
        /* Дизайн сообщений */
        .msg { padding: 14px 18px; border-radius: 18px; max-width: 85%; font-size: 15px; line-height: 1.6; box-shadow: 0 4px 15px rgba(0,0,0,0.1); position: relative; }
        .user { background: var(--user-msg); align-self: flex-end; border-bottom-right-radius: 4px; border: 1px solid #585b70; }
        .bot { background: var(--bot-msg); align-self: flex-start; border-bottom-left-radius: 4px; border-left: 3px solid var(--accent); }
        
        /* Markdown стили */
        .bot p { margin: 0 0 10px 0; }
        .bot ul, .bot ol { margin: 0 0 10px 20px; padding: 0; }
        .bot code { background: #313244; padding: 2px 5px; border-radius: 4px; font-family: monospace; }
        
        /* Инпут */
        #input-area { background: var(--panel); padding: 15px; border-radius: 15px; display: flex; gap: 10px; border: 1px solid #45475a; margin-top: 10px; }
        input { flex: 1; background: transparent; border: none; color: white; outline: none; font-size: 16px; }
        button { background: var(--accent); color: #11111b; border: none; padding: 10px 22px; border-radius: 10px; font-weight: 700; cursor: pointer; transition: 0.2s; }
        button:hover { transform: scale(1.05); background: #fab387; }
    </style>
</head>
<body>
<div id="container">
    <header style="text-align: center; padding-bottom: 10px; border-bottom: 1px solid #313244; margin-bottom: 10px;">
        <h2 style="margin: 0; color: var(--accent);">🦾 TaxAI <span style="font-weight: 200; color: #6c7086;">| ФНС России</span></h2>
    </header>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="messageText" placeholder="Задай вопрос..." autocomplete="off"/>
        <button onclick="sendMessage()">➤</button>
    </div>
</div>

<script>
    async function sendMessage() {
        const input = document.getElementById("messageText");
        const chat = document.getElementById("chat");
        if (!input || !input.value.trim()) return;

        const query = input.value;
        input.value = "";
        chat.innerHTML += '<div class="msg user">' + query + '</div>';
        
        let currentBotMsgDiv = document.createElement("div");
        currentBotMsgDiv.className = "msg bot";
        currentBotMsgDiv.innerHTML = "<b>База знаний ФНС:</b> ⏳...";
        chat.appendChild(currentBotMsgDiv);
        chat.scrollTop = chat.scrollHeight;

        try {
            const response = await fetch('/api/v1/predict/stream', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query: query })
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let sHtml = '';
            let fullText = ""; 

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\\n");

                for (let line of lines) {
                    let trimmed = line.trim();
                    if (!trimmed) continue;
                    
                    try {
                        const data = JSON.parse(trimmed);
                        if (data.type === "metadata") {
                            sHtml = data.sources.map(s => 
                                '🔗 <a href="' + s.url + '" target="_blank" style="color:#89b4fa; font-size:0.85em; text-decoration:none;">' + s.title + '</a>'
                            ).join('<br>');
                        } 
                        else if (data.type === "text") {
                            fullText += data.content;
                            // Рендерим Markdown из накопленного текста
                            currentBotMsgDiv.innerHTML = "<b>База знаний ФНС:</b> 📌 <br>" + marked.parse(fullText);
                        }
                    } catch (e) {
                        // Если JSON склеился, пытаемся выцепить контент регуляркой
                        const match = trimmed.match(/"content":\\s*"(.*?)"/);
                        if (match && match[1]) {
                            let raw = match[1].replace(/\\\\u([0-9a-fA-F]{4})/g, (m, p1) => String.fromCharCode(parseInt(p1, 16)));
                            fullText += raw;
                            currentBotMsgDiv.innerHTML = "<b>База знаний ФНС:</b><br>" + marked.parse(fullText);
                        }
                    }
                }
                chat.scrollTop = chat.scrollHeight;
            }
            
            if (sHtml) {
                currentBotMsgDiv.innerHTML += '<div style="margin-top:10px; padding-top:10px; border-top:1px solid #585b70; font-size:0.9em;">' + sHtml + '</div>';
            }
            
        } catch (err) {
            currentBotMsgDiv.innerHTML += '<br><span style="color:#f38ba8">🚨 Ошибка: ' + err.message + '</span>';
        }
        chat.scrollTop = chat.scrollHeight;
    }

    document.getElementById("messageText").addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
</script>
</body>
</html>
"""


class ChatRequest(BaseModel):
    query: str

@app.post("/api/v1/predict/stream") # Красивый RESTful путь
async def ask_stream(request: ChatRequest):
    return StreamingResponse(
        get_ai_streaming_response(request.query), 
        media_type="text/event-stream"
    )
