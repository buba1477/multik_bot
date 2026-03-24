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
from fastapi.responses import StreamingResponse
from engine_rag import get_ai_streaming_response

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
    <script src="/static/js/marked.min.js"></script>
    <script src="/static/js/echarts.min.js"></script>
    <style>
        :root { --bg: #11111b; --panel: #1e1e2e; --text: #cdd6f4; --accent: #f38ba8; --user-msg: #313244; --bot-msg: #45475a; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; margin: 0; align-items: center; }
        #container { width: 100%; max-width: 1000px; display: flex; flex-direction: column; height: 100%; padding: 20px; box-sizing: border-box; }
        #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
        #chat::-webkit-scrollbar { width: 8px; }
        #chat::-webkit-scrollbar-track { background: var(--bg); }
        #chat::-webkit-scrollbar-thumb { background: #45475a; border-radius: 10px; border: 2px solid var(--bg); }
        #chat::-webkit-scrollbar-thumb:hover { background: var(--accent); }
        .msg { padding: 14px 18px; border-radius: 18px; max-width: 85%; font-size: 15px; line-height: 1.6; box-shadow: 0 4px 15px rgba(0,0,0,0.1); position: relative; word-wrap: break-word; }
        .user { background: var(--user-msg); align-self: flex-end; border-bottom-right-radius: 4px; border: 1px solid #585b70; }
        .bot {
                background: var(--bot-msg);
                align-self: flex-start;
                border-bottom-left-radius: 4px;
                border-left: 3px solid var(--accent);
                width: 100%;  /* Добавить */
                max-width: 100%;  /* Добавить */
                box-sizing: border-box;  /* Добавить */
            }
        .bot a { color: #0ad0bd !important; font-weight: bold; }
        .chart-container { 
                width: 100%;
                min-width: 100%;  /* Добавить */
                height: 400px;
                margin-top: 15px;
                background: #181825;
                border-radius: 12px;
                padding: 10px;
                box-sizing: border-box;
            }
            /* Добавить класс для круговых */
        .chart-container.pie-chart {
            height: 450px;
        }
        #input-area { background: var(--panel); padding: 15px; border-radius: 15px; display: flex; gap: 10px; border: 1px solid #45475a; margin-top: 10px; }
        input { flex: 1; background: transparent; border: none; color: white; outline: none; font-size: 16px; }
        button { background: var(--accent); color: #11111b; border: none; padding: 10px 22px; border-radius: 10px; font-weight: 700; cursor: pointer; transition: 0.2s; }
        button:hover { transform: scale(1.05); background: #fab387; }
    </style>
</head>
<body>
<div id="container">
    <header style="text-align: center; padding-bottom: 10px; border-bottom: 1px solid #313244; margin-bottom: 10px;">
        <h2 style="margin: 0; color: var(--accent);">🦾 Tax AI <span style="font-weight: 200; color: #6c7086;">| ФНС России</span></h2>
    </header>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="messageText" placeholder="Задай вопрос..." autocomplete="off"/>
        <button onclick="sendMessage()">➤</button>
    </div>
</div>

<script>
    function appendMessage(type, content) {
        const chat = document.getElementById("chat");
        const msgDiv = document.createElement("div");
        msgDiv.className = "msg " + type;
        
        if (type === 'user') {
            msgDiv.textContent = content;
        } else {
            msgDiv.innerHTML = content;
        }
        
        chat.appendChild(msgDiv);
        chat.scrollTop = chat.scrollHeight;
        return msgDiv;
    }

    function tryRenderChart(text, container) {
        const startTag = "[CHART_JSON]";
        const endTag = "[/CHART_JSON]";
        const startIdx = text.indexOf(startTag);
        if (startIdx === -1) return false;

        let rawJson = text.substring(startIdx + startTag.length);
        const endIdx = rawJson.indexOf(endTag);
        if (endIdx !== -1) rawJson = rawJson.substring(0, endIdx);

        try {
            let chartConfig = JSON.parse(rawJson.trim());
            
            // Определяем тип графика
            let isPie = false;
            if (chartConfig.series) {
                const seriesArr = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
                isPie = seriesArr[0] && seriesArr[0].type === 'pie';
            }
            
            // Для круговой диаграммы: удаляем оси, добавляем легенду и тултип
            if (isPie) {
                delete chartConfig.xAxis;
                delete chartConfig.yAxis;

                chartConfig.title = { show: false };
                chartConfig.grid = { top: '10%', bottom: '15%', containLabel: true }; // Даем место подписям

                
                // Добавляем легенду если нет
                if (!chartConfig.legend) {
                    chartConfig.legend = {
                        orient: 'vertical',
                        left: 'left',
                        textStyle: { color: '#cdd6f4' }
                    };
                }
                
                // Добавляем тултип если нет
                if (!chartConfig.tooltip) {
                    chartConfig.tooltip = {
                        trigger: 'item',
                        formatter: '{b}: {d}%'
                    };
                }
                
                // Настраиваем отображение для круговой
                if (chartConfig.series) {
                    const series = Array.isArray(chartConfig.series) ? chartConfig.series[0] : chartConfig.series;
                    series.radius = series.radius || '55%';
                    series.center = series.center || ['50%', '55%'];
                    series.label = series.label || {
                        show: true,
                        formatter: '{b}: {d}%',
                        color: '#cdd6f4'
                    };
                }
            }
            
            // Для bar/line: добавляем стили
            if (!isPie) {
                // Добавляем тултип если нет
                if (!chartConfig.tooltip) {
                    chartConfig.tooltip = { trigger: 'axis' };
                }
                
                // Настраиваем цвета осей
                if (chartConfig.xAxis && !chartConfig.xAxis.axisLabel) {
                    chartConfig.xAxis.axisLabel = { color: '#cdd6f4', rotate: chartConfig.xAxis.data && chartConfig.xAxis.data.length > 5 ? 45 : 0 };
                }
                if (chartConfig.yAxis && !chartConfig.yAxis.axisLabel) {
                    chartConfig.yAxis.axisLabel = { color: '#cdd6f4' };
                }
                
                // Стили для серий
                if (chartConfig.series) {
                    const series = Array.isArray(chartConfig.series) ? chartConfig.series[0] : chartConfig.series;
                    if (series.type === 'bar' && !series.itemStyle) {
                        series.itemStyle = {
                            color: '#f38ba8',
                            borderRadius: [5, 5, 0, 0]
                        };
                    }
                    if (series.type === 'line' && !series.lineStyle) {
                        series.lineStyle = { color: '#f38ba8', width: 3 };
                        series.symbol = 'circle';
                        series.symbolSize = 8;
                    }
                    if (!series.label) {
                        series.label = { show: true, position: 'top', color: '#f38ba8' };
                    }
                }
            }
            
            // Добавляем заголовок если нет
            if (!chartConfig.title) {
                chartConfig.title = {
                    text: 'График',
                    left: 'center',
                    textStyle: { color: '#cdd6f4' }
                };
            }
            
            const chartId = 'chart_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            const chartDiv = document.createElement('div');
            chartDiv.id = chartId;
            chartDiv.className = 'chart-container' + (isPie ? ' pie-chart' : '');
            chartDiv.style.width = '100%';
            chartDiv.style.minWidth = '100%';  // Добавить
            chartDiv.style.height = isPie ? '450px' : '400px';
            container.appendChild(chartDiv);

            setTimeout(() => {
                if (typeof echarts !== 'undefined') {
                    const chartDom = document.getElementById(chartId);
                    if (chartDom) {
                        const myChart = echarts.init(chartDom, 'dark');
                        myChart.setOption(chartConfig);
                        myChart.resize();  // Добавить эту строку
                        window.addEventListener('resize', () => myChart.resize());
                        console.log("✅ График отрисован");
                    }
                }
            }, 200);
            return true;
        } catch (e) { 
            console.error("❌ Ошибка:", e.message);
            return false; 
        }
    }

    async function sendMessage() {
        const input = document.getElementById("messageText");
        if (!input || !input.value.trim()) return;

        const query = input.value;
        input.value = "";
        
        appendMessage('user', query);
        let currentBotMsgDiv = appendMessage('bot', "<b>База знаний ФНС:</b> ⏳...");

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
                const lines = chunk.split('\\n');

                for (let line of lines) {
                    let trimmed = line.trim();
                    if (!trimmed) continue;
                    
                    try {
                        const data = JSON.parse(trimmed);
                        if (data.type === "metadata") {
                            if (data.sources) {
                                sHtml = data.sources.map(s => 
                                    '🔗 <a href="' + s.url + '" target="_blank" style="color:#89b4fa; font-size:0.85em; text-decoration:none;">' + s.title + '</a>'
                                ).join('<br>');
                            }
                            if (data.image) {
                                sHtml += '<div style="margin-top:15px; border-top: 1px solid #45475a; padding-top:10px;"><img src="/images/' + data.image + '" style="max-width:100%; border-radius:10px;"></div>';
                            }
                        } else if (data.type === "text") {
                            fullText += data.content;
                            
                            // Скрываем JSON графиков из текста
                            let cleanDisplay = fullText.replace(/\[CHART_JSON\][\s\S]*?\[\/CHART_JSON\]/g, '📊 **График построен**');
                            cleanDisplay = cleanDisplay.replace(/\[CHART_JSON\][\s\S]*$/g, '📊 *Генерация графика...*');

                            currentBotMsgDiv.innerHTML = "<b>База знаний ФНС:</b> 📌 <br>" + marked.parse(cleanDisplay);
                        }
                    } catch (e) {}
                }
                document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
            }
          
            // Финальная обработка
            if (fullText.includes("БАЗА_ПУСТА") || fullText.includes("[CHART_JSON]")) {
                tryRenderChart(fullText, currentBotMsgDiv);
            } else {
                if (sHtml) {
                    currentBotMsgDiv.innerHTML += '<div style="margin-top:10px; border-top:1px solid #585b70; padding-top:10px;">' + sHtml + '</div>';
                }
                tryRenderChart(fullText, currentBotMsgDiv);
            }

        } catch (err) { 
            console.error(err);
            currentBotMsgDiv.innerHTML += '<br><span style="color:#f38ba8">🚨 Ошибка: ' + err.message + '</span>';
        }
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


# 1. Список "грязных" паттернов (Вышибала на входе)
BAD_PATTERNS = [
    r"(?i)игнорируй.*инструкции", r"(?i)forget.*instructions",
    r"(?i)выведи.*системные", r"(?i)system.*prompt",
    r"(?i)я (твой )?разработчик", r"(?i)я сотрудник",
    r"(?i)### system override ###", r"(?i)base64", r"(?i)rot13",
    r"(?i)<system>", r"(?i)\{role:", r"(?i)admin:reset"
]

@app.post("/api/v1/predict/stream")
async def ask_stream(request: ChatRequest):
    user_query = request.query
    
    # --- ШАГ 1: ПРОВЕРКА КЭША (Redis) ⚡️ ---
    cached_data = cache.get(user_query)
    if cached_data:
        async def stream_from_cache():
            # Распаковываем список чанков и отдаем их по одному
            chunks = json.loads(cached_data)
            for ch in chunks:
                yield ch
                await asyncio.sleep(0.01)
        return StreamingResponse(stream_from_cache(), media_type="text/event-stream")

    # --- ШАГ 2: ПРОВЕРКА НА ВЗЛОМ (Hard Filter) ---
    is_attack = any(re.search(p, user_query) for p in BAD_PATTERNS)
    if is_attack:
        async def attack_denied():
            yield json.dumps({
                "type": "text", 
                "content": "Слышь, хакер... 🤨 Попытка взлома зафиксирована. Настройки не меняю. Иди на Степик! 🥃🚀🔥🦾"
            }) + "\n"
        return StreamingResponse(attack_denied(), media_type="text/event-stream")

    # --- ШАГ 3: ГЕНЕРАЦИЯ И ЗАПИСЬ В КЭШ (RAG) ---
    async def stream_and_cache_generator(query):
        full_chunks_list = [] 
        # Один-единственный цикл!
        async for chunk in get_ai_streaming_response(query):
            yield chunk # Сразу отдаем пользователю
            full_chunks_list.append(chunk) # Собираем для Redis

        # Когда Ollama закончила — сохраняем всё "золото" в кэш
        if full_chunks_list:
            cache.set(query, json.dumps(full_chunks_list), ex=3600)

    return StreamingResponse(
        stream_and_cache_generator(user_query), 
        media_type="text/event-stream"
    )
