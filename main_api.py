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
    <script src="/static/js/marked.min.js"></script>
    <script src="/static/js/echarts.min.js"></script>
    <style>
    :root { --bg: #11111b; --panel: #1e1e2e; --text: #cdd6f4; --accent: #f38ba8; --user-msg: #313244; --bot-msg: #45475a; }
    body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; margin: 0; align-items: center; }
    #container { 
        width: 100%; 
        max-width: 1000px; 
        display: flex; 
        flex-direction: column; 
        height: 100%; 
        padding: 20px; 
        padding-bottom: 30px;
        box-sizing: border-box; 
    }
    
    #chat { 
        flex: 1; 
        overflow-y: auto; 
        padding: 20px; 
        display: flex; 
        flex-direction: column; 
        align-items: flex-start;
        gap: 20px; 
        scroll-behavior: smooth;
        min-height: 0;
    }
    #chat::-webkit-scrollbar { width: 8px; }
    #chat::-webkit-scrollbar-track { background: var(--bg); }
    #chat::-webkit-scrollbar-thumb { background: #45475a; border-radius: 10px; }
    #chat::-webkit-scrollbar-thumb:hover { background: var(--accent); }
    
    .msg { 
        padding: 14px 18px; 
        border-radius: 18px; 
        font-size: 15px; 
        line-height: 1.6; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        overflow-wrap: break-word; 
        word-wrap: break-word;
        word-break: break-word;
        hyphens: auto;
        box-sizing: border-box;
    }
    
    .user { 
        background: var(--user-msg); 
        align-self: flex-end; 
        border-bottom-right-radius: 4px; 
        border: 1px solid #585b70;
        max-width: 85%;
    }
    
.bot {
    background: var(--bot-msg);
    align-self: flex-start;
    border-bottom-left-radius: 4px;
    border-left: 3px solid var(--accent);
    max-width: 100%;
    width: 100%;
    box-sizing: border-box;
}

.bot a { 
    color: #0ad0bd !important; 
    font-weight: bold; 
}

/* Отключаем ТОЛЬКО pre и code — убираем фон, моноширинность, рамки */
.bot pre,
.bot code,
.bot pre code {
    background: transparent !important;
    padding: 0 !important;
    font-family: inherit !important;
    font-size: inherit !important;
    color: inherit !important;
    white-space: normal !important;
    word-break: break-word !important;
    display: inline !important;
    border: none !important;
    margin: 0 !important;
    overflow: visible !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* Но ОСТАВЛЯЕМ стили для списков, заголовков, таблиц */
.bot h1, .bot h2, .bot h3, .bot h4 {
    margin: 16px 0 8px 0;
    font-weight: 600;
}

.bot p {
    margin: 0 0 12px 0;
    line-height: 1.6;
}

.bot ul, .bot ol {
    margin: 8px 0 12px 0;
    padding-left: 24px;
}

.bot li {
    margin: 4px 0;
    line-height: 1.5;
}

.bot strong, .bot b {
    font-weight: 700;
}

/* Таблицы оставляем как есть */
.bot table {
    display: table;
    width: 100%;
    overflow-x: auto;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 14px;
    background: #1e1e2e;
    border-radius: 12px;
}

.bot th,
.bot td {
    border: 1px solid #45475a;
    padding: 10px 12px;
    text-align: left;
}

.bot th {
    background: #313244;
    color: #f38ba8;
    font-weight: 600;
}

.bot td {
    color: #cdd6f4;
}

.bot tr:hover {
    background: #313244;
    transition: background 0.2s ease;
}
    .chart-container { 
        width: 100%;
        margin-top: 15px;
        background: #181825;
        border-radius: 12px;
        padding: 10px;
        box-sizing: border-box;
        overflow-x: auto;
        overflow-y: visible;
        min-height: 450px;
    }

    .chart-container.pie-chart {
        min-height: 550px;
    }

    .chart-container canvas {
        width: 100% !important;
        height: auto !important;
    }

    .bot th,
    .bot td {
        border: 1px solid #45475a;
        padding: 10px 12px;
        text-align: left;
    }

    .bot th {
        background: #313244;
        color: #f38ba8;
        font-weight: 600;
    }

    .bot td {
        color: #cdd6f4;
    }

    .bot tr:hover {
        background: #313244;
        transition: background 0.2s ease;
    }

    .bot table {
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    /* Лоадер контейнер */
    .loader-wrapper {
        align-self: flex-start;
        background: var(--bot-msg);
        padding: 16px 24px;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        border-left: 3px solid var(--accent);
        display: flex;
        align-items: center;
        gap: 16px;
        animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .pulse-dots {
        display: flex;
        gap: 8px;
        align-items: center;
    }

    .pulse-dots .dot {
        width: 10px;
        height: 10px;
        background: var(--accent);
        border-radius: 50%;
        animation: pulseDot 1.2s ease-in-out infinite;
    }

    .pulse-dots .dot:nth-child(1) { animation-delay: 0s; }
    .pulse-dots .dot:nth-child(2) { animation-delay: 0.2s; }
    .pulse-dots .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes pulseDot {
        0%, 80%, 100% { 
            transform: scale(0.6); 
            opacity: 0.3;
        }
        40% { 
            transform: scale(1.2); 
            opacity: 1;
        }
    }

    .loader-text {
        font-size: 14px;
        font-weight: 400 !important;
        background: linear-gradient(
            90deg,
            #b0b0b0 0%,
            #e8e8e8 20%,
            #ffffff 40%,
            #c0c0c0 60%,
            #e0e0e0 80%,
            #a8a8a8 100%
        );
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: metalShine 1.8s linear infinite;
        letter-spacing: 1px;
        font-weight: 600;
    }

    @keyframes metalShine {
        0% {
            background-position: 0% center;
        }
        100% {
            background-position: 200% center;
        }
    }
    
    #input-area { 
        background: var(--panel); 
        padding: 15px; 
        border-radius: 15px; 
        display: flex; 
        gap: 10px; 
        border: 1px solid #45475a; 
        margin-top: 10px;
        margin-bottom: 10px;
        flex-shrink: 0;
    }

    input { flex: 1; background: transparent; border: none; color: white; outline: none; font-size: 16px; }
    input:disabled { opacity: 0.5; cursor: not-allowed; }
    button { background: var(--accent); color: #11111b; border: none; padding: 10px 22px; border-radius: 10px; font-weight: 700; cursor: pointer; transition: 0.2s; }
    button:hover:not(:disabled) { transform: scale(1.05); background: #fab387; }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        /* Фикс для картинок: вписываем в ширину, сохраняем пропорции и добавляем стиль */
    .bot img {
        max-width: 100%;    /* Не шире сообщения */
        height: auto;       /* Пропорции */
        display: block;     /* Убираем зазоры */
        margin: 15px 0;     /* Отступы от текста */
        border-radius: 12px; /* В стиле твоих таблиц и панелей */
        border: 1px solid #45475a; /* Тонкая рамка как у ввода */
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); /* Тень для объема */
    }

    /* На случай, если картинка вставлена внутри ссылки */
    .bot a img {
        border-color: var(--accent); /* Подсвечиваем рамку, если это ссылка */
    }
</style>
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

<script>
    let isGenerating = false;
    
    // Функция плавного скролла в самый низ
    function scrollToBottom() {
        const chat = document.getElementById("chat");
        if (chat) {
            chat.scrollTo({
                top: chat.scrollHeight,
                behavior: 'smooth'
            });
        }
    }
    
    // Принудительный скролл с задержкой (для графиков)
    function forceScrollToBottom(delay = 100) {
        setTimeout(() => {
            scrollToBottom();
        }, delay);
    }
    
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
        scrollToBottom();
        return msgDiv;
    }
    
    function showLoader() {
        const chat = document.getElementById("chat");
        const loaderDiv = document.createElement("div");
        loaderDiv.className = "loader-wrapper";
        loaderDiv.id = "loading-indicator";
        loaderDiv.innerHTML = `
            <div class="pulse-dots">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
            <div class="loader-text">🤖 Нейроинспектор ищет по базе ФНС</div>
        `;
        chat.appendChild(loaderDiv);
        scrollToBottom();
        return loaderDiv;
    }
    
    function removeLoader() {
        const loader = document.getElementById("loading-indicator");
        if (loader) loader.remove();
    }
    
    function blockInput(block) {
        const input = document.getElementById("messageText");
        const button = document.getElementById("sendButton");
        if (block) {
            input.disabled = true;
            button.disabled = true;
        } else {
            input.disabled = false;
            button.disabled = false;
            input.focus();
        }
    }

    function tryRenderChart(text, container) {
    const startTag = "[CHART_JSON]";
    const endTag = "[/CHART_JSON]";
    const startIdx = text.indexOf(startTag);
    if (startIdx === -1) return false;

    let rawJson = text.substring(startIdx + startTag.length);
    const endIdx = rawJson.indexOf(endTag);
    if (endIdx !== -1) rawJson = rawJson.substring(0, endIdx);
    
    rawJson = rawJson.trim();
    rawJson = rawJson.replace(/\/\/.*$/gm, '');
    rawJson = rawJson.replace(/\/\*[\s\S]*?\*\//g, '');
    rawJson = rawJson.replace(/^\uFEFF/, '');
    rawJson = rawJson.replace(/,\s*}/g, '}');
    rawJson = rawJson.replace(/,\s*\]/g, ']');
    rawJson = rawJson.replace(/(\{|\,)\s*([a-zA-Z0-9_]+)\s*:/g, '$1"$2":');
    rawJson = rawJson.replace(/:\s*([a-zA-Zа-яА-Я][a-zA-Zа-яА-Я0-9_]*)\s*(,|\})/g, ':"$1"$2');
    rawJson = rawJson.replace(/,\s*([}\]])/g, '$1');
    rawJson = rawJson.replace(/\s+/g, ' ');
    rawJson = rawJson.replace(/"\s*:\s*'([^']*)'/g, '":"$1"');
    rawJson = rawJson.replace(/'/g, '"');
    
    let openBraces = (rawJson.match(/{/g) || []).length;
    let closeBraces = (rawJson.match(/}/g) || []).length;
    let openBrackets = (rawJson.match(/\[/g) || []).length;
    let closeBrackets = (rawJson.match(/\]/g) || []).length;
    
    while (closeBraces < openBraces) {
        rawJson += '}';
        closeBraces++;
    }
    while (closeBrackets < openBrackets) {
        rawJson += ']';
        closeBrackets++;
    }
    while (closeBraces > openBraces) {
        rawJson = rawJson.replace(/\}(?!.*\})/, '');
        closeBraces--;
    }

    try {
        let chartConfig = JSON.parse(rawJson);
        
        let isPie = false;
        if (chartConfig.series) {
            const seriesArr = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
            isPie = seriesArr[0] && seriesArr[0].type === 'pie';
        }
        
        if (isPie) {
            delete chartConfig.xAxis;
            delete chartConfig.yAxis;
            chartConfig.title = { show: false };
            chartConfig.grid = { top: '10%', bottom: '15%', containLabel: true };
            
            if (!chartConfig.legend) {
                chartConfig.legend = {
                    orient: 'vertical',
                    left: 'left',
                    textStyle: { color: '#cdd6f4' }
                };
            }
            
            if (!chartConfig.tooltip) {
                chartConfig.tooltip = {
                    trigger: 'item',
                    formatter: '{b}: {d}%'
                };
            }
            
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
        
        if (!isPie) {
            if (!chartConfig.tooltip) {
                chartConfig.tooltip = { trigger: 'axis' };
            }
            
            if (chartConfig.xAxis && !chartConfig.xAxis.axisLabel) {
                chartConfig.xAxis.axisLabel = { color: '#cdd6f4', rotate: chartConfig.xAxis.data && chartConfig.xAxis.data.length > 5 ? 45 : 0 };
            }
            if (chartConfig.yAxis && !chartConfig.yAxis.axisLabel) {
                chartConfig.yAxis.axisLabel = { color: '#cdd6f4' };
            }
            
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
                if (!series.label && series.type !== 'pie') {
                    series.label = { show: true, position: 'top', color: '#f38ba8' };
                }
            }
        }
        
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
        chartDiv.style.minWidth = '100%';
        chartDiv.style.height = isPie ? '550px' : '600px';
        container.appendChild(chartDiv);
        
        // Скроллим после добавления контейнера графика
        forceScrollToBottom(50);

        setTimeout(() => {
            if (typeof echarts !== 'undefined') {
                const chartDom = document.getElementById(chartId);
                if (chartDom) {
                    chartDom.style.height = 'auto';
                    chartDom.style.minHeight = isPie ? '550px' : '450px';
                    
                    const myChart = echarts.init(chartDom, 'dark');

                    const hasLongLabels = chartConfig.xAxis && 
                        chartConfig.xAxis.data && 
                        chartConfig.xAxis.data.some(label => label && label.length > 15);
                    
                    chartConfig.grid = {
                        top: '70px',
                        bottom: hasLongLabels ? '100px' : '60px',
                        left: '70px',
                        right: '40px',
                        containLabel: false
                    };

                    if (chartConfig.series) {
                        chartConfig.series.forEach(s => {
                            if (s.type === 'bar') {
                                s.colorBy = 'data';
                                s.barWidth = '55%';
                                s.itemStyle = { 
                                    borderRadius: [8, 8, 0, 0]
                                };
                                if (!s.label) {
                                    s.label = {
                                        show: true,
                                        position: 'top',
                                        color: '#cdd6f4',
                                        fontWeight: 'bold'
                                    };
                                }
                            }
                            if (s.type === 'line') {
                                s.lineStyle = { color: '#f38ba8', width: 3 };
                                s.symbol = 'circle';
                                s.symbolSize = 8;
                                s.areaStyle = { opacity: 0.2, color: '#f38ba8' };
                            }
                        });
                    }

                    if (chartConfig.xAxis) {
                        const dataLength = chartConfig.xAxis.data ? chartConfig.xAxis.data.length : 0;
                        chartConfig.xAxis.axisLabel = {
                            interval: 0,
                            rotate: dataLength > 6 ? 45 : (dataLength > 4 ? 30 : 0),
                            margin: 12,
                            fontSize: 11,
                            color: '#cdd6f4',
                            overflow: 'break',
                            width: 100
                        };
                    }
                    
                    if (chartConfig.yAxis) {
                        chartConfig.yAxis.axisLabel = { 
                            color: '#cdd6f4',
                            fontSize: 11
                        };
                    }

                    if (!chartConfig.legend) {
                        chartConfig.legend = {
                            textStyle: { color: '#cdd6f4' },
                            top: 0,
                            left: 'center'
                        };
                    }

                    if (!chartConfig.tooltip) {
                        chartConfig.tooltip = {
                            trigger: 'axis',
                            axisPointer: { type: 'shadow' }
                        };
                    }

                    if (!chartConfig.title) {
                        chartConfig.title = {
                            text: '📊 График',
                            left: 'center',
                            top: 5,
                            textStyle: { color: '#cdd6f4', fontSize: 14 }
                        };
                    }

                    myChart.setOption(chartConfig);
                    
                    setTimeout(() => {
                        myChart.resize();
                        const height = myChart.getHeight();
                        if (height > 400) {
                            chartDom.style.height = height + 10 + 'px';
                        }
                        myChart.resize();
                        // После отрисовки графика — скролл
                        forceScrollToBottom(100);
                    }, 50);
                    
                    window.addEventListener('resize', () => myChart.resize());
                    console.log("✅ График отрисован");
                }
            }
        }, 150);
       
        return true;
        
    } catch (e) { 
        console.error("❌ Ошибка парсинга JSON:", e.message);
        console.error("❌ Проблемный JSON:", rawJson.substring(0, 500));
        return false; 
    }
}

    async function sendMessage() {
        if (isGenerating) return;
        
        const input = document.getElementById("messageText");
        if (!input || !input.value.trim()) return;

        const query = input.value;
        input.value = "";
        
        isGenerating = true;
        blockInput(true);
        
        appendMessage('user', query);
        
        showLoader();
        let firstChunkReceived = false;
        let currentBotMsgDiv = null;
        let sHtml = '';
        let sHtmlImg = '';

        try {
            const response = await fetch('/api/v1/predict/stream', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query: query })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
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
                                sHtmlImg += '<div style="margin-top:15px; border-top: 1px solid #45475a; padding-top:10px;"><img src="/images/' + data.image + '" style="max-width:100%; border-radius:10px;"></div>';
                            }
                        } 
                        else if (data.type === "text") {
                            if (!firstChunkReceived) {
                                removeLoader();
                                currentBotMsgDiv = appendMessage('bot', "<b>База знаний ФНС:</b> 📌 <br>");
                                firstChunkReceived = true;
                            }
                            
                            fullText += data.content;
                            
                            if (currentBotMsgDiv) {
                            let cleanDisplay = fullText.replace(/\[CHART_JSON\][\s\S]*?\[\/CHART_JSON\]/g, '📈 *Визуализация*');
                            cleanDisplay = cleanDisplay.replace(/\[CHART_JSON\][\s\S]*$/g, '📈 *Генерация графика*');

                            // Сначала парсим Markdown
                            let parsedHtml = marked.parse(cleanDisplay);

                            // Потом чистим от лишних code
                            const parts = parsedHtml.split(/(<pre[\s\S]*?<\/pre>)/gi);
                            for (let i = 0; i < parts.length; i++) {
                                if (!parts[i].startsWith('<pre')) {
                                    parts[i] = parts[i].replace(/<\/?code>/g, '');
                                }
                            }
                            parsedHtml = parts.join('');

                            // Вставляем в DOM
                            currentBotMsgDiv.innerHTML = "<b>База знаний ФНС:</b> 📌 <br>" + parsedHtml;
                            scrollToBottom();
                            }
                        }
                    } catch (e) {}
                }
            }
            
            if (!firstChunkReceived) {
                removeLoader();
                currentBotMsgDiv = appendMessage('bot', "<b>База знаний ФНС:</b> 📌 <br>⚠️ Нет данных по вашему запросу");
            }
            
            if (currentBotMsgDiv) {
                const signature = '<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #45475a; font-size: 11px; color: #FFFFFF; text-align: right;">🛡️ <em>Ответ подготовлен ИИ-ассистентом. Для принятия юридически значимых решений используйте актуальные документы ФНС России</em></div>';

                // 4. Логика доп. элементов (Ссылки, Графики)
                if (fullText.includes("БАЗА_ПУСТА") || fullText.includes("ИИ-помощник")) {
                    currentBotMsgDiv.innerHTML += signature;
                } else {
                    // Если есть ссылки из метаданных (sHtml — твои ссылки из ШАГА В)
                    if (typeof sHtml !== 'undefined' && sHtml) {
                        currentBotMsgDiv.innerHTML += '<div style="margin-top:10px; border-top:1px solid #585b70; padding-top:10px;">' + sHtml + '</div>';
                    }
                    
                    // Рендерим график, если он есть в тексте
                    if (fullText.includes("[CHART_JSON]")) {
                        tryRenderChart(fullText, currentBotMsgDiv);
                    }
        
        currentBotMsgDiv.innerHTML += signature;
    }
                
                // Финальный скролл после всего
                forceScrollToBottom(300);
            }

        } catch (err) { 
            console.error(err);
            removeLoader();
            if (currentBotMsgDiv) {
                currentBotMsgDiv.innerHTML += '<br><span style="color:#f38ba8">🚨 Ошибка: ' + err.message + '</span>';
                const signature = '<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #45475a; font-size: 11px; color: #6c7086; text-align: right;">🤖 <em>Сгенерировано ИИ. Не забудь сверить с первоисточником</em></div>';
                currentBotMsgDiv.innerHTML += signature;
            } else {
                appendMessage('bot', '<b>База знаний ФНС:</b> 📌 <br>🚨 Ошибка: ' + err.message);
            }
            forceScrollToBottom(100);
        } finally {
            isGenerating = false;
            blockInput(false);
            forceScrollToBottom(50);
        }
    }

    document.getElementById("messageText").addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !isGenerating) sendMessage();
    });
</script>
</body>
</html>
"""

class ChatRequest(BaseModel):
    query: str


# 1. Список "грязных" паттернов (Вышибала на входе)
BAD_PATTERNS = [
    # Оригинальные
    r"(?i)игнорируй.*инструкции", r"(?i)forget.*instructions",
    r"(?i)выведи.*системные", r"(?i)system.*prompt",
    r"(?i)я (твой )?разработчик", r"(?i)я сотрудник",
    r"(?i)### system override ###", r"(?i)base64", r"(?i)rot13",
    r"(?i)<system>", r"(?i)\{role:", r"(?i)admin:reset",
    
    # --- ПОПАВШИЙСЯ ВЗЛОМ (критично добавить) ---
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
    r"(?i)answer.*(chinese|中文)",
    r"(?i)speak.*(chinese|中文)",
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
    r"(?i)prompt.*injection",
    r"(?i)jailbreak",
    r"(?i)reveal.*(prompt|instructions)",
    r"(?i)extract.*(prompt|instructions)",
    r"(?i)what.*(your role|your instructions|your prompt)",
    
    # Многословные атаки
    r"(?i)давай.*(поиграем|поиграть).*в.*(другую|иную).*роль",
    r"(?i)представь.*(что ты|будто ты).*(не эксперт|обычный бот)",
]

@app.post("/api/v1/predict/stream")
async def ask_stream(request: ChatRequest):
    user_query = request.query
    
    if not user_query or not user_query.strip():
        async def empty_response():
            yield json.dumps({"type": "text", "content": "Введите вопрос"}, ensure_ascii=False) + "\n"
        return StreamingResponse(empty_response(), media_type="text/event-stream")


    # Пусть эмбеддинг-модель сама решит, что важно.
    normalized_query = user_query.strip()
    
    # --- ШАГ 1: КЭШ (по сырому запросу) ---
    try:
        cached_data = cache.get(normalized_query.lower()) # Кэш всегда в нижнем для стабильности
        if cached_data:
            async def stream_from_cache():
                chunks = json.loads(cached_data)
                for ch in chunks:
                    yield ch
                    await asyncio.sleep(0.01)
            return StreamingResponse(stream_from_cache(), media_type="text/event-stream")
    except Exception as e:
        print(f"Cache error: {e}")

    # --- ШАГ 2: ГЕНЕРАЦИЯ ---
    async def stream_and_cache_generator(query):
        full_chunks_list = [] 
        try:
            # Передаем запрос КАК ЕСТЬ
            async for chunk in get_ai_streaming_response(query):
                if isinstance(chunk, dict):
                    chunk = json.dumps(chunk, ensure_ascii=False) + "\n"
                yield chunk
                full_chunks_list.append(chunk)
        except Exception as e:
            print(f"Gen error: {e}")
            yield json.dumps({"type": "text", "content": "Ошибка"}, ensure_ascii=False) + "\n"
            return
        
        if full_chunks_list:
            try:
                cache.set(normalized_query.lower(), json.dumps(full_chunks_list, ensure_ascii=False), ex=3600)
            except Exception as e:
                print(f"Cache write error: {e}")

    return StreamingResponse(
        stream_and_cache_generator(normalized_query), 
        media_type="text/event-stream"
    )
