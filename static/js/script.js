let isGenerating = false;
let chartInstances = []; // Хранилище для инстансов графиков

function scrollToBottom() {
    const chat = document.getElementById("chat");
    if (chat) {
        chat.scrollTo({
            top: chat.scrollHeight,
            behavior: 'smooth'
        });
    }
}

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
    if (endIdx !== -1) {
        rawJson = rawJson.substring(0, endIdx);
    }

    rawJson = rawJson.trim()
        .replace(/```json/g, "").replace(/```/g, "")
        .replace(/\/\/.*$/gm, "").replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/^\uFEFF/, "").replace(/'/g, '"').replace(/\s+/g, " ");

    rawJson = rawJson.replace(/(\{|\,)\s*([a-zA-Z0-9_]+)\s*:/g, '$1"$2":');

    const countOcc = (str, char) => str.split(char).length - 1;

    let nO = countOcc(rawJson, "{"), nC = countOcc(rawJson, "}");
    let bO = countOcc(rawJson, "["), bC = countOcc(rawJson, "]");

    while (bC < bO) { rawJson += "]"; bC++; }
    while (nC < nO) { rawJson += "}"; nC++; }

    try {
        let chartConfig = JSON.parse(rawJson);
        const chartId = 'chart_' + Math.random().toString(36).substr(2, 9);
        
        // ========== ФИКС: если title пришёл строкой, превращаем в объект ==========
        if (chartConfig.title && typeof chartConfig.title === 'string') {
            chartConfig = {
                ...chartConfig,
                title: { text: chartConfig.title }
            };
        }
        
        let isPie = false;
        if (chartConfig.series) {
            const sArr = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
            isPie = sArr[0] && sArr[0].type === 'pie';
        }

        // Создаем контейнер для графика с анимацией
        const chartWrapper = document.createElement('div');
        chartWrapper.className = 'chart-wrapper';
        chartWrapper.style.cssText = 'width: 100%; margin: 20px 0; animation: chartFadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards; opacity: 0; transform: scale(0.95) translateY(10px);';
        
        const chartDiv = document.createElement('div');
        chartDiv.id = chartId;
        chartDiv.style.width = '100%';
        chartDiv.style.height = isPie ? '500px' : '450px';
        chartDiv.style.margin = '0';
        chartWrapper.appendChild(chartDiv);
        container.appendChild(chartWrapper);

        setTimeout(() => {
            if (typeof echarts !== 'undefined') {
                // Initialize with professional theme
                const myChart = echarts.init(document.getElementById(chartId));
                
                // Сохраняем инстанс графика
                chartInstances.push(myChart);
                
                // Modern gradient colors for charts
                const gradientColors = [
                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#00d4ff' },
                        { offset: 1, color: '#00509e' }
                    ]),
                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#7c3aed' },
                        { offset: 1, color: '#4c1d95' }
                    ]),
                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#ffd700' },
                        { offset: 1, color: '#b8860b' }
                    ]),
                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#10b981' },
                        { offset: 1, color: '#059669' }
                    ]),
                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#f59e0b' },
                        { offset: 1, color: '#d97706' }
                    ])
                ];

                const baseOption = {
                    backgroundColor: 'transparent',
                    textStyle: {
                        color: '#1a2c3e',
                        fontWeight: 500,
                        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
                    },
                    title: {
                        text: chartConfig.title?.text || 'Аналитика',
                        left: 'center',
                        top: 10,
                        textStyle: { 
                            color: '#003366', 
                            fontWeight: 700,
                            fontSize: 16,
                            fontFamily: "'Inter', sans-serif"
                        }
                    },
                    tooltip: { 
                        trigger: isPie ? 'item' : 'axis',
                        backgroundColor: 'rgba(255, 255, 255, 0.97)',
                        borderColor: '#d1dce7',
                        borderWidth: 1,
                        textStyle: { 
                            color: '#1a2c3e',
                            fontWeight: 500
                        },
                        extraCssText: 'box-shadow: 0 4px 12px rgba(0, 51, 102, 0.15);'
                    },
                    grid: { 
                        containLabel: true, 
                        bottom: '10%', 
                        top: '20%',
                        left: '10%', 
                        right: '10%' 
                    },
                    legend: {
                        top: 40,
                        left: 'center',
                        itemWidth: 12,
                        itemHeight: 12,
                        textStyle: {
                            fontSize: 12,
                            fontWeight: 500
                        }
                    },
                    animationDuration: 800,
                    animationEasing: 'cubicOut'
                };

                if (isPie) {
                    const s = Array.isArray(chartConfig.series) ? chartConfig.series[0] : chartConfig.series;
                    s.radius = ['45%', '75%'];
                    s.center = ['50%', '55%'];
                    s.itemStyle = { 
                        borderRadius: 12, 
                        borderColor: '#ffffff', 
                        borderWidth: 3,
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 51, 102, 0.2)'
                    };
                    s.label = { 
                        show: true, 
                        color: '#1a2c3e', 
                        formatter: '{b}: {d}%',
                        fontSize: 13,
                        fontWeight: 600,
                        fontFamily: "'Inter', sans-serif"
                    };
                    s.labelLine = {
                        show: true,
                        lineStyle: {
                            color: '#1a2c3e',
                            width: 2
                        },
                        smooth: 0.2,
                        length: 15,
                        length2: 10
                    };
                    s.emphasis = {
                        itemStyle: {
                            shadowBlur: 20,
                            shadowOffsetX: 0,
                            shadowColor: 'rgba(0, 51, 102, 0.3)'
                        }
                    };
                    
                    // Применяем градиенты к сегментам
                    if (s.data) {
                        s.data.forEach((item, index) => {
                            if (typeof item === 'object' && !item.itemStyle) {
                                item.itemStyle = {
                                    color: gradientColors[index % gradientColors.length]
                                };
                            }
                        });
                    }
                } else {
                    if (chartConfig.xAxis) {
                        chartConfig.xAxis.axisLabel = { 
                            color: '#1a2c3e', 
                            rotate: 30,
                            fontSize: 12,
                            fontWeight: 600,
                            fontFamily: "'Inter', sans-serif",
                            margin: 10
                        };
                        chartConfig.xAxis.axisLine = {
                            lineStyle: { color: '#d1dce7', width: 2 }
                        };
                        chartConfig.xAxis.splitLine = {
                            show: false
                        };
                        chartConfig.xAxis.axisTick = {
                            show: false
                        };
                    }
                    if (chartConfig.yAxis) {
                        chartConfig.yAxis.axisLabel = { 
                            color: '#1a2c3e',
                            fontSize: 12,
                            fontWeight: 600,
                            fontFamily: "'Inter', sans-serif",
                            margin: 10
                        };
                        chartConfig.yAxis.axisLine = {
                            show: false
                        };
                        chartConfig.yAxis.axisTick = {
                            show: false
                        };
                        chartConfig.yAxis.splitLine = {
                            lineStyle: { color: '#e8edf2', width: 1 }
                        };
                    }
                    
                    if (chartConfig.series) {
                        const seriesArray = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
                        seriesArray.forEach((s, index) => {
                            if (s.type === 'bar') {
                                s.colorBy = 'data';
                                s.itemStyle = { 
                                    borderRadius: [6, 6, 0, 0],
                                    color: gradientColors[index % gradientColors.length],
                                    shadowBlur: 10,
                                    shadowColor: 'rgba(0, 51, 102, 0.2)'
                                };
                                s.label = { 
                                    show: true, 
                                    position: 'top', 
                                    color: '#1a2c3e',
                                    fontSize: 11,
                                    fontWeight: 600,
                                    fontFamily: "'Inter', sans-serif",
                                    distance: 5
                                };
                                s.barMaxWidth = 50;
                                s.animationDelay = index * 100;
                            } else if (s.type === 'line') {
                                s.itemStyle = { 
                                    color: gradientColors[index % gradientColors.length],
                                    borderWidth: 2
                                };
                                s.lineStyle = { width: 3, shadowBlur: 10, shadowColor: 'rgba(0, 51, 102, 0.2)' };
                                s.symbol = 'circle';
                                s.symbolSize = 10;
                                s.smooth = true;
                                s.areaStyle = {
                                    opacity: 0.1,
                                    color: gradientColors[index % gradientColors.length]
                                };
                                s.animationDelay = index * 100;
                            }
                        });
                    }
                }

                myChart.setOption(Object.assign(baseOption, chartConfig));
                
                // Плавное появление графика
                myChart.setOption({
                    animationDuration: 1000,
                    animationEasing: 'elasticOut'
                });
                
                window.addEventListener('resize', () => myChart.resize());
            }
        }, 100);
        
        return true;
    } catch (e) {
        console.error("❌ JSON Error:", e);
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
    let fullText = ""; 

    try {
        const response = await fetch('/api/v1/predict/stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ query: query })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n'); 

            for (let line of lines) {
                let trimmed = line.trim();
                if (!trimmed) continue;
                
                try {
                    const data = JSON.parse(trimmed);
                    
                    if (data.type === "metadata") {
                        if (data.sources) {
                            sHtml = data.sources.map(s => 
                                '🔗 <a href="' + s.url + '" target="_blank" style="color:#89b4fa; font-size:0.85em; text-decoration:none; font-weight:bold;">' + s.title + '</a>'
                            ).join('<br>');
                        }
                        if (data.image) {
                            sHtmlImg = '<div style="margin-top:15px; border-top: 1px solid #e2e2e2; padding-top:10px;"><img src="/images/' + data.image + '" style="max-width:100%; border-radius:12px; border: 1px solid #e2e2e2;"></div>';
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
                            // 1. Отделяем текст от графиков (чтобы JSON не треснул)
                            let textPart = fullText.split('[CHART_JSON]')[0];
                            let chartPart = fullText.includes('[CHART_JSON]') ? fullText.substring(fullText.indexOf('[CHART_JSON]')) : '';

                            // 🛠️ 2. УЛЬТРА-ОЧИСТКА ТЕКСТА (Для Бударина, Бациева и должностей)
                            const n = "\n";
                            
                            // А) Биографии: Режем по датам (В 2001, С 2013, 2016-2020)
                           // textPart = textPart.replace(/([.!?])\s*(?=(С \d{4}|В \d{4}|\d{4}-\d{4}))/g, '$1' + n + n + '• ');

                            // Б) Полномочия: Режем перед словами "Непосредственно", "Распоряжением", "Координирует"
                            //textPart = textPart.replace(/([.!?])\s*(?=(Непосредственно|Распоряжением|Координирует|Контролирует|Имеет классный))/g, '$1' + n + n + '• ');

                            // В) Разрываем слипшиеся пункты (•), если они уже есть
                            
                            // Любой знак препинания (., !, ?, :, ;) перед • — вставляем перенос
                            textPart = textPart.replace(/([.!?:;])\s*•/g, '$1\n\n•');
                            
                            // Г) Категории должностей (руководители, специалисты)
                            //textPart = textPart.replace(/(\s*)(руководители|специалисты|обеспечивающие специалисты)\s*\(/gi, n + n + '• **$2** (');

                            // Д) Таблицы (пайпы)
                            // textPart = textPart.replace(/([^\n])\|/g, '$1' + n + '|');
                            // textPart = textPart.replace(/\|(\s*)\|/g, '|' + n + '|');


                            // Собираем обратно
                            let display = textPart + chartPart;

                            // 3. Прячем теги графиков для красоты во время стриминга
                            display = display.replace(/\[CHART_JSON\][\s\S]*?\[\/CHART_JSON\]/g, '📈 *Визуализация готова*');
                            display = display.replace(/\[CHART_JSON\][\s\S]*$/g, '📈 *Генерация аналитики...*');

                            // 4. Парсим Markdown и выводим
                            let parsedHtml = marked.parse(display);
                            
                            if (!parsedHtml.includes('<table')) {
                                parsedHtml = parsedHtml.replace(/<pre><code>([\s\S]*?)<\/code><\/pre>/gi, '$1');
                                parsedHtml = parsedHtml.replace(/<code>([\s\S]*?)<\/code>/gi, '$1');
                            }

                            currentBotMsgDiv.innerHTML = "<b>База знаний ФНС:</b> 📌 <br>" + parsedHtml;
                            scrollToBottom();
                        }
                    }
                } catch (e) {}
            }
        }
        
        // ФИНАЛИЗАЦИЯ
        if (currentBotMsgDiv) {
            if (sHtml) currentBotMsgDiv.innerHTML += '<div style="margin-top:10px; border-top:1px solid #e2e2e2; padding-top:10px;">' + sHtml + '</div>';
            if (sHtmlImg) currentBotMsgDiv.innerHTML += sHtmlImg;

            if (fullText.includes("[/CHART_JSON]")) {
                tryRenderChart(fullText, currentBotMsgDiv);
            }

            const signature = '<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #e2e2e2; font-size: 11px; color: #ffffff; text-align: right;">🛡️ <em>Ответ подготовлен ИИ-ассистентом ФНС</em></div>';
            currentBotMsgDiv.innerHTML += signature;
        }

    } catch (err) { 
        console.error(err);
        removeLoader();
    } finally {
        isGenerating = false;
        blockInput(false);
        forceScrollToBottom(100);
    }
}

// Обработчик отправки по Enter
document.getElementById("messageText").addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !isGenerating) sendMessage();
});

// Очистка графики при загрузке страницы
window.addEventListener('beforeunload', () => {
    chartInstances.forEach(chart => {
        if (chart && typeof chart.dispose === 'function') {
            chart.dispose();
        }
    });
});

// Добавляем плавное появление страницы при загрузке
document.addEventListener('DOMContentLoaded', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});
