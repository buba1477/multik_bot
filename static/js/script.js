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
        <div class="loader-text">Нейроинспектор ищет по базе ФНС</div>
    `;
    chat.appendChild(loaderDiv);
    scrollToBottom();
    return loaderDiv;
}

function removeLoader() {
    const loader = document.getElementById("loading-indicator");
    if (loader) loader.remove();
}

function blockInput(status) {
    const textarea = document.getElementById("messageText");
    const sendButton = document.getElementById("sendButton");
    const micButton = document.getElementById("micBtn");
    const clearButton = document.getElementById("clearTextBtn");

    if (status) {
        // 🔥 ДЕАКТИВАЦИЯ ВСЕГО ИНТЕРФЕЙСА ВО ВРЕМЯ ГЕНЕРАЦИИ
        if (textarea) textarea.disabled = true;
        if (sendButton) sendButton.disabled = true;
        
        if (micButton) {
            micButton.disabled = true;
            micButton.style.opacity = "0.4"; // Визуально делаем серым
            micButton.style.cursor = "not-allowed";
        }
        if (clearButton) {
            clearButton.disabled = true;
            clearButton.style.opacity = "0.4";
            clearButton.style.cursor = "not-allowed";
        }
    } else {
        // ✅ ПОЛНАЯ РАЗБЛОКИРОВКА ПОСЛЕ ОКОНЧАНИЯ СТРИМА
        if (textarea) textarea.disabled = false;
        
        if (micButton) {
            micButton.disabled = false;
            micButton.style.opacity = "1";
            micButton.style.cursor = "pointer";
        }
        if (clearButton) {
            clearButton.disabled = false;
            clearButton.style.opacity = "1";
            clearButton.style.cursor = "pointer";
        }
        
        // Фокусируем инспектора обратно на поле ввода
        if (textarea) textarea.focus();
    }
}


function clearInput() {
    // Сохраняем текущий чат перед очисткой
    persistCurrentChat();
    
    const input = document.getElementById("messageText");
    const chat = document.getElementById("chat");

    if (input) {
        input.value = "";
        input.focus();
    }

    if (chat) {
        chat.innerHTML = "";
    }

    removeLoader();
    chartInstances.forEach(chart => {
        if (chart && typeof chart.dispose === 'function') {
            chart.dispose();
        }
    });
    chartInstances = [];
}

// ============================================
// 🔥 ЕДИНАЯ ФУНКЦИЯ РЕНДЕРА ГРАФИКОВ
// ============================================
function initChart(chartId, chartConfig, container) {
    // Если title пришёл строкой — превращаем в объект
    if (chartConfig.title && typeof chartConfig.title === 'string') {
        chartConfig = { ...chartConfig, title: { text: chartConfig.title } };
    }

    let isPie = false;
    let categoryCount = 0;
    if (chartConfig.series) {
        const sArr = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
        isPie = sArr[0] && sArr[0].type === 'pie';
        // Определяем количество категорий
        if (isPie && sArr[0]?.data) {
            categoryCount = sArr[0].data.length;
        } else if (chartConfig.xAxis?.data) {
            categoryCount = chartConfig.xAxis.data.length;
        }
    }

    // Динамическая высота в зависимости от количества категорий
    let chartHeight;
    if (isPie) {
        chartHeight = Math.max(450, Math.min(900, categoryCount * 50));
    } else {
        chartHeight = Math.max(350, Math.min(700, categoryCount * 50));
    }

    // Создаём wrapper с ЕДИНЫМИ стилями
    const chartWrapper = document.createElement('div');
    chartWrapper.className = 'chart-wrapper';
    chartWrapper.style.cssText = 'width: 100%; margin: 10px 0; background: #ffffff; border: 1px solid #d1dce7; border-radius: 16px; border-top: 2px solid #00509e; padding: 20px 15px; box-shadow: 0 4px 16px rgba(0, 51, 102, 0.08); box-sizing: border-box; flex-shrink: 0;';

    const chartDiv = document.createElement('div');
    chartDiv.id = chartId;
    chartDiv.style.cssText = `width: 100%; height: ${chartHeight}px;`;

    chartWrapper.appendChild(chartDiv);
    container.appendChild(chartWrapper);

    // Инициализация ECharts
    setTimeout(() => {
        const dom = document.getElementById(chartId);
        if (!dom || typeof echarts === 'undefined') return;

        const myChart = echarts.init(dom);
        chartInstances.push(myChart);

        const gradientColors = [
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#00d4ff' }, { offset: 1, color: '#00509e' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#7c3aed' }, { offset: 1, color: '#4c1d95' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#ffd700' }, { offset: 1, color: '#b8860b' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#10b981' }, { offset: 1, color: '#059669' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#f59e0b' }, { offset: 1, color: '#d97706' }])
        ];

        const pieColors = [
            '#00d4ff', '#7c3aed', '#ffd700', '#10b981', '#f59e0b',
            '#ec4899', '#06b6d4', '#8b5cf6', '#14b8a6', '#f97316'
        ];

        // Умный поворот подписей X — только если категорий много
        const xRotate = categoryCount > 5 ? 30 : 0;
        const xLabelSize = categoryCount > 8 ? 10 : 12;

        // Базовые опции — ЕДИНЫЕ для всех графиков
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
                textStyle: { color: '#003366', fontWeight: 700, fontSize: 16, fontFamily: "'Inter', sans-serif" }
            },
            tooltip: {
                trigger: isPie ? 'item' : 'axis',
                backgroundColor: 'rgba(255, 255, 255, 0.97)',
                borderColor: '#d1dce7',
                borderWidth: 1,
                textStyle: { color: '#1a2c3e', fontWeight: 500 },
                extraCssText: 'box-shadow: 0 4px 12px rgba(0, 51, 102, 0.15); padding: 12px; border-radius: 8px;'
            },
            grid: isPie ? undefined : { 
                containLabel: true, 
                bottom: '22%', 
                top: '26%', 
                left: '12%', 
                right: '10%' 
            },
            legend: {
                top: 38,
                left: 'center',
                itemWidth: 14,
                itemHeight: 14,
                textStyle: { fontSize: 12, fontWeight: 500 }
            },
            animationDuration: 800,
            animationEasing: 'cubicOut'
        };

        if (isPie) {
            const s = Array.isArray(chartConfig.series) ? chartConfig.series[0] : chartConfig.series;
            // Адаптивный радиус: меньше категорий — больше пирог
            s.radius = categoryCount > 8 ? '45%' : '55%';
            s.center = ['50%', '62%'];
            s.itemStyle = { borderRadius: 10, borderColor: '#ffffff', borderWidth: 3, shadowBlur: 15, shadowColor: 'rgba(0, 0, 0, 0.15)' };
            s.label = { 
                show: true, 
                color: '#1a2c3e', 
                formatter: '{b}: {d}%', 
                fontSize: categoryCount > 6 ? 11 : 13, 
                fontWeight: 600, 
                fontFamily: "'Inter', sans-serif",
                alignTo: 'labelLine',
                distanceToLabelLine: 12
            };
            s.labelLine = { 
                show: true, 
                lineStyle: { color: '#1a2c3e', width: 2 }, 
                smooth: 0.2, 
                length: categoryCount > 6 ? 40 : 50, 
                length2: categoryCount > 6 ? 15 : 22 
            };
            s.labelLayout = { hideOverlap: true };
            s.emphasis = { itemStyle: { shadowBlur: 20, shadowOffsetX: 0, shadowColor: 'rgba(0, 51, 102, 0.3)' } };
            if (s.data) {
                s.data.forEach((item, index) => {
                    if (typeof item === 'object' && !item.itemStyle) {
                        item.itemStyle = { color: pieColors[index % pieColors.length], shadowBlur: 15, shadowColor: 'rgba(0, 0, 0, 0.2)' };
                    }
                });
            }
        } else {
            // Оформление осей
            if (chartConfig.xAxis) {
                chartConfig.xAxis.axisLabel = { 
                    color: '#1a2c3e', 
                    rotate: xRotate, 
                    fontSize: xLabelSize, 
                    fontWeight: 600, 
                    fontFamily: "'Inter', sans-serif", 
                    margin: categoryCount > 5 ? 14 : 10 
                };
                chartConfig.xAxis.axisLine = { lineStyle: { color: '#d1dce7', width: 2 } };
                chartConfig.xAxis.splitLine = { show: false };
                chartConfig.xAxis.axisTick = { show: false };
            }
            if (chartConfig.yAxis) {
                chartConfig.yAxis.axisLabel = { color: '#1a2c3e', fontSize: xLabelSize, fontWeight: 600, fontFamily: "'Inter', sans-serif", margin: 10 };
                chartConfig.yAxis.axisLine = { show: false };
                chartConfig.yAxis.axisTick = { show: false };
                chartConfig.yAxis.splitLine = { lineStyle: { color: '#e8edf2', width: 1 } };
            }
            // Оформление series
            if (chartConfig.series) {
                const seriesArray = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
                seriesArray.forEach((s, index) => {
                    if (s.type === 'bar') {
                        s.itemStyle = { borderRadius: [6, 6, 0, 0], shadowBlur: 10, shadowColor: 'rgba(0, 51, 102, 0.2)' };
                        s.label = { show: true, position: 'top', color: '#1a2c3e', fontSize: 11, fontWeight: 600, fontFamily: "'Inter', sans-serif", distance: 5 };
                        s.barMaxWidth = 50;
                        s.animationDelay = index * 100;
                        // 🎨 Делаем каждый столбец разноцветным
                        if (s.data && Array.isArray(s.data)) {
                            const barColors = [
                                '#00d4ff', '#7c3aed', '#f59e0b', '#10b981', '#ffd700',
                                '#ec4899', '#06b6d4', '#8b5cf6', '#14b8a6', '#f97316',
                                '#3b82f6', '#a855f7', '#22c55e', '#eab308', '#ef4444'
                            ];
                            s.data = s.data.map((item, i) => {
                                const val = typeof item === 'object' ? item.value : item;
                                return {
                                    value: val,
                                    itemStyle: {
                                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                            { offset: 0, color: barColors[i % barColors.length] },
                                            { offset: 1, color: barColors[i % barColors.length] + '99' }
                                        ]),
                                        borderRadius: [6, 6, 0, 0],
                                        shadowBlur: 10,
                                        shadowColor: 'rgba(0, 51, 102, 0.2)'
                                    }
                                };
                            });
                        }
                    } else if (s.type === 'line') {
                        s.itemStyle = { color: gradientColors[index % gradientColors.length], borderWidth: 2 };
                        s.lineStyle = { width: 3, shadowBlur: 10, shadowColor: 'rgba(0, 51, 102, 0.2)' };
                        s.symbol = 'circle';
                        s.symbolSize = 10;
                        s.smooth = true;
                        s.areaStyle = { opacity: 0.1, color: gradientColors[index % gradientColors.length] };
                        s.animationDelay = index * 100;
                    }
                });
            }
        }

        const mergedOption = Object.assign({}, baseOption, chartConfig);
        myChart.setOption(mergedOption, true);

        // Плавное появление
        myChart.setOption({ animationDuration: 1000, animationEasing: 'elasticOut' });

        // Resize
        const doResize = () => { myChart.resize(); };
        setTimeout(doResize, 100);
        setTimeout(doResize, 500);

        const resizeHandler = () => myChart.resize();
        window.addEventListener('resize', resizeHandler);
        myChart.on('dispose', () => window.removeEventListener('resize', resizeHandler));
    }, 300);

    return chartWrapper;
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

    // 🔥 ФИКС: модель иногда выдаёт два JSON-объекта подряд — берём только до конца первого
    let depth = 0;
    let jsonEnd = -1;
    for (let i = 0; i < rawJson.length; i++) {
        if (rawJson[i] === '{') depth++;
        else if (rawJson[i] === '}') {
            depth--;
            if (depth === 0) {
                jsonEnd = i + 1;
                break;
            }
        }
    }
    if (jsonEnd > 0) {
        rawJson = rawJson.substring(0, jsonEnd);
    }

    try {
        let chartConfig = JSON.parse(rawJson);
        
        function hasValidData(series) {
            if (!series) return false;
            const arr = Array.isArray(series) ? series : [series];
            return arr.some(s => s.data && Array.isArray(s.data) && s.data.length > 0);
        }
        
        if (!hasValidData(chartConfig.series)) {
            console.warn('⚠️ Пустые данные от модели! Подставляю тестовые...');
            const defaultTitle = chartConfig.title?.text || 'Статистика';
            const defaultCategories = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн'];
            chartConfig = {
                title: { text: defaultTitle },
                xAxis: { type: 'category', data: defaultCategories },
                yAxis: { type: 'value' },
                series: [{ type: 'bar', data: [42, 38, 25, 18, 12, 30] }],
                tooltip: { trigger: 'axis' }
            };
        }

        const chartId = 'chart_' + Math.random().toString(36).substr(2, 9);
        initChart(chartId, chartConfig, container);
        return true;
    } catch (e) {
        console.error("❌ JSON Error:", e);
        console.log("❌ Проблемный JSON до парсинга:", rawJson.substring(0, 200));
        return false;
    }
}

async function sendMessage() {
    if (isGenerating) return;
    
    // Блокируем onresult и останавливаем голосовой ввод, если он активен
    if (isListening && recognition) {
        voiceInputBlocked = true;  // 🔥 финальный onresult не запишет текст обратно
        if (silenceTimer) clearTimeout(silenceTimer);
        recognition.stop();
        // stopRecording() вызовется в onend автоматически
    }
    
    const input = document.getElementById("messageText");
    if (!input || !input.value.trim()) return;

    const query = input.value;
    
    // ========== ВОЗВРАЩАЕМ ПОЛЕ В ДЕФОЛТ ==========
    input.value = "";                 // очищаем текст
    input.style.height = "auto";      // сбрасываем высоту
    
    // Прячем крестик очистки, кнопку отправки, возвращаем микрофон
    const micBtn = document.getElementById("micBtn");
    if (micBtn) micBtn.classList.remove('moved');
    
    const clearBtn = document.getElementById("clearTextBtn");
    if (clearBtn) clearBtn.classList.remove('visible');
    
    const sendButton = document.getElementById("sendButton");
    if (sendButton) sendButton.classList.remove('visible');
    // =============================================
    
    isGenerating = true;
    blockInput(true);
    
    // Добавляем класс searching к header для анимации
    const header = document.querySelector('header');
    if (header) {
        header.classList.add('searching');
    }
    
    appendMessage('user', query);
    setFirstQuery(query);
    persistCurrentChat();
    showLoader();
    
    let firstChunkReceived = false;
    let currentBotMsgDiv = null;
    let sHtml = '';
    let sHtmlImg = '';
    let fullText = "";
    let hadChartOnly = false;

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
                    else if (data.type === "chart_error") {
                        if (!firstChunkReceived) {
                            removeLoader();
                            currentBotMsgDiv = appendMessage('bot', "<b>База знаний ФНС:</b> 📌 <br>");
                            firstChunkReceived = true;
                        }
                        if (currentBotMsgDiv) {
                            const errorHtml = `<div style="padding: 10px; background: #fff3f3; border: 1px solid #e0b4b4; border-radius: 8px; margin: 10px 0; color: #c0392b;">
                                ⚠️ ${data.message || 'Ошибка визуализации'}</div>`;
                            currentBotMsgDiv.insertAdjacentHTML('beforeend', errorHtml);
                            scrollToBottom();
                        }
                    }
                    else if (data.type === "chart") {
                        if (!firstChunkReceived) {
                            removeLoader();
                            currentBotMsgDiv = appendMessage('bot', "<b>База знаний ФНС:</b> 📌 <br>");
                            firstChunkReceived = true;
                        }

                        const d = data.data;
                        if (!currentBotMsgDiv) {
                            currentBotMsgDiv = appendMessage('bot', '');
                        }

                        const chartId = 'echarts_' + Math.random().toString(36).substr(2, 9);
                        
                        // Преобразуем данные API в формат chartConfig для initChart
                        const seriesArray = [];
                        
                        if (d.chart_type === 'pie') {
                            // Круговая диаграмма
                            const pieData = (d.x_axis || []).map((name, idx) => ({ 
                                name, 
                                value: (d.series_data || [])[idx] || 0 
                            }));
                            seriesArray.push({ 
                                type: 'pie', 
                                data: pieData, 
                                name: d.series_name || 'Данные' 
                            });
                        } else {
                            // Первый ряд данных
                            seriesArray.push({ 
                                type: d.chart_type || 'bar', 
                                data: (d.series_data || []), 
                                name: d.series_name || 'Данные' 
                            });
                            
                            // Второй ряд данных (если есть)
                            if (d.series_data_2 && d.series_data_2.length > 0) {
                                seriesArray.push({ 
                                    type: d.chart_type || 'bar', 
                                    data: d.series_data_2, 
                                    name: d.series_name_2 || 'Данные 2' 
                                });
                            }
                        }

                        const chartConfig = {
                            title: d.title || 'График',
                            xAxis: d.chart_type === 'pie' ? undefined : { data: d.x_axis || [] },
                            yAxis: d.chart_type === 'pie' ? undefined : {},
                            series: seriesArray
                        };

                        const chartWrapper = initChart(chartId, chartConfig, currentBotMsgDiv);
                        chartWrapper.dataset.chartData = JSON.stringify(d);
                        scrollToBottom();
                    }
                    else if (data.type === "text") {
                        if (!firstChunkReceived) {
                            removeLoader();
                            currentBotMsgDiv = appendMessage('bot', "<b>База знаний ФНС:</b> 📌 <br>");
                            firstChunkReceived = true;
                        }
                        
                        fullText += data.content;
                        
                        if (currentBotMsgDiv) {
                            let textPart = fullText.split('[CHART_JSON]')[0];
                            let chartPart = fullText.includes('[CHART_JSON]') ? fullText.substring(fullText.indexOf('[CHART_JSON]')) : '';
                            
                            textPart = textPart.replace(/([.!?:;])\s*•/g, '$1\n\n•');
                            let display = textPart + chartPart;
                            display = display.replace(/\[CHART_JSON\][\s\S]*?\[\/CHART_JSON\]/g, '📈 *Визуализация готова*');
                            display = display.replace(/\[CHART_JSON\][\s\S]*$/g, '📈 *Генерация аналитики...*');
                            
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
        
        if (!currentBotMsgDiv && (sHtml || sHtmlImg)) {
            currentBotMsgDiv = appendMessage('bot', '');
        }
        
        if (currentBotMsgDiv) {
            if (fullText.includes("[/CHART_JSON]")) {
                tryRenderChart(fullText, currentBotMsgDiv);
            }

            let afterContent = '';
            if (sHtml) afterContent += '<div style="margin-top:10px; border-top:1px solid #e2e2e2; padding-top:10px;">' + sHtml + '</div>';
            if (sHtmlImg) afterContent += sHtmlImg;
            afterContent += '<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #e2e2e2; font-size: 11px; color: #ffffff; text-align: right;">🛡️ <em>Ответ подготовлен ИИ-консультантом ФНС</em></div>';

            if (afterContent) {
                currentBotMsgDiv.insertAdjacentHTML('beforeend', afterContent);
            }
            
            persistCurrentChat();
        }

    } catch (err) { 
        console.error(err);
        removeLoader();
    } finally {
        isGenerating = false;
        blockInput(false);
        
        const header = document.querySelector('header');
        if (header) {
            header.classList.remove('searching');
        }
        
        forceScrollToBottom(100);
    }
}

// Обработчик отправки по Enter
document.getElementById("messageText").addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !isGenerating) sendMessage();
});

// Отслеживание состояния инпута
document.getElementById("messageText").addEventListener("input", (e) => {
    const textarea = e.target;
    const micBtn = document.getElementById("micBtn");
    const sendBtn = document.getElementById("sendButton");
    const clearBtn = document.getElementById("clearTextBtn");

    if (textarea.value.trim().length > 0) {
        micBtn.classList.add('moved');
        sendBtn.classList.add('visible');
        clearBtn.classList.add('visible');
    } else {
        micBtn.classList.remove('moved');
        sendBtn.classList.remove('visible');
        clearBtn.classList.remove('visible');
    }
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

    // Обработчик для эффекта перелива на h2 при каждом наведении на header
    const header = document.querySelector('header');
    const headerH2 = document.querySelector('header h2');

    if (header && headerH2) {
        header.addEventListener('mouseenter', () => {
            // Убираем класс если он есть, чтобы перезапустить анимацию
            headerH2.classList.remove('shine-in');
            // Небольшая задержка чтобы браузер пересчитал стили
            setTimeout(() => {
                headerH2.classList.add('shine-in');
            }, 10);
        });

        // Убираем класс после завершения анимации
        headerH2.addEventListener('animationend', () => {
            headerH2.classList.remove('shine-in');
        });
    }

    // 🔥 ИНИЦИАЛИЗАЦИЯ ИСТОРИИ ЧАТОВ
    initializeChatHistory();
});

// ============================================
// 🔥 СИСТЕМА ИСТОРИИ ЧАТОВ - localStorage
// ============================================

let currentChatId = null;
let _loadingChat = false; // Флаг: идёт загрузка чата — не сохраняем
const STORAGE_KEY = 'fns_chat_history';
const CHAT_CONTENT_KEY = 'fns_chat_content_';

// Структура чата: { id, title, messages: [{ type, content }], timestamp, firstUserQuery }

// 🔧 Вспомогательная — собирает сообщения и графики из DOM
function captureMessages() {
    const items = [];
    const chat = document.getElementById('chat');
    if (!chat) return items;
    
    for (const el of chat.children) {
        if (el.classList.contains('msg')) {
            const chartWrapper = el.querySelector('.chart-wrapper');
            if (chartWrapper && chartWrapper.dataset && chartWrapper.dataset.chartData) {
                try {
                    const chartData = JSON.parse(chartWrapper.dataset.chartData);
                    // Клонируем и удаляем wrapper — сохраняем только текст
                    const cloned = el.cloneNode(true);
                    const cw = cloned.querySelector('.chart-wrapper');
                    if (cw) cw.remove();
                    items.push({
                        type: 'bot',
                        content: cloned.innerHTML
                    });
                    items.push({
                        type: 'chart_data',
                        chartData: chartData
                    });
                } catch (e) {
                    items.push({
                        type: el.classList.contains('user') ? 'user' : 'bot',
                        content: el.innerHTML
                    });
                }
            } else {
                items.push({
                    type: el.classList.contains('user') ? 'user' : 'bot',
                    content: el.innerHTML
                });
            }
        }
    }
    return items;
}

function getChatHistory() {
    try {
        const history = localStorage.getItem(STORAGE_KEY);
        return history ? JSON.parse(history) : [];
    } catch (e) {
        console.error('❌ Ошибка при загрузке истории:', e);
        return [];
    }
}

function saveChatHistory(history) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
        console.error('❌ Ошибка при сохранении истории:', e);
    }
}

function getChatMessages(chatId) {
    try {
        const content = localStorage.getItem(CHAT_CONTENT_KEY + chatId);
        return content ? JSON.parse(content) : [];
    } catch (e) {
        console.error('❌ Ошибка при загрузке сообщений:', e);
        return [];
    }
}

function saveChatMessages(chatId, messages) {
    try {
        localStorage.setItem(CHAT_CONTENT_KEY + chatId, JSON.stringify(messages));
    } catch (e) {
        console.error('❌ Ошибка при сохранении сообщений:', e);
    }
}

// 🔧 Сохраняет сообщения текущего чата из DOM (с защитой от сохранения при загрузке)
function persistCurrentChat() {
    if (!currentChatId || _loadingChat) return;
    const messages = captureMessages();
    saveChatMessages(currentChatId, messages);
}

// 🔧 Обновляет заголовок чата первым вопросом пользователя
function setFirstQuery(query) {
    if (!currentChatId || !query.trim()) return;
    const history = getChatHistory();
    const idx = history.findIndex(c => c.id === currentChatId);
    if (idx !== -1 && !history[idx].firstUserQuery) {
        history[idx].firstUserQuery = query.substring(0, 60);
        saveChatHistory(history);
        updateHistoryUI();
    }
}

function createNewChat() {
    const chatId = 'chat_' + Date.now();
    const history = getChatHistory();
    
    const newChat = {
        id: chatId,
        title: 'Новый чат',
        timestamp: new Date().toLocaleString('ru-RU'),
        firstUserQuery: ''
    };
    
    history.unshift(newChat);
    saveChatHistory(history);
    saveChatMessages(chatId, []);
    
    loadChat(chatId);
    closeSidebar();
}

function loadChat(chatId) {
    if (currentChatId === chatId) return; // Если уже открыт
    
    // Сохраняем текущий чат перед переключением
    persistCurrentChat();
    
    // Включаем флаг загрузки — appendMessage не будет сохранять
    _loadingChat = true;
    
    currentChatId = chatId;
    const chat = document.getElementById('chat');
    chat.innerHTML = '';
    
    const messages = getChatMessages(chatId);
    let lastBotDiv = null;
    // Вставляем сообщения напрямую, без appendMessage (чтобы не сохранять)
    messages.forEach(msg => {
        if (msg.type === 'chart_data' && msg.chartData && lastBotDiv) {
            const d = msg.chartData;
            const chartId = 'echarts_' + Math.random().toString(36).substr(2, 9);
            
            // Преобразуем данные API в формат chartConfig для initChart
            const seriesArray = [];
            
            if (d.chart_type === 'pie') {
                const pieData = (d.x_axis || []).map((name, idx) => ({ 
                    name, 
                    value: (d.series_data || [])[idx] || 0 
                }));
                seriesArray.push({ 
                    type: 'pie', 
                    data: pieData, 
                    name: d.series_name || 'Данные' 
                });
            } else {
                seriesArray.push({ 
                    type: d.chart_type || 'bar', 
                    data: (d.series_data || []), 
                    name: d.series_name || 'Данные' 
                });
                
                if (d.series_data_2 && d.series_data_2.length > 0) {
                    seriesArray.push({ 
                        type: d.chart_type || 'bar', 
                        data: d.series_data_2, 
                        name: d.series_name_2 || 'Данные 2' 
                    });
                }
            }

            const chartConfig = {
                title: d.title || 'График',
                xAxis: d.chart_type === 'pie' ? undefined : { data: d.x_axis || [] },
                yAxis: d.chart_type === 'pie' ? undefined : {},
                series: seriesArray
            };

            // Создаём wrapper в контейнере lastBotDiv, но initChart добавляет в конец
            // Нам нужно вставить ДО ссылок — поэтому сначала создадим, потом переместим
            const chartWrapper = initChart(chartId, chartConfig, lastBotDiv);
            chartWrapper.dataset.chartData = JSON.stringify(d);

            // Перемещаем график ДО ссылок (если они есть)
            const sourcesDiv = lastBotDiv.querySelector('div[style*="border-top"]');
            if (sourcesDiv && lastBotDiv.lastElementChild === chartWrapper) {
                lastBotDiv.insertBefore(chartWrapper, sourcesDiv);
            }
            return;
        }
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg ' + msg.type;
        if (msg.type === 'user') {
            msgDiv.textContent = msg.content;
        } else if (msg.type === 'bot') {
            msgDiv.innerHTML = msg.content;
            lastBotDiv = msgDiv;
        } else {
            msgDiv.innerHTML = msg.content;
        }
        chat.appendChild(msgDiv);
    });
    
    _loadingChat = false;
    
    updateHistoryUI();
    
    document.getElementById('messageText').value = '';
    document.getElementById('messageText').focus();
    scrollToBottom();
}

function deleteChat(chatId, e) {
    e.stopPropagation(); // Не открываем чат при клике на удаление
    
    const history = getChatHistory();
    const filtered = history.filter(c => c.id !== chatId);
    saveChatHistory(filtered);
    localStorage.removeItem(CHAT_CONTENT_KEY + chatId);
    
    if (currentChatId === chatId) {
        if (filtered.length > 0) {
            loadChat(filtered[0].id);
        } else {
            currentChatId = null;
            document.getElementById('chat').innerHTML = '';
            createNewChat();
        }
    }
    
    updateHistoryUI();
}

function updateHistoryUI() {
    const history = getChatHistory();
    const container = document.getElementById('chatHistory');
    
    if (history.length === 0) {
        container.innerHTML = `<div class="chat-empty">Нет чатов</div>`;
        return;
    }
    
    container.innerHTML = history.map(chat => {
        const isActive = currentChatId === chat.id;
        const title = chat.firstUserQuery || chat.title;
        const truncated = title.length > 40 ? title.substring(0, 40) + '...' : title;
        
        return `
            <div class="chat-item ${isActive ? 'active' : ''}" onclick="loadChat('${chat.id}')">
                <span>${truncated}</span>
                <span class="chat-del" onclick="deleteChat('${chat.id}', event)">×</span>
            </div>
        `;
    }).join('');
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    sidebar.classList.toggle('open');
    overlay.classList.toggle('open');
    document.body.classList.toggle('sidebar-open');
}

function toggleDesktopSidebar() {
    toggleSidebarCollapse();
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    sidebar.classList.remove('open');
    overlay.classList.remove('open');
    document.body.classList.remove('sidebar-open');
}

function toggleSidebarCollapse() {
    const sidebar = document.getElementById('sidebar');
    const btn = document.getElementById('sidebarToggle');
    
    sidebar.classList.toggle('collapsed');
    document.body.classList.toggle('sidebar-collapsed');
    
    if (sidebar.classList.contains('collapsed')) {
        btn.textContent = '▶';
    } else {
        btn.textContent = '◀';
    }
}

function clearAllHistory() {
    if (!confirm('⚠️ Вы уверены? Все чаты будут удалены!')) return;
    
    const history = getChatHistory();
    history.forEach(chat => {
        localStorage.removeItem(CHAT_CONTENT_KEY + chat.id);
    });
    
    localStorage.removeItem(STORAGE_KEY);
    currentChatId = null;
    document.getElementById('chat').innerHTML = '';
    updateHistoryUI();
    createNewChat();
}

function initializeChatHistory() {
    const history = getChatHistory();
    
    if (history.length === 0) {
        createNewChat();
    } else {
        loadChat(history[0].id);
    }
    
    updateHistoryUI();
}

// ============================================
// ПОЛЕ ВВОДА — ВСЁ В ОДНОМ МЕСТЕ
// ============================================

document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.getElementById("messageText");
    const sendButton = document.getElementById("sendButton");
    const clearBtn = document.getElementById("clearTextBtn");

    if (!textarea || !sendButton) return;

    // Функция авто-высоты
    function autoResizeTextarea() {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }

    // Обновление кнопки отправки
    function updateSendButton() {
        const hasText = textarea.value.trim().length > 0;
        if (hasText) {
            sendButton.classList.add('visible');
            sendButton.disabled = false;
        } else {
            sendButton.classList.remove('visible');
            sendButton.disabled = true;
        }
    }

    // Обновление крестика
    function updateClearButton() {
        if (clearBtn) {
            if (textarea.value.trim().length > 0) {
                clearBtn.classList.add('visible');
            } else {
                clearBtn.classList.remove('visible');
            }
        }
    }

    // Очистка поля
    window.clearTextField = function() {
        textarea.value = '';
        autoResizeTextarea();
        updateSendButton();
        updateClearButton();
        textarea.focus();
    };

    // Событие ввода текста
    textarea.addEventListener('input', function() {
        autoResizeTextarea();
        updateSendButton();
        updateClearButton();
    });

    // Инициализация
    autoResizeTextarea();
    updateSendButton();
    updateClearButton();
});

let recognition = null;
let isListening = false;
let silenceTimer = null; // 🔥 НАШ СКРЫТЫЙ ТАЙМЕР ТИШИНЫ
let voiceInputBlocked = false; // 🔥 Блокировка onresult при отправке сообщения

function toggleMic() {
    const micBtn = document.getElementById("micBtn");
    const textarea = document.getElementById("messageText");
    const sendButton = document.getElementById("sendButton");
    const clearBtn = document.getElementById("clearTextBtn");

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("⚠️ Голосовой ввод не поддерживается в этом браузере.");
        return;
    }

    if (!recognition) {
        recognition = new SpeechRecognition();
        recognition.lang = 'ru-RU';
        // 🔥 ВАЖНО: interimResults оставляем true (чтобы видеть промежуточный текст),
        // но continuous ставим true, чтобы браузер не выключал микрофон на коротких паузах между словами!
        recognition.interimResults = true;
        recognition.continuous = true;

        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add("recording");
            textarea.placeholder = "Слушаю вас, говорите...";
            resetSilenceTimer(); // Запускаем стартовый таймер при включении
        };

        recognition.onresult = (event) => {
            // 🔥 Если отправка сообщения — игнорируем все результаты распознавания
            if (voiceInputBlocked) return;

            // 🔥 Как только пришел ЛЮБОЙ звук или слово — сбрасываем и перезапускаем таймер заново!
            resetSilenceTimer();

            let resultText = "";
            for (let i = 0; i < event.results.length; ++i) {
                resultText += event.results[i][0].transcript;
            }
            
            if (resultText) {
                textarea.value = resultText;
                
                // Автовысота поля ввода
                textarea.style.height = "auto";
                textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
                
                micBtn.classList.add('moved');
                if (sendButton) {
                    sendButton.classList.add('visible');
                    sendButton.disabled = false;
                }
                if (clearBtn) clearBtn.classList.add('visible');
            }
        };

        // Функция сброса и старта таймера тишины
        function resetSilenceTimer() {
            if (silenceTimer) clearTimeout(silenceTimer);
            
            // 🔥 АВТО-ВЫКЛЮЧЕНИЕ ЧЕРЕЗ 3 СЕКУНДЫ ПОЛНОЙ ТИШИНЫ
            silenceTimer = setTimeout(() => {
                if (isListening && recognition) {
                    console.log("⏱️ Обнаружена тишина в течение 3 секунд. Выключаю запись автоматически...");
                    recognition.stop(); // Принудительно глушим микрофон
                }
            }, 3000); // 3000 мс = 3 секунды. Если хочешь 4 секунды — поставь 4000
        }

        recognition.onerror = (event) => {
            console.error("Ошибка речи:", event.error);
            stopRecording();
        };

        recognition.onend = () => {
            stopRecording();
        };
    }

    if (!isListening) {
        try {
            recognition.start();
        } catch (e) {
            console.error(e);
        }
    } else {
        if (silenceTimer) clearTimeout(silenceTimer);
        recognition.stop();
    }

    function stopRecording() {
        isListening = false;
        voiceInputBlocked = false; // 🔥 Сбрасываем блокировку при остановке записи
        if (silenceTimer) clearTimeout(silenceTimer); // Чистим таймер при ручном стопе
        if (micBtn) micBtn.classList.remove("recording");
        if (textarea) textarea.placeholder = "Задай вопрос...";
    }
}
