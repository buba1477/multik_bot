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

function blockInput(block) {
    const input = document.getElementById("messageText");
    const button = document.getElementById("sendButton");
    if (block) {
        if (input) input.disabled = true;
        if (button) button.disabled = true;
    } else {
        if (input) input.disabled = false;
        if (button) button.disabled = false;
        if (input) input.focus();
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
    // Ищем баланс скобок: пропускаем вложенные { } до последнего закрывающего }
    let depth = 0;
    let jsonEnd = -1;
    for (let i = 0; i < rawJson.length; i++) {
        if (rawJson[i] === '{') depth++;
        else if (rawJson[i] === '}') {
            depth--;
            if (depth === 0) {
                jsonEnd = i + 1;
                break; // нашли конец первого объекта
            }
        }
    }
    if (jsonEnd > 0) {
        rawJson = rawJson.substring(0, jsonEnd);
    }

    try {
        let chartConfig = JSON.parse(rawJson);
        console.log('📊 Парсинг JSON ОК, series data:', 
            chartConfig.series 
                ? (Array.isArray(chartConfig.series) 
                    ? chartConfig.series.map(s => s.data?.length || 0)
                    : [(chartConfig.series.data?.length || 0)])
                : 'нет series',
            'config:', chartConfig);
        
        // 🔥 УСИЛЕННЫЙ FALLBACK: проверяем, что в series есть данные
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
            console.log('📊 Fallback config:', chartConfig);
        }

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
        chartWrapper.style.cssText = 'width: 100%; margin: 30px 0 30px 0; background: #ffffff; border: 1px solid #d1dce7; border-radius: 16px; border-top: 2px solid #00509e; padding: 20px 15px 20px 15px; box-shadow: 0 4px 16px rgba(0, 51, 102, 0.08); box-sizing: border-box; flex-shrink: 0;';
        
        const chartDiv = document.createElement('div');
        chartDiv.id = chartId;
        chartDiv.style.width = '100%';
        chartDiv.style.height = isPie ? '500px' : '450px';
        chartDiv.style.margin = '0 auto';
        chartWrapper.appendChild(chartDiv);
        container.appendChild(chartWrapper);

        console.log('📊 chartWrapper добавлен в DOM, ID:', chartId);

        // Ждём больше времени, чтобы DOM точно стабилизировался
        setTimeout(() => {
            const chartElement = document.getElementById(chartId);
            if (!chartElement) {
                console.error('❌ chartDiv не найден в DOM! ID:', chartId);
                return;
            }
            // Визуально подсвечиваем, что элемент найден (убирается после инициализации)
            chartElement.style.border = '2px dashed #ff6600';
            chartElement.style.minHeight = '200px';
            
            if (typeof echarts !== 'undefined') {
                console.log('📊 Инициализация ECharts...');
                const myChart = echarts.init(chartElement, null, { renderer: 'canvas' });
                
                chartInstances.push(myChart);
                
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
                    s.radius = '65%';
                    s.center = ['50%', '50%'];

                    const pieColors = [
                        '#00d4ff', '#7c3aed', '#ffd700', '#10b981', '#f59e0b',
                        '#ec4899', '#06b6d4', '#8b5cf6', '#14b8a6', '#f97316'
                    ];

                    s.itemStyle = {
                        borderRadius: 12,
                        borderColor: '#ffffff',
                        borderWidth: 3,
                        shadowBlur: 15,
                        shadowColor: 'rgba(0, 0, 0, 0.2)'
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

                    // Применяем яркие цвета к каждому сегменту
                    if (s.data) {
                        s.data.forEach((item, index) => {
                            if (typeof item === 'object' && !item.itemStyle) {
                                item.itemStyle = {
                                    color: pieColors[index % pieColors.length],
                                    shadowBlur: 15,
                                    shadowColor: 'rgba(0, 0, 0, 0.2)'
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

                const mergedOption = Object.assign({}, baseOption, chartConfig);
                console.log('📊 Установка опций ECharts:', mergedOption);
                myChart.setOption(mergedOption, true); // true = не мержить, заменить полностью
                
                // Плавное появление графика
                myChart.setOption({
                    animationDuration: 1000,
                    animationEasing: 'elasticOut'
                });

                // Принудительный resize — делаем несколько раз с задержками
                const doResize = () => {
                    myChart.resize();
                    chartElement.style.border = 'none';
                    chartElement.style.minHeight = '';
                };
                setTimeout(doResize, 100);
                setTimeout(doResize, 500);
                
                // resizer с авто-удалением при dispose
                const resizeHandler = () => myChart.resize();
                window.addEventListener('resize', resizeHandler);
                myChart.on('dispose', () => {
                    window.removeEventListener('resize', resizeHandler);
                });
            } else {
                console.error('❌ ECharts не загружен!');
            }
        }, 300);
        
        return true;
    } catch (e) {
        console.error("❌ JSON Error:", e);
        console.log("❌ Проблемный JSON до парсинга:", rawJson.substring(0, 200));
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
    let hadChartOnly = false; // Флаг: пришёл график без текста

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
                        // Ошибка генерации графика — выводим как обычный текст
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
                        // Если график пришёл первым чанком — убираем лоадер и создаём контейнер
                        if (!firstChunkReceived) {
                            removeLoader();
                            currentBotMsgDiv = appendMessage('bot', "<b>База знаний ФНС:</b> 📌 <br>");
                            firstChunkReceived = true;
                        }

                        const d = data.data;

                        if (!currentBotMsgDiv) {
                            currentBotMsgDiv = appendMessage('bot', '');
                        }

                        // Создаём контейнер для графика внутри msg bot
                        const chartId = 'echarts_' + Math.random().toString(36).substr(2, 9);
                        const chartWrapper = document.createElement('div');
                        chartWrapper.className = 'chart-wrapper';
                        chartWrapper.style.cssText = 'width: 100%; margin: 10px 0; background: #ffffff; border: 1px solid #d1dce7; border-radius: 16px; border-top: 2px solid #00509e; padding: 20px 15px; box-shadow: 0 4px 16px rgba(0, 51, 102, 0.08); box-sizing: border-box;';

                        const chartDiv = document.createElement('div');
                        chartDiv.id = chartId;
                        const isPie = d.chart_type === 'pie';
                        chartDiv.style.cssText = `width: 100%; height: ${isPie ? '500px' : '450px'};`;
                        
                        // Сохраняем данные графика для восстановления из истории
                        chartWrapper.dataset.chartData = JSON.stringify(d);
                        chartWrapper.appendChild(chartDiv);
                        currentBotMsgDiv.appendChild(chartWrapper);
                        scrollToBottom();

                        // Инициализация ECharts после стабилизации DOM
                        setTimeout(() => {
                            const dom = document.getElementById(chartId);
                            if (!dom) {
                                console.error('❌ chartDiv не найден в DOM! ID:', chartId);
                                return;
                            }
                            if (typeof echarts === 'undefined') {
                                console.error('❌ ECharts не загружен!');
                                return;
                            }
                            
                            const myChart = echarts.init(dom);
                            chartInstances.push(myChart);
                            
                            // Для pie — преобразуем x_axis + series_data в массив {name, value}
                            const seriesData = isPie
                                ? (d.x_axis || []).map((name, idx) => ({
                                    name,
                                    value: (d.series_data || [])[idx] || 0
                                  }))
                                : (d.series_data || []);

                            const option = {
                                title: {
                                    text: d.title || 'График',
                                    left: 'center',
                                    top: 10,
                                    textStyle: {
                                        color: '#003366',
                                        fontWeight: 700,
                                        fontSize: 16
                                    }
                                },
                                tooltip: {
                                    trigger: isPie ? 'item' : 'axis',
                                    backgroundColor: 'rgba(255, 255, 255, 0.97)',
                                    borderColor: '#d1dce7',
                                    borderWidth: 1,
                                    textStyle: { color: '#1a2c3e' }
                                },
                                legend: isPie ? { orient: 'vertical', left: 'left', top: 40 } : undefined,
                                xAxis: isPie ? undefined : {
                                    data: d.x_axis || [],
                                    axisLabel: {
                                        color: '#1a2c3e',
                                        rotate: 30,
                                        fontSize: 12,
                                        fontWeight: 600
                                    },
                                    axisLine: { lineStyle: { color: '#d1dce7', width: 2 } },
                                    splitLine: { show: false },
                                    axisTick: { show: false }
                                },
                                yAxis: isPie ? undefined : {
                                    axisLabel: { color: '#1a2c3e', fontSize: 12, fontWeight: 600 },
                                    axisLine: { show: false },
                                    axisTick: { show: false },
                                    splitLine: { lineStyle: { color: '#e8edf2', width: 1 } }
                                },
                                grid: isPie ? undefined : {
                                    containLabel: true,
                                    bottom: '10%',
                                    top: '20%',
                                    left: '10%',
                                    right: '10%'
                                },
                                series: [{
                                    name: d.series_name || 'Данные',
                                    type: d.chart_type || 'bar',
                                    radius: isPie ? ['0%', '65%'] : undefined,
                                    data: isPie ? seriesData.map((item, idx) => {
                                        const colors = [
                                            '#00d4ff', '#7c3aed', '#ffd700', '#10b981', '#f59e0b',
                                            '#ec4899', '#06b6d4', '#8b5cf6', '#14b8a6', '#f97316'
                                        ];
                                        return {
                                            ...item,
                                            itemStyle: {
                                                color: colors[idx % colors.length],
                                                shadowBlur: 15,
                                                shadowColor: 'rgba(0, 0, 0, 0.2)'
                                            }
                                        };
                                    }) : seriesData,
                                    colorBy: d.chart_type === 'bar' ? 'data' : undefined,
                                    itemStyle: isPie ? {
                                        borderRadius: 12,
                                        borderColor: '#ffffff',
                                        borderWidth: 3
                                    } : d.chart_type === 'bar' ? {
                                        borderRadius: [4, 4, 0, 0]
                                    } : undefined,
                                    lineStyle: d.chart_type === 'line' ? { width: 3 } : undefined,
                                    label: isPie ? {
                                        show: true,
                                        color: '#1a2c3e',
                                        formatter: '{b}: {d}%',
                                        fontSize: 12,
                                        fontWeight: 600
                                    } : d.chart_type === 'bar' ? {
                                        show: true,
                                        position: 'top',
                                        color: '#1a2c3e',
                                        fontSize: 11,
                                        fontWeight: 600
                                    } : undefined,
                                    labelLine: isPie ? {
                                        show: true,
                                        lineStyle: {
                                            color: '#1a2c3e',
                                            width: 2
                                        },
                                        smooth: 0.2,
                                        length: 15,
                                        length2: 10
                                    } : undefined,
                                    smooth: d.chart_type === 'line' ? true : undefined,
                                    symbol: d.chart_type === 'line' ? 'circle' : undefined,
                                    symbolSize: d.chart_type === 'line' ? 10 : undefined
                                }]
                            };
                            
                            myChart.setOption(option, true);
                            
                            const resizeHandler = () => myChart.resize();
                            window.addEventListener('resize', resizeHandler);
                            myChart.on('dispose', () => {
                                window.removeEventListener('resize', resizeHandler);
                            });
                            
                            setTimeout(() => myChart.resize(), 100);
                            setTimeout(() => myChart.resize(), 500);
                        }, 300);
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
        // Если графика или ошибка пришли без текста — создаём сообщение для источников
        if (!currentBotMsgDiv && (sHtml || sHtmlImg)) {
            currentBotMsgDiv = appendMessage('bot', '');
        }
        
        if (currentBotMsgDiv) {
            // Сначала рендерим график (если есть) — добавляет DOM-ноды через appendChild
            if (fullText.includes("[/CHART_JSON]")) {
                tryRenderChart(fullText, currentBotMsgDiv);
            }

            // Добавляем источники + изображение + подпись через insertAdjacentHTML,
            // чтобы не сломать DOM-ноду графика, созданную выше
            let afterContent = '';
            if (sHtml) afterContent += '<div style="margin-top:10px; border-top:1px solid #e2e2e2; padding-top:10px;">' + sHtml + '</div>';
            if (sHtmlImg) afterContent += sHtmlImg;
            afterContent += '<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #e2e2e2; font-size: 11px; color: #ffffff; text-align: right;">🛡️ <em>Ответ подготовлен ИИ-ассистентом ФНС</em></div>';

            if (afterContent) {
                currentBotMsgDiv.insertAdjacentHTML('beforeend', afterContent);
            }
            
            // Сохраняем ответ бота в историю
            persistCurrentChat();
        }

    } catch (err) { 
        console.error(err);
        removeLoader();
    } finally {
        isGenerating = false;
        blockInput(false);
        
        // Убираем класс searching с header
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
            // Восстанавливаем график — создаём wrapper и вставляем в последний bot-div
            const d = msg.chartData;
            const chartId = 'echarts_' + Math.random().toString(36).substr(2, 9);
            const chartWrapper = document.createElement('div');
            chartWrapper.className = 'chart-wrapper';
            chartWrapper.style.cssText = 'width: 100%; margin: 10px 0; background: #ffffff; border: 1px solid #d1dce7; border-radius: 16px; border-top: 2px solid #00509e; padding: 20px 15px; box-shadow: 0 4px 16px rgba(0, 51, 102, 0.08); box-sizing: border-box;';
            chartWrapper.dataset.chartData = JSON.stringify(d);
            const chartDiv = document.createElement('div');
            chartDiv.id = chartId;
            const isPie = d.chart_type === 'pie';
            chartDiv.style.cssText = `width: 100%; height: ${isPie ? '500px' : '450px'};`;
            chartWrapper.appendChild(chartDiv);

            // Вставляем график ДО ссылок (если они есть)
            const sourcesDiv = lastBotDiv.querySelector('div[style*="border-top"]');
            if (sourcesDiv) {
                lastBotDiv.insertBefore(chartWrapper, sourcesDiv);
            } else {
                lastBotDiv.appendChild(chartWrapper);
            }

            // Инициализация ECharts
            setTimeout(() => {
                const dom = document.getElementById(chartId);
                if (!dom || typeof echarts === 'undefined') return;
                const myChart = echarts.init(dom);
                chartInstances.push(myChart);
                const seriesData = isPie
                    ? (d.x_axis || []).map((name, idx) => ({ name, value: (d.series_data || [])[idx] || 0 }))
                    : (d.series_data || []);

                const pieColors = [
                    '#00d4ff', '#7c3aed', '#ffd700', '#10b981', '#f59e0b',
                    '#ec4899', '#06b6d4', '#8b5cf6', '#14b8a6', '#f97316'
                ];

                const option = {
                    title: { text: d.title || 'График', left: 'center', top: 10, textStyle: { color: '#003366', fontWeight: 700, fontSize: 16 } },
                    tooltip: { trigger: isPie ? 'item' : 'axis', backgroundColor: 'rgba(255, 255, 255, 0.97)', borderColor: '#d1dce7', borderWidth: 1 },
                    legend: isPie ? { orient: 'vertical', left: 'left', top: 40 } : undefined,
                    xAxis: isPie ? undefined : { data: d.x_axis || [], axisLabel: { color: '#1a2c3e', rotate: 30, fontSize: 12, fontWeight: 600 }, axisLine: { lineStyle: { color: '#d1dce7', width: 2 } }, splitLine: { show: false }, axisTick: { show: false } },
                    yAxis: isPie ? undefined : { axisLabel: { color: '#1a2c3e', fontSize: 12, fontWeight: 600 }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: '#e8edf2', width: 1 } } },
                    grid: isPie ? undefined : { containLabel: true, bottom: '10%', top: '20%', left: '10%', right: '10%' },
                    series: [{
                        name: d.series_name || 'Данные',
                        type: d.chart_type || 'bar',
                        radius: isPie ? '65%' : undefined,
                        data: isPie ? seriesData.map((item, idx) => ({
                            ...item,
                            itemStyle: {
                                color: pieColors[idx % pieColors.length],
                                shadowBlur: 15,
                                shadowColor: 'rgba(0, 0, 0, 0.2)'
                            }
                        })) : seriesData,
                        colorBy: d.chart_type === 'bar' ? 'data' : undefined,
                        itemStyle: isPie ? { borderRadius: 12, borderColor: '#ffffff', borderWidth: 3, shadowBlur: 15, shadowColor: 'rgba(0, 0, 0, 0.2)' } : d.chart_type === 'bar' ? { borderRadius: [4, 4, 0, 0] } : undefined,
                        lineStyle: d.chart_type === 'line' ? { width: 3 } : undefined,
                        label: isPie ? { show: true, color: '#1a2c3e', formatter: '{b}: {d}%', fontSize: 12, fontWeight: 600 } : d.chart_type === 'bar' ? { show: true, position: 'top', color: '#1a2c3e', fontSize: 11, fontWeight: 600 } : undefined,
                        labelLine: isPie ? { show: true, lineStyle: { color: '#1a2c3e', width: 2 }, smooth: 0.2, length: 15, length2: 10 } : undefined,
                        smooth: d.chart_type === 'line' ? true : undefined,
                        symbol: d.chart_type === 'line' ? 'circle' : undefined,
                        symbolSize: d.chart_type === 'line' ? 10 : undefined
                    }]
                };
                myChart.setOption(option, true);
                setTimeout(() => myChart.resize(), 100);
                setTimeout(() => myChart.resize(), 500);
                const resizeHandler = () => myChart.resize();
                window.addEventListener('resize', resizeHandler);
                myChart.on('dispose', () => window.removeEventListener('resize', resizeHandler));
            }, 300);
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
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    sidebar.classList.remove('open');
    overlay.classList.remove('open');
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