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
        <div class="loader-text">Нейроконсультант ищет по базе ФНС</div>
    `;
    chat.appendChild(loaderDiv);
    scrollToBottom();
    return loaderDiv;
}

function removeLoader() {
    const loader = document.getElementById("loading-indicator");
    if (loader) loader.remove();
}

function updateInputButtons() {
    const textarea = document.getElementById("messageText");
    const micBtn = document.getElementById("micBtn");
    const sendBtn = document.getElementById("sendButton");
    const clearBtn = document.getElementById("clearTextBtn");

    if (!textarea || !micBtn || !sendBtn || !clearBtn) return;

    if (textarea.value.trim().length > 0) {
        micBtn.classList.add('moved');
        sendBtn.classList.add('visible');
        clearBtn.classList.add('visible');
    } else {
        micBtn.classList.remove('moved');
        sendBtn.classList.remove('visible');
        clearBtn.classList.remove('visible');
    }
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
            micButton.setAttribute("data-tooltip", "⏳ Подождите, идёт генерация ответа");
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
            micButton.setAttribute("data-tooltip", "Микрофон");
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
    updateInputButtons();

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
function initChart(chartId, chartConfig, container, isDarkMode = null) {
    // Если title пришёл строкой — превращаем в объект
    if (chartConfig.title && typeof chartConfig.title === 'string') {
        chartConfig = { ...chartConfig, title: { text: chartConfig.title } };
    }

    // 🔥 ОПРЕДЕЛЯЕМ ТЕМУ ДО setTimeout
    if (isDarkMode === null) {
        isDarkMode = document.body.classList.contains('dark-mode');
    }

    let isPie = false;
    let categoryCount = 0;
    if (chartConfig.series) {
        const sArr = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
        isPie = sArr[0] && sArr[0].type === 'pie';
        if (isPie && sArr[0]?.data) {
            categoryCount = sArr[0].data.length;
        } else if (chartConfig.xAxis?.data) {
            categoryCount = chartConfig.xAxis.data.length;
        }
    }

    let chartHeight;
    if (isPie) {
        chartHeight = Math.max(450, Math.min(900, categoryCount * 50));
    } else {
        chartHeight = categoryCount > 12 ? 650 : (categoryCount > 8 ? 550 : (categoryCount > 5 ? 420 : 350));
    }

    const chartWrapper = document.createElement('div');
    chartWrapper.className = 'chart-wrapper';
    
    const wrapperBg = isDarkMode ? '#1e293b' : '#ffffff';
    const wrapperBorder = isDarkMode ? '#334155' : '#d1dce7';
    const wrapperShadow = isDarkMode ? '0 4px 16px rgba(0, 0, 0, 0.4)' : '0 4px 16px rgba(0, 51, 102, 0.08)';
    const wrapperBorderTop = isDarkMode ? '#3b82f6' : '#00509e';
    
    chartWrapper.style.cssText = `width: 100%; margin: 10px 0; background: ${wrapperBg}; border: 1px solid ${wrapperBorder}; border-radius: 16px; border-top: 2px solid ${wrapperBorderTop}; padding: 20px 15px; box-shadow: ${wrapperShadow}; box-sizing: border-box; flex-shrink: 0;`;

    const chartDiv = document.createElement('div');
    chartDiv.id = chartId;
    chartDiv.style.cssText = `width: 100%; height: ${chartHeight}px;`;

    chartWrapper.appendChild(chartDiv);
    container.appendChild(chartWrapper);

    // 🔥 ЗАПОМИНАЕМ isDarkMode ДЛЯ ИСПОЛЬЗОВАНИЯ В setTimeout
    // 🔥 ТЕПЕРЬ ДАРКМОД ЖЕСТКО БЕРЕТСЯ ИЗ ХРАНИЛИЩА ПРИ ЛЮБОМ ОБНОВЛЕНИИ
    const darkMode = localStorage.getItem('fns_dark_mode') === '1';


    setTimeout(() => {
        const dom = document.getElementById(chartId);
        if (!dom || typeof echarts === 'undefined') return;
        
        dom.style.height = '450px'; 

        const myChart = echarts.init(dom);
        chartInstances.push(myChart);

        // 🔥 ИСПОЛЬЗУЕМ darkMode ВМЕСТО document.body.classList.contains('dark-mode')
        const textColor = darkMode ? '#E2E8F0' : '#1a2c3e';
        const titleColor = darkMode ? '#93c5fd' : '#003366';
        const axisColor = darkMode ? '#94a3b8' : '#1a2c3e';
        const labelColor = darkMode ? '#E2E8F0' : '#1a2c3e';
        const tooltipBg = darkMode ? 'rgba(30, 41, 59, 0.97)' : 'rgba(255, 255, 255, 0.97)';
        const tooltipBorder = darkMode ? '#475569' : '#d1dce7';
        const tooltipText = darkMode ? '#E2E8F0' : '#1a2c3e';
        const splitLineColor = darkMode ? '#334155' : '#e8edf2';
        const axisLineColor = darkMode ? '#475569' : '#d1dce7';

        const darkPieColors = ['#3b82f6', '#60a5fa', '#93c5fd', '#38bdf8', '#0ea5e9', '#2563eb', '#7c3aed', '#8b5cf6'];
        const lightPieColors = ['#00d4ff', '#7c3aed', '#ffd700', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#8b5cf6'];
        const pieColors = darkMode ? darkPieColors : lightPieColors;

        const darkBarColors = ['#3b82f6', '#60a5fa', '#93c5fd', '#38bdf8', '#0ea5e9', '#2563eb', '#7c3aed', '#8b5cf6', '#f59e0b', '#10b981'];
        const lightBarColors = ['#00d4ff', '#7c3aed', '#f59e0b', '#10b981', '#ffd700', '#ec4899', '#06b6d4', '#8b5cf6', '#14b8a6', '#f97316', '#3b82f6', '#a855f7'];
        const barColors = darkMode ? darkBarColors : lightBarColors;

        const gradientColors = darkMode ? [
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#3b82f6' }, { offset: 1, color: '#1e40af' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#8b5cf6' }, { offset: 1, color: '#5b21b6' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#f59e0b' }, { offset: 1, color: '#b45309' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#10b981' }, { offset: 1, color: '#065f46' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#ec4899' }, { offset: 1, color: '#be185d' }])
        ] : [
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#00d4ff' }, { offset: 1, color: '#00509e' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#7c3aed' }, { offset: 1, color: '#4c1d95' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#ffd700' }, { offset: 1, color: '#b8860b' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#10b981' }, { offset: 1, color: '#059669' }]),
            new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#f59e0b' }, { offset: 1, color: '#d97706' }])
        ];

        const xRotate = categoryCount > 5 ? 25 : 0;
        const xLabelSize = categoryCount > 8 ? 10 : 12;
        const xMargin = categoryCount > 5 ? 14 : 10;
        const gridBottom = categoryCount > 8 ? '18%' : '14%';

        const unitLabel = chartConfig.unit || '';

        const baseOption = {
            backgroundColor: 'transparent',
            textStyle: {
                color: textColor,
                fontWeight: 500,
                fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
            },
            title: {
                text: chartConfig.title?.text || 'Аналитика',
                left: 'center',
                top: 10,
                textStyle: { color: titleColor, fontWeight: 700, fontSize: 16, fontFamily: "'Inter', sans-serif" }
            },
            tooltip: {
                trigger: isPie ? 'item' : 'axis',
                backgroundColor: tooltipBg,
                borderColor: tooltipBorder,
                borderWidth: 1,
                textStyle: { color: tooltipText, fontWeight: 500 },
                extraCssText: darkMode
                    ? 'box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); padding: 12px; border-radius: 8px;'
                    : 'box-shadow: 0 4px 12px rgba(0, 51, 102, 0.15); padding: 12px; border-radius: 8px;',
                formatter: isPie 
                    ? unitLabel 
                        ? function(params) { return params.name + '<br/>' + params.value + ' ' + unitLabel; }
                        : undefined
                    : unitLabel
                        ? function(params) {
                            let result = '<b>' + params[0].axisValue + '</b>';
                            params.forEach(function(p) {
                                result += '<br/>' + p.marker + ' ' + p.seriesName + ': ' + p.value + ' ' + unitLabel;
                            });
                            return result;
                        }
                        : undefined
            },
                grid: isPie ? undefined : { 
                containLabel: true, 
                top: '24%',         // 🎨 Возвращаем простор сверху
                bottom: '5%',      // 🎨 Даем место для подписей осей
                left: '10%',         
                right: '8%' 
            },
            legend: {
                top: 40,            // 🎨 Опускаем легенду пониже от заголовка
                left: 'center',
                itemWidth: 14,
                itemHeight: 14,
                textStyle: { color: textColor, fontSize: 12, fontWeight: 500 }
            },
            animationDuration: 800,
            animationEasing: 'cubicOut'
        };

        if (isPie) {
            const s = Array.isArray(chartConfig.series) ? chartConfig.series[0] : chartConfig.series;
            
            // 🔥 РЕШЕНИЕ 1: Уменьшаем радиус круга (было 55%), чтобы освободить место по бокам для текста
            s.radius = categoryCount > 8 ? '50%' : '58%';
            s.center = ['50%', '58%']; // Чуть приподнимаем к центру нового холста
            
            s.itemStyle = { 
                borderRadius: 10, 
                borderColor: darkMode ? '#1e293b' : '#ffffff', 
                borderWidth: 3, 
                shadowBlur: 15, 
                shadowColor: darkMode ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.15)' 
            };
            s.label = { 
                show: true, 
                color: labelColor, 
                // 🎨 Текст категории сверху, процент снизу через перенос строки \n
                formatter: '{b}\n{d}%', 
                fontSize: categoryCount > 6 ? 11 : 12, 
                fontWeight: 400, 
                fontFamily: "'Inter', sans-serif",
                
                // 🔥 РЕШЕНИЕ 2: Жестко запрещаем обрезать текст троеточием!
                overflow: 'break', 
                width: 140,         // Максимальная ширина текстового блока в пикселях перед переносом строки
                
                alignTo: 'labelLine',
                distanceToLabelLine: 10
            };
            s.labelLine = { 
                show: true, 
                lineStyle: { color: axisColor, width: 2 }, 
                smooth: 0.2, 
                // 🔥 РЕШЕНИЕ 3: Сокращаем ублюдочные длинные линии (было 40-50 пикселей!)
                length: categoryCount > 6 ? 15 : 20, 
                length2: categoryCount > 6 ? 10 : 12 
            };
            s.labelLayout = { hideOverlap: true };
            s.emphasis = { 
                itemStyle: { 
                    shadowBlur: 20, 
                    shadowOffsetX: 0, 
                    shadowColor: darkMode ? 'rgba(59, 130, 246, 0.4)' : 'rgba(0, 51, 102, 0.3)' 
                } 
            };
            if (s.data) {
                s.data.forEach((item, index) => {
                    if (typeof item === 'object' && !item.itemStyle) {
                        item.itemStyle = { 
                            color: pieColors[index % pieColors.length], 
                            shadowBlur: 15, 
                            shadowColor: darkMode ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 0, 0, 0.2)' 
                        };
                    }
                });
            }
            
        } else {
            if (chartConfig.xAxis) {
                chartConfig.xAxis.axisLabel = { 
                    color: axisColor, 
                    rotate: 35,            // 🔥 Снижаем угол до 35 градусов (так текст займет меньше места по высоте)
                    fontSize: 11,          // 🔥 Чуть-чуть уменьшаем шрифт для идеальной компактности
                    fontWeight: 600, 
                    fontFamily: "'Inter', sans-serif", 
                    margin: 18,            // 🔥 Сдвигаем текст вниз от синей линии осей, чтобы буквы не пересекали её
                    interval: 0,           
                    overflow: 'break',     
                    width: 140             // 🔥 РЕШЕНИЕ: Увеличиваем ширину до 140px! Теперь «Консолидированный» влезет целиком!
                };
                chartConfig.xAxis.axisLine = { lineStyle: { color: axisLineColor, width: 2 } };
                chartConfig.xAxis.splitLine = { show: false };
                chartConfig.xAxis.axisTick = { show: false };
            }

            if (chartConfig.yAxis) {
                chartConfig.yAxis.axisLabel = { 
                    color: axisColor, 
                    fontSize: xLabelSize, 
                    fontWeight: 600, 
                    fontFamily: "'Inter', sans-serif", 
                    margin: 10,
                    formatter: unitLabel ? function(val) { return val + ' ' + unitLabel; } : undefined
                };
                chartConfig.yAxis.axisLine = { show: false };
                chartConfig.yAxis.axisTick = { show: false };
                chartConfig.yAxis.splitLine = { lineStyle: { color: splitLineColor, width: 1 } };
            }
            if (chartConfig.series) {
                const seriesArray = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
                seriesArray.forEach((s, index) => {
                    if (s.type === 'bar') {
                        s.itemStyle = { 
                            borderRadius: [6, 6, 0, 0], 
                            shadowBlur: 10, 
                            shadowColor: darkMode ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 51, 102, 0.2)' 
                        };
                        s.label = { 
                            show: true, 
                            position: 'top', 
                            color: labelColor, 
                            fontSize: 11, 
                            fontWeight: 600, 
                            fontFamily: "'Inter', sans-serif", 
                            distance: 5,
                            formatter: unitLabel ? function(params) { return params.value + ' ' + unitLabel; } : undefined
                        };
                        s.barMaxWidth = 50;
                        s.animationDelay = index * 100;
                        if (s.data && Array.isArray(s.data)) {
                            s.data = s.data.map((item, i) => {
                                const val = typeof item === 'object' ? item.value : item;
                                return {
                                    value: val,
                                    itemStyle: {
                                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                            { offset: 0, color: barColors[i % barColors.length] },
                                            { offset: 1, color: barColors[i % barColors.length] + (darkMode ? '66' : '99') }
                                        ]),
                                        borderRadius: [6, 6, 0, 0],
                                        shadowBlur: 10,
                                        shadowColor: darkMode ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 51, 102, 0.2)'
                                    }
                                };
                            });
                        }
                    } else if (s.type === 'line') {
                        s.itemStyle = { color: gradientColors[index % gradientColors.length], borderWidth: 2 };
                        s.lineStyle = { width: 3, shadowBlur: 10, shadowColor: darkMode ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 51, 102, 0.2)' };
                        s.symbol = 'circle';
                        s.symbolSize = 10;
                        s.smooth = true;
                        s.areaStyle = { opacity: 0.1, color: gradientColors[index % gradientColors.length] };
                        s.animationDelay = index * 100;
                    }
                });
                if (!isPie && chartConfig.series && chartConfig.xAxis && chartConfig.xAxis.data && chartConfig.xAxis.data.length > 0) {
                    const sArr = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
                    if (sArr.length === 1 && sArr[0].type === 'bar') {
                        const originalSeries = sArr[0];
                        const categories = chartConfig.xAxis.data;
                        const newSeries = categories.map((catName, idx) => {
                            const dataItem = Array.isArray(originalSeries.data) ? originalSeries.data[idx] : null;
                            const value = typeof dataItem === 'object' ? dataItem.value : (dataItem != null ? dataItem : 0);
                            const origItemStyle = typeof dataItem === 'object' && dataItem.itemStyle ? dataItem.itemStyle : undefined;
                            
                            const seriesData = new Array(categories.length).fill(null);
                            seriesData[idx] = value;
                            
                            return {
                                type: 'bar',
                                name: catName,
                                data: seriesData,
                                barGap: '-100%',
                                barWidth: '70%',
                                itemStyle: origItemStyle || {
                                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                        { offset: 0, color: barColors[idx % barColors.length] },
                                        { offset: 1, color: barColors[idx % barColors.length] + (darkMode ? '66' : '99') }
                                    ]),
                                    borderRadius: [6, 6, 0, 0],
                                    shadowBlur: 10,
                                    shadowColor: darkMode ? 'rgba(0, 0, 0, 0.4)' : 'rgba(0, 51, 102, 0.2)'
                                },
                                label: originalSeries.label ? { ...originalSeries.label } : undefined,
                                barMaxWidth: originalSeries.barMaxWidth || 50,
                                animationDelay: idx * 100
                            };
                        });
                        chartConfig.series = newSeries;
                        baseOption.tooltip.trigger = 'item';
                        baseOption.tooltip.formatter = function(params) {
                            return params.name + '<br/>' + params.value + (unitLabel ? ' ' + unitLabel : '');
                        };
                    }
                }
            }
        }

        const mergedOption = Object.assign({}, baseOption, chartConfig);
        myChart.setOption(mergedOption, true);
        myChart.setOption({ animationDuration: 1000, animationEasing: 'elasticOut' });

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
    updateInputButtons();
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
                            series: seriesArray,
                            unit: d.unit || '',
                            y_axis_label: d.y_axis_label || ''
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
document.getElementById("messageText").addEventListener("input", () => {
    updateInputButtons();
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
    if (currentChatId === chatId) return;
    
    persistCurrentChat();
    _loadingChat = true;
    
    currentChatId = chatId;
    const chat = document.getElementById('chat');
    chat.innerHTML = '';
    
    const messages = getChatMessages(chatId);
    let lastBotDiv = null;
    
    // 🌗 Определяем текущую тему ДО загрузки графиков
    const isDarkMode = document.body.classList.contains('dark-mode');
    
    messages.forEach(msg => {
        if (msg.type === 'chart_data' && msg.chartData && lastBotDiv) {
            const d = msg.chartData;
            const chartId = 'echarts_' + Math.random().toString(36).substr(2, 9);
            
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
                series: seriesArray,
                unit: d.unit || '',
                y_axis_label: d.y_axis_label || ''
            };

            // 🎨 Передаём текущую тему в initChart
            const chartWrapper = initChart(chartId, chartConfig, lastBotDiv, isDarkMode);
            chartWrapper.dataset.chartData = JSON.stringify(d);

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
    updateInputButtons();
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
        updateInputButtons();
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

    micBtn.setAttribute("data-tooltip", "Микрофон");

    if (!recognition) {
        recognition = new SpeechRecognition();
        recognition.lang = 'ru-RU';
        recognition.interimResults = true;
        recognition.continuous = true;

        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add("recording");
            
            const micSvg = micBtn.querySelector('svg');
            
            // 🔥 ДОБАВЛЯЕМ КЛАСС ДЛЯ ПУЛЬСАЦИИ — ОН БУДЕТ РАБОТАТЬ ВСЕГДА!
            if (micSvg) {
                // Убираем старый класс text-slate-500, который мог мешать
                micSvg.classList.remove("text-slate-500");
                // Добавляем классы для пульсации (они не зависят от hover)
                micSvg.classList.add("mic-recording", "text-red-500", "drop-shadow-[0_0_10px_rgba(239,68,68,0.7)]");
                // Если ты используешь Tailwind с анимациями — добавь:
                micSvg.classList.add("animate-pulse");
            }
            
            micBtn.setAttribute("data-tooltip", "⏹ Остановить запись");
            textarea.placeholder = "Слушаю вас, говорите...";
            resetSilenceTimer();
        };

        recognition.onresult = (event) => {
            if (voiceInputBlocked) return;
            resetSilenceTimer();

            let resultText = "";
            for (let i = 0; i < event.results.length; ++i) {
                resultText += event.results[i][0].transcript;
            }
            
            if (resultText) {
                textarea.value = resultText;
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

        function resetSilenceTimer() {
            if (silenceTimer) clearTimeout(silenceTimer);
            silenceTimer = setTimeout(() => {
                if (isListening && recognition) {
                    console.log("⏱️ Тишина 3 секунды. Выключаю запись...");
                    recognition.stop();
                }
            }, 3000);
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
        voiceInputBlocked = false;
        if (silenceTimer) clearTimeout(silenceTimer);
        
        if (micBtn) {
            micBtn.classList.remove("recording");
            
            const micSvg = micBtn.querySelector('svg');
            
            // 🔥 УБИРАЕМ ВСЕ КЛАССЫ ПУЛЬСАЦИИ И ВОЗВРАЩАЕМ ОБЫЧНЫЙ СТИЛЬ
            if (micSvg) {
                micSvg.classList.remove("mic-recording", "text-red-500", "drop-shadow-[0_0_10px_rgba(239,68,68,0.7)]", "animate-pulse");
                micSvg.classList.add("text-slate-500");
            }
            
            micBtn.setAttribute("data-tooltip", "Микрофон");
        }
        if (textarea) textarea.placeholder = "Задай вопрос...";
    }
}

// ============================================
// 🌗 ПЕРЕКЛЮЧЕНИЕ ТЕМЫ (DARK MODE)
// ============================================

// 1. Выносим ультимативное обновление графиков в отдельную функцию
function applyChartsTheme(isDarkMode) {
    const textColor = isDarkMode ? '#E2E8F0' : '#1a2c3e';     
    const titleColor = isDarkMode ? '#93c5fd' : '#003366';    
    const axisLineColor = isDarkMode ? '#475569' : '#cbd5e1'; 
    const splitLineColor = isDarkMode ? '#334155' : '#f1f5f9';

    const darkPieColors = ['#3b82f6', '#60a5fa', '#93c5fd', '#38bdf8', '#0ea5e9'];
    const lightPieColors = ['#003366', '#004080', '#0059b3', '#0073e6', '#3399ff'];
    const piePalette = isDarkMode ? darkPieColors : lightPieColors;

    chartInstances.forEach(ch => {
        if (!ch || typeof ch.setOption !== 'function') return;
        try {
            let optionConfig = {
                textStyle: { color: textColor },
                title: { textStyle: { color: titleColor } },
                legend: { textStyle: { color: textColor } },
                tooltip: {
                    backgroundColor: isDarkMode ? '#1e293b' : '#ffffff',
                    borderColor: isDarkMode ? '#475569' : '#cbd5e1',
                    textStyle: { color: textColor }
                }
            };

            const currentOption = ch.getOption();
            const hasPie = currentOption && currentOption.series && currentOption.series.some(s => s.type === 'pie');

            if (hasPie) {
                optionConfig.color = piePalette;
                optionConfig.series = [{
                    label: { color: textColor },
                    labelLine: { lineStyle: { color: axisLineColor } }
                }];
            } else {
                optionConfig.xAxis = {
                    axisLabel: { color: textColor },
                    axisLine: { lineStyle: { color: axisLineColor } }
                };
                optionConfig.yAxis = {
                    axisLabel: { color: textColor },
                    axisLine: { lineStyle: { color: axisLineColor } },
                    splitLine: { lineStyle: { color: splitLineColor } }
                };
            }

            ch.setOption(optionConfig);
            ch.resize();
        } catch(e) {
            console.error("Ошибка обновления графика:", e);
        }
    });
}

// 2. Основная функция переключения по клику
function toggleDarkMode() {
    const body = document.body;
    const toggleButton = document.getElementById('themeToggle');
    
    body.classList.toggle('dark-mode');
    
    const isDarkMode = body.classList.contains('dark-mode');
    localStorage.setItem('fns_dark_mode', isDarkMode ? '1' : '0');
    
    if (toggleButton) {
        toggleButton.textContent = isDarkMode ? '☀️' : '🌙';
    }
    
    // Дергаем перекраску графиков
    applyChartsTheme(isDarkMode);
}

// 3. Восстанавливаем тему при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('fns_dark_mode');
    const toggleButton = document.getElementById('themeToggle');
    const isDarkMode = (savedTheme === '1');
    
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        if (toggleButton) toggleButton.textContent = '☀️';
    } else {
        document.body.classList.remove('dark-mode');
        if (toggleButton) toggleButton.textContent = '🌙';
    }
    
    // ТАКТИЧЕСКИЙ ХАК: Даем ECharts 100 миллисекунд, чтобы они успели инициализироваться на странице,
    // и сразу после этого жестко накатываем нужные цвета из localStorage
    setTimeout(() => {
        applyChartsTheme(isDarkMode);
    }, 100);
});


// Восстанавливаем тему при загрузке
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('fns_dark_mode');
    const toggleButton = document.getElementById('themeToggle');
    
    if (savedTheme === '1') {
        document.body.classList.add('dark-mode');
        if (toggleButton) {
            toggleButton.textContent = '☀️';
        }
    } else {
        document.body.classList.remove('dark-mode');
        if (toggleButton) {
            toggleButton.textContent = '🌙';
        }
    }
});