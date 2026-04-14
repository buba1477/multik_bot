
    let isGenerating = false;
    
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
            const chartConfig = JSON.parse(rawJson);
            const chartId = 'chart_' + Math.random().toString(36).substr(2, 9);
            
            let isPie = false;
            if (chartConfig.series) {
                const sArr = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
                isPie = sArr[0] && sArr[0].type === 'pie';
            }

            const chartDiv = document.createElement('div');
            chartDiv.id = chartId;
            chartDiv.style.width = '100%';
            chartDiv.style.height = isPie ? '500px' : '450px';
            chartDiv.style.margin = '20px 0';
            container.appendChild(chartDiv);

            setTimeout(() => {
                if (typeof echarts !== 'undefined') {
                    const myChart = echarts.init(document.getElementById(chartId), 'dark');
                    const theme = { text: '#cdd6f4', accent: '#f38ba8' };

                    const baseOption = {
                        backgroundColor: 'transparent',
                        title: {
                            text: chartConfig.title?.text || 'Аналитика',
                            left: 'center',
                            textStyle: { color: theme.text }
                        },
                        tooltip: { trigger: isPie ? 'item' : 'axis' },
                        grid: { containLabel: true, bottom: '15%', top: '15%' }
                    };

                    if (isPie) {
                        const s = Array.isArray(chartConfig.series) ? chartConfig.series[0] : chartConfig.series;
                        s.radius = ['40%', '70%'];
                        s.itemStyle = { borderRadius: 10, borderColor: '#1e1e2e', borderWidth: 2 };
                        s.label = { show: true, color: theme.text, formatter: '{b}: {d}%' };
                    } else {
                        if (chartConfig.xAxis) chartConfig.xAxis.axisLabel = { color: theme.text, rotate: 30 };
                        if (chartConfig.yAxis) chartConfig.yAxis.axisLabel = { color: theme.text };
                        
                        if (chartConfig.series) {
                            const seriesArray = Array.isArray(chartConfig.series) ? chartConfig.series : [chartConfig.series];
                            seriesArray.forEach(s => {
                                if (s.type === 'bar') {
                                    s.colorBy = 'data';
                                    s.itemStyle = { borderRadius: [5, 5, 0, 0] };
                                    s.label = { show: true, position: 'top', color: theme.text };
                                }
                            });
                        }
                    }

                    myChart.setOption(Object.assign(baseOption, chartConfig));
                    window.addEventListener('resize', () => myChart.resize());
                }
            }, 50);
            
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
                            sHtmlImg = '<div style="margin-top:15px; border-top: 1px solid #45475a; padding-top:10px;"><img src="/images/' + data.image + '" style="max-width:100%; border-radius:12px; border: 1px solid #45475a;"></div>';
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
            if (sHtml) currentBotMsgDiv.innerHTML += '<div style="margin-top:10px; border-top:1px solid #585b70; padding-top:10px;">' + sHtml + '</div>';
            if (sHtmlImg) currentBotMsgDiv.innerHTML += sHtmlImg;

            if (fullText.includes("[/CHART_JSON]")) {
                tryRenderChart(fullText, currentBotMsgDiv);
            }

            const signature = '<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #45475a; font-size: 11px; color: #ffffff; text-align: right;">🛡️ <em>Ответ подготовлен ИИ-ассистентом ФНС</em></div>';
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



    document.getElementById("messageText").addEventListener("keypress", (e) => {
        if (e.key === "Enter" && !isGenerating) sendMessage();
    });
