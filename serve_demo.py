#!/usr/bin/env python3
"""
Простой HTTP сервер для демонстрации фронтенда с историей чатов
Без зависимостей, использует встроенный Python http.server
"""

import http.server
import socketserver
import os
from pathlib import Path
import webbrowser
import threading
import time

PORT = 3000
DIRECTORY = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def do_GET(self):
        # Если запрос к /, возвращаем HTML с историей
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = self._get_chat_html()
            self.wfile.write(html.encode('utf-8'))
            return
        
        # Для всех остальных файлов используем стандартную логику
        super().do_GET()
    
    def _get_chat_html(self):
        """Возвращает HTML с историей чатов"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Нейроинспектор | ФНС России</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="/static/css/style.css">
    <script src="/static/js/marked.min.js"></script>
    <script src="/static/js/echarts.min.js"></script>
</head>
<body>
<!-- 🔥 БОКОВАЯ ПАНЕЛЬ ИСТОРИИ -->
<div id="sidebar" class="sidebar">
    <div class="sidebar-header">
        <h3>📋 История</h3>
        <button class="sidebar-close-btn" onclick="toggleSidebar()">✕</button>
    </div>
    <button id="newChatBtn" class="new-chat-btn" onclick="createNewChat()">
        ➕ Новый чат
    </button>
    <div id="chatHistory" class="chat-history"></div>
    <div class="sidebar-footer">
        <button class="clear-history-btn" onclick="clearAllHistory()">🗑️ Очистить историю</button>
    </div>
</div>

<!-- Оверлей для закрытия сайдбара -->
<div id="sidebarOverlay" class="sidebar-overlay" onclick="toggleSidebar()"></div>

<div id="container">
    <header style="display: flex; align-items: center; justify-content: center; border-bottom: 4px solid #e2e2e2; margin-bottom: 10px; position: relative;">
        <!-- 🔥 Кнопка меню -->
        <button id="menuBtn" class="menu-toggle-btn" onclick="toggleSidebar()" title="История чатов">☰</button>
        
        <!-- 🔥 Логотип -->
        <a href="/" style="display: flex; align-items: center; text-decoration: none; margin-right: auto; padding-left: 5px;">
        <img src="/static/logo.png" alt="Лого" style="height: 35px; width: auto; object-fit: contain; margin-right: auto; padding-left: 5px;">
        </a>
        <!-- 🔥 Заголовок строго по центру -->
        <h2 style="margin: 0; color: #003366; display: flex; align-items: center; font-size: 22px; position: absolute; left: 50%; transform: translateX(-50%);">
            Нейроинспектор <span style="font-weight: 200; color: #003366; margin-left: 8px;">| ФНС России</span>
        </h2>
    </header>
    
    <div id="chat"></div>
    
    <!-- Главный контейнер панели ввода -->
    <div style="width: 100%; position: relative;">
        <div id="input-area" style="width: 100%; margin-bottom: 0; display: flex; align-items: center; position: relative;">
            <input type="text" id="messageText" placeholder="Задай вопрос..." autocomplete="off" style="width: 100%; height: 44px; box-sizing: border-box;"/>
            <button id="sendButton" onclick="sendMessage()" style="height: 44px; box-sizing: border-box;">➤</button>
            
            <!-- Кнопка очистки -->
            <div style="position: absolute; right: -56px; height: 44px; display: flex; align-items: center;">
                <button id="clearButton" class="clear-btn" data-tooltip="Очистить чат" onclick="clearInput()" style="background: #f1f3f5; border: 1px solid #d1dce7; border-radius: 8px; font-size: 18px; cursor: pointer; height: 44px; width: 44px; display: flex; align-items: center; justify-content: center; box-sizing: border-box; transition: all 0.2s;">🗑️</button>
            </div>
        </div>
    </div>
</div>

<script src="/static/js/script.js"></script>
</body>
</html>'''

if __name__ == '__main__':
    os.chdir(DIRECTORY)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════╗
║  🚀 Нейроинспектор - Демо Сервер запущен!                ║
╠════════════════════════════════════════════════════════════╣
║  📍 Адрес: http://localhost:{PORT}/                        ║
║  📁 Директория: {DIRECTORY}                              
║  ✨ История чатов: включена (localStorage)                ║
║                                                            ║
║  Нажми: Ctrl+C для остановки сервера                      ║
╚════════════════════════════════════════════════════════════╝
""")
        
        # Открываем браузер автоматически через 1 секунду
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}/')
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✅ Сервер остановлен.")
