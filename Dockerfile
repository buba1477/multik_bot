# 1. Берем легкий образ Python
FROM python:3.11-slim

# 2. Системные зависимости (добавил build-essential для сборки сложных либ)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Рабочая директория
WORKDIR /app

# 4. Копируем зависимости
COPY requirements.txt .

# 5. ВАЖНО: СТАВИМ ЗЕРКАЛО ДЛЯ PIP!
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=100 --no-cache-dir -r requirements.txt && \
    pip install --default-timeout=100 --no-cache-dir uvicorn fastapi

# 6. Копируем весь твой код (включая engine_rag.py и main_api.py)
COPY . .

# 7. Дефолтная команда (переопределяется в docker-compose.yml для API)
CMD ["python", "main.py"]