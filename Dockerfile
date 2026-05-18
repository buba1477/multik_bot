FROM python:3.11-slim

# 1. Системные зависимости
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Установка PyTorch CPU (Бетонная ссылка)
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu


# 3. Установка стабильной связки библиотек
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "qdrant-client==1.9.0" \
    "llama-index-core==0.10.55" \
    "llama-index-vector-stores-qdrant==0.1.4" \
    "llama-index-embeddings-huggingface" \
    "llama-index-llms-ollama==0.1.3" \
    "llama-index-postprocessor-sbert-rerank==0.1.3" \
    "sentence-transformers==3.1.1" \
    "transformers==4.44.2" \
    "ollama==0.3.3" \
    "FlagEmbedding" \
    "peft" \
    "rank-bm25" \
    "nltk" \
    "fastapi" \
    "uvicorn"

# 4. 🔥 КРИТИЧНО: Предзагрузка словарей NLTK
# Сначала копируем только скрипт настройки
COPY setup_nltk.py .
# Запускаем его нормально
RUN python setup_nltk.py

# 5. Копируем требования (если есть доп. либы)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# 6. Копируем код
# Благодаря твоему новому .dockerignore сюда попадет только код!
COPY . .

CMD ["python", "main.py"]
