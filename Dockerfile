FROM python:3.11-slim

# Debian mirror
# deb.debian.org недоступен с текущего хоста, поэтому используем рабочее зеркало
RUN sed -i 's|http://deb.debian.org/debian|http://ftp.nl.debian.org/debian|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|http://deb.debian.org/debian-security|http://ftp.nl.debian.org/debian-security|g' /etc/apt/sources.list.d/debian.sources

# 1. Системные зависимости
# Включая зависимости Chromium для Playwright
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libnss3 \
    libnspr4 \
    libatk1.0-0t64 \
    libatk-bridge2.0-0t64 \
    libcups2t64 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 3. Основной стек Python
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
    "fastapi==0.115.0" \
    "uvicorn==0.30.6"

# 4. NLTK DATA — локально, без скачивания из интернета
# LlamaIndex содержит собственный NLTK cache
ENV NLTK_DATA=/usr/local/lib/python3.11/site-packages/llama_index/core/_static/nltk_cache

# 5. Дополнительные зависимости проекта
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# 6. Bundled Chromium для Playwright HTML -> PDF
RUN python -m playwright install chromium 2>&1 \
    || echo "Playwright browsers install skipped"

# 7. Копируем код проекта
COPY . .

CMD ["python", "main.py"]