FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# --- ШАГ 2: СТАВИМ СТАБИЛЬНУЮ СВЯЗКУ ---
RUN pip install --no-cache-dir \
    "transformers>=4.44.0" \
    "sentence-transformers>=3.1.0" \
    "llama-index-core" \
    "llama-index-embeddings-huggingface" \
    "llama-index-llms-ollama" \
    "llama-index-postprocessor-flag-embedding-reranker" \
    "FlagEmbedding" \
    fastapi uvicorn

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
