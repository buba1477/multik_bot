FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# --- ШАГ 2: СТАВИМ СТАБИЛЬНУЮ СВЯЗКУ ---
RUN pip install --no-cache-dir \
    "numpy<2" \
    "transformers<4.36.0" \
    "fastapi" "uvicorn" \
    "llama-index-core==0.14.15" \
    "llama-index-llms-ollama==0.9.1" \
    "llama-index-embeddings-huggingface==0.6.1" \
    "sentence-transformers==2.6.1"


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
