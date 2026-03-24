from sentence_transformers import SentenceTransformer
import os
import torch

# Твой путь к кешу
CACHE_DIR = "./hf_cache/multilingual-e5-base"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

print("🚀 Начинаю загрузку e5-base строго на CPU...")

# Явно указываем device='cpu'
model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')

model.save(CACHE_DIR)
print(f"✅ Модель сохранена в: {CACHE_DIR}")