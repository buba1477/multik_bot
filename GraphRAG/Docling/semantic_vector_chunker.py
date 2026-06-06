import json
import os
import re
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# =========================================================
# CONFIGURATION & PATHS
# =========================================================
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/ru-en-RoSBERTa"
INPUT_PDF = "грязный_приказ_без_структуры.pdf"
OUTPUT_FILE = "146-FZ-pure-semantic.jsonl"
URL = "http://kremlin.ru"

# Жесткие лимиты по книге и возможностям твоей 1660 Ti
MAX_TOKENS = 512
MIN_TOKENS = 50
ABSOLUTE_MAX_TOKENS = 512
# Процент отсечения (перцентиль): чем выше число, тем реже бьются чанки (только на жестких стыках тем)
BREAKPOINT_PERCENTILE = 65 

# =========================================================
# INITIALIZATION (AIR-GAPPED ENVIRONMENT)
# =========================================================
os.environ.update({
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1"
})

# Загружаем атомы токенизации и весов Сбера
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)

# Переводим модель Сбера на GPU, если CUDA доступна
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================================================
# MATHEMATICAL & EMBEDDING CORE
# =========================================================
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def get_embedding(text: str) -> np.ndarray:
    """Генерирует вектор для предложения через локальную ru-en-RoSBERTa"""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Беру среднее по эмбеддингам токенов (Mean Pooling) по канонам SberAI
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()[0]

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Вычисляет косинусное сходство между двумя векторами"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def truncate_to_limit(text: str, max_tokens: int) -> str:
    """[BOOK RECIPE 2.3] Предохранитель от битых символов-ромбиков кириллицы"""
    if not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True).strip()

# =========================================================
# TEXT CLEANING & PREPARATION
# =========================================================
def clean_text_basic(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_into_sentences(text: str) -> list:
    """Хирургически режет текст на предложения, учитывая сокращения РФ"""
    text = clean_text_basic(text)
    # Защищаем точки в сокращениях (ФЗ, РФ, г, руб), чтобы не рвать их
    text = re.sub(r'(?<=\s_st46_c4)(ст|пункт|п|г|руб| фз| рф)\.', r'\1__DOT__', text, flags=re.I)
    sentences = re.split(r'(?<=[\.\!\?;])\s+(?=[А-ЯA-Z0-9])', text)
    return [s.replace("__DOT__", ".").strip() for s in sentences if s.strip()]

# =========================================================
# ADVANCED MATHEMATICAL SEMANTIC CHUNKING
# =========================================================
def build_pure_semantic_chunks(sentences: list, prefix: str) -> list:
    """
    [BOOK RECIPE 4.2] Динамический семантический чанкинг на основе 
    косинусного расстояния между соседними предложениями.
    """
    if not sentences:
        return []
        
    print(f"🧠 Расчёт векторов Сбера для {len(sentences)} предложений...")
    embeddings = [get_embedding(s) for s in sentences]
    
    # 1. Считаем семантические расстояния между соседними предложениями
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(embeddings[i], embeddings[i+1])
        distances.append(1.0 - similarity) # Дистанция = 1 - схожесть
        
    if not distances:
        return ["\n\n".join(sentences)]
        
    # 2. Находим порог математического разрыва (Breakpoint Threshold)
    breakpoint_threshold = np.percentile(distances, BREAKPOINT_PERCENTILE)
    
    chunks = []
    current_sentences = [sentences[0]]
    
    # 3. Скользящий обход и нарезка по точкам излома смысла
    for i, distance in enumerate(distances):
        next_sentence = sentences[i + 1]
        
        # Проверяем, сколько токенов займет кусок, если мы добавим это предложение
        test_text = prefix + "\n\n" + " ".join(current_sentences + [next_sentence])
        
        # Условия закрытия чанка: либо прыжок дистанции (смена темы), либо упор в лимит 512 токенов
        if distance > breakpoint_threshold or count_tokens(test_text) > MAX_TOKENS:
            # Сохраняем текущий чанк
            chunk_body = " ".join(current_sentences)
            if count_tokens(prefix + "\n\n" + chunk_body) >= MIN_TOKENS:
                chunks.append(prefix + "\n\n" + chunk_body)
            current_sentences = [next_sentence]
        else:
            current_sentences.append(next_sentence)
            
    if current_sentences:
        chunk_body = " ".join(current_sentences)
        chunks.append(prefix + "\n\n" + chunk_body)
        
    # 4. Финальный прогон через truncate_to_limit для тотальной безопасности
    safe_chunks = []
    for chunk in chunks:
        safe_chunk = truncate_to_limit(chunk, ABSOLUTE_MAX_TOKENS)
        if count_tokens(safe_chunk) >= MIN_TOKENS:
            safe_chunks.append(safe_chunk)
            
    return safe_chunks

# =========================================================
# MAIN EXECUTIVE CONTUR
# =========================================================
def main():
    print(f"🚀 Запуск запасного семантического чанкера для: {INPUT_PDF}")
    
    if not Path(INPUT_PDF).exists():
        print(f"❌ Файл {INPUT_PDF} не найден на диске!")
        return

    # Локальный Air-Gapped Docling без интернета
    converter = DocumentConverter(format_options={
        "pdf": PdfFormatOption(pipeline_options=PdfPipelineOptions(enable_remote_services=False))
    })
    
    try:
        markdown = converter.convert(INPUT_PDF).document.export_to_markdown()
    except Exception as e:
        print(f"❌ Ошибка конвертации Docling: {e}")
        return
        
    print(f"📄 Документ распарсен. Длина: {len(markdown)} символов.")
    
    # Нарезаем весь плоский текст на неделимые атомы-предложения
    sentences = split_into_sentences(markdown)
    print(f"📑 Выделено предложений для анализа: {len(sentences)}")
    
    doc_id = Path(INPUT_PDF).stem.lower().replace(" ", "_").strip()
    prefix = f"[{doc_id.upper()}] [АНАЛИТИЧЕСКИЙ ОБЗОР]"
    
    # Запускаем математический семантический чанкинг
    final_chunks = build_pure_semantic_chunks(sentences, prefix)
    
    print(f"💾 Запись отвалидированных чанков в {OUTPUT_FILE}...")
    total_written = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(final_chunks, 1):
            node = {
                "id": f"{doc_id}_semantic_c{idx}",
                "title": "Аналитические материалы ведомства",
                "text": chunk,
                "local_img": "",
                "url": URL
            }
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
            total_written += 1
            
    print(f"\n{'='*50}")
    print(f"📊 ИТОГИ ЧАНКИНГА ПО КНИГЕ (РЕЦЕПТ 4.2):")
    print(f"✅ Успешно создано математических чанков: {total_written}")
    print(f"🛡️ Все чанки жестко ограничены лимитом в {ABSOLUTE_MAX_TOKENS} токенов.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
 