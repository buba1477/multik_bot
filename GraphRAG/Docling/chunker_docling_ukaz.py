import json
import re
from pathlib import Path
import fitz
from transformers import AutoTokenizer

# --- НАСТРОЙКИ ---
MODEL_PATH = '/home/amlin04/multik_bot/hf_cache/ru-en-RoSBERTa'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MAX_TOKENS = 480
INPUT_PDF = "UKAZ_763.pdf"
OUTPUT_FILE = "UKAZ_763.jsonl"

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        full.append(text)
    return "\n".join(full)

def split_by_main_points(text: str):
    """Режем по основным пунктам: 1., 2., 3., 5., 51., 5¹."""
    pattern = r'(?=\n\d+\.\s|\n\d+[¹²³]\.\s|\n\d+\s*\.\s)'
    sections = re.split(pattern, text)
    return [s for s in sections if s.strip()]

def split_by_sentences(text: str, max_tokens: int) -> list:
    """Режет длинный текст по предложениям, НО НЕ РАЗРЫВАЕТ подпункты."""
    lines = text.split('\n')
    result = []
    current_chunk = []
    for line in lines:
        # Проверяем, не начало ли это подпункта (а), б), 5.1, 5¹.)
        is_subpoint = bool(re.match(r'^\s*[а-я]\)|\d+\.\d+|\d+[¹²³]\.', line))
        test_chunk = current_chunk + [line]
        if count_tokens('\n'.join(test_chunk)) <= max_tokens:
            current_chunk.append(line)
        else:
            # Если не влезает, сохраняем текущий и начинаем новый
            if current_chunk:
                result.append('\n'.join(current_chunk))
            current_chunk = [line]
    if current_chunk:
        result.append('\n'.join(current_chunk))
    return result

def main():
    print(f"📄 Обрабатываю {INPUT_PDF}...")
    raw_text = extract_text_from_pdf(INPUT_PDF)
    
    # 1. Режем по основным пунктам
    rough_sections = split_by_main_points(raw_text)
    
    total = 0
    doc_id = Path(INPUT_PDF).stem.lower()
    all_final_chunks = []
    
    for sec in rough_sections:
        sec = sec.strip()
        if len(sec) < 50:
            continue
        
        first_line = sec.split('\n')[0].strip()
        num_match = re.match(r'^(\d+\.?\s*|\d+[¹²³]\.\s*)', first_line)
        if num_match:
            header = num_match.group(0).strip()
        else:
            header = first_line[:60].strip()
            if not header:
                header = "Пункт"
        
        # Если секция уже влезает в лимит — сохраняем целиком
        if count_tokens(sec) <= MAX_TOKENS:
            chunk_data = {
                "id": f"{doc_id}_{re.sub(r'[^a-zA-Z0-9а-яА-Я]', '_', header)[:30]}_p{1}",
                "title": header[:120],
                "text": f"[УКАЗ № 763] [{header}]\n{sec}",
                "local_img": "",
                "url": "http://kremlin.ru"
            }
            all_final_chunks.append(chunk_data)
            total += 1
        else:
            # Если не влезает — режем по предложениям
            chunks = split_by_sentences(sec, MAX_TOKENS)
            for ci, chunk in enumerate(chunks, 1):
                chunk_data = {
                    "id": f"{doc_id}_{re.sub(r'[^a-zA-Z0-9а-яА-Я]', '_', header)[:30]}_p{ci}",
                    "title": header[:120],
                    "text": f"[УКАЗ № 763] [{header}]\n{chunk}",
                    "local_img": "",
                    "url": "http://kremlin.ru"
                }
                all_final_chunks.append(chunk_data)
                total += 1
    
    # Сохраняем в JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for chunk in all_final_chunks:
            f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"\n🏁 ГОТОВО! Создано {total} чанков → {OUTPUT_FILE}")
    # Проверка токенов
    over_limit = [c for c in all_final_chunks if count_tokens(c['text']) > MAX_TOKENS]
    print(f"\n📊 Статистика: создано {total} чанков, из них {len(over_limit)} превышают лимит {MAX_TOKENS} токенов.")

if __name__ == "__main__":
    main()