import json
import re
from pathlib import Path
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter

# --- CONFIG ---
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/multilingual-e5-large"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
MAX_TOKENS = 400
INPUT_PDF = "79-FZ.pdf"
OUTPUT_FILE = "final_base_v57.jsonl"

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def extract_full_article_number(raw_header):
    """
    Извлекает полный номер статьи и ОСТАТОК названия.
    Возвращает (номер, чистое_название)
    """
    # Ищем "Статья" и цифры/точки после неё
    match = re.search(r'Статья\s+([\d\.\s]+)', raw_header)
    if not match:
        return None, None
    
    raw_num_part = match.group(0) # "Статья 1 " или "Статья 60 1"
    num_only = match.group(1).strip()
    
    # Нормализуем номер для ID (60 1 -> 60.1)
    safe_num = re.sub(r'\s+', '.', num_only).rstrip('.')
    
    # Вырезаем "Статья N" из заголовка, чтобы получить только текст названия
    title_text = raw_header.replace(raw_num_part, '', 1).strip()
    # Убираем решетки и лишние точки в начале названия
    title_text = re.sub(r'^[.#\s]+', '', title_text).strip()
    
    return safe_num, title_text

def split_text_strictly(text, prefix, max_t):
    """Режет текст ПРИНУДИТЕЛЬНО, сохраняя лимит токенов."""
    header = prefix if prefix.endswith('\n') else prefix + "\n"
    header_t = count_tokens(header)
    limit = max_t - header_t
    
    parts = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_t = 0

    for part in parts:
        part_t = count_tokens(part)
        
        if part_t > limit:
            if current_chunk:
                chunks.append(header + " ".join(current_chunk))
                current_chunk = []
                current_t = 0
            
            words = part.split()
            sub_chunk = []
            sub_t = 0
            for w in words:
                w_t = count_tokens(w + " ")
                if sub_t + w_t > limit:
                    chunks.append(header + " ".join(sub_chunk))
                    sub_chunk = [w]
                    sub_t = w_t
                else:
                    sub_chunk.append(w)
                    sub_t += w_t
            if sub_chunk:
                current_chunk = sub_chunk
                current_t = sub_t
            continue

        if current_t + part_t > limit:
            chunks.append(header + " ".join(current_chunk))
            current_chunk = [part]
            current_t = part_t
        else:
            current_chunk.append(part)
            current_t += part_t

    if current_chunk:
        chunks.append(header + " ".join(current_chunk))
    
    return chunks

def main():
    print(f"🧐 Взламываю {INPUT_PDF}...")
    converter = DocumentConverter()
    result = converter.convert(INPUT_PDF)
    content = result.document.export_to_markdown()
    
    # Предварительная чистка индексов во всем тексте
    content = re.sub(r'(\d+)\s+(\d+)\s*(?=\.)', r'\1.\2', content)
    content = re.sub(r'(\d+)\s+(\d+)', r'\1.\2', content)
    
    # Режем на разделы
    sections = re.split(r'\n(?=###\s+Статья|##\s+Статья|#\s+Глава|Раздел\s+|Статья\s+\d+)', content)
    
    doc_id = Path(INPUT_PDF).stem.lower()
    total = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for idx, sec in enumerate(sections):
            sec = sec.strip()
            if not sec or len(sec) < 30:
                continue
            
            lines = sec.split('\n')
            header_raw = lines[0].strip()
            
            # Пропускаем главы и разделы (идем по статьям)
            if 'Глава' in header_raw or 'Раздел' in header_raw:
                continue
            
            # Извлекаем нормализованный номер и чистое название
            art_num, art_title = extract_full_article_number(header_raw)
            
            if art_num:
                # Формируем КРАСИВЫЙ заголовок с пробелом (Статья 1 Название)
                clean_header = f"Статья {art_num} {art_title}".strip()
                # Формируем УНИКАЛЬНЫЙ ID для базы (79-fz_st1)
                safe_id = art_num.replace('.', '_')
                base_id = f"{doc_id}_st{safe_id}"
            else:
                # Fallback для неопознанных блоков
                clean_header = re.sub(r'#{1,3}\s*', '', header_raw)[:100].strip()
                base_id = f"{doc_id}_st_unknown_{idx}"
            
            prefix = f"[{doc_id.upper()}] [{clean_header}]"
            body = "\n".join(lines[1:]).strip()
            
            final_segments = split_text_strictly(body, prefix, MAX_TOKENS)
            
            for i, chunk_text in enumerate(final_segments, 1):
                chunk = {
                    "id": f"{base_id}_p{i}",
                    "title": clean_header,
                    "text": chunk_text.strip(),
                    "local_img": "",
                    "url": "http://kremlin.ru"
                }
                f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total += 1
                if total % 50 == 0:
                    print(f"   ... обработано {total} чанков")

    print(f"\n🏁 ИТОГО: {total} чанков. Пробелы в заголовках и уникальные ID пофикшены!")

if __name__ == "__main__":
    main()
