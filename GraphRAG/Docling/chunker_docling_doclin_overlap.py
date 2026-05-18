import json
import re
import os
from pathlib import Path
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# --- CONFIG ---
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/ru-en-RoSBERTa"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

MAX_TOKENS = 400
OVERLAP_TOKENS = 60
MIN_CHARS = 100
INPUT_PDF = "58-ФЗ.pdf"
OUTPUT_FILE = "58-FZ.jsonl"

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def normalize_text(text):
    if not text: return ""
    return " ".join(text.split())

def is_garbage(line):
    garbage_titles = {
        'события', 'структура', 'видео и фото', 'документы', 'контакты', 'поиск',
        'вконтакте', 'rutube', 'telegram-канал', 'youtube', 'max', 'подписаться',
        'введите запрос', 'найти', 'все события', 'английском', 'english'
    }
    s = line.strip().lower()
    return len(s) < 3 or s in garbage_titles or s.startswith(('http', 'www', '©'))

def normalize_article_number(art_num: str) -> str:
    """591 → 59.1, 59 1 → 59.1, 59-1 → 59.1, 59<sup>3</sup> → 59.3"""
    art_num = re.sub(r'<[^>]+>', '', art_num)
    art_num = re.sub(r'[\s\-]+', '.', art_num)
    art_num = re.sub(r'\.+', '.', art_num)
    
    if '.' in art_num:
        return art_num
    
    if len(art_num) == 3 and art_num.isdigit():
        return f"{art_num[:2]}.{art_num[2]}"
    
    if len(art_num) == 4 and art_num.isdigit():
        main = art_num[:2]
        sub = art_num[2:]
        if sub.isdigit():
            return f"{main}.{sub}"
    
    return art_num

def extract_article_title(first_line: str) -> tuple:
    """
    Извлекает номер статьи и название.
    Возвращает (номер_статьи, полный_заголовок)
    """
    # Убираем лишние символы
    first_line = re.sub(r'<[^>]+>', '', first_line)
    first_line = normalize_text(first_line)
    
    # Ищем номер статьи
    match = re.search(r'Статья\s+([\d\.\s\-]+)', first_line)
    if not match:
        return None, None
    
    raw_num = match.group(1).strip()
    art_num = normalize_article_number(raw_num)
    
    # Всё, что после номера — это название статьи
    title_part = first_line[match.end():].strip()
    # Убираем префиксы типа "1." в начале названия
    title_part = re.sub(r'^\d+\.\s*', '', title_part)
    # Убираем точку в конце
    title_part = re.sub(r'\.$', '', title_part)
    
    if title_part:
        full_title = f"Статья {art_num} {title_part}"
    else:
        full_title = f"Статья {art_num}"
    
    return art_num, full_title

def split_into_chunks(text, prefix, limit, overlap):
    header = prefix + "\n\n"
    header_len = count_tokens(header)
    max_body_len = limit - header_len
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        s_len = count_tokens(sent)
        
        if s_len > max_body_len:
            if current_chunk:
                chunks.append(header + " ".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            words = sent.split()
            tmp = []
            tmp_len = 0
            for w in words:
                w_len = count_tokens(w + " ")
                if tmp_len + w_len > max_body_len:
                    chunks.append(header + " ".join(tmp))
                    tmp = tmp[-(overlap//10):] if len(tmp) > (overlap//10) else []
                    tmp_len = count_tokens(" ".join(tmp))
                tmp.append(w)
                tmp_len += w_len
            current_chunk = tmp
            current_len = tmp_len
            continue

        if current_len + s_len > max_body_len:
            chunks.append(header + " ".join(current_chunk))
            current_chunk = current_chunk[-1:] if current_chunk else []
            current_len = count_tokens(" ".join(current_chunk)) if current_chunk else 0
            
        current_chunk.append(sent)
        current_len += s_len
        
    if current_chunk:
        chunks.append(header + " ".join(current_chunk))
    
    valid_chunks = []
    for c in chunks:
        c_len = count_tokens(c)
        if c_len > 510:
            mid = len(c) // 2
            valid_chunks.append(c[:mid])
            valid_chunks.append(header + c[mid:])
        else:
            valid_chunks.append(c)
            
    return valid_chunks

def main():
    print(f"🚀 Старт: {INPUT_PDF}")
    converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=PdfPipelineOptions(enable_remote_services=False))})
    
    try:
        result = converter.convert(INPUT_PDF)
        raw_md = result.document.export_to_markdown()
    except Exception as e:
        print(f"❌ Ошибка Docling: {e}")
        return

    lines = raw_md.split('\n')
    valid_lines = [l for l in lines if not is_garbage(l)]
    clean_md = "\n".join(valid_lines)

    sections = re.split(r'(?=Статья\s+[\d\.]+)', clean_md)
    doc_id = Path(INPUT_PDF).stem.lower().replace(' ', '_')
    total = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for idx, sec in enumerate(sections):
            sec = sec.strip()
            if len(sec) < 30:
                continue
            
            lines_in_sec = sec.split('\n')
            first_line = lines_in_sec[0]
            
            art_num, full_title = extract_article_title(first_line)
            
            if art_num:
                base_id = f"{doc_id}_st{art_num.replace('.', '')}"
                # Тело — все строки после первой, но без дублирования заголовка
                body = " ".join(lines_in_sec[1:])
                # Если тело начинается с того же текста, что и заголовок — убираем дубль
                title_clean = re.sub(r'Статья\s+[\d\.]+\.?\s*', '', full_title)
                if body.startswith(title_clean) and len(title_clean) > 5:
                    body = body[len(title_clean):].strip()
                body = re.sub(r'^[.,;:\s]+', '', body)
            else:
                full_title = "Вводная часть"
                base_id = f"{doc_id}_intro_{idx}"
                body = sec

            if not body:
                continue

            prefix = f"[{doc_id.upper()}] [{full_title}]"
            body_clean = normalize_text(body)
            
            final_chunks = split_into_chunks(body_clean, prefix, MAX_TOKENS, OVERLAP_TOKENS)
            
            for p_idx, text in enumerate(final_chunks, 1):
                clean_text = normalize_text(text)
                if len(clean_text) < MIN_CHARS:
                    continue
                
                node = {
                    "id": f"{base_id}_p{p_idx}",
                    "title": full_title,
                    "text": clean_text,
                    "local_img": "",
                    "url": "http://kremlin.ru"
                }
                f.write(json.dumps(node, ensure_ascii=False) + '\n')
                total += 1

    print(f"✅ Готово! Создано чанков: {total}")

if __name__ == "__main__":
    main()