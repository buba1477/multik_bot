import json
import re
import os
from pathlib import Path
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# --- OFFLINE FORCE ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- CONFIG ---
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/FRIDA"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MAX_TOKENS = 400
INPUT_PDF = "58-ФЗ.pdf"
OUTPUT_FILE = "58-FZ.jsonl"

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def extract_full_article_number(raw_header):
    match = re.search(r'Статья\s+([\d\.\s]+)', raw_header)
    if not match:
        return None, None
    raw_num_part = match.group(0) 
    num_only = match.group(1).strip()
    safe_num = re.sub(r'\s+', '.', num_only).rstrip('.')
    title_text = raw_header.replace(raw_num_part, '', 1).strip()
    title_text = re.sub(r'^[.#\s]+', '', title_text).strip()
    return safe_num, title_text

def split_text_strictly(text, prefix, max_t):
    """Умный нарезчик текста по логическим пунктам НПА без разрыва слов"""
    header = prefix if prefix.endswith('\n') else prefix + "\n"
    header_t = count_tokens(header)
    limit = max_t - header_t

    # Режем тело статьи на параграфы (пункты закона) по началу строк
    para_pattern = re.compile(r'(?m)^(?=(?:\d+(?:\.\d+|\^[\d\w]+)?[\.\)])\s+|[а-я]\)\s+)')
    paragraphs = para_pattern.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_t = 0

    for para in paragraphs:
        para_t = count_tokens(para)

        # Если один пункт гигантский и превышает лимит — бьем аккуратно по предложениям
        if para_t > limit:
            if current_chunk:
                chunks.append(header + "\n".join(current_chunk))
                current_chunk = []
                current_t = 0
            
            # Добавляем локальный контекст пункта в префикс для разорванных кусков
            item_match = re.match(r'^(\d+(?:\.\d+|\^[\d\w]+)?[\.\)]|[а-я]\)\s+)', para)
            local_context = f" (в части пункта {item_match.group(1).strip()})" if item_match else " (продолжение)"
            extended_header = header.strip() + local_context + "\n"
            ext_limit = max_t - count_tokens(extended_header)

            # Делим длинный пункт по предложениям
            sentences = re.split(r'(?<=[\.\!\?;])\s+(?=[А-ЯA-Z0-9])', para)
            sub_chunk = []
            sub_t = 0
            for sent in sentences:
                sent_t = count_tokens(sent + " ")
                if sub_t + sent_t > ext_limit:
                    if sub_chunk:
                        chunks.append(extended_header + " ".join(sub_chunk))
                    sub_chunk = [sent]
                    sub_t = sent_t
                else:
                    sub_chunk.append(sent)
                    sub_t += sent_t
            if sub_chunk:
                current_chunk = sub_chunk
                current_t = sub_t
            continue

        # Собираем обычные пункты вместе, пока они влезают в лимит токенов
        if current_t + para_t > limit:
            chunks.append(header + "\n".join(current_chunk))
            current_chunk = [para]
            current_t = para_t
        else:
            current_chunk.append(para)
            current_t += para_t

    if current_chunk:
        chunks.append(header + "\n".join(current_chunk))
    return chunks

def main():
    print(f"🧐 Взламываю {INPUT_PDF} в режиме OFFLINE...")
    
    # --- НАСТРОЙКА OFFLINE DOCLING ---
    pipeline_options = PdfPipelineOptions()
    pipeline_options.enable_remote_services = False  # Запрет интернета
    
    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(INPUT_PDF)
    content = result.document.export_to_markdown()
    
    # --- ТОТАЛЬНАЯ ЗАЧИСТКА МУСОРА И АРТЕФАКТОВ РАЗМЕТКИ ---
    # Счищаем мусорные номера строк от Docling перед буквами и цифрами (типа "4. а)" -> "а)")
    content = re.sub(r'(?m)^\s*\d+\.\s+([а-яА-Яa-zA-Z]\))', r'\1', content)
    content = re.sub(r'(?m)^\s*\d+\.\s+(\d+\))', r'\1', content)
    content = re.sub(r'публичноправовой', 'публично-правовой', content)

    
    # Схлопываем множественные пробелы и табы в один пробел
    content = re.sub(r'[ \t]+', ' ', content)
    # Нормализуем переносы строк
    content = re.sub(r'\r\n?', '\n', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Разделение по статьям (учитываем # заголовки Markdown от Docling и начало строк)
    sections = re.split(r'\n(?=###\s+Статья|##\s+Статья|^Статья\s+\d+)', content)
    doc_id = Path(INPUT_PDF).stem.lower().replace(" ", "_").strip()
    total = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for idx, sec in enumerate(sections):
            sec = sec.strip()
            if not sec or len(sec) < 30: 
                continue
                
            lines = sec.split('\n')
            header_raw = lines[0].strip()
            
            # Пропускаем технические разделы и главы, нам нужны только статьи
            if 'Глава' in header_raw or 'Раздел' in header_raw: 
                continue
                
            art_num, art_title = extract_full_article_number(header_raw)
            if art_num:
                clean_header = f"Статья {art_num}. {art_title}".strip()
                safe_id = art_num.replace('.', '_')
                base_id = f"{doc_id}_st{safe_id}"
            else:
                # Если номер статьи не определился (например, преамбула), берем кусок заголовка
                clean_header = re.sub(r'#{1,3}\s*', '', header_raw)[:100].strip()
                base_id = f"{doc_id}_st_unknown_{idx}"
            
            prefix = f"[{doc_id.upper()}] [{clean_header}]"
            body = "\n".join(lines[1:]).strip()
            
            # Запускаем безопасную нарезку тела статьи на чанки
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
                
        print(f"✅ Готово! Успешно создано {total} чистых чанков без разрывов.")

if __name__ == "__main__":
    main()
