import json
import re
import os
from pathlib import Path
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter

# --- КОНФИГ ---
MODEL_NAME = 'intfloat/multilingual-e5-large'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS = 450

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def split_body_smart(body, prefix):
    """Твоя логика: нарезка с сохранением локального контекста (двоеточий)"""
    if not body: return []
    
    # Пытаемся выцепить преамбулу до первого списка
    preamble_match = re.search(r'^(.*?)(?=\n\d+\.\s)', body, re.DOTALL)
    local_context = preamble_match.group(1).strip() if preamble_match else ""
    if len(local_context) > 150: local_context = local_context[:147] + "..."
    
    lines = body.split('\n')
    chunks = []
    current_batch = []
    
    def format_chunk(ctx, lines_list):
        header = f"{prefix}КОНТЕКСТ: {ctx}\n" if ctx else prefix
        return header + "\n".join(lines_list)

    current_tokens = count_tokens(format_chunk(local_context, []))

    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Логика "двоеточия" — обновляем контекст
        if line.endswith(':') and len(line) > 10:
            if current_batch:
                chunks.append(format_chunk(local_context, current_batch))
                current_batch = []
            local_context = line
            current_tokens = count_tokens(format_chunk(local_context, []))
            continue

        line_tokens = count_tokens(line)
        if current_tokens + line_tokens > MAX_TOKENS and current_batch:
            chunks.append(format_chunk(local_context, current_batch))
            current_batch = [line]
            current_tokens = count_tokens(format_chunk(local_context, [])) + line_tokens
        else:
            current_batch.append(line)
            current_tokens += line_tokens
            
    if current_batch:
        chunks.append(format_chunk(local_context, current_batch))
    return chunks

def process_with_docling(input_pdf, output_jsonl, doc_metadata):
    print(f"🚀 Запуск Docling для: {input_pdf}")
    
    # 1. Конвертация PDF в структурированный Markdown через Docling
    converter = DocumentConverter()
    result = converter.convert(input_pdf)
    md_content = result.document.export_to_markdown()

    # 2. Твоя логика разбора глав и статей (уже по Markdown)
    content = re.sub(r'[ \t]+', ' ', md_content)
    # Docling делает заголовки через # или ##
    articles = re.split(r'\n(?=#+\s+Статья|#+\s+Глава|#+\s+Раздел)', content)
    
    final_chunks = []
    doc_prefix = doc_metadata.get("doc_id_prefix").lower()
    current_gl_num = "0"

    for art in articles:
        art = art.strip()
        if not art: continue
        
        lines = art.split('\n')
        header = lines[0].strip()

        # Обработка Глав
        if 'Глава' in header or 'Раздел' in header:
            gl_match = re.search(r'(\d+)', header)
            current_gl_num = gl_match.group(1) if gl_match else "0"
            continue

        # Обработка Статей
        art_match = re.search(r'Статья\s+(\d+)', header)
        art_id = art_match.group(1) if art_match else "info"
        clean_header = header.replace('#', '').strip()
        
        # Формируем префикс как в V56
        prefix = f"[{doc_metadata['doc_display_name']}] [{clean_header}]\n"
        
        body = "\n".join(lines[1:]).strip()
        
        # Нарезка
        chunks_text = split_body_smart(body, prefix)
        
        for i, chunk_text in enumerate(chunks_text, 1):
            final_chunks.append({
                "id": f"{doc_prefix}_gl{current_gl_num}_st{art_id}_p{i}",
                "title": clean_header[:100],
                "text": chunk_text.strip(),
                "url": doc_metadata.get("url", ""),
                "page": "N/A" # У Docling можно вытащить страницу, если нужно
            })

    # 3. Сохранение
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for c in final_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    print(f"✅ Готово! Нарезано {len(final_chunks)} чанков через Docling + V56 Logic.")

if __name__ == "__main__":
    doc_config = {
        "file": "79-FZ.pdf", 
        "out": "79fz_docling_v57.jsonl", 
        "meta": {
            "doc_id_prefix": "79фз", 
            "doc_display_name": "79-ФЗ", 
            "url": "http://www.kremlin.ru/acts/bank/21210"
        }
    }
    process_with_docling(doc_config["file"], doc_config["out"], doc_config["meta"])
