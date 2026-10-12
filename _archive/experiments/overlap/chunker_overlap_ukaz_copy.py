import json
import re
import os
from transformers import AutoTokenizer

# --- КОНФИГ ---
MODEL_PATH = '/home/amlin04/multik_bot/hf_cache/multilingual-e5-large'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MAX_TOKENS = 400 

def get_tokens_count(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def process_income_docs(input_file, output_jsonl, meta):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Чистим двойные пробелы
    content = re.sub(r'[ \t]+', ' ', content)
    
    # 🔥 УМНЫЙ СПЛИТТЕР: Режем по ### и по **Заголовкам**
    # Ловит: \n### Заголовок или \n**Жирный Заголовок**
    sections = re.split(r'\n(?=###\s+|\*\*[^*]+\*\*)', content)
    
    final_data = []
    doc_name = meta.get("doc_display_name", "Документ")
    prefix = meta.get("doc_id_prefix", "doc")

    for section in sections:
        section = section.strip()
        if not section: continue
        
        lines = section.split('\n')
        raw_header = lines[0]
        # Очищаем заголовок для метаданных и ID
        clean_header = raw_header.replace('###', '').replace('**', '').strip()
        body = "\n".join(lines[1:]).strip()
        
        # 🔥 ФИКС ID: Разрешаем кириллицу + убираем длинные подчеркивания
        safe_id = re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁ]', '_', clean_header[:40].lower())
        safe_id = re.sub(r'_+', '_', safe_id).strip('_')
        if not safe_id: safe_id = "section"

        # Режем тело раздела на пункты (якорь: "171. " или "1. " в начале строки)
        points = re.split(r'\n(?=\d+\.\s)', body)
        
        current_chunk_parts = []
        part_idx = 1

        for p in points:
            p = p.strip()
            if not p: continue
            
            # Контекстная шапка чанка
            header_context = f"ИСТОЧНИК: {doc_name}\nРАЗДЕЛ: {clean_header}\n\n"
            
            test_content = "\n\n".join(current_chunk_parts + [p])
            
            if get_tokens_count(header_context + test_content) > MAX_TOKENS:
                # Если батч не пустой - сохраняем
                if current_chunk_parts:
                    final_data.append({
                        "id": f"{prefix}_{safe_id}_p{part_idx}",
                        "title": clean_header,
                        "text": header_context + "\n\n".join(current_chunk_parts),
                        "url": meta.get("url", "")
                    })
                    part_idx += 1
                    
                    # 🔥 OVERLAP: Берем последний пункт для связки
                    overlap_p = current_chunk_parts[-1]
                    current_chunk_parts = [overlap_p, p]
                    
                    # Проверка на случай если оверлап + новый пункт сразу вылетают за лимит
                    if get_tokens_count(header_context + "\n\n".join(current_chunk_parts)) > MAX_TOKENS:
                        current_chunk_parts = [p]
                
                # Аварийная нарезка гигантского пункта по предложениям
                if get_tokens_count(header_context + p) > MAX_TOKENS:
                    sentences = re.split(r'(?<=[.!?])\s+', p)
                    sub_batch = []
                    for sent in sentences:
                        if get_tokens_count(header_context + "\n".join(sub_batch + [sent])) > MAX_TOKENS:
                            if sub_batch:
                                final_data.append({
                                    "id": f"{prefix}_{safe_id}_p{part_idx}",
                                    "title": clean_header,
                                    "text": header_context + "\n".join(sub_batch),
                                    "url": meta.get("url", "")
                                })
                                part_idx += 1
                            sub_batch = [sent]
                        else:
                            sub_batch.append(sent)
                    current_chunk_parts = sub_batch
            else:
                current_chunk_parts.append(p)

        # Сохраняем остаток раздела
        if current_chunk_parts:
            final_data.append({
                "id": f"{prefix}_{safe_id}_p{part_idx}",
                "title": clean_header,
                "text": header_context + "\n\n".join(current_chunk_parts),
                "url": meta.get("url", "")
            })

    # Пишем в JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ Готово! Нарезано {len(final_data)} чанков. ID исправлены, заголовки разделены.")


if __name__ == "__main__":
    docs = [
        {"file": "metod_income.md", "out": "metod_income.jsonl", "meta": {
            "doc_id_prefix": "income",
            "doc_display_name": "Методические рекомендации",
            "url": "https://mintrud.gov.ru/ministry/anticorruption/Methods/13"
        }},
    ]
    
    for d in docs:
        process_income_docs(d["file"], d["out"], d["meta"])