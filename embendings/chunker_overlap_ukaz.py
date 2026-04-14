import json
import re
import os
from transformers import AutoTokenizer
from pathlib import Path

# --- КОНФИГ ---
MODEL_PATH = '/home/amlin04/multik_bot/hf_cache/multilingual-e5-large'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MAX_TOKENS = 800  # Увеличен для длинных пунктов

def get_tokens_count(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def normalize_heading(heading):
    heading = re.sub(r'\*+', '', heading)
    heading = re.sub(r'\s+', ' ', heading).strip()
    return heading

def extract_roman_numeral(text):
    cleaned = re.sub(r'[^\w\s\.]', ' ', text)
    match = re.match(r'\s*([IVX]+)\.\s*', cleaned)
    if match:
        return match.group(1)
    return None

def is_table_start(line):
    return (re.match(r'^[Тт]аблица\s+№\s*\d+', line) or 
            re.match(r'^Перечень ситуаций', line) or
            re.match(r'^[Пп]ример', line))

def collect_table_block(lines, start_idx):
    table_lines = [lines[start_idx].strip()]
    idx = start_idx + 1
    while idx < len(lines):
        current = lines[idx].strip()
        if (re.match(r'^\d+\.\s+', current) or 
            current.startswith('#') or
            is_table_start(current)):
            break
        if current:
            table_lines.append(current)
        idx += 1
    return '\n'.join(table_lines), idx

def process_method_recommendations(input_file, output_jsonl, meta):
    if not Path(input_file).exists():
        print(f"❌ Файл {input_file} не найден")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'[ \t]+', ' ', content)
    
    final_data = []
    doc_name = meta.get("doc_display_name", "Методические рекомендации")
    doc_prefix = meta.get("doc_id_prefix", "method")
    
    current_section = "Введение"
    current_subsection = ""
    current_parent_point = None
    
    point_counter = 0
    table_counter = 0
    text_counter = 0
    
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Заголовки
        if line.startswith('# ') and not line.startswith('##'):
            current_section = normalize_heading(line[2:])
            current_subsection = ""
            current_parent_point = None
            i += 1
            continue
        
        if line.startswith('## '):
            current_subsection = normalize_heading(line[3:])
            current_parent_point = None
            i += 1
            continue
        
        if line.startswith('### '):
            current_subsection = normalize_heading(line[4:])
            current_parent_point = None
            i += 1
            continue
        
        # Римские цифры
        roman = extract_roman_numeral(line)
        if roman and ('I' in roman or 'V' in roman or 'X' in roman):
            text_part = re.sub(r'^\s*[\*\s]*' + roman + r'[\*\s]*\.\s*', '', line)
            text_part = normalize_heading(text_part)
            current_section = f"{roman}. {text_part}" if text_part else f"{roman}."
            current_subsection = ""
            current_parent_point = None
            i += 1
            continue
        
        # Таблицы
        if is_table_start(line):
            full_table, next_idx = collect_table_block(lines, i)
            table_counter += 1
            table_type = f"Таблица №{table_counter}" if 'Таблица' in line else "Таблица"
            hierarchy = f"{current_section}"
            if current_subsection:
                hierarchy += f" → {current_subsection}"
            if current_parent_point:
                hierarchy += f" → Пункт {current_parent_point}"
            hierarchy += f" → {table_type}"
            header_context = f"ИСТОЧНИК: {doc_name}\nРАЗДЕЛ: {hierarchy}\n\n"
            final_data.append({
                "id": f"{doc_prefix}_таблица_{table_counter:03d}",
                "title": f"{doc_name}: {hierarchy[:200]}",
                "text": header_context + full_table,
                "url": meta.get("url", "")
            })
            i = next_idx
            continue
        
        # Нумерованные пункты (ОСНОВНОЕ ИСПРАВЛЕНИЕ)
        item_match = re.match(r'^(\d+)\.\s+(.*)$', line)
        if item_match:
            point_counter += 1
            original_num = item_match.group(1)
            item_lines = [item_match.group(2)]
            i += 1
            
            # Собираем пункт
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                
                # Условия ОСТАНОВКИ:
                # 1. Новый пункт (цифра с точкой в начале) - но НЕ подпункт внутри отступа!
                # Проверяем: если строка начинается с цифры и точки, но при этом НЕ имеет отступа в начале — это новый пункт
                if re.match(r'^\d+\.\s+', next_line):
                    # Дополнительная проверка: если предыдущая строка была подпунктом (содержит "1." в начале строки без отступа)
                    # Или если строка не имеет пробелов в начале — это действительно новый пункт
                    if not next_line.startswith(' ') and not next_line.startswith('\t'):
                        break
                
                # 2. Заголовок
                if next_line.startswith('#'):
                    break
                
                # 3. Таблица или пример
                if is_table_start(next_line):
                    break
                
                # ВСЁ ОСТАЛЬНОЕ — добавляем в пункт (включая строки с 1., 2., 3. с отступом или без)
                item_lines.append(next_line)
                i += 1
            
            full_item = '\n'.join(item_lines)
            current_parent_point = original_num
            
            hierarchy = f"{current_section}"
            if current_subsection:
                hierarchy += f" → {current_subsection}"
            hierarchy += f" → Пункт {original_num}"
            
            safe_id = f"p{int(original_num):03d}"
            header_context = f"ИСТОЧНИК: {doc_name}\nРАЗДЕЛ: {hierarchy}\n\n"
            
            clean_text = full_item
            clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)
            clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean_text)
            
            # Если текст очень длинный, разбиваем на части
            total_tokens = get_tokens_count(header_context + clean_text)
            if total_tokens <= MAX_TOKENS:
                final_data.append({
                    "id": f"{doc_prefix}_{safe_id}",
                    "title": f"{doc_name}: {hierarchy[:200]}",
                    "text": header_context + clean_text,
                    "url": meta.get("url", "")
                })
            else:
                # Нарезка по предложениям
                sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z0-9])', clean_text)
                chunks = []
                current_chunk = []
                current_len = get_tokens_count(header_context)
                for sent in sentences:
                    sent_len = get_tokens_count(sent)
                    if current_len + sent_len <= MAX_TOKENS:
                        current_chunk.append(sent)
                        current_len += sent_len
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_len = get_tokens_count(header_context) + sent_len
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                for idx, chunk in enumerate(chunks, 1):
                    final_data.append({
                        "id": f"{doc_prefix}_{safe_id}_part{idx}",
                        "title": f"{doc_name}: {hierarchy[:150]} (часть {idx})",
                        "text": header_context + chunk,
                        "url": meta.get("url", "")
                    })
            continue
        
        # Обычный текст
        if len(line) > 20 and not re.match(r'^\d+\.', line) and not line.startswith('**'):
            if current_parent_point:
                i += 1
                continue
            text_counter += 1
            hierarchy = f"{current_section}"
            if current_subsection:
                hierarchy += f" → {current_subsection}"
            header_context = f"ИСТОЧНИК: {doc_name}\nРАЗДЕЛ: {hierarchy}\n\n"
            final_data.append({
                "id": f"{doc_prefix}_text_{text_counter:03d}",
                "title": f"{doc_name}: {hierarchy[:200]}",
                "text": header_context + line,
                "url": meta.get("url", "")
            })
        
        i += 1
    
    # Удаление дублей
    seen_ids = set()
    unique_data = []
    for entry in final_data:
        if entry["id"] not in seen_ids:
            seen_ids.add(entry["id"])
            unique_data.append(entry)
    
    unique_data.sort(key=lambda x: x["id"])
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in unique_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ {doc_name}")
    print(f"   └── {len(unique_data)} уникальных чанков")
    
    if unique_data:
        token_counts = [get_tokens_count(e['text']) for e in unique_data]
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"   └── Средний размер: {avg_tokens:.0f} токенов")

if __name__ == "__main__":
    docs = [
        {"file": "metod_income.md", "out": "metod_income.jsonl", "meta": {
            "doc_id_prefix": "income",
            "doc_display_name": "Методические рекомендации",
            "url": "https://mintrud.gov.ru/ministry/anticorruption/Methods/13"
        }},
    ]
    
    for d in docs:
        process_method_recommendations(d["file"], d["out"], d["meta"])