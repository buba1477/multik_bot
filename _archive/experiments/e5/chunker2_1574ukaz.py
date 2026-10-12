import json
import re
from transformers import AutoTokenizer
from pathlib import Path

# --- КОНФИГ ---
MODEL_NAME = 'intfloat/multilingual-e5-large'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS = 450  # Оставляем запас

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def process_1574_to_jsonl(input_md, output_jsonl, doc_metadata):
    if not Path(input_md).exists(): return

    with open(input_md, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc_prefix = doc_metadata.get("doc_id_prefix", "указ1574")
    processed_lines = []
    
    # Переменные контекста (будут обновляться по мере чтения файла)
    curr_razdel = "Общие сведения"
    curr_cat = "Без категории"
    curr_group = "Без группы"

    # 1. ПЕРВИЧНАЯ СБОРКА АТОМАРНЫХ СТРОК
    for line in lines:
        l = line.strip().strip('|').strip()
        # Чистим мусор Гаранта сразу
        if not l or '---' in l or 'Система ГАРАНТ' in l or 'Указ Президента' in l or '06.04.2026' in l:
            continue
        
        # Обновляем контекст заголовков
        if 'Раздел' in l: curr_razdel = l; continue
        if 'Должности категории' in l: curr_cat = l; continue
        if 'группа должностей' in l.lower(): curr_group = l; continue
        
        # Если в строке есть код (находим через паттерн)
        code_match = re.search(r'\d+-\d+-\d+-\d+[\d\.\*]*', l)
        if code_match:
            parts = l.split('|')
            name = parts[0].strip()
            code = code_match.group(0)
            # Сшиваем "золотую строку"
            atom = f"• {name} | КОД: `{code}` | ГРУППА: {curr_group} | КАТЕГОРИЯ: {curr_cat} | {curr_razdel}"
            processed_lines.append(atom)

    # 2. НАРЕЗКА НА ЧАНКИ ПО ТОКЕНАМ
    final_chunks = []
    current_accum = ""
    chunk_idx = 1
    
    prefix = f"[{doc_metadata['doc_display_name']}] [РЕЕСТР ДОЛЖНОСТЕЙ]\n"

    for atom in processed_lines:
        # Проверяем лимит
        if count_tokens(prefix + current_accum + "\n" + atom) > MAX_TOKENS:
            final_chunks.append({
                "id": f"{doc_prefix}_p{chunk_idx}",
                "title": doc_metadata['doc_display_name'],
                "text": (prefix + current_accum).strip(),
                "url": doc_metadata['url']
            })
            current_accum = atom + "\n"
            chunk_idx += 1
        else:
            current_accum += atom + "\n"

    if current_accum:
        final_chunks.append({
            "id": f"{doc_prefix}_p{chunk_idx}",
            "title": doc_metadata['doc_display_name'],
            "text": (prefix + current_accum).strip(),
            "url": doc_metadata['url']
        })

    # 3. ЗАПИСЬ В JSONL (строго в одну линию)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for c in final_chunks:
            f.write(json.dumps({**c, "local_img": ""}, ensure_ascii=False) + '\n')

    print(f"✅ Успех! Создано {len(final_chunks)} идеальных чанков в {output_jsonl}")

# --- ЗАПУСК ---
meta = {
    "doc_id_prefix": "указ1574",
    "doc_display_name": "Указ Президента №1574 (Реестр должностей)",
    "url": "http://www.kremlin.ru/acts/bank/23342"
}
process_1574_to_jsonl("1574ukaz.md", "ukaz_1574_final.jsonl", meta)
