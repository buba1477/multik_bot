import json
import re
from transformers import AutoTokenizer
from pathlib import Path

# --- КОНФИГ ---
MODEL_NAME = 'intfloat/multilingual-e5-large' 
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def process_law_v25_atomic(input_file, output_jsonl, doc_metadata):
    if not Path(input_file).exists():
        print(f"❌ Файл {input_file} не найден")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Предварительная чистка (убираем лишние пробелы, сохраняем абзацы)
    content = re.sub(r'[ \t]+', ' ', content)
    
    # 2. Делим на Статьи
    articles = re.split(r'\n(?=###\s+Статья|#\s+Глава|Раздел\s+)', content)
    
    final_chunks = []
    doc_prefix = doc_metadata.get("doc_id_prefix").lower()
    current_gl_name = "Общие сведения"
    current_gl_num = "0"

    for art in articles:
        art = art.strip()
        if not art: continue
        
        lines = art.split('\n')
        header = lines[0].strip()

        # Обновляем контекст Главы/Раздела
        if header.startswith('# ') or header.startswith('Раздел'):
            gl_match = re.search(r'(?:Глава|Раздел)\s+(\d+)', header)
            current_gl_num = gl_match.group(1) if gl_match else "0"
            current_gl_name = header.replace('#', '').strip()
            continue

        # Парсим номер статьи
        art_match = re.search(r'Статья\s+(\d+)', header)
        art_id = art_match.group(1) if art_match else "info"
        
        # Префикс (Паспорт чанка)
        prefix = f"[{doc_metadata['doc_display_name']}] [{current_gl_name}] [{header.replace('#', '').strip()}]\n"
        
        body = "\n".join(lines[1:]).strip()
        
        # ГЛАВНЫЙ ХОД: Делим тело статьи на ЧАСТИ (1. , 2. , 3. и т.д.)
        # Используем lookahead, чтобы не удалять сами цифры
        parts = re.split(r'\n(?=\d+\.\s)', body)
        
        for idx, p in enumerate(parts, 1):
            p = p.strip()
            if not p: continue
            
            # Если в куске несколько частей (например, короткие), 
            # но мы хотим строго "одна часть = один чанк", 
            # то просто берем этот кусок как есть.
            
            # Формируем финальный текст чанка
            chunk_text = prefix + p
            
            # Проверяем на пустые части (бывает при кривом маркдауне)
            if len(p) < 10: continue

            final_chunks.append({
                "id": f"{doc_prefix}_gl{current_gl_num}_st{art_id}_p{idx}",
                "title": header.replace('#', '').strip()[:80],
                "text": chunk_text.strip(),
                "url": doc_metadata.get("url", ""),
                "local_img": ""
            })

    # Запись в JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for c in final_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    print(f"💎 V25 Atomic: {doc_metadata['doc_display_name']} -> {len(final_chunks)} чанков (по частям статей).")

if __name__ == "__main__":
    docs = [
        # {
        #     "file": "79-FZ_markdown.md", 
        #     "out": "79fz_final.jsonl", 
        #     "meta": {"doc_id_prefix": "79фз", "doc_display_name": "79-ФЗ", "url": "http://www.kremlin.ru/acts/bank/21210"}
        # },
        # {
        #     "file": "58-FZ_markdown.md", 
        #     "out": "58fz_final.jsonl", 
        #     "meta": {"doc_id_prefix": "58фз", "doc_display_name": "58-ФЗ", "url": "http://www.kremlin.ru/acts/bank/19524"}
        # },
        {
            "file": "1574_fixed.md", 
            "out": "ukaz_1574_final.jsonl", 
            "meta": {"doc_id_prefix": "указ1574", "doc_display_name": "Указ №1574", "url": "http://www.kremlin.ru/acts/bank/23342"}
        }
    ]
    
    for d in docs:
        process_law_v25_atomic(d["file"], d["out"], d["meta"])
