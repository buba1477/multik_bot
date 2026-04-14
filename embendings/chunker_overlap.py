import json
import re
from transformers import AutoTokenizer
from pathlib import Path

# --- КОНФИГ ---
MODEL_NAME = 'intfloat/multilingual-e5-large' 
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS = 450 

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def split_body_with_recursive_context(body, prefix):
    """Умная нарезка: наследует преамбулу и локальные заголовки перед списками"""
    if not body: return []
    
    # 1. Берем общую преамбулу статьи (до первого пункта)
    preamble_match = re.search(r'^(.*?)(?=\n1\.\s)', body, re.DOTALL)
    global_preamble = preamble_match.group(1).strip() if preamble_match else ""
    if len(global_preamble) > 150: global_preamble = global_preamble[:147] + "..."
    
    # 2. Делим на строки, чтобы отслеживать "двоеточия"
    lines = body.split('\n')
    chunks = []
    current_batch = []
    # Локальный контекст (обновляется, когда видим :)
    local_context = global_preamble 
    
    # Базовый заголовок для каждого чанка
    def get_header(ctx):
        return f"{prefix}КОНТЕКСТ: {ctx}\n" if ctx else prefix

    current_tokens = count_tokens(get_header(local_context))

    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Если строка заканчивается на ":" — это новый локальный контекст (например, "В судах рассматриваются:")
        if line.endswith(':') and len(line) > 10:
            # Если уже что-то накопили — сбрасываем в чанк перед сменой контекста
            if current_batch:
                chunks.append(get_header(local_context) + "\n".join(current_batch))
                current_batch = []
            local_context = line
            current_tokens = count_tokens(get_header(local_context))
            continue

        line_tokens = count_tokens(line)

        # Если лимит превышен — закрываем чанк
        if current_tokens + line_tokens > MAX_TOKENS and current_batch:
            chunks.append(get_header(local_context) + "\n".join(current_batch))
            current_batch = [line]
            current_tokens = count_tokens(get_header(local_context)) + line_tokens
        else:
            current_batch.append(line)
            current_tokens += line_tokens
            
    if current_batch:
        chunks.append(get_header(local_context) + "\n".join(current_batch))
            
    return chunks

def process_law_v56_recursive(input_file, output_jsonl, doc_metadata):
    if not Path(input_file).exists():
        print(f"❌ Файл {input_file} не найден")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'[ \t]+', ' ', content)
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

        if header.startswith('# ') or header.startswith('Раздел'):
            gl_match = re.search(r'(?:Глава|Раздел)\s+(\d+)', header)
            current_gl_num = gl_match.group(1) if gl_match else "0"
            current_gl_name = header.replace('#', '').strip()
            continue

        art_match = re.search(r'Статья\s+(\d+)', header)
        art_id = art_match.group(1) if art_match else "info"
        clean_header = header.replace('#', '').strip()
        prefix = f"[{doc_metadata['doc_display_name']}] [{clean_header}]\n"
        
        body = "\n".join(lines[1:]).strip()
        chunks_text = split_body_with_recursive_context(body, prefix)
        
        for i, chunk_text in enumerate(chunks_text, 1):
            final_chunks.append({
                "id": f"{doc_prefix}_gl{current_gl_num}_st{art_id}_p{i}",
                "title": clean_header[:80],
                "text": chunk_text.strip(),
                "url": doc_metadata.get("url", ""),
                "local_img": ""
            })

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for c in final_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    print(f"🚀 V56 Recursive: {doc_metadata['doc_display_name']} -> {len(final_chunks)} чанков.")

if __name__ == "__main__":
    docs = [
        {"file": "79-FZ_markdown.md", "out": "79fz_final.jsonl", "meta": {"doc_id_prefix": "79фз", "doc_display_name": "79-ФЗ", "url": "http://www.kremlin.ru/acts/bank/21210"}},
        {"file": "58-FZ_markdown.md", "out": "58fz_final.jsonl", "meta": {"doc_id_prefix": "58фз", "doc_display_name": "58-ФЗ", "url": "http://www.kremlin.ru/acts/bank/19524"}}
    ]
    for d in docs:
        process_law_v56_recursive(d["file"], d["out"], d["meta"])
