import json
import re
from pathlib import Path
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
import os

# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/ru-en-RoSBERTa"
INPUT_PDF = "146-ФЗ .pdf"
OUTPUT_FILE = "146-FZ-semantic.jsonl"
URL = "http://kremlin.ru"

MAX_TOKENS = 512
TARGET_TOKENS = 350
MIN_TOKENS = 50
ABSOLUTE_MAX_TOKENS = 512

# =========================================================
# INITIALIZATION
# =========================================================

os.environ.update({
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1"
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# =========================================================
# UTILITIES & BOOK RECIPE 2.3 (TOKEN MANAGEMENT)
# =========================================================

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_to_limit(text: str, max_tokens: int) -> str:
    """
    [BOOK RECIPE 2.3] Безопасно обрезает русский текст строго по лимиту токенов,
    полностью исключая появление битых символов-ромбиков () в UTF-8.
    """
    if not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True).strip()

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_garbage(line: str) -> bool:
    s = line.strip().lower()
    garbage = {"события", "структура", "контакты", "документы", "поиск",
               "rutube", "telegram", "youtube", "введите запрос", "найти", "для сми"}
    return len(s) < 2 or s in garbage or s.startswith("http") or "pravo.gov.ru" in s

def is_orphan_number(text: str) -> bool:
    """Проверяет, является ли чанк одиноким номером пункта"""
    cleaned = re.sub(r'^\[[^\]]+\]\[[^\]]+\]', '', text)
    cleaned = re.sub(r'[\s\n]', '', cleaned)
    cleaned = cleaned.strip()
    if re.match(r'^\d+(?:\.\d+)*\.?$', cleaned):
        return True
    if len(cleaned) < 10 and re.match(r'^\d+(?:\.\d+)*$', cleaned):
        return True
    return False

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_legal_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r'\r\n?', '\n', text)

    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\]\([^)]+\)', ' ', text)
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    text = re.sub(r'(?m)^\s*-\s+', '', text)

    text = re.sub(r'<[^>]+>', ' ', text)

    text = re.sub(r'([а-яa-z])\n-\s*([а-яa-z])', r'\1\2', text, flags=re.I)
    text = re.sub(r'([а-яa-z,;:])\n([а-яa-z])', r'\1 \2', text, flags=re.I)

    patterns = [r'\(В редакции[^)]*\)', r'\(Дополнение[^)]*\)',
                r'\(Утратил силу[^)]*\)', r'\(Наименование[^)]*\)']
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.I)

    text = re.sub(r'(?mi)^\s*(?:\d+\.\s*)?Абзац\.\s*$', '', text)
    text = re.sub(r'k6cl[a-zA-Z0-9:=&/?._-]+', ' ', text)

    # Исправления номеров статей и законов без дублирования
    text = re.sub(r'(\d+)\s+(\d+)\s+\.', r'\1.\2.', text)
    text = re.sub(r'(\d+)\s+(\d+)[-\s]*(\d*)[\s\n]*\.', r'\1.\2\3.', text)
    text = re.sub(r'\s+(\d+(?:\.\d+)*\.?)\s*$', '', text)
    text = re.sub(r'(\d+\.\d+)\s+\.', r'\1.', text)
    text = re.sub(r'(\d+)\s+(\d+)(?=\s*(?:настоящего|статьи|Кодекса|пункта|главы|части|раздела|счета|договора))', r'\1.\2', text)
    text = re.sub(r'^\s*(\d+(?:\.\d+)*\.)\s+', r'\1 ', text, flags=re.MULTILINE)
    text = re.sub(r'(?m)^\s*-\s*(\d+\))', r'\1', text)
    text = re.sub(r'(?m)^\s*-\s*([а-яa-z]\))', r'\1', text, flags=re.I)
    text = re.sub(r'(\d{2,3})ФЗ', r'\1-ФЗ', text)
    text = re.sub(r'№\s*\n\s*(\d+-\w+)', r'№ \1', text)
    text = re.sub(r'(Федерального закона от \d{2}\.\d{2}\.\d{4} №)\s*\n\s*(\d+-\w+)', r'\1 \2', text)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def extract_article_title(text: str):
    text = normalize_text(text)
    m = re.search(r'Статья\s+([\d\.\-\s]+)', text)
    if not m:
        return None, None

    raw_num = m.group(1)
    art_num = normalize_article_number(raw_num)

    title = text[m.end():].strip()
    title = re.sub(r'^\.+', '', title)

    if art_num.endswith('.'):
        full_title = f"Статья {art_num} {title}".strip()
    else:
        full_title = f"Статья {art_num}. {title}".strip()

    return art_num, full_title

def normalize_article_number(num: str) -> str:
    num = re.sub(r'<[^>]+>', '', num)
    num = num.replace("-", ".")
    num = re.sub(r'\s+', '.', num)
    num = re.sub(r'\.+', '.', num)
    num = re.sub(r'\.+$', '.', num)
    return num.strip(".")

# =========================================================
# TEXT SPLITTING — АВТОНОМНАЯ ЛОКАЛЬНАЯ ВЕРСИЯ
# =========================================================

def split_by_paragraphs(text: str):
    if not text:
        return []
    
    protected = {}
    def protect(m):
        idx = len(protected)
        placeholder = f"__PROTECTED_{idx}__"
        protected[placeholder] = m.group(0)
        return placeholder
    
    text = re.sub(r'№\s*\d+-\w+', protect, text)
    text = re.sub(r'Федерального закона от \d{2}\.\d{2}\.\d{4} № \d+-\w+', protect, text)
    
    pattern = re.compile(r'(?=^\d+(?:\.\d+)*\.\s+[А-ЯA-Z])', re.MULTILINE)
    parts = re.split(pattern, text)
    
    for placeholder, original in protected.items():
        parts = [p.replace(placeholder, original) for p in parts]
    
    result = []
    for part in parts:
        part = part.strip()
        if len(part) > 20:
            result.append(part)
            
    return result

def split_large_paragraph(para: str, limit_tokens: int):
    sentences = re.split(r'(?<=[\.\!\?;])\s+(?=[А-ЯA-Z0-9])', para)
    chunks, current = [], []
    
    for sent in sentences:
        candidate = " ".join(current + [sent])
        if count_tokens(candidate) > limit_tokens:
            if current:
                chunks.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
    
    if current:
        chunks.append(" ".join(current))
        
    return chunks

def build_chunks_respectful(paragraphs, prefix, max_tokens=MAX_TOKENS):
    """Собирает чанки локально, ПРЕДОТВРАЩАЯ появление оторванных сирот"""
    chunks = []
    current = []
    prefix_tokens = count_tokens(prefix)
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        para_with_prefix = prefix + "\n\n" + para
        para_tokens = count_tokens(para_with_prefix)
        
        if para_tokens > max_tokens:
            if current:
                chunks.append(prefix + "\n\n" + "\n\n".join(current))
                current = []
            
            safe_limit = max_tokens - prefix_tokens - 4
            sub_paragraphs = split_large_paragraph(para, safe_limit)
            
            for sub in sub_paragraphs:
                if sub.strip():
                    safe_sub = truncate_to_limit(sub, safe_limit)
                    chunks.append(prefix + "\n\n" + safe_sub)
            continue
        
        test_candidate = prefix + "\n\n" + "\n\n".join(current + [para])
        if count_tokens(test_candidate) <= max_tokens:
            current.append(para)
        else:
            if current:
                chunks.append(prefix + "\n\n" + "\n\n".join(current))
            current = [para]
            
    # Защита финальных хвостов статей
    if current:
        last_chunk = prefix + "\n\n" + "\n\n".join(current)
        if count_tokens(last_chunk) < MIN_TOKENS and chunks:
            prev_chunk = chunks.pop()
            clean_tail = "\n\n".join(current)
            merged_chunk = prev_chunk + "\n\n" + clean_tail
            
            if count_tokens(merged_chunk) <= ABSOLUTE_MAX_TOKENS:
                chunks.append(merged_chunk)
            else:
                chunks.append(prev_chunk)
                safe_last = truncate_to_limit(last_chunk, ABSOLUTE_MAX_TOKENS)
                chunks.append(safe_last)
        else:
            safe_last = truncate_to_limit(last_chunk, ABSOLUTE_MAX_TOKENS)
            chunks.append(safe_last)
            
    return chunks

def validate_chunks(chunks):
    seen, final = set(), []

    for chunk in chunks:
        chunk = normalize_text(chunk)

        if not chunk:
            continue

        token_count = count_tokens(chunk)

        if token_count < MIN_TOKENS:
            continue

        if token_count > ABSOLUTE_MAX_TOKENS:
            continue

        if is_orphan_number(chunk):
            continue

        chunk_hash = hash(chunk)
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            final.append(chunk)

    return final

# =========================================================
# MAIN PROCESSING
# =========================================================

def main():
    print(f"🚀 Processing: {INPUT_PDF}")

    if not Path(INPUT_PDF).exists():
        print(f"❌ File {INPUT_PDF} not found!")
        return

    converter = DocumentConverter(format_options={
        "pdf": PdfFormatOption(pipeline_options=PdfPipelineOptions(enable_remote_services=False))
    })

    try:
        markdown = converter.convert(INPUT_PDF).document.export_to_markdown()
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return

    print(f"📄 Content length: {len(markdown)} chars")

    clean_lines = [line for line in markdown.split("\n") if not is_garbage(line)]
    clean_md = clean_legal_text("\n".join(clean_lines))

    articles = re.split(r'(?=Статья\s+[\d.-\s]+)', clean_md)
    print(f"📑 Found {len(articles)} sections")

    doc_id = Path(INPUT_PDF).stem.lower().replace(" ", "_").strip()
    total_chunks = 0
    small_chunks = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for article in articles:
            article = normalize_text(article)
            if len(article) < 50:
                continue

            article_match = re.match(r'(Статья\s+[\d.-\s]+.?\s*[^\n]*)', article, flags=re.I)
            if not article_match:
                continue

            art_num, title = extract_article_title(article_match.group(1).strip())
            if not art_num:
                continue

            body = clean_legal_text(article[len(article_match.group(1)):].strip())
            if not body or len(body) < 30:
                continue

            prefix = f"[{doc_id.upper()}] [{title}]"
            full_text = prefix + "\n\n" + body
            total_tokens = count_tokens(full_text)

            print(f"📌 Article {art_num}: {total_tokens} tokens")

            paragraphs = split_by_paragraphs(body)
            if not paragraphs:
                paragraphs = [p.strip() for p in body.split('\n\n') if p.strip()]

            if total_tokens <= MAX_TOKENS:
                chunks = [full_text]
            else:
                chunks = build_chunks_respectful(paragraphs, prefix)

            chunks = validate_chunks(chunks)
            safe_art_num = art_num.replace(".", "_")

            for idx, chunk in enumerate(chunks, 1):
                token_count = count_tokens(chunk)
                if token_count < MIN_TOKENS:
                    small_chunks.append((f"{doc_id}_st{safe_art_num}_c{idx}", token_count))
                    continue

                node = {
                    "id": f"{doc_id}_st{safe_art_num}_c{idx}",
                    "title": title,
                    "text": chunk,
                    "local_img": "",
                    "url": URL
                }
                f.write(json.dumps(node, ensure_ascii=False) + "\n")
                total_chunks += 1

            print(f"✅ Created {len(chunks)} valid chunks")

    print(f"\n{'='*50}")
    print(f"📊 ИТОГИ ПРОВЕРКИ:")
    print(f"✅ Всего чанков: {total_chunks}")
    print(f"🚩 Отфильтровано малышей (<{MIN_TOKENS}): {len(small_chunks)}")
    if small_chunks:
        print(f"⚠️ Примеры отфильтрованных:")
        for sid, scount in small_chunks[:10]:
            print(f" - {sid}: {scount} токенов")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()