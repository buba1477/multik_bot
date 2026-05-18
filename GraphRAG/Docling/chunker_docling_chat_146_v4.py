import json
import os
import re
from pathlib import Path

from transformers import AutoTokenizer

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)

# =========================================================
# OFFLINE
# =========================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/ru-en-RoSBERTa"

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

INPUT_PDF = "146-ФЗ.pdf"

OUTPUT_FILE = "146-FZ.jsonl"

URL = "http://www.kremlin.ru/acts/bank/12755"

# =========================================================
# LEGAL RAG SETTINGS
# =========================================================

MAX_TOKENS = 350

ABSOLUTE_MAX_TOKENS = 500

MIN_CHARS = 50

# =========================================================
# TOKEN COUNT
# =========================================================

def count_tokens(text: str):

    return len(
        TOKENIZER.encode(
            text,
            add_special_tokens=False
        )
    )

# =========================================================
# NORMALIZE
# =========================================================

def normalize_text(text: str):

    if not text:
        return ""

    text = text.replace("\xa0", " ")

    text = re.sub(
        r'\r\n?',
        '\n',
        text
    )

    text = re.sub(
        r'[ \t]+',
        ' ',
        text
    )

    text = re.sub(
        r'\n{3,}',
        '\n\n',
        text
    )

    return text.strip()

# =========================================================
# GARBAGE FILTER
# =========================================================

def is_garbage(line: str):

    s = line.strip().lower()

    garbage = {
        "события",
        "структура",
        "контакты",
        "документы",
        "поиск",
        "rutube",
        "telegram",
        "youtube",
        "введите запрос",
        "найти",
    }

    if len(s) < 2:
        return True

    if s in garbage:
        return True

    if s.startswith("http"):
        return True

    return False

# =========================================================
# CLEAN LEGAL TEXT
# =========================================================

def clean_legal_text(text: str):

    if not text:
        return ""

    text = text.replace("\xa0", " ")

    text = re.sub(
        r'\r\n?',
        '\n',
        text
    )

    # =====================================================
    # MARKDOWN
    # =====================================================

    # markdown links
    text = re.sub(
        r'\[([^\]]+)\]\([^)]+\)',
        r'\1',
        text
    )

    # broken markdown
    text = re.sub(
        r'\]\([^)]+\)',
        ' ',
        text
    )

    # markdown headers
    text = re.sub(
        r'(?m)^#{1,6}\s*',
        '',
        text
    )

    # markdown bullet lists
    text = re.sub(
        r'(?m)^\s*-\s+',
        '',
        text
    )

    # =====================================================
    # HTML
    # =====================================================

    text = re.sub(
        r'<[^>]+>',
        ' ',
        text
    )

    # =====================================================
    # FIX PDF LINE BREAKS
    # =====================================================

    # переносы внутри слов
    text = re.sub(
        r'([а-яa-z])\n-\s*([а-яa-z])',
        r'\1\2',
        text,
        flags=re.I
    )

    # переносы внутри предложения
    text = re.sub(
        r'([а-яa-z,;:])\n([а-яa-z])',
        r'\1 \2',
        text,
        flags=re.I
    )

    # =====================================================
    # REMOVE EDITORIAL GARBAGE
    # =====================================================

    text = re.sub(
        r'\(В редакции[^)]*\)',
        ' ',
        text,
        flags=re.I
    )

    text = re.sub(
        r'\(Дополнение[^)]*\)',
        ' ',
        text,
        flags=re.I
    )

    text = re.sub(
        r'\(Утратил силу[^)]*\)',
        ' ',
        text,
        flags=re.I
    )

    text = re.sub(
        r'\(Наименование[^)]*\)',
        ' ',
        text,
        flags=re.I
    )

    # orphan "Абзац."
    text = re.sub(
        r'(?mi)^\s*Абзац\.\s*$',
        '',
        text
    )

    # =====================================================
    # PDF / DOCLING GARBAGE
    # =====================================================

    text = re.sub(
        r'k6cl[a-zA-Z0-9:=&/?._-]+',
        ' ',
        text
    )

    # =====================================================
    # NORMALIZE LISTS
    # =====================================================

    # "- 1)" -> "1)"
    text = re.sub(
        r'(?m)^\s*-\s*(\d+\))',
        r'\1',
        text
    )

    # "- а)" -> "а)"
    text = re.sub(
        r'(?m)^\s*-\s*([а-яa-z]\))',
        r'\1',
        text,
        flags=re.I
    )

    # "3 1 ." -> "3.1."
    text = re.sub(
        r'(\d+)\s+(\d+)\s*\.',
        r'\1.\2.',
        text
    )

    # =====================================================
    # SPACES
    # =====================================================

    text = re.sub(
        r'[ \t]+',
        ' ',
        text
    )

    text = re.sub(
        r'\n{3,}',
        '\n\n',
        text
    )

    return text.strip()

# =========================================================
# ARTICLE NUMBER
# =========================================================

def normalize_article_number(num: str):

    num = re.sub(
        r'<[^>]+>',
        '',
        num
    )

    num = num.replace("-", ".")

    num = re.sub(
        r'\s+',
        '.',
        num
    )

    num = re.sub(
        r'\.+',
        '.',
        num
    )

    return num.strip(".")

# =========================================================
# EXTRACT ARTICLE TITLE
# =========================================================

def extract_article_title(text: str):

    text = normalize_text(text)

    m = re.search(
        r'Статья\s+([\d\.\-\s]+)',
        text
    )

    if not m:
        return None, None

    raw_num = m.group(1)

    art_num = normalize_article_number(raw_num)

    title = text[m.end():].strip()

    title = re.sub(
        r'^\.+',
        '',
        title
    )

    full_title = f"Статья {art_num}"

    if title:
        full_title += f". {title}"

    return art_num, full_title

# =========================================================
# REMOVE DUPLICATE HEADER
# =========================================================

def remove_duplicate_header(text):

    lines = text.splitlines()

    if not lines:
        return text

    header = lines[0].strip()

    cleaned = [lines[0]]

    for line in lines[1:]:

        if line.strip() == header:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

# =========================================================
# SENTENCE SPLIT
# =========================================================

def split_sentences(text: str):

    text = normalize_text(text)

    if not text:
        return []

    sentences = re.split(
        r'(?<=[\.\!\?;])\s+(?=[А-ЯA-Z0-9])',
        text
    )

    result = []

    for s in sentences:

        s = s.strip()

        if s:
            result.append(s)

    return result

# =========================================================
# SPLIT LEGAL BLOCKS
# =========================================================

def split_legal_blocks(text: str):

    text = clean_legal_text(text)

    if not text:
        return []

    pattern = re.compile(
        r'(?=^(?:'
        r'\d+(?:\.\d+)*\.\s|'   # 1. 1.1. 1.1.1.
        r'\d+\)\s|'             # 1)
        r'[а-яa-z]\)\s'         # а)
        r'))',
        flags=re.MULTILINE | re.IGNORECASE
    )

    parts = re.split(pattern, text)

    blocks = []

    for part in parts:

        part = normalize_text(part)

        if len(part) < 20:
            continue

        blocks.append(part)

    return blocks

# =========================================================
# SPLIT LARGE BLOCK
# =========================================================

def split_large_block(block, limit_tokens):

    # =====================================================
    # SPLIT BY PARAGRAPHS
    # =====================================================

    paragraphs = re.split(
        r'\n\n+',
        block
    )

    chunks = []

    current = []

    for part in paragraphs:

        candidate = "\n\n".join(
            current + [part]
        )

        if count_tokens(candidate) > limit_tokens:

            if current:

                chunks.append(
                    "\n\n".join(current)
                )

            current = [part]

        else:

            current.append(part)

    if current:

        chunks.append(
            "\n\n".join(current)
        )

    # =====================================================
    # SPLIT BY SENTENCES
    # =====================================================

    final = []

    for chunk in chunks:

        if count_tokens(chunk) <= limit_tokens:

            final.append(chunk)

            continue

        sentences = split_sentences(chunk)

        temp = []

        for sentence in sentences:

            candidate = " ".join(
                temp + [sentence]
            )

            if count_tokens(candidate) > limit_tokens:

                if temp:

                    final.append(
                        " ".join(temp)
                    )

                temp = [sentence]

            else:

                temp.append(sentence)

        if temp:

            final.append(
                " ".join(temp)
            )

    return final

# =========================================================
# BUILD CHUNKS
# =========================================================

def build_chunks(
    blocks,
    prefix,
    max_tokens=MAX_TOKENS
):

    chunks = []

    current_blocks = []

    prefix_tokens = count_tokens(prefix)

    for block in blocks:

        candidate = (
            prefix
            + "\n\n"
            + "\n\n".join(
                current_blocks + [block]
            )
        )

        candidate_tokens = count_tokens(candidate)

        # =================================================
        # FITS
        # =================================================

        if candidate_tokens <= max_tokens:

            current_blocks.append(block)

            continue

        # =================================================
        # FLUSH CURRENT
        # =================================================

        if current_blocks:

            chunk_text = (
                prefix
                + "\n\n"
                + "\n\n".join(current_blocks)
            )

            chunks.append(chunk_text)

        # =================================================
        # HUGE BLOCK
        # =================================================

        block_full = (
            prefix
            + "\n\n"
            + block
        )

        if count_tokens(block_full) > max_tokens:

            huge_parts = split_large_block(
                block,
                max_tokens - prefix_tokens
            )

            for hp in huge_parts:

                chunks.append(
                    prefix
                    + "\n\n"
                    + hp
                )

            current_blocks = []

        else:

            current_blocks = [block]

    # =====================================================
    # FINAL FLUSH
    # =====================================================

    if current_blocks:

        chunks.append(
            prefix
            + "\n\n"
            + "\n\n".join(current_blocks)
        )

    return chunks

# =========================================================
# VALIDATE
# =========================================================

def validate_chunks(chunks):

    final = []

    seen = set()

    for chunk in chunks:

        chunk = normalize_text(chunk)

        if not chunk:
            continue

        if len(chunk) < MIN_CHARS:
            continue

        tokens = count_tokens(chunk)

        if tokens > ABSOLUTE_MAX_TOKENS:
            continue

        # deduplicate
        chunk_hash = hash(chunk)

        if chunk_hash in seen:
            continue

        seen.add(chunk_hash)

        final.append(chunk)

    return final

# =========================================================
# MAIN
# =========================================================

def main():

    print(f"🚀 Processing: {INPUT_PDF}")

    if not Path(INPUT_PDF).exists():

        print(f"❌ Файл {INPUT_PDF} не найден!")

        return

    converter = DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    enable_remote_services=False
                )
            )
        }
    )

    try:

        result = converter.convert(INPUT_PDF)

        markdown = result.document.export_to_markdown()

    except Exception as e:

        print(f"❌ Ошибка конвертации: {e}")

        return

    print(
        f"📄 Длина контента: "
        f"{len(markdown)} символов"
    )

    # =====================================================
    # CLEAN RAW LINES
    # =====================================================

    lines = markdown.split("\n")

    clean_lines = []

    for line in lines:

        if not is_garbage(line):
            clean_lines.append(line)

    clean_md = "\n".join(clean_lines)

    clean_md = clean_legal_text(clean_md)

    # =====================================================
    # SPLIT ARTICLES
    # =====================================================

    articles = re.split(
        r'(?=Статья\s+[\d\.\-\s]+)',
        clean_md
    )

    print(f"📑 Найдено секций: {len(articles)}")

    doc_id = (
        Path(INPUT_PDF)
        .stem
        .lower()
        .replace(" ", "_")
    )

    total_chunks = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for article in articles:

            article = normalize_text(article)

            if len(article) < 30:
                continue

            article_match = re.match(
                r'(Статья\s+[\d\.\-\s]+\.?\s*[^\n]*)',
                article,
                flags=re.I
            )

            if not article_match:
                continue

            article_header = (
                article_match.group(1)
                .strip()
            )

            art_num, title = extract_article_title(
                article_header
            )

            if not art_num:
                continue

            body = article[
                len(article_header):
            ].strip()

            body = clean_legal_text(body)

            if not body:
                continue

            prefix = (
                f"[{doc_id.upper()}] "
                f"[{title}]"
            )

            full_text = (
                prefix
                + "\n\n"
                + body
            )

            total_tokens = count_tokens(full_text)

            print(
                f"📌 Статья {art_num}: "
                f"{total_tokens} токенов"
            )

            # =================================================
            # SMALL ARTICLE
            # =================================================

            if total_tokens <= MAX_TOKENS:

                chunks = [full_text]

            else:

                blocks = split_legal_blocks(body)

                if not blocks:

                    chunks = [full_text]

                else:

                    chunks = build_chunks(
                        blocks,
                        prefix
                    )

            chunks = validate_chunks(chunks)

            safe_art_num = art_num.replace(".", "_")

            article_chunks = 0

            for idx, chunk in enumerate(
                chunks,
                start=1
            ):

                chunk = normalize_text(chunk)

                chunk = remove_duplicate_header(chunk)

                tokens = count_tokens(chunk)

                if tokens > ABSOLUTE_MAX_TOKENS:
                    continue

                node = {
                    "id": (
                        f"{doc_id}_"
                        f"st{safe_art_num}_"
                        f"c{idx}"
                    ),

                    "title": (
                        f"{doc_id.upper()} | "
                        f"{title}"
                    ),

                    "text": chunk,

                    "local_img": "",

                    "url": URL,

                    # metadata
                    "law": doc_id.upper(),

                    "article": art_num,

                    "chunk_index": idx,

                    "tokens": tokens
                }

                f.write(
                    json.dumps(
                        node,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_chunks += 1
                article_chunks += 1

            print(
                f"✅ Создано чанков: "
                f"{article_chunks}"
            )

    print("\n✅ Готово!")

    print(
        f"📦 Всего чанков: "
        f"{total_chunks}"
    )

# =========================================================

if __name__ == "__main__":
    main()