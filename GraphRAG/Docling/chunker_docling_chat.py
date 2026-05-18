import json
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
import os
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

# =========================================================
# CHUNK SETTINGS
# =========================================================

MAX_TOKENS = 350
ABSOLUTE_MAX_TOKENS = 512

OVERLAP_BLOCKS = 1

MIN_CHARS = 80
MIN_CHUNK_TOKENS = 80

URL = "http://kremlin.ru"

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
# HARD TOKEN SAFETY
# =========================================================

def enforce_token_limit(
    text,
    max_tokens=ABSOLUTE_MAX_TOKENS
):

    tokens = count_tokens(text)

    if tokens <= max_tokens:
        return [text]

    words = text.split()

    chunks = []
    temp = []

    for word in words:

        candidate = temp + [word]

        candidate_text = " ".join(candidate)

        if count_tokens(candidate_text) > max_tokens:

            chunks.append(
                " ".join(temp)
            )

            overlap = temp[-20:]

            while (
                overlap
                and count_tokens(
                    " ".join(overlap)
                ) > max_tokens // 3
            ):
                overlap = overlap[1:]

            temp = overlap + [word]

        else:
            temp.append(word)

    if temp:
        chunks.append(" ".join(temp))

    return chunks

# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize_text(text: str):

    if not text:
        return ""

    text = text.replace("\xa0", " ")

    text = re.sub(r'#+', ' ', text)

    text = re.sub(r'\r\n?', '\n', text)

    text = re.sub(
        r'(?<!\n)-\s+',
        r'\n- ',
        text
    )

    text = re.sub(
        r'\n{3,}',
        '\n\n',
        text
    )

    text = re.sub(
        r'[ \t]+',
        ' ',
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
# ARTICLE NUMBER
# =========================================================

def normalize_article_number(num: str):

    num = re.sub(r'<[^>]+>', '', num)

    num = num.replace("-", ".")

    num = re.sub(r'\s+', '.', num)

    num = re.sub(r'\.+', '.', num)

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

    title = re.sub(r'^\.+', '', title)

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
# CLEAN LEGAL TEXT
# =========================================================
def clean_legal_text(text: str):
    
    """
    Минимальная безопасная очистка для legal RAG.

    НЕ меняем текст закона.
    Удаляем только технический мусор PDF/markdown.
    """

    if not text:
        return ""

    # unicode spaces
    text = text.replace("\xa0", " ")

    # line endings
    text = re.sub(r'\r\n?', '\n', text)

    # markdown links:
    # [text](url) -> text
    text = re.sub(
        r'\[([^\]]+)\]\([^)]+\)',
        r'\1',
        text
    )

    # broken markdown leftovers
    text = re.sub(
        r'\]\([^)]+\)',
        ' ',
        text
    )

    # html tags
    text = re.sub(
        r'<[^>]+>',
        ' ',
        text
    )

    # docling/pdf garbage
    text = re.sub(
        r'k6cl[a-zA-Z0-9:=&/?._-]+',
        ' ',
        text
    )

    # multiple spaces
    text = re.sub(
        r'[ \t]+',
        ' ',
        text
    )

    # too many newlines
    text = re.sub(
        r'\n{3,}',
        '\n\n',
        text
    )

    return text.strip()
# =========================================================
# SPLIT LEGAL BLOCKS
# =========================================================

def split_legal_blocks(text: str):

    text = clean_legal_text(text)

    if not text:
        return []

    pattern = (
        r'(?='
        r'(?:^|\n)'
        r'(?:'
        r'\d+\.\s|'
        r'\d+\.\d+\.\s'
        r')'
        r')'
    )

    raw_blocks = re.split(
        pattern,
        text,
        flags=re.MULTILINE
    )

    blocks = []

    for block in raw_blocks:

        block = normalize_text(block)

        if len(block) < 5:
            continue

        blocks.append(block)

    if not blocks and text.strip():
        return [text.strip()]

    return blocks

# =========================================================
# SPLIT HUGE BLOCK
# =========================================================

def split_huge_block(
    block,
    limit_tokens
):

    words = block.split()

    chunks = []
    temp = []

    for w in words:

        candidate = temp + [w]

        candidate_text = " ".join(candidate)

        candidate_tokens = count_tokens(
            candidate_text
        )

        if candidate_tokens > limit_tokens:

            chunk_text = " ".join(temp)

            chunk_text = normalize_text(
                chunk_text
            )

            if count_tokens(chunk_text) > limit_tokens:

                safe_parts = enforce_token_limit(
                    chunk_text,
                    limit_tokens
                )

                chunks.extend(safe_parts)

            else:
                chunks.append(chunk_text)

            overlap_words = temp[-20:]

            while (
                overlap_words
                and count_tokens(
                    " ".join(overlap_words)
                ) > limit_tokens // 3
            ):
                overlap_words = overlap_words[1:]

            temp = overlap_words + [w]

        else:
            temp.append(w)

    if temp:

        chunk_text = " ".join(temp)

        chunk_text = normalize_text(
            chunk_text
        )

        if count_tokens(chunk_text) > limit_tokens:

            safe_parts = enforce_token_limit(
                chunk_text,
                limit_tokens
            )

            chunks.extend(safe_parts)

        else:
            chunks.append(chunk_text)

    return chunks

# =========================================================
# STRUCTURE-AWARE CHUNKING
# =========================================================

def build_chunks(
    blocks,
    prefix,
    max_tokens=MAX_TOKENS
):

    header = prefix + "\n\n"

    header_tokens = count_tokens(header)

    body_limit = max_tokens - header_tokens

    if body_limit < 50:
        body_limit = 200

    chunks = []

    current_blocks = []
    current_tokens = 0

    for block in blocks:

        block_tokens = count_tokens(block)

        # HUGE BLOCK
        if block_tokens > body_limit:

            if current_blocks:

                chunk_text = (
                    header
                    + "\n\n".join(current_blocks)
                )

                chunks.append(chunk_text)

                current_blocks = []
                current_tokens = 0

            huge_parts = split_huge_block(
                block,
                body_limit
            )

            for hp in huge_parts:

                chunk_text = header + hp

                if (
                    count_tokens(chunk_text)
                    > ABSOLUTE_MAX_TOKENS
                ):

                    safe_parts = enforce_token_limit(
                        chunk_text,
                        ABSOLUTE_MAX_TOKENS
                    )

                    chunks.extend(safe_parts)

                else:
                    chunks.append(chunk_text)

            continue

        projected = (
            current_tokens
            + block_tokens
        )

        if projected > body_limit:

            chunk_text = (
                header
                + "\n\n".join(current_blocks)
            )

            if (
                count_tokens(chunk_text)
                > ABSOLUTE_MAX_TOKENS
            ):

                safe_parts = enforce_token_limit(
                    chunk_text,
                    ABSOLUTE_MAX_TOKENS
                )

                chunks.extend(safe_parts)

            else:
                chunks.append(chunk_text)

            overlap = (
                current_blocks[-OVERLAP_BLOCKS:]
                if current_blocks
                else []
            )

            current_blocks = overlap.copy()

            current_tokens = count_tokens(
                "\n\n".join(current_blocks)
            )

        current_blocks.append(block)

        current_tokens += block_tokens

    if current_blocks:

        chunk_text = (
            header
            + "\n\n".join(current_blocks)
        )

        if (
            count_tokens(chunk_text)
            > ABSOLUTE_MAX_TOKENS
        ):

            safe_parts = enforce_token_limit(
                chunk_text,
                ABSOLUTE_MAX_TOKENS
            )

            chunks.extend(safe_parts)

        else:
            chunks.append(chunk_text)

    return chunks

# =========================================================
# MERGE SMALL CHUNKS
# =========================================================

def merge_small_chunks(
    chunks,
    min_tokens=MIN_CHUNK_TOKENS,
    max_tokens=MAX_TOKENS
):

    if not chunks:
        return []

    merged = []

    buffer = chunks[0]

    for chunk in chunks[1:]:

        buffer_tokens = count_tokens(buffer)

        combined = (
            buffer
            + "\n\n"
            + chunk
        )

        combined_tokens = count_tokens(combined)

        if (
            buffer_tokens < min_tokens
            and combined_tokens <= max_tokens
        ):

            buffer = combined

        else:

            merged.append(buffer)

            buffer = chunk

    if buffer:

        if merged:

            combined = (
                merged[-1]
                + "\n\n"
                + buffer
            )

            combined_tokens = count_tokens(combined)

            if (
                count_tokens(buffer) < min_tokens
                and combined_tokens <= max_tokens
            ):

                merged[-1] = combined

            else:
                merged.append(buffer)

        else:
            merged.append(buffer)

    return merged

# =========================================================
# FINAL VALIDATION
# =========================================================

def validate_chunks(chunks):

    final_chunks = []

    for chunk in chunks:

        chunk = normalize_text(chunk)

        if not chunk:
            continue

        tokens = count_tokens(chunk)

        if tokens > ABSOLUTE_MAX_TOKENS:

            safe_parts = enforce_token_limit(
                chunk,
                ABSOLUTE_MAX_TOKENS
            )

            final_chunks.extend(safe_parts)

        else:
            final_chunks.append(chunk)

    return final_chunks

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
    # CLEAN LINES
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
    articles_processed = 0
    articles_skipped = []

    # =====================================================
    # SAVE
    # =====================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for article_idx, article in enumerate(articles):

            article = normalize_text(article)

            if len(article) < 30:
                continue

            # =================================================
            # EXTRACT TITLE + BODY
            # =================================================

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

            # =================================================
            # BODY
            # =================================================

            body = article[
                len(article_header):
            ].strip()

            body = clean_legal_text(body)

            # =================================================
            # FIX EMPTY BODY
            # =================================================

            if not body:

                split_once = re.split(
                    r'(Статья\s+[\d\.\-\s]+\.?.*?)',
                    article,
                    maxsplit=1,
                    flags=re.I
                )

                if len(split_once) >= 3:

                    body = split_once[2].strip()

                    body = clean_legal_text(body)

            # =================================================
            # EMPTY CHECK
            # =================================================

            if (
                not body
                or len(body.strip()) < 5
            ):

                print(
                    f"⚠️ Статья {art_num}: "
                    f"пустое тело"
                )

                articles_skipped.append(
                    f"{art_num} (пустое тело)"
                )

                continue

            # =================================================
            # PREFIX
            # =================================================

            prefix = (
                f"[{doc_id.upper()}] "
                f"[{title}]"
            )

            full_text = (
                prefix
                + "\n\n"
                + body
            )

            total_tokens = count_tokens(
                full_text
            )

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

                legal_blocks = split_legal_blocks(
                    body
                )

                if not legal_blocks:

                    chunks = [full_text]

                else:

                    chunks = build_chunks(
                        legal_blocks,
                        prefix
                    )

            # =================================================
            # MERGE
            # =================================================

            chunks = merge_small_chunks(
                chunks,
                MIN_CHUNK_TOKENS,
                MAX_TOKENS
            )

            chunks = validate_chunks(chunks)

            if not chunks:

                articles_skipped.append(
                    f"{art_num} (0 чанков)"
                )

                continue

            # =================================================
            # SAVE CHUNKS
            # =================================================

            safe_art_num = (
                art_num
                .replace(".", "_")
            )

            article_chunks = 0

            for idx, chunk in enumerate(
                chunks,
                start=1
            ):

                chunk = normalize_text(chunk)

                chunk = remove_duplicate_header(
                    chunk
                )

                if len(chunk) < MIN_CHARS:
                    continue

                chunk_tokens = count_tokens(
                    chunk
                )

                if (
                    chunk_tokens
                    > ABSOLUTE_MAX_TOKENS
                ):
                    continue

                node = {
                    "id": (
                        f"{doc_id}_"
                        f"st{safe_art_num}_"
                        f"c{idx}"
                    ),
                    "title": title,
                    "text": chunk,
                    "local_img": "",
                    "url": URL
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
                f"  ✅ Создано чанков: "
                f"{article_chunks}"
            )

            articles_processed += 1

            if article_idx % 10 == 0:

                print(
                    f"   Обработано статей: "
                    f"{article_idx}"
                )

    # =====================================================
    # FINAL
    # =====================================================

    print("\n✅ Готово!")

    print(
        f"   Обработано статей: "
        f"{articles_processed}"
    )

    print(
        f"   Создано чанков: "
        f"{total_chunks}"
    )

    print(
        f"   Пропущено статей: "
        f"{len(articles_skipped)}"
    )

    if (
        articles_skipped
        and len(articles_skipped) <= 20
    ):

        print(
            "   Список пропущенных: "
            + ", ".join(articles_skipped)
        )

# =========================================================

if __name__ == "__main__":
    main()