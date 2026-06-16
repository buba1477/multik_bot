import json
import re
import os
from pathlib import Path
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# --- OFFLINE FORCE ---
os.environ["HF_HUB_OFFSET"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- CONFIG ---
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/FRIDA"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MAX_TOKENS = 512
INPUT_PDF = "79-ФЗ.pdf"
OUTPUT_FILE = INPUT_PDF.replace(".pdf", ".jsonl")

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))

def extract_full_article_number(raw_header):
    raw_header = re.sub(r'^#{1,6}\s+', '', raw_header)
    match = re.search(r'Статья\s+([\d\.\s]+)', raw_header)
    if not match:
        return None, None
    raw_num_part = match.group(0) 
    num_only = match.group(1).strip()
    safe_num = re.sub(r'\s+', '.', num_only).rstrip('.')
    title_text = raw_header.replace(raw_num_part, '', 1).strip()
    title_text = re.sub(r'^[.#\s\-:]+', '', title_text).strip()
    return safe_num, title_text

def split_by_tokens(text, limit):
    ids = TOKENIZER.encode(text, add_special_tokens=False)

    chunks = []

    for i in range(0, len(ids), limit):
        chunk_ids = ids[i:i + limit]

        chunks.append(
            TOKENIZER.decode(
                chunk_ids,
                skip_special_tokens=True
            )
        )

    return chunks


def split_article_to_blocks(text):
    blocks = re.split(
        r'(?=^\s*-?\s*\d+(?:\.\d+)*\.\s)',
        text,
        flags=re.MULTILINE
    )

    return [
        b.strip()
        for b in blocks
        if b.strip()
    ]


def split_block_to_subpoints(text):
    blocks = re.split(
        r'(?=^\s*-?\s*\d+\))',
        text,
        flags=re.MULTILINE
    )

    return [
        b.strip()
        for b in blocks
        if b.strip()
    ]


def split_text_strictly(text, prefix, max_tokens):
    
    header = prefix + "\n"

    header_tokens = count_tokens(header)

    limit = max_tokens - header_tokens - 20
    if limit < 100:
        limit = 100

    chunks = []

    current = []

    current_tokens = 0

    article_blocks = split_article_to_blocks(text)

    for block in article_blocks:

        block_tokens = count_tokens(block)

        if block_tokens <= limit:

            if current_tokens + block_tokens <= limit:

                current.append(block)
                current_tokens += block_tokens
                continue

            chunks.append(
                header +
                "\n\n".join(current)
            )

            current = [block]
            current_tokens = block_tokens

            continue

        if current:

            chunks.append(
                header +
                "\n\n".join(current)
            )

            current = []
            current_tokens = 0

        subpoints = split_block_to_subpoints(block)

        if len(subpoints) <= 1:

            for piece in split_by_tokens(
                block,
                limit
            ):
                chunks.append(
                    header + piece
                )

            continue

        sub_current = []

        sub_tokens = 0

        for sub in subpoints:

            t = count_tokens(sub)

            if t <= limit:

                if sub_tokens + t <= limit:

                    sub_current.append(sub)
                    sub_tokens += t
                    continue

                chunks.append(
                    header +
                    "\n\n".join(sub_current)
                )

                sub_current = [sub]
                sub_tokens = t

            else:

                if sub_current:

                    chunks.append(
                        header +
                        "\n\n".join(sub_current)
                    )

                sub_current = []
                sub_tokens = 0

                for piece in split_by_tokens(
                    sub,
                    limit
                ):
                    chunks.append(
                        header + piece
                    )

        if sub_current:

            chunks.append(
                header +
                "\n\n".join(sub_current)
            )

    if current:

        chunks.append(
            header +
            "\n\n".join(current)
        )

    result = []

    for chunk in chunks:

        tok = count_tokens(chunk)

        if tok <= max_tokens:

            result.append(chunk)
            continue

        body = chunk[len(header):]

        for piece in split_by_tokens(
            body,
            limit - 30
        ):
            result.append(
                header + piece
            )

    final_result = []

    for chunk in result:

        tok = count_tokens(chunk)

        if tok <= max_tokens:
            final_result.append(chunk)
            continue

        header_only = header

        body = chunk[len(header_only):]

        safe_limit = max(
            50,
            max_tokens - count_tokens(header_only) - 5
        )

        for piece in split_by_tokens(body, safe_limit):
            final_result.append(
                header_only + piece
            )

    return final_result

def main():
    print(f"🧐 Обрабатываю {INPUT_PDF}...")
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.enable_remote_services = False
    
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    try:
        result = converter.convert(INPUT_PDF)
        content = result.document.export_to_markdown()
        content = re.sub(
    r'(- 1\).*?)'
    r'(- 2\).*?)'
    r'(- 3\).*?)'
    r'(\|.*?\|\n(?:\|.*?\|\n)+)'
    r'(законодательством Российской Федерации;)',
    r'\1\4\n\2\3\5',
    content,
    flags=re.S
)
      
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        return
    
    # --- ПОСТ-ОБРАБОТКА ---
    # 1. Чиним структуру — приклеиваем таблицу к пункту 1
    
    
    # 2. Чиним разорванные абзацы
 
    content = re.sub(
        r'Стр\.\s+\d+\s+из\s+\d+',
        '',
        content
    )
    
    content = re.sub(
        r'\b4\.\s*8\s*1\s*\.',
        '8.1.',
        content
    )

    content = re.sub(
        r'[ \t]+',
        ' ',
        content
    )

    content = re.sub(
        r'\r\n?',
        '\n',
        content
    )

    content = re.sub(
        r'\n{4,}',
        '\n\n',
        content
        )

    
    # Разделение по статьям
    sections = re.split(
        r'(?=^\s*#{0,6}\s*Статья\s+\d+)',
        content,
        flags=re.MULTILINE
    )
    
    doc_id = Path(INPUT_PDF).stem.lower().replace(" ", "_").strip()
    total = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for idx, sec in enumerate(sections):
            sec = sec.strip()
            if not sec or len(sec) < 30:
                continue
            
            lines = sec.split('\n')
            header_raw = lines[0].strip()
            
            if 'Глава' in header_raw or 'Раздел' in header_raw:
                continue
            
            art_num, art_title = extract_full_article_number(header_raw)
            
            if not art_num:
                continue
            
            clean_header = f"Статья {art_num}. {art_title}".strip()
            safe_id = art_num.replace('.', '_')
            base_id = f"{doc_id}_st{safe_id}"
            
            prefix = f"[{doc_id.upper()}] [{clean_header}]"
            body = "\n".join(lines[1:]).strip()
            
            final_segments = split_text_strictly(body, prefix, MAX_TOKENS)

            max_chunk = max(
            count_tokens(x)
            for x in final_segments
        )

            print(
                f"{clean_header}: max={max_chunk}"
            )
            
            for seg in final_segments:
                tok = count_tokens(seg)

                if tok > MAX_TOKENS:
                    raise RuntimeError(
                    f"OVERSIZE: {clean_header} -> {tok}"
                )
            for i, chunk_text in enumerate(final_segments, 1):
                chunk = {
                    "id": f"{base_id}_p{i}",
                    "title": clean_header,
                    "text": chunk_text.strip(),
                    "local_img": "",
                    "url": "http://www.kremlin.ru/acts/bank/21210"
                }
                f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total += 1
    
    print(f"✅ Готово! Записано {total} чанков.")

if __name__ == "__main__":
    main()