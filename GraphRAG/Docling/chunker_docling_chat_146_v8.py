import json
import re
import os
from pathlib import Path
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# --- OFFLINE FORCE ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- CONFIG ---
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/FRIDA"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
MAX_TOKENS = 400
TARGET_TOKENS = 350

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = SCRIPT_DIR / "raw"
CHUNKS_DIR = SCRIPT_DIR / "chunks"

def count_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def extract_full_article_number(raw_header):
    match = re.search(r'Статья\s+([\d\.\s]+)', raw_header)
    if not match:
        return None, None
    raw_num_part = match.group(0) 
    num_only = match.group(1).strip()
    safe_num = re.sub(r'\s+', '.', num_only).rstrip('.')
    title_text = raw_header.replace(raw_num_part, '', 1).strip()
    title_text = re.sub(r'^[.#\s]+', '', title_text).strip()
    return safe_num, title_text

def split_text_strictly(text, prefix, max_t, target_t=None):
    if target_t is None:
        target_t = max_t
    
    header = prefix if prefix.endswith('\n') else prefix + "\n"
    header_t = len(TOKENIZER.encode(header, add_special_tokens=False))
    hard_limit = max_t - header_t
    soft_limit = target_t - header_t
    
    para_pattern = re.compile(r'(?m)\n+|^(?=(?:\d+(?:\.\d+|\^[\d\w]+)?[\.\)])\s+|[а-я]\)\s+)')
    paragraphs = para_pattern.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_t = 0
    
    for para in paragraphs:
        para_t = len(TOKENIZER.encode(para, add_special_tokens=False))
        
        if para_t > hard_limit:
            if current_chunk:
                chunks.append(header + "\n".join(current_chunk))
                current_chunk = []
                current_t = 0
            
            item_match = re.match(r'^(\d+(?:\.\d+|\^[\d\w]+)?[\.\)]|[а-я]\)\s+)', para)
            local_context = f" (в части пункта {item_match.group(1).strip()})" if item_match else " (продолжение)"
            extended_header = header.strip() + local_context + "\n"
            ext_limit = max_t - len(TOKENIZER.encode(extended_header, add_special_tokens=False))
            
            sentences = re.split(r'(?<=[\.\!\?;])\s+(?=[А-ЯA-Z0-9а-яa-z])', para)
            
            sub_chunk = []
            sub_t = 0
            for sent in sentences:
                sent_t = len(TOKENIZER.encode(sent + " ", add_special_tokens=False))
                
                if sent_t > ext_limit:
                    if sub_chunk:
                        chunks.append(extended_header + " ".join(sub_chunk))
                        sub_chunk = []
                        sub_t = 0
                    
                    words = sent.split()
                    word_chunk = []
                    word_t = 0
                    for w in words:
                        w_t = len(TOKENIZER.encode(w + " ", add_special_tokens=False))
                        if word_t + w_t > ext_limit:
                            chunks.append(extended_header + " ".join(word_chunk))
                            word_chunk = [w]
                            word_t = w_t
                        else:
                            word_chunk.append(w)
                            word_t += w_t
                    if word_chunk:
                        sub_chunk = word_chunk
                        sub_t = word_t
                    continue
                
                if sub_t + sent_t > ext_limit:
                    if sub_chunk:
                        chunks.append(extended_header + " ".join(sub_chunk))
                    sub_chunk = [sent]
                    sub_t = sent_t
                else:
                    sub_chunk.append(sent)
                    sub_t += sent_t
            
            if sub_chunk:
                current_chunk = sub_chunk
                current_t = sub_t
            continue
        
        if current_t == 0:
            if para_t > hard_limit:
                chunks.append(header + para)
                continue
            current_chunk.append(para)
            current_t = para_t
        elif current_t + para_t > soft_limit:
            chunks.append(header + "\n".join(current_chunk))
            current_chunk = [para]
            current_t = para_t
        else:
            current_chunk.append(para)
            current_t += para_t
    
    if current_chunk:
        chunks.append(header + "\n".join(current_chunk))
    return chunks

def validate_jsonl(filepath):
    errors = []
    chunk_count = 0
    ids = set()
    token_counts = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON - {e}")
                continue
            
            text = obj.get("text", "")
            if not text:
                errors.append(f"Line {line_num}: empty text")
                continue
            
            chunk_id = obj.get("id", "")
            if chunk_id in ids:
                errors.append(f"Line {line_num}: duplicate ID '{chunk_id}'")
            ids.add(chunk_id)
            
            t_count = len(TOKENIZER.encode(text, add_special_tokens=False))
            token_counts.append(t_count)
            if t_count > 400:
                errors.append(f"Line {line_num}: {t_count} tokens > 400 (id={chunk_id})")
            
            chunk_count += 1
    
    return errors, chunk_count, token_counts


def process_single_pdf(pdf_path):
    print(f"  Docling extraction...")
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.enable_remote_services = False
    pipeline_options.do_table_structure = False
    
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    result = converter.convert(str(pdf_path))
    content = result.document.export_to_markdown()
    
    content = re.sub(r'(?i)Стр\.\s+\d+\s+из\s+\d+', '', content)
    content = re.sub(r'(?i)Страница\s+\d+', '', content)
    content = re.sub(r'(?m)^\s*\d+\.\s+([а-яА-Яa-zA-Z]\))', r'\1', content)
    content = re.sub(r'(?m)^\s*\d+\.\s+(\d+\))', r'\1', content)
    content = re.sub(r'[ \t]+', ' ', content)
    content = re.sub(r'\r\n?', '\n', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    sections = re.split(r'\n(?=###\s+Статья|##\s+Статья|^Статья\s+\d+)', content, flags=re.MULTILINE)
    
    doc_id = pdf_path.stem.lower().replace(" ", "_").strip()
    output_path = CHUNKS_DIR / f"{pdf_path.stem}.jsonl"
    total = 0
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
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
            
            final_segments = split_text_strictly(body, prefix, MAX_TOKENS, TARGET_TOKENS)
            
            for i, chunk_text in enumerate(final_segments, 1):
                chunk = {
                    "id": f"{base_id}_p{i}",
                    "title": clean_header,
                    "text": chunk_text.strip(),
                    "local_img": "",
                    "url": ""
                }
                f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total += 1
    
    return total, output_path

def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Нет PDF-файлов в {RAW_DIR}")
        return
    
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    n = len(pdf_files)
    ok_count = 0
    error_count = 0
    total_chunks_all = 0
    
    print(f"Найдено {n} PDF в {RAW_DIR}")
    print(f"Целевой размер: {TARGET_TOKENS} токенов, макс: {MAX_TOKENS} токенов")
    print()
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx}/{n}] Processing {pdf_path.name}...")
        try:
            chunk_count, output_path = process_single_pdf(pdf_path)
            
            val_errors, val_count, token_counts = validate_jsonl(output_path)
            
            if val_errors:
                print(f"  Валидация: {len(val_errors)} ошибок")
                for err in val_errors[:5]:
                    print(f"    {err}")
                error_count += 1
            else:
                ok_count += 1
            
            total_chunks_all += val_count
            
            if token_counts:
                min_t = min(token_counts)
                max_t = max(token_counts)
                avg_t = sum(token_counts) / len(token_counts)
                print(f"  chunks: {val_count}, tokens: min={min_t}, avg={avg_t:.1f}, max={max_t}")
            
            print(f"[OK] {pdf_path.name} -> {output_path.name} -> {val_count} chunks")
            
        except Exception as e:
            print(f"[ERROR] {pdf_path.name} -> {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
        
        print()
    
    print("=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print(f"   PDF найдено:     {n}")
    print(f"   Успешно:         {ok_count}")
    print(f"   Ошибок:          {error_count}")
    print(f"   Всего chunks:    {total_chunks_all}")
    print("=" * 60)


if __name__ == "__main__":
    main()
