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

MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/FRIDA"
INPUT_PDF = "79-ФЗ.pdf"
OUTPUT_FILE = "79-FZ.jsonl"
URL = "http://www.kremlin.ru/acts/bank/21210"

MAX_TOKENS = 512
# 🔥 УВЕЛИЧИЛ ДО 1024, ЧТОБЫ БОЛЬШИЕ ПУНКТЫ (СТАТЬЯ 17) НЕ РЕЗАЛИСЬ!
SAFE_MAX_TOKENS = 460
TARGET_TOKENS = 350
MIN_TOKENS = 0
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
# UTILITIES
# =========================================================

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\d+)\s+(\d+)\.', r'\1.\2.', text)
    return text.strip()


def is_garbage(line: str) -> bool:
    s = line.strip().lower()
    garbage = {"события", "структура", "контакты", "документы", "поиск",
               "rutube", "telegram", "youtube", "введите запрос", "найти", "для сми"}
    return len(s) < 2 or s in garbage or s.startswith("http") or "pravo.gov.ru" in s


def is_orphan_number(text: str) -> bool:
    """Проверяет, является ли чанк одиноким номером пункта"""
    cleaned = re.sub(r'^\[[^\]]+\]\[Статья[^\]]+\]', '', text)
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

def fix_nested_numbering(text: str) -> str:
    """
    Исправляет сквозную нумерацию от Docling, восстанавливая вложенные списки.
    Превращает:
        4. а) ... 5. б) ... 6. в) ... 8. а) ... 9. б) ... 10. в) ...
    в:
        а) ... б) ... в) ... а) ... б) ... в) ...
    """
    # Удаляем цифры с точкой перед буквой со скобкой
    text = re.sub(r'(?m)^\s*\d+\.\s+([а-я]\)\s*)', r'   \1', text)
    return text


def clean_legal_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r'\r\n?', '\n', text)

    # Remove markdown
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\]\([^)]+\)', ' ', text)
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    text = re.sub(r'(?m)^\s*-\s+', '', text)

    # Remove HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # Fix line breaks
    text = re.sub(r'([а-яa-z])\n-\s*([а-яa-z])', r'\1\2', text, flags=re.I)
    text = re.sub(r'([а-яa-z,;:])\n([а-яa-z])', r'\1 \2', text, flags=re.I)

    # Remove editorial marks
    patterns = [r'\(В редакции[^)]*\)', r'\(Дополнение[^)]*\)',
                r'\(Утратил силу[^)]*\)', r'\(Наименование[^)]*\)']
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.I)

    text = re.sub(r'(?mi)^\s*(?:\d+\.\s*)?Абзац\.\s*$', '', text)
    text = re.sub(r'k6cl[a-zA-Z0-9:=&/?._-]+', ' ', text)

    # ========== ОСНОВНЫЕ ФИКСЫ ==========
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

    # 🔥 ИСПРАВЛЯЕМ СКВОЗНУЮ НУМЕРАЦИЮ ОТ DOCLING
    text = fix_nested_numbering(text)

    # Final spacing
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_article_title(text: str):
    text = normalize_text(text)
    m = re.search(r'Статья\s+([\d.\s\-]+)', text)
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
# TEXT SPLITTING
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

    # Режем по основным пунктам (1), 2), 3) и т.д.
    pattern = re.compile(r'(?m)^(?=\d+\)\s+[А-ЯA-Z])')
    parts = re.split(pattern, text)

    for placeholder, original in protected.items():
        parts = [p.replace(placeholder, original) for p in parts]

    result = [p.strip() for p in parts if len(p.strip()) > 5]
    
    if not result:
        return [text]
    
    return result


def split_large_paragraph(para: str, limit_tokens: int):
    """Разбивает большой параграф по предложениям"""
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


def build_chunks_respectful(paragraphs, prefix, max_tokens=SAFE_MAX_TOKENS):
    """Собирает чанки, НЕ РАЗРЫВАЯ параграфы."""
    chunks = []
    current = []
    prefix_tokens = count_tokens(prefix)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_with_prefix = prefix + "\n\n" + para
        para_tokens = count_tokens(para_with_prefix)

        if para_tokens > max_tokens:
            if current:
                chunks.append(prefix + "\n\n" + "\n\n".join(current))
                current = []

            sub_paragraphs = split_large_paragraph(para, max_tokens - prefix_tokens - 4)
            for sub in sub_paragraphs:
                if sub.strip():
                    chunks.append(prefix + "\n\n" + sub)
            continue

        candidate = prefix + "\n\n" + "\n\n".join(current + [para])
        if count_tokens(candidate) <= max_tokens:
            current.append(para)
        else:
            if current:
                chunks.append(prefix + "\n\n" + "\n\n".join(current))
            current = [para]

    if current:
        chunks.append(prefix + "\n\n" + "\n\n".join(current))

    return chunks


def validate_chunks(chunks):
    """Финальная валидация чанков"""
    seen, final = set(), []

    for chunk in chunks:
        chunk = normalize_text(chunk)
        if not chunk:
            continue

        token_count = count_tokens(chunk)
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
    print(f"🚀 Стартуем умный чанкинг: {INPUT_PDF}")
    if not Path(INPUT_PDF).exists():
        print(f"❌ Файл {INPUT_PDF} не найден!")
        return

    converter = DocumentConverter(format_options={
        "pdf": PdfFormatOption(pipeline_options=PdfPipelineOptions(enable_remote_services=False))
    })

    try:
        markdown = converter.convert(INPUT_PDF).document.export_to_markdown()
    except Exception as e:
        print(f"❌ Ошибка конвертации Docling: {e}")
        return

    # Предварительная очистка строк от мусора
    clean_lines = [line for line in markdown.split("\n") if not is_garbage(line)]
    clean_text = "\n".join(clean_lines)
    clean_text = re.sub(r'===== Page \d+ =====', '', clean_text)
    clean_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean_text)

    # 🔥 ФИКС ДЕЛЕНИЯ НА СТАТЬИ: 
    # Ищем слово "Статья" с учетом возможных решеток Markdown (#) и любых пробелов
    article_pattern = re.compile(r'(?i)(?:\n+|\A)(?=(?:#{1,6}\s+)?Статья\s+\d+)')
    articles = article_pattern.split(clean_text)
    
    # Первую часть (все, что идет ДО Статьи 1 — преамбула, дата, название закона) сохраняем отдельно
    preamble = articles[0].strip() if articles else ""
    articles = [a.strip() for a in articles[1:] if a.strip() and len(a.strip()) > 10]

    print(f"📑 Найдено изолированных статей: {len(articles)}")
    
    doc_id = Path(INPUT_PDF).stem.lower().replace(" ", "_")
    total_chunks = 0
    skipped_articles = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Сначала обработаем преамбулу как отдельный чанк, чтобы не потерять название закона
        if preamble and len(preamble) > 15:
            prefix = f"[{doc_id.upper()}] [Преамбула и общие данные]"
            node = {
                "id": f"{doc_id}_preamble_c1",
                "title": "Общие положения и преамбула закона",
                "text": f"{prefix}\n\n{clean_legal_text(preamble)}",
                "local_img": "",
                "url": URL
            }
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
            total_chunks += 1
            print("📝 Записан чанк преамбулы закона")

        # Теперь идем по реальным статьям
        for idx, art_raw in enumerate(articles):
            art_raw = normalize_text(art_raw)
            
            # 🔥 ФИКС ИЗВЛЕЧЕНИЯ ЗАГОЛОВКА: Учитываем решетки Markdown перед словом Статья
            art_header_match = re.match(r'^((?:#{1,6}\s+)?Статья\s+[\d\s-]+\.?\s*[^\n]*)', art_raw, flags=re.I)
            if not art_header_match:
                print(f"⚠️ Пропущена строка {idx+1}, дебаг-превью: {art_raw[:80]}...")
                skipped_articles += 1
                continue

            raw_header = art_header_match.group(1).strip()
            # Очищаем сам заголовок от решеток для красивого вывода в title
            clean_header = re.sub(r'^#{1,6}\s*', '', raw_header)
            
            art_num, title = extract_article_title(clean_header)
            if not art_num:
                print(f"⚠️ Не удалось распарсить номер из заголовка: {clean_header[:50]}")
                skipped_articles += 1
                continue

            # Зачищаем тело статьи (отрезаем заголовок)
            body = clean_legal_text(art_raw[len(raw_header):].strip())
            prefix = f"[{doc_id.upper()}] [{title}]"
            
            # Делим тело на параграфы по структуре НПА
            paragraphs = split_by_paragraphs(body)
            if not paragraphs:
                paragraphs = [p.strip() for p in body.split('\n\n') if p.strip()]

            # Формируем безопасные чанки с учетом лимитов токенов
            chunks = build_chunks_respectful(paragraphs, prefix)
            safe_art_num = art_num.replace(".", "_")

            # Валидация и запись в JSONL
            for chunk_idx, chunk in enumerate(chunks, 1):
                chunk = normalize_text(chunk)
                if not chunk or count_tokens(chunk) > ABSOLUTE_MAX_TOKENS:
                    continue

                node = {
                    "id": f"{doc_id}_st{safe_art_num}_c{chunk_idx}",
                    "title": title,
                    "text": chunk,
                    "local_img": "",
                    "url": URL
                }
                f.write(json.dumps(node, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"\n{'=' * 50}")
    print(f"📊 ИТОГИ СЕМАНТИЧЕСКОГО ЧАНКИНГА:")
    print(f"✅ Всего чанков записано в файл: {total_chunks}")
    print(f"⏳ Безопасно пропущено блоков: {skipped_articles}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()