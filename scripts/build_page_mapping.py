#!/usr/bin/env python3
"""
build_page_mapping.py - сопоставление чанков с номерами страниц PDF.

Для каждого чанка (из chunks/*.jsonl) ищет точное вхождение первых 200 символов
текста чанка (probe) в тексте каждой страницы соответствующего PDF.

Результат:
  mapping/chunk_to_page.json   - {chunk_id: page_number | null}
  mapping/ambiguous.json        - список неоднозначных совпадений

Запуск:
  python3 scripts/build_page_mapping.py          # полный прогон
  python3 scripts/build_page_mapping.py --dry-run  # 5 чанков, без записи

Требования:
  - PyMuPDF (fitz) - pip install pymupdf
  - venv/ - уже есть
"""

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

RAW_DIR = Path("raw")
CHUNKS_DIR = Path("chunks")
MAPPING_DIR = Path("mapping")
OUTPUT_FILE = MAPPING_DIR / "chunk_to_page.json"
AMBIGUOUS_FILE = MAPPING_DIR / "ambiguous.json"

PROBE_MIN_LENGTH = 50
DRY_RUN_LIMIT = 5

# Скользящее окно: смещения (в символах) для probe длиной 200.
# Каждый offset — попытка найти точное совпадение, сдвинув начало probe
# мимо «шумного» начала (номер подпункта, сноска, префикс).
SLIDING_OFFSETS = [0, 50, 100, 150, 200, 250, 300]

# Разделитель страниц: один символ, не встречается в текстах.
PAGE_DELIM = chr(0)  # NUL



def _normalize(text: str) -> str:
    """
    - Удалить заголовки страниц вида "Стр. N из M" (есть в PDF, нет в чанках).
    - NBSP (U+00A0) и др. Unicode-пробелы -> обычный пробел.
    - Сжать все \\s+ в один пробел.
    - Убрать ведущие/замыкающие пробелы.
    """
    text = re.sub(r'\u0421\u0442\u0440\.\s*\d+\s*\u0438\u0437\s*\d+', "", text)
    text = re.sub(r"[\xa0\u200b\u202f\u2060\u3000]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_search(text: str) -> str:
    """
    Нормализация для поискового probe.
    В дополнение к _normalize() удаляет артефакты \\| (разметка таблиц).
    Исходный текст чанка и PDF не изменяются — нормализация применяется
    только к копиям в момент поиска.
    """
    text = _normalize(text)
    text = re.sub(r'\\[|]', ' ', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def _strip_chunk_prefix(text: str) -> str:
    """
    Удалить [doc_display] [section_title] из начала текста чанка.
    Удаляется только строгий двухблочный префикс в начале строки.
    Квадратные скобки внутри текста закона не затрагиваются.
    """
    return re.sub(r'^\[[^\]]+\]\s*\[[^\]]+\]\s*\n?', '', text, count=1)



def build_doc_map() -> dict[str, tuple[Path, Path]]:
    pdfs: dict[str, list[Path]] = {}
    for f in sorted(RAW_DIR.glob("*.pdf")):
        stem = f.stem.lower()
        pdfs.setdefault(stem, []).append(f)

    jsonls: dict[str, list[Path]] = {}
    for f in sorted(CHUNKS_DIR.glob("*.jsonl")):
        stem = f.stem.lower()
        jsonls.setdefault(stem, []).append(f)

    doc_map: dict[str, tuple[Path, Path]] = {}
    all_stems = sorted(set(pdfs) | set(jsonls))

    for stem in all_stems:
        pdf_list = pdfs.get(stem, [])
        jsonl_list = jsonls.get(stem, [])

        if len(pdf_list) != 1:
            print(f"[WARNING] Stem='{stem}': PDF " + chr(1085) + chr(1072) + chr(1081) + chr(1076) + chr(1077) + chr(1085) + chr(1086) + " {len(pdf_list)}", file=sys.stderr)
            continue
        if len(jsonl_list) != 1:
            print(f"[WARNING] Stem='{stem}': JSONL " + chr(1085) + chr(1072) + chr(1081) + chr(1076) + chr(1077) + chr(1085) + chr(1086) + " {len(jsonl_list)}", file=sys.stderr)
            continue

        doc_map[stem] = (pdf_list[0], jsonl_list[0])

    print(f"\nPDF-JSONL matching: {len(doc_map)}")
    return doc_map


def count_nonempty_lines(doc_map: dict[str, tuple[Path, Path]]) -> int:
    total = 0
    errs = 0
    for _stem, (_pdf_path, jsonl_path) in doc_map.items():
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total += 1
        except Exception as exc:
            print(f"[WARNING] Cannot count {jsonl_path.name}: {exc}", file=sys.stderr)
            errs += 1
    if errs:
        print(f"[WARNING] {errs} JSONL files could not be counted", file=sys.stderr)
    return total


def extract_pages(pdf_path: Path) -> list[str]:
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = [doc.load_page(i).get_text() for i in range(len(doc))]
    doc.close()
    return pages


def _build_concat_index(pages: list[str]) -> tuple[str, list[int]]:
    """
    Склеивает страницы через \\x00 (не встречается в русском тексте)
    и возвращает массив накопленных длин для bisect.

    cum_lens[i] = начало страницы i (0-based) в склеенной строке.
    cum_lens[i+1] = позиция сразу после разделителя после страницы i.
    """
    cum_lens = [0]
    for pt in pages:
        cum_lens.append(cum_lens[-1] + len(pt) + 1)  # +1 за \\x00
    concat_text = PAGE_DELIM.join(pages)
    return concat_text, cum_lens


def _find_pages_in_concat(probe_norm: str, concat_text: str, cum_lens: list[int]) -> list[int]:
    """
    Быстрый поиск probe_norm в склеенном тексте.
    Использует str.find() (O(n) C-level) и bisect для определения страницы.
    Проверяет, что совпадение не пересекает границу страниц.
    """
    found: set[int] = set()
    search_start = 0
    probe_len = len(probe_norm)
    while True:
        pos = concat_text.find(probe_norm, search_start)
        if pos == -1:
            break
        page_idx = bisect.bisect_right(cum_lens, pos) - 1
        if 0 <= page_idx < len(cum_lens) - 1:
            if pos + probe_len <= cum_lens[page_idx + 1]:
                found.add(page_idx + 1)
        search_start = pos + 1
        if search_start >= len(concat_text):
            break
    return sorted(found)


def _pages_for_offset(
    text: str,
    offset: int,
    concat_text: str,
    cum_lens: list[int],
) -> list[int]:
    """
    Найти страницы для probe = text[offset:offset+200].
    Возвращает уникальные отсортированные страницы.
    """
    probe = text[offset:offset + 200]
    if len(probe) < PROBE_MIN_LENGTH:
        return []

    probe = _strip_chunk_prefix(probe)
    if len(probe) < PROBE_MIN_LENGTH:
        return []

    probe_norm = _normalize_search(probe)
    if len(probe_norm) < PROBE_MIN_LENGTH:
        return []

    return _find_pages_in_concat(probe_norm, concat_text, cum_lens)


def find_page_for_chunk(
    chunk: dict,
    pages_norm: list[str],
    pages_norm_search: list[str],
    concat_text: str | None = None,
    cum_lens: list[int] | None = None,
) -> tuple[int | None, list[int]]:
    """
    Поиск страницы чанка с приоритетом offset=0.

    1. offset=0 (probe = text[:200]):
       - ровно 1 страница -> однозначное сопоставление;
       - больше 1          -> неоднозначно (page=None);
       - ни одной         -> перейти к скользящему окну.
    2. Скользящее окно (SLIDING_OFFSETS, без повторного offset=0):
       собирает страницы со всех смещений; 1 -> однозначно, >1 -> ambiguous.

    Нормализация `\\|` применяется только к поисковому представлению.

    Возвращает: (page_number | None, list[int] найденных страниц)
    """
    text = chunk.get("text", "")
    if not text or not concat_text or not cum_lens:
        return None, []

    # Шаг 1: приоритет offset=0
    offset0_pages = _pages_for_offset(text, 0, concat_text, cum_lens)
    if len(offset0_pages) == 1:
        return offset0_pages[0], offset0_pages
    if len(offset0_pages) > 1:
        return None, offset0_pages

    # Шаг 2: скользящее окно (offset=0 уже пуст)
    all_pages: set[int] = set()
    for offset in SLIDING_OFFSETS:
        if offset == 0:
            continue
        all_pages.update(_pages_for_offset(text, offset, concat_text, cum_lens))

    if not all_pages:
        return None, []

    sorted_pages = sorted(all_pages)
    if len(sorted_pages) > 1:
        return None, sorted_pages
    return sorted_pages[0], sorted_pages


def _process_chunk(
    chunk: dict,
    pages_norm: list[str] | None,
    pages_norm_search: list[str] | None,
    concat_text: str | None,
    cum_lens: list[int] | None,
    mapping: dict[str, int | None],
    ambiguous_list: list[dict],
    counters: dict,
    is_dry_run: bool = False,
) -> None:
    chunk_id = chunk.get("id")
    if not chunk_id:
        print("[WARNING] Chunk without id, skipped", file=sys.stderr)
        counters["skipped_no_id_count"] += 1
        return

    if chunk_id in mapping:
        print(f"[WARNING] Duplicate chunk_id '{chunk_id}': already mapped, skipping", file=sys.stderr)
        counters["duplicate_count"] += 1
        return

    counters["total_chunks"] += 1

    # PDF не прочитан - все чанки unmapped без поиска
    if pages_norm is None:
        mapping[chunk_id] = None
        counters["unmapped_count"] += 1
        if is_dry_run:
            print(f"  DRY: {chunk_id} | [PDF ERROR] page=null")
        return

    # Счётчик short_probe: проверяем ДО вызова find_page_for_chunk
    text = chunk.get("text", "")
    probe_raw = text[:200]
    probe_stripped = _strip_chunk_prefix(probe_raw)
    if len(probe_stripped) < PROBE_MIN_LENGTH:
        mapping[chunk_id] = None
        counters["short_probe_count"] += 1
        counters["unmapped_count"] += 1
        if is_dry_run:
            print(f"  DRY: {chunk_id} | SKIP (short probe, len={len(probe_stripped)})")
        return

    page, ambiguous_pages = find_page_for_chunk(chunk, pages_norm, pages_norm_search, concat_text, cum_lens)

    if page is None and not ambiguous_pages:
        mapping[chunk_id] = None
        counters["unmapped_count"] += 1
    elif page is not None and len(ambiguous_pages) == 1:
        # Однозначное совпадение
        mapping[chunk_id] = page
        counters["mapped_count"] += 1
    else:
        # Неоднозначное: найдено несколько страниц, ни одна не выбрана
        mapping[chunk_id] = None
        counters["unmapped_count"] += 1

    if len(ambiguous_pages) > 1:
        counters["ambiguous_count"] += 1
        ambiguous_list.append({
            "chunk_id": chunk_id,
            "probe": text[:200],
            "pages": sorted(ambiguous_pages),
        })

    if is_dry_run:
        amb = f"ambiguous={ambiguous_pages}" if len(ambiguous_pages) > 1 else ""
        print(f"  DRY: {chunk_id} | page={page} {amb}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map chunks to PDF page numbers")
    parser.add_argument("--dry-run", action="store_true",
        help=f"Process only first {DRY_RUN_LIMIT} chunks, no file writes")
    args = parser.parse_args()
    dry_run = args.dry_run

    if dry_run:
        print("=" * 60)
        print(f"DRY-RUN MODE: only {DRY_RUN_LIMIT} chunks, no writes")
        print("=" * 60)

    doc_map = build_doc_map()
    if not doc_map:
        print("No documents to process.")
        return

    total_nonempty_lines = 0
    if not dry_run:
        print("Pre-counting non-empty JSONL lines...")
        total_nonempty_lines = count_nonempty_lines(doc_map)
        print(f"Total non-empty JSONL lines: {total_nonempty_lines}")

    counters = {
        "total_chunks": 0,
        "mapped_count": 0,
        "unmapped_count": 0,
        "short_probe_count": 0,
        "ambiguous_count": 0,
        "pdf_error_count": 0,
        "jsonl_error_count": 0,
        "json_parse_errors": 0,
        "skipped_no_id_count": 0,
        "duplicate_count": 0,
    }

    mapping: dict[str, int | None] = {}
    ambiguous_list: list[dict] = []
    next_progress = 100

    for stem, (pdf_path, jsonl_path) in sorted(doc_map.items()):
        pages_norm: list[str] | None = None
        concat_text: str | None = None
        cum_lens: list[int] | None = None
        try:
            pages_raw = extract_pages(pdf_path)
            pages_norm = [_normalize(pt) for pt in pages_raw]
            pages_norm_search = [_normalize_search(pt) for pt in pages_raw]
            concat_text, cum_lens = _build_concat_index(pages_norm_search)
            print(f"{pdf_path.name}: {len(pages_norm)} pages")
        except Exception as exc:
            print(f"[ERROR] Cannot read PDF {pdf_path.name}: {exc}", file=sys.stderr)
            counters["pdf_error_count"] += 1

        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"[WARNING] Bad JSON line in {jsonl_path.name}: {exc}", file=sys.stderr)
                        counters["json_parse_errors"] += 1
                        continue

                    _process_chunk(
                        chunk, pages_norm, pages_norm_search,
                        concat_text, cum_lens,
                        mapping, ambiguous_list,
                        counters, is_dry_run=dry_run,
                    )

                    if not dry_run:
                        n = counters["total_chunks"]
                        if n >= next_progress:
                            print(f"Processed {n}/{total_nonempty_lines}")
                            next_progress += 100

                    if dry_run and counters["total_chunks"] >= DRY_RUN_LIMIT:
                        break

        except Exception as exc:
            print(f"[ERROR] Cannot read JSONL {jsonl_path.name}: {exc}", file=sys.stderr)
            counters["jsonl_error_count"] += 1
            continue

        if dry_run and counters["total_chunks"] >= DRY_RUN_LIMIT:
            break

    if not dry_run and counters["total_chunks"] > 0:
        print(f"Processed {counters["total_chunks"]}/{total_nonempty_lines}")

    if dry_run:
        print("")
        print("=" * 60)
        print("DRY-RUN COMPLETE. No files or directories created.")
    else:
        MAPPING_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as fp:
            json.dump(mapping, fp, ensure_ascii=False, indent=2)
        print(f"")
        print(f"Written {OUTPUT_FILE} ({len(mapping)} entries)")
        with open(AMBIGUOUS_FILE, "w", encoding="utf-8") as fp:
            json.dump(ambiguous_list, fp, ensure_ascii=False, indent=2)
        print(f"Written {AMBIGUOUS_FILE} ({len(ambiguous_list)} entries)")

    tc = counters["total_chunks"]
    pct = (counters["mapped_count"] / tc * 100) if tc else 0.0
    print("")
    print("=" * 60)
    print("STATISTICS:")
    print(f"  Total chunks processed:            {tc}")
    print(f"  Mapped (page found):               {counters["mapped_count"]}")
    print(f"  Mapped percent:                    {pct:.1f}%")
    print(f"  Unmapped (page = null):            {counters["unmapped_count"]}")
    print(f"  Short probes (< {PROBE_MIN_LENGTH}):            {counters["short_probe_count"]}")
    print(f"  Ambiguous (multiple pages):        {counters["ambiguous_count"]}")
    print(f"  PDF read errors:                   {counters["pdf_error_count"]}")
    print(f"  JSONL read errors:                 {counters["jsonl_error_count"]}")
    print(f"  JSON parse errors:                 {counters["json_parse_errors"]}")
    print(f"  Skipped (no id):                   {counters["skipped_no_id_count"]}")
    print(f"  Duplicate chunk_ids:               {counters["duplicate_count"]}")

    mapped_ex = [(k, v) for k, v in mapping.items() if v is not None][:5]
    unmapped_ex = [(k, v) for k, v in mapping.items() if v is None][:5]

    if tc:
        print(f"")
        print(f"  5 MAPPED examples:")
        for cid, pg in mapped_ex:
            print(f"    {cid} -> page {pg}")
        print(f"")
        print(f"  5 UNMAPPED examples:")
        for cid, pg in unmapped_ex:
            print(f"    {cid} -> {pg}")

if __name__ == "__main__":
    main()

