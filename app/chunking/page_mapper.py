"""page_mapper.py - map chunks to PDF page numbers.

Core algorithm from scripts/build_page_mapping.py plus incremental update.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


PROBE_MIN_LENGTH = 50
SLIDING_OFFSETS = [0, 50, 100, 150, 200, 250, 300]
PAGE_DELIM = chr(0)


def _normalize(text: str) -> str:
    text = re.sub(r"Стр\.\s*\d+\s*из\s*\d+", "", text)
    text = re.sub(r"[ ​ ⁠　]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_search(text: str) -> str:
    text = _normalize(text)
    text = re.sub(r"\\[|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_chunk_prefix(text: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*\[[^\]]+\]\s*\n?", "", text, count=1)


def extract_pages(pdf_path: Path) -> list[str]:
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = [doc.load_page(i).get_text() for i in range(len(doc))]
    doc.close()
    return pages


def _build_concat_index(pages: list[str]) -> tuple[str, list[int]]:
    cum_lens = [0]
    for pt in pages:
        cum_lens.append(cum_lens[-1] + len(pt) + 1)
    concat_text = PAGE_DELIM.join(pages)
    return concat_text, cum_lens


def _find_pages_in_concat(probe_norm: str, concat_text: str, cum_lens: list[int]) -> list[int]:
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


def _pages_for_offset(text: str, offset: int, concat_text: str, cum_lens: list[int]) -> list[int]:
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


def find_page_for_chunk(chunk: dict, concat_text: str, cum_lens: list[int]) -> tuple[Optional[int], list[int]]:
    text = chunk.get("text", "")
    if not text or not concat_text or not cum_lens:
        return None, []
    offset0_pages = _pages_for_offset(text, 0, concat_text, cum_lens)
    if len(offset0_pages) == 1:
        return offset0_pages[0], offset0_pages
    if len(offset0_pages) > 1:
        return None, offset0_pages
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


def _find_page_for_chunk_text(chunk: dict, pages_norm_search, concat_text, cum_lens):
    if pages_norm_search is None or concat_text is None or cum_lens is None:
        return None, []
    text = chunk.get("text", "")
    probe_raw = text[:200]
    probe_stripped = _strip_chunk_prefix(probe_raw)
    if len(probe_stripped) < PROBE_MIN_LENGTH:
        return None, []
    return find_page_for_chunk(chunk, concat_text, cum_lens)


def build_mapping_for_jsonl(jsonl_path: Path, pdf_path: Path) -> dict[str, Optional[int]]:
    if not pdf_path.exists():
        logger.warning("PDF not found: %s", pdf_path)
        return {}
    try:
        pages_raw = extract_pages(pdf_path)
    except Exception as exc:
        logger.error("Cannot read PDF %s: %s", pdf_path.name, exc)
        return {}
    if not pages_raw:
        logger.warning("PDF %s: no text extracted", pdf_path.name)
        return {}
    pages_norm_search = [_normalize_search(pt) for pt in pages_raw]
    concat_text, cum_lens = _build_concat_index(pages_norm_search)
    result: dict[str, Optional[int]] = {}
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Bad JSON in %s: %s", jsonl_path.name, exc)
                    continue
                chunk_id = chunk.get("id")
                if not chunk_id:
                    continue
                page, ambiguous_pages = _find_page_for_chunk_text(chunk, pages_norm_search, concat_text, cum_lens)
                result[chunk_id] = page
                if len(ambiguous_pages) > 1:
                    logger.info("Ambiguous: %s pages=%s", chunk_id, ambiguous_pages)
    except Exception as exc:
        logger.error("Cannot read JSONL %s: %s", jsonl_path.name, exc)
        return {}
    return result


def load_mapping(mapping_path: Path) -> dict[str, Optional[int]]:
    if not mapping_path.exists():
        return {}
    try:
        with open(mapping_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Cannot load %s: %s", mapping_path, exc)
        return {}


def save_mapping(mapping: dict[str, Optional[int]], mapping_path: Path) -> None:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = mapping_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, mapping_path)
    except Exception as exc:
        logger.error("Failed to save %s: %s", mapping_path, exc)
        if tmp_path.exists():
            try: tmp_path.unlink()
            except OSError: pass
        raise


def get_chunk_ids_from_jsonl(jsonl_path: Path) -> set[str]:
    ids: set[str] = set()
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                cid = chunk.get("id")
                if cid:
                    ids.add(cid)
    except Exception as exc:
        logger.error("Cannot read IDs from %s: %s", jsonl_path.name, exc)
    return ids


def update_mapping_for_document(existing, jsonl_path, new_entries):
    if not new_entries:
        logger.warning("No new entries for %s, mapping preserved", jsonl_path.name)
        return None
    doc_stem = jsonl_path.stem.lower()
    prefix = f"{doc_stem}_"
    new_ids = set(new_entries.keys())
    old_ids = {k for k in existing if k.startswith(prefix)}
    ids_to_remove = old_ids | new_ids
    result = {k: v for k, v in existing.items() if k not in ids_to_remove}
    result.update(new_entries)
    removed_count = len(old_ids)
    logger.info("Updated %s: removed %d old, added %d new", doc_stem, removed_count, len(new_entries))
    return result


def find_pdf_for_jsonl(jsonl_path: Path, raw_dir: Path) -> Path | None:
    stem = jsonl_path.stem
    candidate = raw_dir / f"{stem}.pdf"
    if candidate.exists():
        return candidate
    stem_lower = stem.lower()
    # Case-insensitive поиск: проверяем все расширения .pdf/.PDF/.Pdf и т.д.
    for f in sorted(raw_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() != '.pdf':
            continue
        if f.stem.lower() == stem_lower:
            return f
    return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("page_mapper.py - module, not a script")
    print("Use: from app.chunking.page_mapper import ...")