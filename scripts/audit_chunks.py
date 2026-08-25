#!/usr/bin/env python3
"""Аудит качества chunking: проверка chunks/*.jsonl, structure/*.json, markdown/*.md.

Потоковый аудит без загрузки всех данных в память.
Консоль — краткий summary. Подробности — в audit_reports/audit_chunks_report.json.
Не изменяет production-код и данные.
"""
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import ijson

# ─── Константы ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
STRUCTURE_DIR = PROJECT_DIR / "structure"
CHUNKS_DIR = PROJECT_DIR / "chunks"
MARKDOWN_DIR = PROJECT_DIR / "markdown"
AUDIT_REPORTS_DIR = PROJECT_DIR / "audit_reports"

MAX_TOKENS = 400
SAMPLE_SEED = 42
MAX_EXAMPLES = 20
REQUIRED_FIELDS = {"id", "title", "text", "local_img", "url"}

# ─── Tokenizer (копия из legal_chunker) ──────────────────────────────────────
_TOKENIZER = None


def _load_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER or None
    model_dir = PROJECT_DIR / "hf_cache" / "FRIDA"
    if not model_dir.exists():
        _TOKENIZER = False
        return None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        _TOKENIZER = tok
        return tok
    except Exception:
        _TOKENIZER = False
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tok = _load_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, (len(text) + 3) // 4)


# ─── Утилиты ─────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    """Ленивый генератор строк JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ─── Анализ ID ───────────────────────────────────────────────────────────────

ID_PATTERN = re.compile(
    r"^([a-z0-9-]+)_(pre|st(\d+(?:_\d+)*(?:-\d+)*)|sec(\d+(?:_\d+)*(?:-\d+)*)|secs(\d+(?:_\d+)*(?:-\d+)*)|app(\d+(?:-\d+)*))_p(\d+)$"
)


def parse_chunk_id(chunk_id: str) -> dict | None:
    m = ID_PATTERN.match(chunk_id)
    if not m:
        return None
    section_type_raw = m.group(2)
    # Определяем чистый тип секции без номера
    if section_type_raw.startswith("pre"):
        t = "pre"
    elif section_type_raw.startswith("st"):
        t = "st"
    elif section_type_raw.startswith("secs"):
        t = "secs"
    elif section_type_raw.startswith("sec"):
        t = "sec"
    elif section_type_raw.startswith("app"):
        t = "app"
    else:
        t = section_type_raw
    if m.group(3) is not None:
        section_num = m.group(3).replace("_", ".")
    elif m.group(4) is not None:
        section_num = m.group(4).replace("_", ".")
    elif m.group(5) is not None:
        section_num = m.group(5).replace("_", ".")
    elif m.group(6) is not None:
        section_num = m.group(6)
    else:
        section_num = ""
    return {"doc": m.group(1), "section_type": t,
            "section_raw": section_type_raw, "section_num": section_num,
            "chunk_num": int(m.group(7))}


def extract_article_number(title: str) -> str | None:
    """Извлечь номер статьи из title, нормализуя пробелы в точки.

    'Статья 246 1. Освобождение'    -> '246.1'
    'Статья 246 1-1. Освобождение'  -> '246.1-1'
    'Статья 333 34-1. Особенности'  -> '333.34-1'
    'Статья 1. Основные термины'    -> '1'
    'Статья 12. Учёт'               -> '12'
    """
    m = re.search(r"Статья\s+(.+?)\s*\.\s|Статья\s+(.+)$", title)
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    parts = re.split(r"\s+", raw.strip())
    cleaned = [p.rstrip(".") for p in parts]
    return ".".join(cleaned)


# ─── Префикс чанка ───────────────────────────────────────────────────────────

_CHUNK_PREFIX_PATTERN = re.compile(r"^\[[^\]]+\]\s*\[[^\]]+\]\s*\n?")


def strip_chunk_prefix(text: str) -> str:
    """Удалить документный префикс [DOC] [TITLE] из текста чанка."""
    return _CHUNK_PREFIX_PATTERN.sub("", text, count=1)


# ─── Проверки ────────────────────────────────────────────────────────────────

def check_file_correspondence() -> dict:
    md_stems = {f.stem for f in MARKDOWN_DIR.glob("*.md")}
    struct_stems = {f.stem for f in STRUCTURE_DIR.glob("*.json")}
    chunk_stems = {f.stem for f in CHUNKS_DIR.glob("*.jsonl")}
    all_ok = (md_stems == struct_stems == chunk_stems)
    return {"markdown": len(md_stems), "structure": len(struct_stems),
            "chunks": len(chunk_stems),
            "missing_structure": sorted(md_stems - struct_stems),
            "missing_chunks": sorted(md_stems - chunk_stems),
            "extra_structure": sorted(struct_stems - md_stems),
            "extra_chunks": sorted(chunk_stems - md_stems),
            "status": "PASS" if all_ok else "FAIL"}
def check_jsonl_schema_and_tokens() -> dict:
    total_chunks = 0
    schema_errors = []
    empty_id_examples = []
    empty_title_examples = []
    empty_text_examples = []
    seen_ids = set()
    duplicate_id_examples = []
    token_counts = []
    min_tok = None
    max_tok = None
    sum_tok = 0.0
    buckets = {"0-39": 0, "40-99": 0, "100-199": 0, "200-299": 0,
            "300-349": 0, "350-379": 0, "380-399": 0, "=400": 0, ">400": 0}
    doc_token_agg = {}

    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        doc_name = chunk_path.stem
        dag = doc_token_agg.setdefault(doc_name, {"count": 0, "sum": 0, "min": None, "max": None})
        for i, obj in enumerate(iter_jsonl(chunk_path), 1):
            total_chunks += 1
            obj_keys = set(obj.keys())
            if obj_keys != REQUIRED_FIELDS and len(schema_errors) < MAX_EXAMPLES:
                schema_errors.append({"file": chunk_path.name, "line": i,
                    "id": obj.get("id", ""),
                    "extra_fields": sorted(obj_keys - REQUIRED_FIELDS),
                    "missing_fields": sorted(REQUIRED_FIELDS - obj_keys)})
            cid = str(obj.get("id", ""))
            title = str(obj.get("title", ""))
            text = str(obj.get("text", ""))
            if not cid.strip() and len(empty_id_examples) < MAX_EXAMPLES:
                empty_id_examples.append({"file": chunk_path.name, "line": i})
            if not title.strip() and len(empty_title_examples) < MAX_EXAMPLES:
                empty_title_examples.append({"file": chunk_path.name, "line": i, "id": cid})
            if not text.strip() and len(empty_text_examples) < MAX_EXAMPLES:
                empty_text_examples.append({"file": chunk_path.name, "line": i, "id": cid})
            if cid and cid in seen_ids and len(duplicate_id_examples) < MAX_EXAMPLES:
                duplicate_id_examples.append({"file": chunk_path.name, "line": i, "id": cid})
            if cid:
                seen_ids.add(cid)
            tok_count = count_tokens(text)
            token_counts.append(tok_count)
            sum_tok += tok_count
            dag["count"] += 1
            dag["sum"] += tok_count
            if dag["min"] is None or tok_count < dag["min"]:
                dag["min"] = tok_count
            if dag["max"] is None or tok_count > dag["max"]:
                dag["max"] = tok_count
            if min_tok is None or tok_count < min_tok:
                min_tok = tok_count
            if max_tok is None or tok_count > max_tok:
                max_tok = tok_count
            # Buckets — отдельно "=400" и ">400"
            if tok_count > MAX_TOKENS:
                buckets[">400"] += 1
            elif tok_count == MAX_TOKENS:
                buckets["=400"] += 1
            elif tok_count >= 380:
                buckets["380-399"] += 1
            elif tok_count >= 350:
                buckets["350-379"] += 1
            elif tok_count >= 300:
                buckets["300-349"] += 1
            elif tok_count >= 200:
                buckets["200-299"] += 1
            elif tok_count >= 100:
                buckets["100-199"] += 1
            elif tok_count >= 40:
                buckets["40-99"] += 1
            else:
                buckets["0-39"] += 1

    avg_tok = round(sum_tok / total_chunks, 2) if total_chunks else 0
    sorted_tokens = sorted(token_counts)
    median_tok = statistics.median(sorted_tokens) if sorted_tokens else 0

    def percentile(data, p):
        if not data:
            return 0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = k - f
        return round(data[f] + (c * (data[f+1] - data[f])) if f+1 < len(data) else data[f], 1)

    p10 = percentile(sorted_tokens, 10)
    p25 = percentile(sorted_tokens, 25)
    p50 = percentile(sorted_tokens, 50)
    p75 = percentile(sorted_tokens, 75)
    p90 = percentile(sorted_tokens, 90)
    p95 = percentile(sorted_tokens, 95)
    p99 = percentile(sorted_tokens, 99)
    token_dist = {"min": min_tok, "max": max_tok, "avg": avg_tok,
        "median": median_tok, "p10": p10, "p25": p25, "p50": p50,
        "p75": p75, "p90": p90, "p95": p95, "p99": p99,
        "buckets": buckets, "over_400": buckets.get(">400", 0), "exact_400": buckets.get("=400", 0)}
    token_status = "FAIL" if buckets.get(">400", 0) > 0 else "PASS"
    schema_has_fail = bool(schema_errors or empty_id_examples or empty_text_examples or duplicate_id_examples)
    return {"total_chunks": total_chunks,
        "schema_errors": {"count": len(schema_errors), "examples": schema_errors},
        "empty_id": {"count": len(empty_id_examples), "examples": empty_id_examples},
        "empty_title": {"count": len(empty_title_examples), "examples": empty_title_examples},
        "empty_text": {"count": len(empty_text_examples), "examples": empty_text_examples},
        "duplicate_ids": {"count": len(duplicate_id_examples), "examples": duplicate_id_examples},
        "token_statistics": token_dist, "doc_token_agg": doc_token_agg,
        "schema_status": "FAIL" if schema_has_fail else "PASS", "token_status": token_status,
        "status": "FAIL" if (schema_has_fail or token_status == "FAIL") else "PASS"}
def check_duplicates_sha256(schema_data: dict) -> dict:
    text_hash_map = {}
    duplicate_texts = []
    cross_doc_dups = set()
    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        doc_name = chunk_path.stem
        for obj in iter_jsonl(chunk_path):
            cid = obj.get("id", "")
            text = obj.get("text", "")
            if not text:
                continue
            h = sha256_hex(text)
            if h in text_hash_map:
                prev = text_hash_map[h]
                is_cross = prev["doc"] != doc_name
                if len(duplicate_texts) < MAX_EXAMPLES:
                    duplicate_texts.append({"hash": h, "first_id": prev["id"],
                        "first_doc": prev["doc"], "second_id": cid,
                        "second_doc": doc_name, "cross_doc": is_cross})
                if is_cross:
                    cross_doc_dups.add((prev["doc"], doc_name))
            else:
                text_hash_map[h] = {"doc": doc_name, "id": cid}
    return {"duplicate_texts": {"count": len(duplicate_texts),
        "examples": duplicate_texts, "cross_doc_pairs": len(cross_doc_dups)},
        "status": "WARN" if duplicate_texts else "PASS"}


def check_document_boundaries() -> dict:
    errors = []
    warnings = []
    status = "PASS"
    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        doc_name = chunk_path.stem
        expected_prefix = doc_name.lower()
        urls_in_doc = set()
        for obj in iter_jsonl(chunk_path):
            cid = str(obj.get("id", ""))
            url = str(obj.get("url", ""))
            if cid and not cid.lower().startswith(expected_prefix.lower() + "_"):
                if len(errors) < MAX_EXAMPLES:
                    errors.append({"file": chunk_path.name, "id": cid,
                        "expected_prefix": f"{expected_prefix}_", "issue": "id_prefix_mismatch"})
                status = "FAIL"
            urls_in_doc.add(url)
        if len(urls_in_doc) > 1:
            if len(warnings) < MAX_EXAMPLES:
                warnings.append({"file": chunk_path.name, "urls": sorted(urls_in_doc),
                    "issue": "multiple_urls_in_doc"})
            status = "WARN" if status == "PASS" else status
    return {"errors": errors, "warnings": warnings, "status": status}


def check_article_boundaries() -> dict:
    violations = []
    warnings_list = []
    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for obj in iter_jsonl(chunk_path):
            cid = str(obj.get("id", ""))
            title = str(obj.get("title", ""))
            text = str(obj.get("text", ""))
            parsed = parse_chunk_id(cid)
            if not parsed or parsed["section_type"] != "st":
                continue
            expected_article_num = parsed["section_num"]
            title_article = extract_article_number(title)
            if title_article and title_article != expected_article_num:
                if len(violations) < MAX_EXAMPLES:
                    violations.append({"file": chunk_path.name, "id": cid,
                        "expected_article": expected_article_num,
                        "title_article": title_article, "issue": "title_mismatch"})
                continue
            if title and "Статья" not in title and "статья" not in title.lower():
                if len(warnings_list) < MAX_EXAMPLES:
                    warnings_list.append({"file": chunk_path.name, "id": cid,
                        "title": title[:80], "issue": "no_article_in_title"})
    status = "FAIL" if violations else ("WARN" if warnings_list else "PASS")
    return {"violations": {"count": len(violations), "examples": violations},
        "warnings": {"count": len(warnings_list), "examples": warnings_list},
        "status": status}
def check_id_sequence() -> dict:
    gaps = []
    repeats = []
    status = "PASS"
    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        section_chunks = {}
        seen_ids = {}
        for obj in iter_jsonl(chunk_path):
            cid = str(obj.get("id", ""))
            # Full-ID duplicates
            if cid in seen_ids and len(repeats) < MAX_EXAMPLES:
                repeats.append({"file": chunk_path.name, "id": cid,
                    "count": seen_ids[cid] + 1, "type": "duplicate_id"})
                status = "WARN"
                seen_ids[cid] += 1
            else:
                seen_ids[cid] = 1
            parsed = parse_chunk_id(cid)
            if not parsed:
                continue
            key = parsed["section_raw"]
            section_chunks.setdefault(key, []).append(parsed["chunk_num"])
        for section, nums in sorted(section_chunks.items()):
            nums = sorted(set(nums))
            if len(nums) <= 1:
                continue
            expected = list(range(nums[0], nums[-1] + 1))
            missing = sorted(set(expected) - set(nums))
            if missing and len(gaps) < MAX_EXAMPLES:
                gaps.append({"file": chunk_path.name, "section": section,
                    "expected_range": f"{nums[0]}-{nums[-1]}",
                    "missing_numbers": missing[:10]})
                status = "WARN"
            num_counts = Counter(nums)
            dup_nums = {n: c for n, c in num_counts.items() if c > 1}
            if dup_nums and len(repeats) < MAX_EXAMPLES:
                repeats.append({"file": chunk_path.name, "section": section,
                    "duplicates": {str(k): v for k, v in sorted(dup_nums.items())}})
                status = "WARN"
    return {"gaps": {"count": len(gaps), "examples": gaps},
        "repeats": {"count": len(repeats), "examples": repeats}, "status": status}


def check_url() -> dict:
    empty_urls = []
    multi_urls = []
    status = "PASS"
    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        urls = set()
        for obj in iter_jsonl(chunk_path):
            url = str(obj.get("url", ""))
            if not url.strip():
                if len(empty_urls) < MAX_EXAMPLES:
                    empty_urls.append({"file": chunk_path.name, "id": obj.get("id", "")})
                status = "WARN"
            urls.add(url)
        if len(urls) > 1:
            if len(multi_urls) < MAX_EXAMPLES:
                multi_urls.append({"file": chunk_path.name, "urls": sorted(urls)})
            status = "WARN" if status != "FAIL" else "FAIL"
    return {"empty_urls": {"count": len(empty_urls), "examples": empty_urls},
        "multi_urls": {"count": len(multi_urls), "examples": multi_urls}, "status": status}


def check_local_img() -> dict:
    total = 0
    with_img = 0
    empty_img = 0
    no_field = 0
    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for obj in iter_jsonl(chunk_path):
            total += 1
            if "local_img" not in obj:
                no_field += 1
            elif obj.get("local_img") in (None, "", []):
                empty_img += 1
            else:
                with_img += 1
    status = "INFO" if empty_img == total else "WARN"
    return {"total": total, "with_local_img": with_img,
        "empty_local_img": empty_img, "no_field": no_field, "status": status}


def check_structure_chunk_correlation(schema_data: dict) -> dict:
    table = []
    doc_token_agg = schema_data.get("doc_token_agg", {})
    for struct_path in sorted(STRUCTURE_DIR.glob("*.json")):
        doc_name = struct_path.stem
        records_count = 0
        with open(struct_path, "rb") as f:
            for _ in ijson.items(f, "records.item"):
                records_count += 1
        chunks_path = CHUNKS_DIR / f"{doc_name}.jsonl"
        chunks_count = count_jsonl_lines(chunks_path) if chunks_path.exists() else 0
        agg = doc_token_agg.get(doc_name, {})
        table.append({"document": doc_name, "records": records_count,
            "chunks": chunks_count, "min_tokens": agg.get("min"),
            "avg_tokens": round(agg["sum"] / agg["count"], 2) if agg.get("count") else None,
            "max_tokens": agg.get("max")})
    table.sort(key=lambda r: r["chunks"], reverse=True)
    return {"table": table, "status": "PASS"}
def check_text_integrity_sample(schema_data: dict) -> dict:
    import random
    rng = random.Random(SAMPLE_SEED)
    sample_results = []
    not_found = []
    suspicious = []
    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        doc_name = chunk_path.stem
        struct_path = STRUCTURE_DIR / f"{doc_name}.json"
        if not struct_path.exists():
            continue
        all_chunks = []
        for obj in iter_jsonl(chunk_path):
            tok = count_tokens(obj.get("text", ""))
            all_chunks.append({"obj": obj, "tokens": tok})
        if not all_chunks:
            continue
        sorted_by_tok = sorted(all_chunks, key=lambda x: x["tokens"])
        sample_indices = {0, len(all_chunks) - 1}
        smallest_idx = next(i for i, c in enumerate(all_chunks) if c["tokens"] == sorted_by_tok[0]["tokens"])
        sample_indices.add(smallest_idx)
        largest_idx = next(i for i, c in enumerate(all_chunks) if c["tokens"] == sorted_by_tok[-1]["tokens"])
        sample_indices.add(largest_idx)
        available = [i for i in range(len(all_chunks)) if i not in sample_indices]
        if available:
            random_choices = rng.sample(available, min(2, len(available)))
            sample_indices.update(random_choices)
        for idx in sorted(sample_indices):
            chunk = all_chunks[idx]
            cid = chunk["obj"]["id"]
            ctext = chunk["obj"].get("text", "")
            ctext_stripped = strip_chunk_prefix(ctext)
            norm_ctext = norm_ws(ctext_stripped)
            if len(norm_ctext) <= 20:
                continue
            found = False
            with open(struct_path, "rb") as f:
                for rec in ijson.items(f, "records.item"):
                    norm_rtext = norm_ws(rec.get("text", ""))
                    if norm_ctext in norm_rtext:
                        found = True
                        break
            entry = {"document": doc_name, "chunk_id": cid,
                "title": chunk["obj"].get("title", "")[:100],
                "token_count": chunk["tokens"],
                "text_preview": ctext[:300]}
            if found:
                entry["status"] = "PASS"
            else:
                entry["status"] = "WARN"
                if len(not_found) < MAX_EXAMPLES:
                    not_found.append(entry)
            sample_results.append(entry)
    status = "FAIL" if suspicious else ("WARN" if not_found else "PASS")
    return {"sampled_chunks": len(sample_results),
        "sample_results": sample_results,
        "not_found_in_records": {"count": len(not_found), "examples": not_found},
        "suspicious": {"count": len(suspicious), "examples": suspicious},
        "status": status}


def check_large_source_records() -> dict:
    max_record = None
    max_tokens_val = 0
    tok = _load_tokenizer()
    for struct_path in sorted(STRUCTURE_DIR.glob("*.json")):
        doc_name = struct_path.stem
        with open(struct_path, "rb") as f:
            for rec in ijson.items(f, "records.item"):
                text = rec.get("text", "")
                if not text:
                    continue
                tok_count = count_tokens(text)
                if tok_count > max_tokens_val:
                    max_tokens_val = tok_count
                    max_record = {"document": doc_name,
                        "node_id": rec.get("node_id", ""),
                        "record_type": str(rec.get("type", "unknown")),
                        "record_tokens": tok_count,
                        "record_chars": len(text),
                        "title": rec.get("title", "")[:100],
                        "text_preview": text[:200]}
    if max_record:
        doc = max_record["document"]
        chunks_path = CHUNKS_DIR / f"{doc}.jsonl"
        if chunks_path.exists():
            chunk_count = 0
            largest_chunk_tokens = 0
            for obj in iter_jsonl(chunks_path):
                text = obj.get("text", "")
                if not text:
                    continue
                chunk_count += 1
                tc = count_tokens(text)
                if tc > largest_chunk_tokens:
                    largest_chunk_tokens = tc
            max_record["chunks_in_doc"] = chunk_count
            max_record["largest_chunk_tokens"] = largest_chunk_tokens
            max_record["model_max_length"] = tok.model_max_length if tok else None
    return {"largest_record": max_record, "status": "INFO" if max_record else "PASS"}


# ─── Оркестрация ─────────────────────────────────────────────────────────────
def run_audit() -> dict:
    """Запустить все проверки и вернуть словарь с результатами."""
    print("═══ Аудит чанкинга ═══")
    print()

    print("[1/12] File correspondence...")
    r1 = check_file_correspondence()
    print(f"  -> {r1['status']}: {r1['markdown']} md / {r1['structure']} struct / {r1['chunks']} chunks")

    print("[2/12] Schema + tokens...")
    r2 = check_jsonl_schema_and_tokens()
    print(f"  -> schema={r2['schema_status']} tokens={r2['token_status']}: "
          f"{r2['total_chunks']} total, {r2['token_statistics']['over_400']} over {MAX_TOKENS}")

    print("[3/12] Duplicates SHA256...")
    r3 = check_duplicates_sha256(r2)
    print(f"  -> {r3['status']}: {r3['duplicate_texts']['count']} dup texts")

    print("[4/12] Document boundaries...")
    r4 = check_document_boundaries()
    print(f"  -> {r4['status']}: {len(r4['errors'])} errors, {len(r4['warnings'])} warnings")

    print("[5/12] Article boundaries...")
    r5 = check_article_boundaries()
    print(f"  -> {r5['status']}: {r5['violations']['count']} violations, {r5['warnings']['count']} warnings")

    print("[6/12] ID sequence...")
    r6 = check_id_sequence()
    print(f"  -> {r6['status']}: {r6['gaps']['count']} gaps, {r6['repeats']['count']} repeats")

    print("[7/12] URL...")
    r7 = check_url()
    print(f"  -> {r7['status']}: {r7['empty_urls']['count']} empty, {r7['multi_urls']['count']} multi")

    print("[8/12] Local img...")
    r8 = check_local_img()
    print(f"  -> {r8['status']}: {r8['with_local_img']} with, {r8['empty_local_img']} empty, {r8['no_field']} no field")

    print("[9/12] Structure-chunk correlation...")
    r9 = check_structure_chunk_correlation(r2)
    print(f"  -> {r9['status']}: {len(r9['table'])} documents in table")

    print("[10/12] Text integrity sample...")
    r10 = check_text_integrity_sample(r2)
    print(f"  -> {r10['status']}: {r10['not_found_in_records']['count']} not found")

    print("[11/12] Large source records...")
    r11 = check_large_source_records()
    if r11.get("largest_record"):
        print(f"  -> {r11['status']}: max record = {r11['largest_record']['record_tokens']} tokens")
    else:
        print(f"  -> {r11['status']}: no records found")

    print("[12/12] Done.")
    print()

    return {
        "check_file_correspondence": r1,
        "check_jsonl_schema_and_tokens": r2,
        "check_duplicates_sha256": r3,
        "check_document_boundaries": r4,
        "check_article_boundaries": r5,
        "check_id_sequence": r6,
        "check_url": r7,
        "check_local_img": r8,
        "check_structure_chunk_correlation": r9,
        "check_text_integrity_sample": r10,
        "check_large_source_records": r11,
    }


def compute_verdict(results: dict) -> str:
    """Вычислить общий вердикт: FAIL если хоть один FAIL, иначе WARN, иначе PASS."""
    for key, r in results.items():
        st = r.get("status", "PASS")
        if st == "FAIL":
            return "FAIL"
    for key, r in results.items():
        st = r.get("status", "PASS")
        if st == "WARN":
            return "WARN"
    return "PASS"


def format_summary(results: dict) -> str:
    """Форматировать краткий summary для консоли."""
    lines = []
    lines.append("═══ AUDIT SUMMARY ═══")
    r2 = results.get("check_jsonl_schema_and_tokens", {})
    lines.append(f"Total chunks  : {r2.get('total_chunks', '?')}")
    ts = r2.get("token_statistics", {})
    lines.append(f"Token range   : {ts.get('min', '?')} - {ts.get('max', '?')}")
    lines.append(f"Token avg     : {ts.get('avg', '?')}")
    lines.append(f"Token median  : {ts.get('median', '?')}")
    lines.append(f"Over {MAX_TOKENS} tokens : {ts.get('over_400', '?')}")
    lines.append(f"")
    for key, r in results.items():
        st = r.get("status", "?")
        lines.append(f"  {st:5s} | {key}")
    verdict = compute_verdict(results)
    lines.append(f"")
    lines.append(f"VERDICT: {verdict}")
    return "\n".join(lines)


def main() -> None:
    AUDIT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_audit()
    summary = format_summary(results)
    verdict = compute_verdict(results)
    print(summary)
    report = {
        "meta": {
            "project_dir": str(PROJECT_DIR),
            "audit_timestamp": __import__("datetime").datetime.now().isoformat(),
            "total_checks": len(results),
            "verdict": verdict,
        },
        "results": results,
    }
    report_path = AUDIT_REPORTS_DIR / "audit_chunks_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПодробный отчёт: {report_path}")


if __name__ == "__main__":
    main()
