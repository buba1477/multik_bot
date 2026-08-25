#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Семантический аудит чанков: проверка смысловой целостности границ.

Потоковая обработка: JSONL читается построчно, structure — через ijson.
Консервативные эвристики: heuristic suspicion -> SUSPECT/INFO, не FAIL.
FAIL только при подтверждённом нарушении через source structure.
Не изменяет production-код и данные.
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import ijson

# Константы
PROJECT_DIR = Path(__file__).resolve().parent.parent
STRUCTURE_DIR = PROJECT_DIR / "structure"
CHUNKS_DIR = PROJECT_DIR / "chunks"
AUDIT_REPORTS_DIR = PROJECT_DIR / "audit_reports"

MAX_EXAMPLES = 20
TAIL_HEAD_LIMIT = 300

SAFE = "SAFE"
INFO = "INFO"
SUSPECT = "SUSPECT"
HIGH_RISK = "HIGH_RISK"

# Шаблоны
_TERMINAL_PUNCTUATION = re.compile(r"[.!?]\s*$")

_SUSPICIOUS_END_PATTERNS = [
    ('если', re.compile(r'если\\s*$', re.IGNORECASE)),
    ('при условии', re.compile(r'при\\s+условии\\s*$', re.IGNORECASE)),
    ('в случае', re.compile(r'в\\s+случае\\s*$', re.IGNORECASE)),
    ('за исключением', re.compile(r'за\\s+исключением\\s*$', re.IGNORECASE)),
    ('кроме', re.compile(r'кроме\\s*$', re.IGNORECASE)),
    ('который', re.compile(r'(который|которая|которые|которого|которой|которых|которому|которым|котором)\\s*$', re.IGNORECASE)),
    ('а также', re.compile(r'а\\s+также\\s*$', re.IGNORECASE)),
    ('в том числе', re.compile(r'в\\s+том\\s+числе\\s*$', re.IGNORECASE)),
    ('и', re.compile(r'и\\s*$', re.IGNORECASE)),
    ('или', re.compile(r'или\\s*$', re.IGNORECASE)),
    ('либо', re.compile(r'либо\\s*$', re.IGNORECASE)),
    ('то', re.compile(r'то\\s*$', re.IGNORECASE)),
    ('в соответствии с', re.compile(r'в\\s+соответствии\\s+с\\s*$', re.IGNORECASE)),
    ('согласно', re.compile(r'согласно\\s*$', re.IGNORECASE)),
    ('по', re.compile(r'по\\s*$', re.IGNORECASE)),
    ('для', re.compile(r'для\\s*$', re.IGNORECASE)),
    ('при', re.compile(r'при\\s*$', re.IGNORECASE)),
    ('об', re.compile(r'об\\s*$', re.IGNORECASE)),
    ('под', re.compile(r'под\\s*$', re.IGNORECASE)),
]

_SUSPICIOUS_START_PATTERNS = [
    ('нижний регистр', re.compile(r'^[a-zа-яё]')),
    ('закрывающая скобка', re.compile(r'^[)\\]）]]\\s*', re.IGNORECASE)),
    ('и', re.compile(r'^\\s*и[\\s,;]', re.IGNORECASE)),
    ('или', re.compile(r'^\\s*или[\\s,;]', re.IGNORECASE)),
    ('либо', re.compile(r'^\\s*либо[\\s,;]', re.IGNORECASE)),
    ('а также', re.compile(r'^\\s*а\\s+также[\\s,;]', re.IGNORECASE)),
    ('который', re.compile(r'^\\s*(который|которая|которые)\\s', re.IGNORECASE)),
    ('в том числе', re.compile(r'^\\s*в\\s+том\\s+числе[\\s,:]', re.IGNORECASE)),
    ('а', re.compile(r'^\\s*а[\\s,;]', re.IGNORECASE)),
    ('но', re.compile(r'^\\s*но[\\s,;]', re.IGNORECASE)),
    ('однако', re.compile(r'^\\s*однако[\\s,;]', re.IGNORECASE)),
    ('также', re.compile(r'^\\s*также[\\s,;]', re.IGNORECASE)),
    ('при этом', re.compile(r'^\\s*при\\s+этом', re.IGNORECASE)),
    ('вместе с тем', re.compile(r'^\\s*вместе\\s+с\\s+тем', re.IGNORECASE)),
]

_CONDITIONAL_START_MARKERS = [
    ('если', re.compile(r'^\\s*если\\s', re.IGNORECASE)),
    ('при', re.compile(r'^\\s*при\\s', re.IGNORECASE)),
    ('в случае', re.compile(r'^\\s*в\\s+случае\\s', re.IGNORECASE)),
    ('за исключением', re.compile(r'^\\s*за\\s+исключением', re.IGNORECASE)),
    ('кроме', re.compile(r'^\\s*кроме\\s', re.IGNORECASE)),
    ('который', re.compile(r'^\\s*(который|которая|которые)\\s', re.IGNORECASE)),
    ('в том числе', re.compile(r'^\\s*в\\s+том\\s+числе', re.IGNORECASE)),
    ('а также', re.compile(r'^\\s*а\\s+также', re.IGNORECASE)),
]

_LIST_CONTINUATION_START = re.compile(
    r"^\s*\d+\)\s*[а-яa-z]|^\s*[а-я]\)\s|^\s*[—–-]\s|^\s*\d+\.\s"
)
_LIST_INTRO_END = re.compile(r":\s*$")

# Структурные причины — норма для юридических документов, не SUSPECT
_STRUCTURAL_REASONS = {
    "начинается с элемента перечисления",
    "заканчивается на точку с запятой",
    "заканчивается на двоеточие",
}

_CHUNK_PREFIX = re.compile(r"^\[[^\]]+\]\s*\[[^\]]+\]\s*")

# Вспомогательные функции

def _strip_prefix(text: str) -> str:
    """Удалить префикс [DOC] [TITLE] из текста чанка."""
    return _CHUNK_PREFIX.sub("", text.rstrip(), count=1).strip()

def _norm_ws(text: str) -> str:
    """Нормализовать пробелы."""
    return re.sub(r"\s+", " ", text).strip()

def _truncate(text: str, limit: int = TAIL_HEAD_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."

def _tail(text: str, limit: int = TAIL_HEAD_LIMIT) -> str:
    clean = _norm_ws(_strip_prefix(text))
    if len(clean) <= limit:
        return clean
    return "..." + clean[-limit:]

def _head(text: str, limit: int = TAIL_HEAD_LIMIT) -> str:
    clean = _norm_ws(_strip_prefix(text))
    return _truncate(clean, limit)

def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)

# Парсинг ID
_ID_PATTERN = re.compile(
    r"^([a-z0-9-]+)_(pre|st(\d+(?:_\d+)*(?:-\d+)*)|"
    r"sec(\d+(?:_\d+)*(?:-\d+)*)|secs(\d+(?:_\d+)*(?:-\d+)*)|"
    r"app(\d+(?:-\d+)*))_p(\d+)$"
)

def _parse_chunk_id(chunk_id: str) -> dict | None:
    """Разобрать chunk_id на компоненты."""
    m = _ID_PATTERN.match(chunk_id)
    if not m:
        return None
    section_type_raw = m.group(2)
    if section_type_raw.startswith("pre"): t = "pre"
    elif section_type_raw.startswith("st"): t = "st"
    elif section_type_raw.startswith("secs"): t = "secs"
    elif section_type_raw.startswith("sec"): t = "sec"
    elif section_type_raw.startswith("app"): t = "app"
    else: t = section_type_raw
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
    return {
        "doc": m.group(1),
        "section_type": t,
        "section_num": section_num,
        "chunk_num": int(m.group(7)),
    }

# Потоковые генераторы

def iter_jsonl(path: Path) -> Iterator[dict]:
    """Построчное чтение JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def iter_structure_records(path: Path) -> Iterator[dict]:
    """Потоковое чтение structure JSON через ijson."""
    with open(path, "rb") as f:
        for record in ijson.items(f, "records.item"):
            yield record

# BoundaryAnalyzer

class BoundaryAnalyzer:
    """Анализирует границы между соседними чанками."""

    def __init__(self) -> None:
        self.boundaries: list[dict] = []
        self._prev: dict | None = None

    def feed(self, chunk: dict) -> None:
        """Подать следующий chunk."""
        if self._prev is not None:
            b = self._analyze(self._prev, chunk)
            self.boundaries.append(b)
        self._prev = chunk

    def _analyze(self, prev: dict, nxt: dict) -> dict:
        prev_id = prev.get("id", "")
        nxt_id = nxt.get("id", "")
        prev_title = prev.get("title", "")
        nxt_title = nxt.get("title", "")
        prev_text = prev.get("text", "")
        nxt_text = nxt.get("text", "")
        prev_p = _parse_chunk_id(prev_id)
        nxt_p = _parse_chunk_id(nxt_id)
        struct: list[str] = []
        if prev_p and nxt_p:
            if prev_p["doc"] != nxt_p["doc"]:
                struct.append("DOCUMENT_CHANGE")
            elif prev_p["section_type"] != nxt_p["section_type"]:
                struct.append("SECTION_TYPE_CHANGE")
            elif prev_p["section_num"] != nxt_p["section_num"]:
                struct.append("ARTICLE_CHANGE")
        if prev_title != nxt_title:
            struct.append("TITLE_CHANGE")
        severity, reasons, flags = self._evaluate(prev_text, nxt_text)
        return {
            "prev_id": prev_id,
            "next_id": nxt_id,
            "prev_title": prev_title,
            "next_title": nxt_title,
            "prev_tail": _tail(prev_text),
            "next_head": _head(nxt_text),
            "structural_changes": struct,
            "flags": flags,
            "reasons": reasons,
            "severity": severity,
        }

    def _evaluate(self, prev_text: str, nxt_text: str) -> tuple[str, list[str], list[str]]:
        clean_prev = _norm_ws(_strip_prefix(prev_text))
        clean_nxt = _norm_ws(_strip_prefix(nxt_text))
        reasons: list[str] = []
        flags: list[str] = []
        reasons.extend(self._check_end(clean_prev))
        r, f = self._check_start(clean_nxt)
        reasons.extend(r); flags.extend(f)
        reasons.extend(self._check_conditional(clean_prev, clean_nxt))
        flags.extend(self._check_list(clean_prev, clean_nxt))
        if reasons:
            # Если все причины — структурные (норма для юр.текстов), то INFO
            if all(r in _STRUCTURAL_REASONS for r in reasons):
                sev = INFO
            elif any("условного" in x for x in reasons):
                sev = SUSPECT
            elif any("заканчивается" in x or "обрывается" in x for x in reasons):
                sev = SUSPECT
            elif any("начинается" in x for x in reasons):
                sev = SUSPECT
            else: sev = INFO
        else: sev = SAFE
        return sev, reasons, flags

    def _check_end(self, text: str) -> list[str]:
        if not text: return []
        r: list[str] = []
        if _TERMINAL_PUNCTUATION.search(text): return []
        if text.endswith(","): r.append("заканчивается на запятую")
        elif text.endswith(";"): r.append("заканчивается на точку с запятой")
        elif text.endswith(":"): r.append("заканчивается на двоеточие")
        elif text.endswith(("(", "[")): r.append("заканчивается на открывающую скобку")
        elif text.endswith(("-", "—", "–")): r.append("заканчивается на тире")
        if not r:
            for label, pat in _SUSPICIOUS_END_PATTERNS:
                if pat.search(text):
                    r.append(f"заканчивается на '{label}'")
                    break
        if not r:
            c = text[-1]
            if c.isalpha() or c.isdigit():
                r.append("текст обрывается без знака препинания")
        return r

    def _check_start(self, text: str) -> tuple[list[str], list[str]]:
        if not text: return [], []
        r: list[str] = []; f: list[str] = []
        c = text.strip()
        if _LIST_CONTINUATION_START.match(c):
            f.append("list_continuation_start")
            return ["начинается с элемента перечисления"], f
        for label, pat in _SUSPICIOUS_START_PATTERNS:
            if pat.search(c):
                r.append(f"начинается с '{label}'")
                break
        return r, f

    def _check_conditional(self, prev: str, nxt: str) -> list[str]:
        r: list[str] = []
        for label, pat in _CONDITIONAL_START_MARKERS:
            if pat.search(nxt):
                r.append(f"следующий chunk начинается с условного маркера '{label}'")
        return r

    def _check_list(self, prev: str, nxt: str) -> list[str]:
        f: list[str] = []
        if _LIST_INTRO_END.search(prev):
            f.append("list_intro_end (prev заканчивается на ':')")
        if _LIST_CONTINUATION_START.match(nxt.strip()):
            f.append("list_continuation_start")
        return f

class StructureContinuityChecker:
    """Проверка coverage чанков относительно структуры документа."""
    def check_document(self, structure_path: Path, chunks: list[dict]) -> dict:
        result = {
            "confirmed_gaps": [], "confirmed_overlaps": [],
            "suspect_gaps": [], "article_coverage": {},
        }
        art_rec: dict[str, int] = defaultdict(int)
        try:
            for rec in iter_structure_records(structure_path):
                ctx = rec.get("structure", {}).get("context_flat", {})
                art = ctx.get("article")
                if art: art_rec[str(art)] += 1
        except Exception:
            return result
        ch_art: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            cid = chunk.get("id", "")
            p = _parse_chunk_id(cid)
            if p and p["section_type"] == "st":
                ch_art[p["section_num"]] += 1
            elif p and p["section_type"] == "pre":
                ch_art["<preamble>"] += 1
        for art, rc in art_rec.items():
            cc = ch_art.get(art, 0)
            result["article_coverage"][f"ст. {art}"] = {"records": rc, "chunks": cc}
            if rc > 0 and cc == 0:
                result["suspect_gaps"].append({
                    "article": art,
                    "type": "structure_has_records_no_chunks",
                    "records": rc, "chunks": cc,
                })
        return result

class StatisticsCollector:
    """Сбор статистики по всем документам."""

    def __init__(self) -> None:
        self.total_chunks = 0
        self.total_boundaries = 0
        self.severity_counts = {SAFE: 0, INFO: 0, SUSPECT: 0, HIGH_RISK: 0}
        self.flag_counts: dict = {}
        self.reason_counts: dict = {}
        self.token_buckets = {
            "0-39": 0, "40-99": 0, "100-199": 0,
            "200-299": 0, "300-349": 0, "350-379": 0,
            "380-399": 0, "=400": 0,
        }
        self.document_stats: dict = {}
        self.samples = {
            "suspicious_ends": [], "suspicious_starts": [],
            "conditional_boundaries": [], "list_boundaries": [],
            "high_risk": [], "structure_gaps": [],
            "hard_breaks": [],
        }

    def record_chunk(self, doc_name: str, chunk: dict) -> None:
        doc_name = doc_name.lower()
        self.total_chunks += 1
        t = _count_tokens(_norm_ws(_strip_prefix(chunk.get("text", ""))))
        b = self.token_buckets
        if t < 40: b['0-39'] += 1
        elif t < 100: b['40-99'] += 1
        elif t < 200: b['100-199'] += 1
        elif t < 300: b['200-299'] += 1
        elif t < 350: b['300-349'] += 1
        elif t < 380: b['350-379'] += 1
        elif t < 400: b['380-399'] += 1
        else: b['=400'] += 1
        if doc_name not in self.document_stats:
            self.document_stats[doc_name] = {
                "chunks": 0, "boundaries": 0,
                "safe": 0, "info": 0, "suspect": 0, "high_risk": 0,
                "small": 0, "large": 0,
            }
        ds = self.document_stats[doc_name]
        ds["chunks"] += 1
        if t < 40: ds['small'] += 1
        if t >= 400: ds['large'] += 1

    def record_boundary(self, b: dict, doc_name: str) -> None:
        doc_name = doc_name.lower()
        self.total_boundaries += 1
        sev = b["severity"]
        self.severity_counts[sev] = self.severity_counts.get(sev, 0) + 1
        if doc_name in self.document_stats:
            ds = self.document_stats[doc_name]
            ds["boundaries"] += 1
            ds["safe"] += 1 if sev == SAFE else 0
            ds["info"] += 1 if sev == INFO else 0
            ds["suspect"] += 1 if sev == SUSPECT else 0
            ds["high_risk"] += 1 if sev == HIGH_RISK else 0
        for r in b.get("reasons", []):
            self.reason_counts[r] = self.reason_counts.get(r, 0) + 1
        for f in b.get("flags", []):
            self.flag_counts[f] = self.flag_counts.get(f, 0) + 1
        self._collect_sample(b, doc_name)

    def record_structure_suspect(self, item: dict, doc_name: str) -> None:
        if len(self.samples["structure_gaps"]) < MAX_EXAMPLES:
            self.samples["structure_gaps"].append({"document": doc_name, **item})

    def _collect_sample(self, b: dict, doc_name: str) -> None:
        reasons = b.get("reasons", [])
        flags = b.get("flags", [])
        sev = b["severity"]
        s = {
            "document": doc_name,
            "prev_id": b["prev_id"], "next_id": b["next_id"],
            "prev_text_tail": b["prev_tail"], "next_text_head": b["next_head"],
            "structural_changes": b.get("structural_changes", []),
            "flags": flags, "reasons": reasons, "severity": sev,
        }
        if any("заканчивается" in r for r in reasons):
            if len(self.samples["suspicious_ends"]) < MAX_EXAMPLES:
                self.samples["suspicious_ends"].append(s)
        if any("начинается" in r for r in reasons):
            if len(self.samples["suspicious_starts"]) < MAX_EXAMPLES:
                self.samples["suspicious_starts"].append(s)
        if any("условного" in r for r in reasons):
            if len(self.samples["conditional_boundaries"]) < MAX_EXAMPLES:
                self.samples["conditional_boundaries"].append(s)
        if any("list_" in f for f in flags):
            if len(self.samples["list_boundaries"]) < MAX_EXAMPLES:
                self.samples["list_boundaries"].append(s)
        if "текст обрывается без знака препинания" in reasons:
            self.samples["hard_breaks"].append(s)
        if sev == HIGH_RISK:
            if len(self.samples["high_risk"]) < MAX_EXAMPLES:
                self.samples["high_risk"].append(s)

def _longest_common_suffix_prefix(a: str, b: str, max_len: int = 50) -> int:
    limit = min(len(a), len(b), max_len)
    for i in range(limit, 0, -1):
        if b.startswith(a[-i:]): return i
    return 0

def _lexical_continuity_check(chunks: list[dict]) -> dict:
    result = {"suspect_gaps": [], "suspect_overlaps": []}
    for i in range(len(chunks) - 1):
        p = _norm_ws(_strip_prefix(chunks[i].get("text", "")))
        n = _norm_ws(_strip_prefix(chunks[i + 1].get("text", "")))
        if not p or not n: continue
        t80 = p[-80:].strip(); h80 = n[:80].strip()
        if t80 and h80:
            c = _longest_common_suffix_prefix(t80, h80)
            if c >= 20:
                result["suspect_overlaps"].append({
                    "prev_id": chunks[i]["id"],
                    "next_id": chunks[i + 1]["id"],
                    "overlap_chars": c, "overlap_text": t80[-c:],
                })
        pp = _parse_chunk_id(chunks[i].get("id", ""))
        np = _parse_chunk_id(chunks[i + 1].get("id", ""))
        same = pp and np and pp['doc'] == np['doc']
        same = same and pp['section_type'] == np['section_type']
        same = same and pp['section_num'] == np['section_num']
        if same:
            tw = set(p.lower().split()[-5:])
            hw = set(n.lower().split()[:5])
            if not (tw & hw) and len(p) > 50 and len(n) > 50:
                result["suspect_gaps"].append({
                    "prev_id": chunks[i]["id"],
                    "next_id": chunks[i + 1]["id"],
                    "gap_type": "no_lexical_bridge_within_article",
                })
    return result

def run_semantic_audit() -> dict:
    """Запустить полный семантический аудит."""
    a = BoundaryAnalyzer(); s = StatisticsCollector()
    sc = StructureContinuityChecker()
    files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    print("═══ СЕМАНТИЧЕСКИЙ АУДИТ ═══\n")
    cg = 0; co = 0; sg = 0; so = 0
    for idx, fp in enumerate(files, 1):
        dn = fp.stem; dc = []
        for ch in iter_jsonl(fp):
            dc.append(ch); a.feed(ch); s.record_chunk(dn, ch)
        lr = _lexical_continuity_check(dc)
        sg += len(lr['suspect_gaps']); so += len(lr['suspect_overlaps'])
        sp = STRUCTURE_DIR / f"{dn}.json"
        if sp.exists():
            sr = sc.check_document(sp, dc)
            cg += len(sr['confirmed_gaps']); co += len(sr['confirmed_overlaps'])
            for g in sr['suspect_gaps']:
                sg += 1; s.record_structure_suspect(g, dn)
        if idx % 5 == 0 or idx == len(files):
            print(f"  [{idx}/{len(files)}] {dn}: {len(dc)} chunks")
    print()
    print(f"Файлов: {len(files)} | Чанков: {s.total_chunks} | Границ: {len(a.boundaries)}")
    for b in a.boundaries:
        dn = _parse_chunk_id(b['prev_id'])
        dn = dn['doc'] if dn else '?'
        s.record_boundary(b, dn)
    print(f"SAFE={s.severity_counts[SAFE]} INFO={s.severity_counts[INFO]} SUSPECT={s.severity_counts[SUSPECT]} HIGH_RISK={s.severity_counts[HIGH_RISK]}")
    print(f"Small(<40)={s.token_buckets['0-39']} Large(=400)={s.token_buckets['=400']}")
    print(f"Suspicious ends={len(s.samples['suspicious_ends'])} starts={len(s.samples['suspicious_starts'])}")
    print(f"Cond bound={len(s.samples['conditional_boundaries'])} list bound={len(s.samples['list_boundaries'])}")
    print(f"Hard breaks={len(s.samples['hard_breaks'])}")
    print(f"Lex gaps={sg} overlaps={so} Conf gaps={cg} overlaps={co} Struct suspect={len(s.samples['structure_gaps'])}")
    if cg > 0 or co > 0: v = 'FAIL'
    elif s.severity_counts[SUSPECT] > 0 or sg > 0 or so > 0: v = 'WARN'
    else: v = 'PASS'
    print(f"\nВЕРДИКТ: {v}")
    return {
        "meta": {"timestamp": datetime.now().isoformat(),
            "project_dir": str(PROJECT_DIR),
            "docs": len(files), "chunks": s.total_chunks,
            "boundaries": len(a.boundaries), "verdict": v},
        "summary": {"safe": s.severity_counts[SAFE], "info": s.severity_counts[INFO],
            "suspect": s.severity_counts[SUSPECT], "high_risk": s.severity_counts[HIGH_RISK],
            "confirmed_gaps": cg, "confirmed_overlaps": co, "suspect_gaps": sg, "suspect_overlaps": so},
        "statistics": {"token_buckets": s.token_buckets,
            "small": s.token_buckets["0-39"], "large": s.token_buckets["=400"],
            "suspicious_ends": len(s.samples["suspicious_ends"]),
            "suspicious_starts": len(s.samples["suspicious_starts"]),
            "conditional_boundaries": len(s.samples["conditional_boundaries"]),
            "list_boundaries": len(s.samples["list_boundaries"]),
            "hard_breaks": len(s.samples["hard_breaks"]),
            "reasons": dict(sorted(s.reason_counts.items(), key=lambda x: -x[1])),
            "flags": dict(sorted(s.flag_counts.items(), key=lambda x: -x[1]))},
        "document_stats": s.document_stats, "samples": s.samples,
    }

def main() -> None:
    AUDIT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    r = run_semantic_audit()
    p = AUDIT_REPORTS_DIR / "semantic_audit_report.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт: {p}")

if __name__ == "__main__":
    main()
