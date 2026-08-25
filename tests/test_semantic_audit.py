"""Tests for semantic_audit.py."""
import json
from pathlib import Path

import pytest

from scripts.semantic_audit import (
    SAFE,
    SUSPECT,
    INFO,
    HIGH_RISK,
    BoundaryAnalyzer,
    StatisticsCollector,
    StructureContinuityChecker,
    _norm_ws,
    _count_tokens,
    _parse_chunk_id,
    _lexical_continuity_check,
    _longest_common_suffix_prefix,
)


class TestNormWs:
    def test_normalize_spaces(self) -> None:
        assert _norm_ws("  hello   world  ") == "hello world"

    def test_empty(self) -> None:
        assert _norm_ws("") == ""

    def test_no_change(self) -> None:
        assert _norm_ws("hello world") == "hello world"


class TestCountTokens:
    def test_empty(self) -> None:
        assert _count_tokens("") == 0

    def test_short(self) -> None:
        assert _count_tokens("hi") == 1

    def test_five_chars(self) -> None:
        assert _count_tokens("abcde") == 2


class TestParseChunkId:
    def test_regular_article(self) -> None:
        r = _parse_chunk_id("117-fz_st143_p1")
        assert r is not None
        assert r["doc"] == "117-fz"
        assert r["section_type"] == "st"
        assert r["section_num"] == "143"
        assert r["chunk_num"] == 1

    def test_hyphenated_article(self) -> None:
        r = _parse_chunk_id("117-fz_st246_1-1_p1")
        assert r is not None
        assert r["section_num"] == "246.1-1"

    def test_preamble(self) -> None:
        r = _parse_chunk_id("117-fz_pre_p3")
        assert r is not None
        assert r["section_type"] == "pre"

class TestBoundaryAnalyzer:
    def test_single_chunk_no_boundary(self) -> None:
        ba = BoundaryAnalyzer()
        ba.feed({"id": "doc_st1_p1", "title": "t1", "text": "Some text."})
        assert len(ba.boundaries) == 0

    def test_two_chunks_one_boundary(self) -> None:
        ba = BoundaryAnalyzer()
        ba.feed({"id": "doc_st1_p1", "title": "t1", "text": "First chunk."})
        ba.feed({"id": "doc_st1_p2", "title": "t1", "text": "Second chunk."})
        assert len(ba.boundaries) == 1
        b = ba.boundaries[0]
        assert b["prev_id"] == "doc_st1_p1"
        assert b["next_id"] == "doc_st1_p2"

    def test_document_change(self) -> None:
        ba = BoundaryAnalyzer()
        ba.feed({"id": "doc1_st1_p1", "title": "t1", "text": "Text."})
        ba.feed({"id": "doc2_st1_p1", "title": "t2", "text": "More."})
        assert "DOCUMENT_CHANGE" in ba.boundaries[0]["structural_changes"]

    def test_article_change(self) -> None:
        ba = BoundaryAnalyzer()
        ba.feed({"id": "doc_st1_p1", "title": "t1", "text": "Text."})
        ba.feed({"id": "doc_st2_p1", "title": "t2", "text": "More."})
        assert "ARTICLE_CHANGE" in ba.boundaries[0]["structural_changes"]

    def test_suspicious_end(self) -> None:
        ba = BoundaryAnalyzer()
        ba.feed({"id": "doc_st1_p1", "title": "t1", "text": "Text ohne Punkt"})
        ba.feed({"id": "doc_st1_p2", "title": "t1", "text": "More text."})
        sev = ba.boundaries[0]["severity"]
class TestStatisticsCollector:
    def test_record_chunk(self) -> None:
        sc = StatisticsCollector()
        sc.record_chunk("test", {"text": "Short text."})
        assert sc.total_chunks == 1
        assert sc.token_buckets["0-39"] == 1

    def test_record_chunk_large(self) -> None:
        sc = StatisticsCollector()
        sc.record_chunk("test", {"text": "x" * 2000})
        assert sc.token_buckets["=400"] == 1

    def test_record_boundary(self) -> None:
        sc = StatisticsCollector()
        b = {"prev_id": "a", "next_id": "b", "severity": SAFE,
             "prev_tail": "", "next_head": "", "reasons": [], "flags": [],
             "structural_changes": [], "prev_title": "", "next_title": ""}
        sc.document_stats["test"] = {"chunks": 0, "boundaries": 0,
            "safe": 0, "info": 0, "suspect": 0, "high_risk": 0, "small": 0, "large": 0}
        sc.record_boundary(b, "test")
        assert sc.total_boundaries == 1
        assert sc.severity_counts[SAFE] == 1


class TestLexicalContinuity:
    def test_no_overlap(self) -> None:
        chunks = [{"id": "doc_st1_p1", "text": "First chunk."},
                  {"id": "doc_st1_p2", "text": "Second chunk."}]
        r = _lexical_continuity_check(chunks)
        assert len(r["suspect_overlaps"]) == 0

    def test_longest_common_suffix_prefix(self) -> None:
        assert _longest_common_suffix_prefix("abcde", "cdefg") == 3
        assert _longest_common_suffix_prefix("abcde", "xyz") == 0


class TestStructureContinuityChecker:
    def test_empty_records(self, tmp_path: Path) -> None:
        sp = tmp_path / "test.json"
        sp.write_text(json.dumps({"records": []}), encoding="utf-8")
        c = StructureContinuityChecker()
        r = c.check_document(sp, [])
        assert r["confirmed_gaps"] == []
        assert r["suspect_gaps"] == []


class TestBoundaryAnalyzerSafe:
    def test_safe_boundary(self) -> None:
        ba = BoundaryAnalyzer()
        ba.feed({"id": "doc_st1_p1", "title": "t1", "text": "First sentence. Second."})
        ba.feed({"id": "doc_st2_p1", "title": "t2", "text": "New article text."})
        assert ba.boundaries[0]["severity"] == SAFE

