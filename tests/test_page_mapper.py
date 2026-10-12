"""Tests for page_mapper module."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.chunking.page_mapper import (
    _normalize, _normalize_search, _strip_chunk_prefix,
    _build_concat_index, _find_pages_in_concat,
    find_page_for_chunk, load_mapping, save_mapping,
    get_chunk_ids_from_jsonl, update_mapping_for_document,
    find_pdf_for_jsonl,
)

# Build normalized test pages for proper probe matching
def _norm_pages() -> tuple[str, list[int]]:
    raw = [
        "A. Page one content. " * 20 + "END1.",
        "B. Page two content. " * 20 + "END2.",
        "C. Page three content. " * 20 + "END3.",
    ]
    return _build_concat_index([_normalize_search(p) for p in raw])

CONCAT_TEXT, CUM_LENS = _norm_pages()


class TestNormalize:
    def test_whitespace_compaction(self):
        assert _normalize("  Hello  World  ") == "Hello World"
    def test_normalize_nbsp(self):
        assert _normalize("Hello\u00a0World") == "Hello World"

class TestStripPrefix:
    def test_strip_two_brackets(self):
        assert _strip_chunk_prefix("[doc] [title] content") == "content"
    def test_no_prefix(self):
        assert _strip_chunk_prefix("plain text") == "plain text"
    def test_only_one_bracket(self):
        assert _strip_chunk_prefix("[only] text") == "[only] text"

class TestConcatIndex:
    def test_simple_pages(self):
        pages = ["a", "bb", "ccc"]
        concat, cum = _build_concat_index(pages)
        assert cum == [0, 2, 5, 9]
    def test_find_pages_basic(self):
        pages = ["alpha beta", "gamma delta"]
        concat, cum = _build_concat_index(pages)
        assert _find_pages_in_concat("beta", concat, cum) == [1]
    def test_no_match_returns_empty(self):
        assert _find_pages_in_concat("NONEXISTENT", CONCAT_TEXT, CUM_LENS) == []

class TestFindPageForChunk:
    def test_exact_match_page_2(self):
        text = "B. Page two content. " * 4 + "END2."
        chunk = {"id": "t_p1", "text": text}
        page, a = find_page_for_chunk(chunk, CONCAT_TEXT, CUM_LENS)
        assert page == 2, f"Expected 2, got {page}"
        assert a == [2]
    def test_short_text_returns_none(self):
        chunk = {"id": "t_p2", "text": "Short"}
        page, a = find_page_for_chunk(chunk, CONCAT_TEXT, CUM_LENS)
        assert page is None
    def test_empty_text_returns_none(self):
        chunk = {"id": "t_p3", "text": ""}
        page, a = find_page_for_chunk(chunk, CONCAT_TEXT, CUM_LENS)
        assert page is None

class TestMappingIO:
    def test_save_and_load_roundtrip(self):
        data = {"a_p1": 1, "a_p2": None}
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        p = Path(tmp.name); tmp.close()
        try:
            save_mapping(data, p)
            assert load_mapping(p) == data
        finally:
            p.unlink(missing_ok=True)
    def test_load_nonexistent_returns_empty(self):
        assert load_mapping(Path("/tmp/_nonexist_map_test_.json")) == {}
    def test_save_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmpd:
            p = Path(tmpd) / "m.json"
            save_mapping({"x": 1}, p)
            assert p.exists()
            assert not p.with_suffix(".json.tmp").exists()
            assert json.loads(p.read_text()) == {"x": 1}

class TestGetChunkIds:
    def test_basic_ids(self):
        ll = [json.dumps({"id": "a_p1", "text": "x"}), json.dumps({"id": "a_p2", "text": "y"})]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(chr(10).join(ll)); p = Path(f.name)
        try:
            assert get_chunk_ids_from_jsonl(p) == {"a_p1", "a_p2"}
        finally:
            p.unlink(missing_ok=True)

class TestUpdateMapping:
    def test_replace_document(self):
        existing = {"a_p1": 1, "a_p2": None, "b_p1": 5}
        new = {"a_p1": 2, "a_p3": 3}
        r = update_mapping_for_document(existing, Path("a.jsonl"), new)
        assert r["b_p1"] == 5 and r["a_p1"] == 2 and r["a_p3"] == 3
        assert "a_p2" not in r and len(r) == 3
    def test_empty_new_returns_none(self):
        assert update_mapping_for_document({"a": 1}, Path("x.jsonl"), {}) is None
    def test_preserves_other_documents(self):
        existing = {"a_p1": 1, "b_p1": 10}
        r = update_mapping_for_document(existing, Path("a.jsonl"), {"a_p1": 5})
        assert r["b_p1"] == 10 and r["a_p1"] == 5

class TestFindPDF:
    def test_exact_match(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); (d / "test.pdf").write_text("pdf")
            assert find_pdf_for_jsonl(Path("test.jsonl"), d) == d / "test.pdf"
    def test_case_insensitive(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); (d / "TEST.PDF").write_text("pdf")
            r = find_pdf_for_jsonl(Path("test.jsonl"), d)
            assert r == d / "TEST.PDF", f"Got {r}"
    def test_no_pdf_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            assert find_pdf_for_jsonl(Path("nonexistent.jsonl"), Path(td)) is None