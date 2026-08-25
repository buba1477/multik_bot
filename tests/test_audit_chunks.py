"""Тесты для audit_chunks.

Unit-тесты на чистые функции: parse_chunk_id, count_tokens, sha256_hex, norm_ws,
compute_verdict, format_summary.
"""

from scripts.audit_chunks import (
    compute_verdict,
    count_tokens,
    extract_article_number,
    format_summary,
    norm_ws,
    parse_chunk_id,
    sha256_hex,
    strip_chunk_prefix,
)

# ============================================================================
# parse_chunk_id
# ============================================================================


class TestParseChunkId:
    def test_preambula(self):
        p = parse_chunk_id("79-fz_pre_p1")
        assert p is not None
        assert p["doc"] == "79-fz"
        assert p["section_type"] == "pre"
        assert p["chunk_num"] == 1

    def test_statya_simple(self):
        p = parse_chunk_id("79-fz_st1_p1")
        assert p is not None
        assert p["doc"] == "79-fz"
        assert p["section_type"] == "st"
        assert p["section_num"] == "1"
        assert p["chunk_num"] == 1

    def test_statya_dotted(self):
        p = parse_chunk_id("79-fz_st1_2_3_p5")
        assert p is not None
        assert p["section_num"] == "1.2.3"
        assert p["chunk_num"] == 5

    def test_sekciya(self):
        p = parse_chunk_id("79-fz_sec3_1_p2")
        assert p is not None
        assert p["section_type"] == "sec"
        assert p["section_num"] == "3.1"

    def test_podsekciya(self):
        p = parse_chunk_id("79-fz_secs1_2_p1")
        assert p is not None
        assert p["section_type"] == "secs"
        assert p["section_num"] == "1.2"

    def test_prilozhenie(self):
        p = parse_chunk_id("79-fz_app1_p3")
        assert p is not None
        assert p["section_type"] == "app"
        assert p["section_num"] == "1"
        assert p["chunk_num"] == 3

    def test_statya_underscore_one(self):
        """st145_1 -> статья 145.1."""
        p = parse_chunk_id("117-fz_st145_1_p1")
        assert p is not None
        assert p["section_num"] == "145.1"
        assert p["chunk_num"] == 1
        assert p["doc"] == "117-fz"

    def test_statya_underscore_two(self):
        """st246_1 -> статья 246.1."""
        p = parse_chunk_id("117-fz_st246_1_p1")
        assert p is not None
        assert p["section_num"] == "246.1"

    def test_statya_twelve(self):
        """st12 -> статья 12."""
        p = parse_chunk_id("117-fz_st12_p1")
        assert p is not None
        assert p["section_num"] == "12"

    def test_statya_double_underscore(self):
        """st246_1_2 -> статья 246.1.2."""
        p = parse_chunk_id("117-fz_st246_1_2_p3")
        assert p is not None
        assert p["section_num"] == "246.1.2"
        assert p["chunk_num"] == 3

    def test_statya_triple_underscore(self):
        """st333_34_1 -> статья 333.34.1."""
        p = parse_chunk_id("117-fz_st333_34_1_p1")
        assert p is not None
        assert p["section_num"] == "333.34.1"

    def test_statya_hyphenated(self):
        """st246_1-1_p1 -> статья 246.1-1, часть 1."""
        p = parse_chunk_id("117-fz_st246_1-1_p1")
        assert p is not None
        assert p["section_type"] == "st"
        assert p["section_num"] == "246.1-1"
        assert p["chunk_num"] == 1

    def test_statya_hyphenated_p3(self):
        """st418_5-1_p3 -> статья 418.5-1, часть 3."""
        p = parse_chunk_id("117-fz_st418_5-1_p3")
        assert p is not None
        assert p["section_num"] == "418.5-1"
        assert p["chunk_num"] == 3

    def test_statya_hyphenated_complex(self):
        """st333_34-1_p1 -> статья 333.34-1."""
        p = parse_chunk_id("117-fz_st333_34-1_p1")
        assert p is not None
        assert p["section_num"] == "333.34-1"


# ============================================================================
# extract_article_number
# ============================================================================


class TestExtractArticleNumber:
    def test_article_with_space(self):
        """Статья 246 1. ... -> 246.1."""
        assert extract_article_number("Статья 246 1. Освобождение") == "246.1"

    def test_article_space_hyphen(self):
        """Статья 246 1-1. ... -> 246.1-1."""
        assert extract_article_number("Статья 246 1-1. Освобождение") == "246.1-1"

    def test_article_simple_one(self):
        """Статья 1. ... -> 1."""
        assert extract_article_number("Статья 1. Основные термины") == "1"

    def test_article_simple_twelve(self):
        """Статья 12. ... -> 12."""
        assert extract_article_number("Статья 12. Учёт") == "12"

    def test_article_complex_three_levels(self):
        """Статья 333 34-1. ... -> 333.34-1."""
        assert extract_article_number("Статья 333 34-1. Особенности") == "333.34-1"

    def test_article_no_dot_space(self):
        """Статья 246 1 (без точки) -> 246.1."""
        assert extract_article_number("Статья 246 1") == "246.1"

    def test_non_article(self):
        """Преамбула -> None."""
        assert extract_article_number("Преамбула") is None

    def test_empty_title(self):
        assert extract_article_number("") is None


# ============================================================================
# strip_chunk_prefix
# ============================================================================


class TestStripChunkPrefix:
    def test_standard_prefix(self):
        """Удаляет [DOC] [TITLE] в начале строки."""
        result = strip_chunk_prefix("[117-ФЗ] [Преамбула]\nКодекс Российской Федерации")
        assert result == "Кодекс Российской Федерации"

    def test_prefix_with_long_title(self):
        """Удаляет [117-ФЗ] [Статья 246 1-1. ...]."""
        result = strip_chunk_prefix("[117-ФЗ] [Статья 246 1-1. Освобождение...]\n1. Организация")
        assert result == "1. Организация"

    def test_no_prefix(self):
        """Без префикса — строка не меняется."""
        assert strip_chunk_prefix("Простой текст") == "Простой текст"

    def test_empty(self):
        assert strip_chunk_prefix("") == ""

    def test_only_prefix(self):
        """Только префикс без содержимого."""
        result = strip_chunk_prefix("[117-ФЗ] [Преамбула]\n")
        assert result == ""

    def test_prefix_without_newline(self):
        """Префикс без \n."""
        result = strip_chunk_prefix("[117-ФЗ] [Преамбула]Кодекс")
        assert result == "Кодекс"


class TestCountTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_non_empty_fallback(self):
        """Без FRIDA-токенайзера считает //4."""
        c = count_tokens("привет мир как дела")
        assert c == 5  # (19+3)//4 = 5

    def test_short_text(self):
        assert count_tokens("а") >= 1


# ============================================================================
# sha256_hex
# ============================================================================


class TestSha256Hex:
    def test_empty(self):
        import hashlib
        assert sha256_hex("") == hashlib.sha256(b"").hexdigest()

    def test_output(self):
        h = sha256_hex("hello")
        assert len(h) == 64
        assert h == sha256_hex("hello")  # детерминизм


# ============================================================================
# norm_ws
# ============================================================================


class TestNormWs:
    def test_collapse(self):
        assert norm_ws("  много   пробелов ") == "много пробелов"

    def test_tabs_newlines(self):
        assert norm_ws("a\nb\tc") == "a b c"

    def test_empty(self):
        assert norm_ws("") == ""

    def test_only_whitespace(self):
        assert norm_ws("   \n\t   ") == ""


# ============================================================================
# compute_verdict
# ============================================================================


class TestComputeVerdict:
    def test_all_pass(self):
        assert compute_verdict({"a": {"status": "PASS"}, "b": {"status": "PASS"}}) == "PASS"

    def test_warn(self):
        assert compute_verdict({"a": {"status": "PASS"}, "b": {"status": "WARN"}}) == "WARN"

    def test_fail_dominates(self):
        r = {"a": {"status": "PASS"}, "b": {"status": "WARN"}, "c": {"status": "FAIL"}}
        assert compute_verdict(r) == "FAIL"

    def test_empty(self):
        assert compute_verdict({}) == "PASS"

    def test_info_ignored(self):
        assert compute_verdict({"a": {"status": "INFO"}}) == "PASS"


# ============================================================================
# format_summary
# ============================================================================


class TestFormatSummary:
    def test_contains_verdict(self):
        fake_results = {
            "check_jsonl_schema_and_tokens": {
                "total_chunks": 100,
                "token_statistics": {"min": 1, "max": 400, "avg": 200, "median": 200,
                                     "over_400": 0, "buckets": {}},
                "schema_status": "PASS",
                "token_status": "PASS",
            },
            "check_file_correspondence": {"status": "PASS"},
        }
        s = format_summary(fake_results)
        assert "VERDICT" in s

    def test_lines_count(self):
        fake_results = {
            "check_jsonl_schema_and_tokens": {
                "total_chunks": 50,
                "token_statistics": {"min": 1, "max": 399, "avg": 150, "median": 120,
                                     "over_400": 0, "buckets": {}},
                "schema_status": "PASS",
                "token_status": "PASS",
            },
            "check_duplicates_sha256": {"status": "PASS"},
        }
        s = format_summary(fake_results)
        assert s.count("\n") >= 3