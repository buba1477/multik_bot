"""Тесты для markdown_structure_parser.

Standalone — не требуют Qdrant, Ollama, RAG. Только parser + markdown-фикстуры.
"""
import json
from pathlib import Path

import pytest

from app.ingestion.markdown_structure_parser import (
    _classify_block,
    _md_heading_level,
    _make_node_id,
    _norm_number,
    _parse_table_rows,
    _strip_md_heading,
    build_context_recursive,
    build_linear,
    build_records,
    build_tree,
    classify_content,
    classify_heading,
    exact_reconstruct,
    is_table_line,
    normalize_text,
    parse_blocks,
    read_markdown,
    split_segments,
)

FIXTURES = [
    ("markdown/79-FZ.md", "79-FZ"),
    ("markdown_test/polozhenie.md", "polozhenie"),
    ("markdown_test/postanovlenie.md", "postanovlenie"),
    ("markdown_test/prikaz.md", "prikaz"),
    ("markdown_test/rasporyazhenie.md", "rasporyazhenie"),
    ("markdown_test/ukaz.md", "ukaz"),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# 1. Unit-тесты существующих функций
# ============================================================================

class TestNormNumber:
    def test_simple(self):
        assert _norm_number("1") == "1"
        assert _norm_number(" 1.2 ") == "1.2"
        assert _norm_number("1.2.3") == "1.2.3"


class TestClassifyHeading:
    def test_chapter(self):
        t, num, title, conf = classify_heading("Глава 1. Общие положения")
        assert t == "chapter"
        assert num == "1"
        assert title == "Общие положения"
        assert conf == 0.95

    def test_section_word(self):
        t, num, title, conf = classify_heading("Раздел I. Общие положения")
        assert t == "section"
        assert num == "I"

    def test_article(self):
        t, num, title, conf = classify_heading("Статья 1. Основные термины")
        assert t == "article"
        assert num == "1"

    def test_appendix(self):
        t, num, title, conf = classify_heading("Приложение № 1 Таблица")
        assert t == "appendix"
        assert num == "1"

    def test_numbered_section(self):
        t, num, title, conf = classify_heading("1. Общие положения")
        assert t == "section"
        assert num == "1"

    def test_numbered_subsection(self):
        t, num, title, conf = classify_heading("1.1 Термины")
        assert t == "subsection"
        assert num == "1.1"

    def test_roman_heading(self):
        t, num, title, conf = classify_heading("I. Общие положения")
        assert t == "section"
        assert num == "I"

    def test_unknown_heading(self):
        t, num, title, conf = classify_heading("Произвольный текст")
        assert t == "unknown"


class TestClassifyContent:
    def test_blockquote(self):
        assert classify_content(["> цитата"])[0] == "blockquote"

    def test_item_with_paren(self):
        t, num, conf = classify_content(["1) пункт"])
        assert t == "item"
        assert num == "1"

    def test_subparagraph_letter_paren(self):
        t, num, conf = classify_content(["а) подпункт"])
        assert t == "subparagraph"
        assert num == "а"

    def test_paragraph_number_dot(self):
        t, num, conf = classify_content(["1. параграф"])
        assert t == "paragraph"
        assert num == "1"

    def test_plain_text(self):
        assert classify_content(["Обычный текст"])[0] == "text"


class TestIsTableLine:
    def test_with_pipe(self):
        assert is_table_line("| cell1 | cell2 |") is True

    def test_no_pipe(self):
        assert is_table_line("plain text") is False

    def test_empty(self):
        assert is_table_line("") is False


class TestSplitSegments:
    def test_simple(self):
        result = split_segments(["a", "b", "", "c", "d", "", "e"])
        assert result == [[0, 1], [3, 4], [6]]

    def test_no_blanks(self):
        result = split_segments(["a", "b"])
        assert result == [[0, 1]]

    def test_trailing_blanks(self):
        result = split_segments(["a", "b", "", ""])
# ============================================================================
# 2. Тесты нового pipeline
# ============================================================================

class TestMdHeadingLevel:
    def test_h1(self):
        assert _md_heading_level("# Title") == 1

    def test_h2(self):
        assert _md_heading_level("## Title") == 2

    def test_not_heading(self):
        assert _md_heading_level("Text") == 0


class TestStripMdHeading:
    def test_basic(self):
        assert _strip_md_heading("## Глава 1") == "Глава 1"

    def test_no_marker(self):
        assert _strip_md_heading("Глава 1") == "Глава 1"


class TestNodeId:
    def test_padding(self):
        assert _make_node_id(0) == "n0000"
        assert _make_node_id(5) == "n0005"
        assert _make_node_id(42) == "n0042"


class TestParseTableRows:
    def test_simple_table(self):
        rows = _parse_table_rows(["| a | b |", "| c | d |"])
        assert rows == [["a", "b"], ["c", "d"]]

    def test_no_leading_trailing(self):
        rows = _parse_table_rows(["a | b"])
        assert rows == [["a", "b"]]


# ============================================================================
# 3. Интеграционные тесты на всех фикстурах
# ============================================================================

class TestFullPipeline:
    @pytest.mark.parametrize("rel_path,stem", FIXTURES)
    def test_pipeline_runs_and_reconstructs(self, rel_path, stem):
        md_path = PROJECT_ROOT / rel_path
        assert md_path.exists(), f"Фикстура не найдена: {md_path}"

        lines = read_markdown(md_path)
        assert len(lines) > 0, f"Пустой файл: {md_path}"

        blocks = parse_blocks(lines)
        assert len(blocks) > 0, "Нет блоков"
        assert len(blocks) == len(split_segments(lines))

        linear = build_linear(blocks)
        assert len(linear) == len(blocks), "Каждый блок → ровно одна Node"

        tree_root = build_tree(linear)
        assert tree_root["type"] == "document"
        assert "children" in tree_root

        build_context_recursive(tree_root)

        def _check_ctx(node):
            assert "context" in node
            assert "context_flat" in node
            assert "ancestors" in node["context"]
            for f in ["chapter", "section", "subsection", "article", "paragraph", "item", "appendix"]:
                assert f in node["context_flat"]
            for c in node["children"]:
                _check_ctx(c)

        _check_ctx(tree_root)

        records = build_records(tree_root)
        assert len(records) > 0
        for rec in records:
            assert "node_id" in rec
            assert "text" in rec
            assert "structure" in rec
            assert "type" in rec["structure"]
            assert "context_flat" in rec["structure"]

        recon = exact_reconstruct(linear, total_source_lines=len(lines))
        original_stripped = md_path.read_text(encoding="utf-8").rstrip("\n")
        assert original_stripped == recon, f"Exact reconstruction FAIL for {stem}"

        assert normalize_text(original_stripped) == normalize_text(recon)

    @pytest.mark.parametrize("rel_path,stem", FIXTURES)
    def test_no_duplicate_blocks(self, rel_path, stem):
        md_path = PROJECT_ROOT / rel_path
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)

        ranges = [(n["source_lines"]["start"], n["source_lines"]["end"]) for n in linear]
        ranges.sort()
        for i in range(1, len(ranges)):
            assert ranges[i][0] > ranges[i - 1][1], (
                f"Перекрытие блоков: {ranges[i-1]} и {ranges[i]}"
            )

    @pytest.mark.parametrize("rel_path,stem", FIXTURES)
    def test_content_is_verbatim(self, rel_path, stem):
        md_path = PROJECT_ROOT / rel_path
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)

        for node in linear:
            start = node["source_lines"]["start"]
            end = node["source_lines"]["end"]
            reconstructed_lines = "\n".join(lines[start:end + 1])
            assert node["content"] == reconstructed_lines, (
                f"Node {node['id']}: content не совпадает с исходными строками"
            )


# ============================================================================
# 4. Тест coverage records
# ============================================================================

class TestRecordsCoverage:
    def test_article_paragraph_item_table_have_records(self):
        md_path = PROJECT_ROOT / "markdown/79-FZ.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree_root = build_tree(linear)
        build_context_recursive(tree_root)
        records = build_records(tree_root)
        record_types = {r["structure"]["type"] for r in records}
        assert "article" in record_types
        assert "paragraph" in record_types
        assert "item" in record_types
        assert "table" in record_types

    def test_chapter_no_records(self):
        md_path = PROJECT_ROOT / "markdown/79-FZ.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree_root = build_tree(linear)
        build_context_recursive(tree_root)
        records = build_records(tree_root)
        chapter_records = [r for r in records if r["structure"]["type"] == "chapter"]
        assert len(chapter_records) == 0


# ============================================================================
# 5. Тест конкретного узла
# ============================================================================

class TestLinearNodeStructure:
    def test_first_node_79_fz(self):
        md_path = PROJECT_ROOT / "markdown/79-FZ.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        node = linear[0]
        assert node["id"] == "n0000"
        assert node["type"] == "text"
        assert node["source_lines"] == {"start": 0, "end": 0}
        assert node["content"].startswith("Федеральный закон")
        assert node["confidence"] == 0.5

    def test_glava_node_79_fz(self):
        md_path = PROJECT_ROOT / "markdown/79-FZ.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        chapters = [n for n in linear if n["type"] == "chapter"]
        assert len(chapters) > 0
        assert chapters[0]["number"] == "1"
        assert chapters[0]["title"] == "Общие положения"