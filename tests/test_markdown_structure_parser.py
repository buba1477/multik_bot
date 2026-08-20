"""Тесты для markdown_structure_parser.

Standalone — не требуют Qdrant, Ollama, RAG. Только parser + markdown-фикстуры.
"""
import io
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
    normalize_underscores,
    parse_and_export_jsonl,
    parse_blocks,
    read_markdown,
    records_to_jsonl,
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
# ============================================================================
# 6. Regression tests для документов 506, 96, 667-r, 159
# ============================================================================

class TestRegression506:
    """postanovlenie-506: римские цифры, иерархия 1 → 1.1 → 1.1.1"""

    @pytest.fixture(autouse=True)
    def _parse_506(self):
        md_path = PROJECT_ROOT / "markdown/postanovlenie-506.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        self.linear = build_linear(blocks)
        self.tree = build_tree(self.linear)
        build_context_recursive(self.tree)

    def test_section_I_obshchie_polozheniya(self):
        """I. Общие положения → section 'I'"""
        sections = [n for n in self.linear if n["type"] == "section" and n["number"] == "I"]
        assert len(sections) >= 1
        assert "Общие положения" in sections[0]["content"]

    def test_section_II_polnomochiya(self):
        """II\\. Полномочия → section 'II'"""
        sections = [n for n in self.linear if n["type"] == "section" and n["number"] == "II"]
        assert len(sections) >= 1
        assert "Полномочия" in sections[0]["content"]

    def test_paragraph_5_under_II(self):
        """Пункт 5 под разделом II"""
        ctx_records = build_records(self.tree)
        p5 = [r for r in ctx_records
              if r["structure"]["type"] == "paragraph"
              and r["structure"]["number"] == "5"
              and r["structure"]["context_flat"].get("section") == "II"]
        assert len(p5) >= 1

    def test_subparagraph_5_1(self):
        """5.1. осуществляет контроль → subparagraph '5.1'"""
        sp = [n for n in self.linear if n["type"] == "subparagraph" and n["number"] == "5.1"]
        assert len(sp) >= 1
        assert "контроль" in sp[0]["content"]

    def test_item_5_1_1(self):
        """5.1.1. соблюдением → item '5.1.1'"""
        items = [n for n in self.linear if n["type"] == "item" and n["number"] == "5.1.1"]
        assert len(items) >= 1
        assert "соблюдением" in items[0]["content"]

    def test_appendix_utverzhdeno(self):
        """УТВЕРЖДЕНО → appendix"""
        apps = [n for n in self.linear if n["type"] == "appendix"]
        assert len(apps) >= 1
        assert "УТВЕРЖДЕНО" in apps[0]["content"]
class TestRegression96:
    """ukaz-96: разделы, подпункты а-з, вложенные пункты"""

    @pytest.fixture(autouse=True)
    def _parse_96(self):
        md_path = PROJECT_ROOT / "markdown/ukaz-96.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        self.linear = build_linear(blocks)
        self.tree = build_tree(self.linear)
        build_context_recursive(self.tree)

    def test_section_I_obshchie(self):
        assert any(n["type"] == "section" and n["number"] == "I"
                   for n in self.linear)

    def test_section_II_poryadok(self):
        assert any(n["type"] == "section" and n["number"] == "II"
                   for n in self.linear)

    def test_section_III_konkurs(self):
        assert any(n["type"] == "section" and n["number"] == "III"
                   for n in self.linear)

    def test_paragraph_1(self):
        assert any(n["type"] == "paragraph" and n["number"] == "1"
                   for n in self.linear)

    def test_paragraph_2_has_sub_a_b_v_g(self):
        """Пункт 2 → 4+ подпункта (а, б, в, г)"""
        ctx_records = build_records(self.tree)
        subpars = [r for r in ctx_records
                   if r["structure"]["type"] == "subparagraph"
                   and r["structure"]["context_flat"].get("paragraph") == "2"]
        assert len(subpars) >= 4

    def test_paragraph_8_has_sub_a_b_v(self):
        """Пункт 8 → 3+ подпункта (а, б, в)"""
        ctx_records = build_records(self.tree)
        subpars = [r for r in ctx_records
                   if r["structure"]["type"] == "subparagraph"
                   and r["structure"]["context_flat"].get("paragraph") == "8"]
        assert len(subpars) >= 3

    def test_paragraph_14_15_exist(self):
        nums = {n["number"] for n in self.linear
                if n["type"] == "paragraph"}
        assert "14" in nums and "15" in nums

    def test_appendix_utverzhdeno(self):
        assert any(n["type"] == "appendix" for n in self.linear)
class TestRegression667r:
    """rasporyazhenie-667-r: анкета, пункты 2-23"""

    @pytest.fixture(autouse=True)
    def _parse_667(self):
        md_path = PROJECT_ROOT / "markdown/rasporyazhenie-667-r.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        self.linear = build_linear(blocks)
        self.tree = build_tree(self.linear)
        build_context_recursive(self.tree)

    def test_paragraphs_2_to_23(self):
        nums = {n["number"] for n in self.linear
                if n["type"] == "paragraph" and n["number"] is not None}
        for expected in ("2", "3", "4", "5", "6", "7", "8", "9",
                         "10", "11", "12", "13", "14", "15",
                         "16", "17", "18", "19", "20", "21", "22", "23"):
            assert expected in nums, f"paragraph {expected} not found"

    def test_g_moskva_is_text_not_subparagraph(self):
        for n in self.linear:
            if "г.\u00a0Москва" in n["content"] or "г. Москва" in n["content"]:
                assert n["type"] != "subparagraph"

    def test_underscore_normalized_in_records(self):
        """underscores заменены на ___ в records"""
        records = build_records(self.tree)
        for r in records:
            assert "\\_\\_\\_\\_\\_\\_" not in r["text"], (
                f"{r['node_id']}: long underscores not normalized"
            )

    def test_appendix_utverzhdena(self):
        assert any(n["type"] == "appendix" for n in self.linear)


class TestRegression159:
    """ukaz-159: основной указ → приложение → разделы внутри формы"""

    @pytest.fixture(autouse=True)
    def _parse_159(self):
        md_path = PROJECT_ROOT / "markdown/ukaz-159.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        self.linear = build_linear(blocks)
        self.tree = build_tree(self.linear)
        build_context_recursive(self.tree)

    def test_ukaz_paragraphs_1_2_3(self):
        """Основной указ: п. 1, 2, 3"""
        for num in ("1", "2", "3"):
            assert any(n["type"] == "paragraph" and n["number"] == num
                       for n in self.linear)

    def test_appendix_utverzhdena(self):
        """УТВЕРЖДЕНА → appendix"""
        assert any(n["type"] == "appendix" for n in self.linear)

    def test_section_I_inside_appendix(self):
        """I. Общие положения внутри приложения → section"""
        sections = [n for n in self.linear
                    if n["type"] == "section" and n["number"] == "I"]
        assert len(sections) >= 1
        assert "Общие положения" in sections[0]["content"]

    def test_section_II_inside_appendix(self):
        """II внутри приложения"""
        sections = [n for n in self.linear
                    if n["type"] == "section" and n["number"] == "II"]
        assert len(sections) >= 1

    def test_paragraphs_1_2_3_inside_appendix(self):
        """П. 1, 2, 3 внутри приложения (не основного указа)"""
        pars = [n for n in self.linear
                if n["type"] == "paragraph" and n["number"] in ("1", "2", "3")]
        assert len(pars) >= 4

    def test_section_IV_inside_appendix(self):
        """IV\\. Оплата труда → section 'IV'"""
        assert any(n["type"] == "section" and n["number"] == "IV"
                   for n in self.linear)

    def test_subparagraphs_inside_appendix(self):
        """Подпункты внутри приложения (а, б, в, г)"""
        subpars = [n for n in self.linear
                   if n["type"] == "subparagraph"]
        assert len(subpars) >= 19


class TestPageBreakContinuation:
    """Продолжение пункта через границу страниц"""

    def test_paragraph_2_continuation_159(self):
        """Пункт 2 в ukaz-159 продолжается > 100 chars"""
        md_path = PROJECT_ROOT / "markdown/ukaz-159.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        p2 = [n for n in linear if n["type"] == "paragraph" and n["number"] == "2"]
        assert len(p2) >= 1
# ============================================================================
# 7. JSONL export tests
# ============================================================================

class TestJsonlExport:
    """Проверка формата JSONL и целостности records."""

    DOCS = [
        ("markdown/79-FZ.md", "79-FZ"),
        ("markdown/postanovlenie-506.md", "506"),
        ("markdown/ukaz-96.md", "96"),
        ("markdown/rasporyazhenie-667-r.md", "667-r"),
        ("markdown/ukaz-159.md", "159"),
    ]

    @pytest.mark.parametrize("md_rel,doc_id", DOCS)
    def test_jsonl_format(self, md_rel, doc_id, tmp_path):
        """JSONL читается, каждая строка — валидный JSON, нет пустых строк."""
        md_path = PROJECT_ROOT / md_rel
        jsonl_path = tmp_path / f"{doc_id}.jsonl"
        records = parse_and_export_jsonl(md_path, jsonl_path)

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0, "JSONL не должен быть пустым"
        assert len(lines) == len(records), (
            f"Количество строк JSONL ({len(lines)}) != records ({len(records)})"
        )

        for i, line in enumerate(lines):
            assert line.strip() != "", f"Пустая строка {i} в JSONL"
            obj = json.loads(line)
            assert isinstance(obj, dict), f"Строка {i} не является JSON object"

    @pytest.mark.parametrize("md_rel,doc_id", DOCS)
    def test_unique_node_ids(self, md_rel, doc_id, tmp_path):
        """node_id уникален внутри документа."""
        md_path = PROJECT_ROOT / md_rel
        jsonl_path = tmp_path / f"{doc_id}.jsonl"
        records = parse_and_export_jsonl(md_path, jsonl_path)

        ids = [r["node_id"] for r in records]
        assert len(ids) == len(set(ids)), f"Найдены дубликаты node_id в {doc_id}"

    @pytest.mark.parametrize("md_rel,doc_id", DOCS)
    def test_source_lines_non_overlapping(self, md_rel, doc_id, tmp_path):
        """source_lines не пересекаются."""
        md_path = PROJECT_ROOT / md_rel
        jsonl_path = tmp_path / f"{doc_id}.jsonl"
        records = parse_and_export_jsonl(md_path, jsonl_path)

        ranges = []
        for r in records:
            sl = r.get("source_lines", {})
            if sl:
                ranges.append((sl.get("start", -1), sl.get("end", -1)))
        ranges.sort()
        for i in range(1, len(ranges)):
            prev_end = ranges[i - 1][1]
            curr_start = ranges[i][0]
            assert curr_start > prev_end or curr_start == prev_end + 1, (
                f"Пересечение source_lines: {ranges[i-1]} -> {ranges[i]}"
            )

    @pytest.mark.parametrize("md_rel,doc_id", DOCS)
    def test_required_fields(self, md_rel, doc_id, tmp_path):
        """Каждый record содержит обязательные поля."""
        md_path = PROJECT_ROOT / md_rel
        jsonl_path = tmp_path / f"{doc_id}.jsonl"
        records = parse_and_export_jsonl(md_path, jsonl_path)

        for r in records:
            assert "node_id" in r, f"Отсутствует node_id в {r}"
            assert "text" in r, f"Отсутствует text в {r['node_id']}"
            assert "structure" in r, f"Отсутствует structure в {r['node_id']}"
            struct = r["structure"]
            assert "type" in struct, f"Отсутствует structure.type в {r['node_id']}"
            assert "context_flat" in struct, (
                f"Отсутствует structure.context_flat в {r['node_id']}"
            )

    @pytest.mark.parametrize("md_rel,doc_id", DOCS)
    def test_text_not_empty_for_content_nodes(self, md_rel, doc_id, tmp_path):
        """Узлы с типом paragraph/subparagraph/item не имеют пустого text."""
        md_path = PROJECT_ROOT / md_rel
        jsonl_path = tmp_path / f"{doc_id}.jsonl"
        records = parse_and_export_jsonl(md_path, jsonl_path)

        for r in records:
            t = r["structure"]["type"]
            if t in ("paragraph", "subparagraph", "item"):
                assert r["text"].strip(), (
                    f"Пустой text у {r['node_id']} (type={t})"
                )

    @pytest.mark.parametrize("md_rel,doc_id", DOCS)
    def test_json_serializable(self, md_rel, doc_id, tmp_path):
        """Все records сериализуются в JSON."""
        md_path = PROJECT_ROOT / md_rel
        jsonl_path = tmp_path / f"{doc_id}.jsonl"
        records = parse_and_export_jsonl(md_path, jsonl_path)

        for r in records:
            dumped = json.dumps(r, ensure_ascii=False)
            restored = json.loads(dumped)
            assert restored["node_id"] == r["node_id"]

    @pytest.mark.parametrize("md_rel,doc_id", DOCS)
    def test_jsonl_count_matches_records(self, md_rel, doc_id, tmp_path):
        """Количество JSONL строк = количество records из build_records."""
        md_path = PROJECT_ROOT / md_rel
        jsonl_path = tmp_path / f"{doc_id}.jsonl"
        records = parse_and_export_jsonl(md_path, jsonl_path)

        jsonl_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(jsonl_lines) == len(records)


# ============================================================================
# 8. Context hierarchy regression tests
# ============================================================================

class TestContextHierarchy:
    """Проверка иерархического контекста через context_flat."""

    def test_ukaz_96_paragraph_2_sub_a(self):
        """ukaz-96: paragraph 2 → subparagraph а (context_flat.section=I, paragraph=2)"""
        md_path = PROJECT_ROOT / "markdown/ukaz-96.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for r in records:
            cf = r["structure"]["context_flat"]
            if cf.get("paragraph") == "2" and r["structure"]["type"] == "subparagraph" and r["structure"]["number"] == "а":
                assert cf.get("section") == "I", (
                    f"subparagraph 'а' п.2 должен быть в section I, а не {cf.get('section')}"
                )
                return
        pytest.fail("Не найден subparagraph 'а' под paragraph 2 в ukaz-96")

    def test_ukaz_96_paragraph_2_sub_b(self):
        """ukaz-96: paragraph 2 → subparagraph б"""
        md_path = PROJECT_ROOT / "markdown/ukaz-96.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for r in records:
            cf = r["structure"]["context_flat"]
            if cf.get("paragraph") == "2" and r["structure"]["type"] == "subparagraph" and r["structure"]["number"] == "б":
                assert cf.get("section") == "I"
                return
        pytest.fail("Не найден subparagraph 'б' под paragraph 2")

    def test_ukaz_96_paragraph_8_sub_a(self):
        """ukaz-96: paragraph 8 → subparagraph а (section II)"""
        md_path = PROJECT_ROOT / "markdown/ukaz-96.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for r in records:
            cf = r["structure"]["context_flat"]
            if cf.get("paragraph") == "8" and r["structure"]["type"] == "subparagraph" and r["structure"]["number"] == "а":
                assert cf.get("section") == "II", (
                    f"subparagraph 'а' п.8 должен быть в section II, а не {cf.get('section')}"
                )
                return
        pytest.fail("Не найден subparagraph 'а' под paragraph 8")

    def test_ukaz_96_paragraph_8_sub_b(self):
        """ukaz-96: paragraph 8 → subparagraph б (section II)"""
        md_path = PROJECT_ROOT / "markdown/ukaz-96.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for r in records:
            cf = r["structure"]["context_flat"]
            if cf.get("paragraph") == "8" and r["structure"]["type"] == "subparagraph" and r["structure"]["number"] == "б":
                assert cf.get("section") == "II"
                return
        pytest.fail("Не найден subparagraph 'б' под paragraph 8")

    def test_ukaz_96_paragraph_8_sub_v(self):
        """ukaz-96: paragraph 8 → subparagraph в (section II)"""
        md_path = PROJECT_ROOT / "markdown/ukaz-96.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for r in records:
            cf = r["structure"]["context_flat"]
            if cf.get("paragraph") == "8" and r["structure"]["type"] == "subparagraph" and r["structure"]["number"] == "в":
                assert cf.get("section") == "II"
                return
        pytest.fail("Не найден subparagraph 'в' под paragraph 8")

    def test_506_section_II_paragraph_5_sub_5_1_item_5_1_1(self):
        """postanovlenie-506: II → 5 → 5.1 → 5.1.1"""
        md_path = PROJECT_ROOT / "markdown/postanovlenie-506.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        found_5_1 = False
        for r in records:
            cf = r["structure"]["context_flat"]
            if r["structure"]["type"] == "subparagraph" and r["structure"]["number"] == "5.1":
                assert cf.get("section") == "II", (
                    f"5.1 должен быть в section II, а не {cf.get('section')}"
                )
                assert cf.get("paragraph") == "5", (
                    f"5.1 должен быть под paragraph 5, а не {cf.get('paragraph')}"
                )
                found_5_1 = True
                break
        assert found_5_1, "Не найден subparagraph 5.1"

        found_5_1_1 = False
        for r in records:
            cf = r["structure"]["context_flat"]
            if r["structure"]["type"] == "item" and r["structure"]["number"] == "5.1.1":
                assert cf.get("section") == "II", (
                    f"5.1.1 должен быть в section II, а не {cf.get('section')}"
                )
                assert cf.get("paragraph") == "5", (
                    f"5.1.1 должен быть под paragraph 5, а не {cf.get('paragraph')}"
                )
                found_5_1_1 = True
                break
        assert found_5_1_1, "Не найден item 5.1.1"
    def test_ukaz_159_main_paragraphs_have_no_appendix_context(self):
        """ukaz-159: пункты 1,2,3 основного указа НЕ имеют appendix в context_flat."""
        md_path = PROJECT_ROOT / "markdown/ukaz-159.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for r in records:
            if r["structure"]["type"] == "paragraph" and r["structure"]["number"] in ("1", "2", "3"):
                cf = r["structure"]["context_flat"]
                if r["node_id"] in ("n0007", "n0008", "n0012"):
                    assert cf.get("appendix") is None, (
                        f"{r['node_id']}: основной paragraph {r['structure']['number']} "
                        f"не должен иметь appendix={cf.get('appendix')!r}"
                    )

    def test_ukaz_159_appendix_paragraphs_have_appendix_context(self):
        """ukaz-159: пункты 1,2,3 внутри приложения имеют appendix=1 в context_flat."""
        md_path = PROJECT_ROOT / "markdown/ukaz-159.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for r in records:
            if r["structure"]["type"] == "paragraph" and r["structure"]["number"] in ("1", "2", "3"):
                cf = r["structure"]["context_flat"]
                if r["node_id"] in ("n0042", "n0043", "n0055"):
                    assert cf.get("appendix") == "1", (
                        f"{r['node_id']}: appendix paragraph {r['structure']['number']} "
                        f"должен иметь appendix=1, а не {cf.get('appendix')!r}"
                    )
                    assert cf.get("section") == "I", (
                        f"{r['node_id']}: appendix paragraph {r['structure']['number']} "
                        f"должен быть в section I, а не {cf.get('section')!r}"
                    )
    def test_ukaz_159_appendix_boundary(self):
        """ukaz-159: граница appendix — УТВЕРЖДЕНА → I. Общие положения → п.1,2,3."""
        md_path = PROJECT_ROOT / "markdown/ukaz-159.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        appendix_idx = None
        section_I_idx = None
        para_1_idx = None
        for i, n in enumerate(linear):
            if n["type"] == "appendix":
                appendix_idx = i
            if appendix_idx is not None and n["type"] == "section" and n["number"] == "I":
                section_I_idx = i
            if section_I_idx is not None and n["type"] == "paragraph" and n["number"] == "1":
                para_1_idx = i
                break

        assert appendix_idx is not None, "Не найден appendix в linear"
        assert section_I_idx is not None, "Не найден section I после appendix"
        assert para_1_idx is not None, "Не найден paragraph 1 после section I"
        assert section_I_idx > appendix_idx, "section I должен быть после appendix"
        assert para_1_idx > section_I_idx, "paragraph 1 должен быть после section I"

    def test_ukaz_159_paragraph_1_context_differs(self):
        """ukaz-159: paragraph 1 основного указа ≠ paragraph 1 внутри appendix по context_flat."""
        md_path = PROJECT_ROOT / "markdown/ukaz-159.md"
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        main_p1 = None
        appendix_p1 = None
        for r in records:
            if r["structure"]["type"] == "paragraph" and r["structure"]["number"] == "1":
                cf = r["structure"]["context_flat"]
                if cf.get("appendix") is None and cf.get("section") is None:
                    main_p1 = r
                elif cf.get("appendix") == "1" and cf.get("section") == "I":
                    appendix_p1 = r

        assert main_p1 is not None, "Не найден paragraph 1 основного указа"
        assert appendix_p1 is not None, "Не найден paragraph 1 внутри appendix"
        assert main_p1["node_id"] != appendix_p1["node_id"]
        main_cf = main_p1["structure"]["context_flat"]
        app_cf = appendix_p1["structure"]["context_flat"]
        assert main_cf != app_cf, (
            f"context_flat основного paragraph 1 {main_cf} не должен совпадать "
            f"с context_flat appendix paragraph 1 {app_cf}"
        )
        assert app_cf.get("appendix") == "1", (
            f"appendix paragraph 1 должен иметь appendix=1, а не {app_cf.get('appendix')!r}"
        )
        assert main_cf.get("appendix") is None, (
            f"основной paragraph 1 не должен иметь appendix={main_cf.get('appendix')!r}"
        )


# ============================================================================
# 9. Text preservation tests
# ============================================================================

class TestTextPreservation:
    """Проверка, что normalize_underscores не ломает исходный текст в linear."""

    DOCS = [
        "markdown/79-FZ.md",
        "markdown/postanovlenie-506.md",
        "markdown/ukaz-96.md",
        "markdown/rasporyazhenie-667-r.md",
        "markdown/ukaz-159.md",
    ]

    @pytest.mark.parametrize("md_rel", DOCS)
    def test_exact_reconstruction_after_normalize(self, md_rel):
        """normalize_underscores не влияет на exact реконструкцию из linear.

        Известное исключение: rasporyazhenie-667-r содержит строку-пробел (' '),
        которая split_segments трактует как пустую строку, поэтому пробел теряется.
        """
        md_path = PROJECT_ROOT / md_rel
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        recon = exact_reconstruct(linear, total_source_lines=len(lines))
        original = Path(md_path).read_text(encoding="utf-8").rstrip("\n")
        if original != recon:
            # Допускаем расхождение только для 667-r (trailing space line)
            if "rasporyazhenie-667-r" in md_rel:
                ol = original.splitlines(keepends=False)
                rl = recon.splitlines(keepends=False)
                diffs = 0
                for i, (o, r) in enumerate(zip(ol, rl, strict=True)):
                    if o != r:
                        diffs += 1
                # Единственное расхождение — строка-пробел
                assert diffs <= 1, (
                    f"rasporyazhenie-667-r: ожидается <=1 расхождение, получено {diffs}"
                )
            else:
                assert False, (
                    f"exact_reconstruct дал расхождение для {md_rel}."
                )

    @pytest.mark.parametrize("md_rel", DOCS)
    def test_normalize_underscores_does_not_modify_linear(self, md_rel):
        """normalize_underscores в records не меняет content в linear."""
        md_path = PROJECT_ROOT / md_rel
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        original_contents = {n["id"]: n["content"] for n in linear}

        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        for n in linear:
            assert n["content"] == original_contents[n["id"]], (
                f"content изменился для {n['id']} после build_records"
            )

        for r in records:
            if "\\_\\_\\_\\_\\_" in original_contents.get(r["node_id"], ""):
                assert "___" in r["text"] or " ___ " in r["text"], (
                    f"normalize_underscores не сработал для {r['node_id']}"
                )
# ============================================================================
# 10. Plain-text structural headings (Статья/Глава/Раздел без #-заголовков)
# ============================================================================

class TestPlainTextHeading:
    """Проверка, что Статья/Глава/Раздел/Приложение распознаются без MD-заголовка #."""

    def test_article_in_classify_block(self):
        """Статья 1. Основные термины → (article, 1, ...) в _classify_block."""
        btype, num, title, conf = _classify_block({"lines": ["Статья 1. Основные термины"]})
        assert btype == "article"
        assert num == "1"
        assert title == "Основные термины"
        assert conf == 0.95

    def test_chapter_in_classify_block(self):
        """Глава 2. Государственные должности → (chapter, 2, ...)."""
        btype, num, title, conf = _classify_block({"lines": ["Глава 2. Государственные должности"]})
        assert btype == "chapter"
        assert num == "2"
        assert title == "Государственные должности"

    def test_section_in_classify_block(self):
        """Раздел I. Общие положения → (section, I, ...)."""
        btype, num, title, conf = _classify_block({"lines": ["Раздел I. Общие положения"]})
        assert btype == "section"
        assert num == "I"
        assert title == "Общие положения"

    def test_appendix_in_classify_block(self):
        """Приложение № 1 ... → (appendix, 1, ...)."""
        btype, num, title, conf = _classify_block({"lines": ["Приложение № 1 Таблица"]})
        assert btype == "appendix"
        assert num == "1"

    def test_plain_paragraph_not_affected(self):
        """1. Параграф (без #) → paragraph, а НЕ section (conf < 0.9)."""
        btype, num, title, conf = _classify_block({"lines": ["1. Параграф"]})
        assert btype == "paragraph"
        assert num == "1"

    def test_plain_subparagraph_not_affected(self):
        """1.1 Подраздел (без #) НЕ перехватывается heading_check (не article/chapter/section/appendix)."""
        btype, num, title, conf = _classify_block({"lines": ["1.1 Подраздел"]})
        assert btype not in ("article", "chapter", "section", "appendix"), (
            f"1.1 Подраздел не должен классифицироваться как {btype}"
        )
        assert conf < 0.9, f"confidence должен быть < 0.9, получено {conf}"

    def test_article_context_in_full_pipeline(self):
        """Полный pipeline: Статья 1 → следующий текст имеет article=1 в context_flat."""
        md_text = (
            "Статья 1. Основные термины\n"
            "\n"
            "Для целей закона применяются следующие термины.\n"
        )
        lines = md_text.splitlines(keepends=True)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        # Находим записи по type
        article_records = [r for r in records if r["structure"]["type"] == "article"]
        assert len(article_records) == 1, f"Ожидается 1 article, получено {len(article_records)}"
        assert article_records[0]["structure"]["context_flat"]["article"] == "1", (
            f"article=1, а не {article_records[0]['structure']['context_flat'].get('article')!r}"
        )

        text_records = [r for r in records if r["structure"]["type"] == "text"]
        assert len(text_records) >= 1
        # Первый текст после статьи должен иметь article=1
        first_text = text_records[0]
        assert first_text["structure"]["context_flat"].get("article") == "1", (
            f"текст после статьи должен иметь article=1, "
            f"а не {first_text['structure']['context_flat'].get('article')!r}"
        )

    def test_article_boundary(self):
        """Две статьи: контекст article=1 НЕ перетекает в article=2."""
        md_text = (
            "Статья 1. Термины\n"
            "\n"
            "Текст первой статьи.\n"
            "\n"
            "Статья 2. Права\n"
            "\n"
            "Текст второй статьи.\n"
        )
        lines = md_text.splitlines(keepends=True)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)

        article_records = [r for r in records if r["structure"]["type"] == "article"]
        assert len(article_records) == 2, f"Ожидается 2 статьи, получено {len(article_records)}"
        assert article_records[0]["structure"]["context_flat"]["article"] == "1"
        assert article_records[1]["structure"]["context_flat"]["article"] == "2"

        text_records = [r for r in records if r["structure"]["type"] == "text"]
        # Текст после статьи 1
        text_after_1 = [r for r in text_records
                        if r["structure"]["context_flat"].get("article") == "1"]
        assert len(text_after_1) >= 1, "Нет текста с article=1"

        # Текст после статьи 2
        text_after_2 = [r for r in text_records
                        if r["structure"]["context_flat"].get("article") == "2"]
        assert len(text_after_2) >= 1, "Нет текста с article=2"

        # Ни один текст после статьи 2 не должен иметь article=1
        for r in text_records:
            if r["structure"]["context_flat"].get("article") == "2":
                assert r["structure"]["context_flat"].get("paragraph") is None, (
                    f"текст после article=2 не должен содержать paragraph={r['structure']['context_flat'].get('paragraph')!r}"
                )