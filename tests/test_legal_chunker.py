"""Тесты для legal_chunker.

Standalone — не требуют Qdrant, Ollama, RAG, Docling. Только чанкер + structure JSON.
"""
import json
from pathlib import Path

from app.chunking.legal_chunker import (
    _build_chunks_for_segment,
    _clean_header,
    _group_segments,
    _is_editorial,
    _pack_blocks,
    _split_oversized,
    count_tokens,
    convert,
    process_markdown_file,
    batch_convert,
    validate_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRUCTURE_79FZ = PROJECT_ROOT / "structure" / "79-FZ.json"
REQUIRED_KEYS = {"id", "title", "text", "local_img", "url"}


def _rec(node_id, rtype, num, article, text, chapter=None, appendix=None, section=None, paragraph=None):
    return {
        "node_id": node_id,
        "text": text,
        "structure": {
            "type": rtype,
            "number": num,
            "context_flat": {
                "chapter": chapter,
                "section": section,
                "subsection": None,
                "article": article,
                "paragraph": paragraph,
                "item": None,
                "appendix": appendix,
            },
        },
    }


# ============================================================================
# Unit-тесты вспомогательных функций
# ============================================================================

class TestCountTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_non_empty(self):
        assert count_tokens("привет мир") >= 1


class TestCleanHeader:
    def test_strips_md_heading(self):
        assert _clean_header("## Статья 1. Основные термины") == "Статья 1. Основные термины"

    def test_collapses_whitespace(self):
        assert _clean_header("  Статья   1.   Термины  ") == "Статья 1. Термины"


class TestIsEditorial:
    def test_revision_block(self):
        assert _is_editorial("(В редакции федеральных законов от 02.02.2006 № 19-ФЗ, от ...)")

    def test_normal_text(self):
        assert not _is_editorial("Для целей настоящего закона термин означает:")


class TestSplitOversized:
    def test_respects_limit(self):
        text = "АААА. ББББ. ВВВВ. ГГГГ. ДДДД. ЕЕЕЕ. ЖЖЖЖ. ЗЗЗЗ."
        parts = _split_oversized(text, limit=8)
        assert len(parts) >= 2
        for p in parts:
            assert count_tokens(p) <= 8

    def test_splits_oversized_sentence(self):
        """Одно длинное предложение без разрывов — проверяет bulk-нарезку по токенам."""
        text = "слово " * 1000
        parts = _split_oversized(text, limit=200)
        assert len(parts) >= 2
        for p in parts:
            assert count_tokens(p) <= 200
        assert all(p.strip() for p in parts)

class TestPackBlocks:
    def test_never_exceeds_max(self):
        prefix_tokens = count_tokens("[79-ФЗ] [Статья 1. Термины]")
        blocks = [("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" * 3, False) for _ in range(50)]
        parts = _pack_blocks(blocks, prefix_tokens, max_tokens=400, target_tokens=350,
                             min_tokens=40)
        for p_text, p_indices in parts:
            assert prefix_tokens + count_tokens(p_text) <= 400

    def test_editorial_block_capped_not_split(self):
        prefix_tokens = count_tokens("[79-ФЗ] [Преамбула]")
        # большой редакционный блок, который не должен стать множеством чанков
        big_ed = "(В редакции федеральных законов от 01.01.2000 № 1-ФЗ, " + \
            "от 02.02.2001 № 2-ФЗ, " * 200 + ")"
        parts = _pack_blocks([(big_ed, True)], prefix_tokens, max_tokens=400,
                             target_tokens=350, min_tokens=40)
        assert len(parts) == 1  # ровно один чанк, а не множество
        # _pack_blocks гарантирует body-часть в пределах max_tokens - prefix
        assert count_tokens(parts[0][0]) <= 400 - prefix_tokens


# ============================================================================
# Группировка records в сегменты
# ============================================================================

class TestGroupSegments:
    def test_preamble_then_articles(self):
        records = [
            _rec("n0", "text", None, None, "Федеральный закон № 79-ФЗ"),
            _rec("n2", "article", "1", "1", "## Статья 1. Термины", chapter="1"),
            _rec("n3", "item", "1", "1", "1) термин один", chapter="1"),
            _rec("n4", "article", "2", "2", "## Статья 2. Предмет", chapter="1"),
            _rec("n5", "paragraph", "1", "2", "1. Предмет регулирования", chapter="1"),
        ]
        segs = _group_segments(records)
        assert [s["type"] for s in segs] == ["preamble", "article", "article"]
        assert segs[1]["title"] == "Статья 1. Термины"
        assert segs[1]["article"] == "1"
        # заголовок статьи НЕ попадает в body
        assert segs[1]["body"][0]["node_id"] == "n3"


class TestBuildChunksForSegment:
    def test_article_schema(self):
        records = [
            _rec("n1", "text", None, "1", "Для целей настоящего закона термин означает:", chapter="1"),
            _rec("n2", "item", "1", "1", "1) государственная должность", chapter="1"),
        ]
        seg = {
            "type": "article", "number": "1", "title": "Статья 1. Термины",
            "article": "1", "appendix": None, "body": records,
        }
        chunks = _build_chunks_for_segment(
            seg, "79-фз", "79-ФЗ", "raw/79-FZ.pdf", 400, 350, 40)
        assert len(chunks) == 1
# ============================================================================
# Новые тесты: структурная сегментация для НПА без статей
# ============================================================================

class TestGroupSegmentsNoArticles:
    """Test 1: Документ без article — paragraph-контекст сохраняется."""

    def test_paragraphs_grouped_in_preamble(self):
        """paragraph 1, paragraph 2, paragraph 3 → один сегмент с Пункт 1."""
        records = [
            _rec("n0", "text", None, None, "В соответствии с законом № 79-ФЗ"),
            _rec("n1", "paragraph", "1", None, "1. Утвердить прилагаемую форму", paragraph="1"),
            _rec("n2", "paragraph", "2", None, "2. Руководителям обеспечить", paragraph="2"),
            _rec("n3", "text", None, None, "руководствоваться утвержденной формой", paragraph="2"),
            _rec("n4", "paragraph", "3", None, "3. Настоящий Указ вступает в силу", paragraph="3"),
        ]
        segs = _group_segments(records)
        assert len(segs) == 1, f"Expected 1 segment, got {len(segs)}"
        assert segs[0]["title"] == "Пункт 1", f"Expected Пункт 1, got {segs[0]['title']!r}"
        assert len(segs[0]["body"]) == 5, f"Expected 5 body records, got {len(segs[0]['body'])}"

    def test_paragraph_text_inheritance(self):
        """Test 2: text records наследуют paragraph-контекст."""
        records = [
            _rec("n0", "paragraph", "2", None, "2. Руководителям обеспечить", paragraph="2"),
            _rec("n1", "text", None, None, "руководствоваться утвержденной формой", paragraph="2"),
            _rec("n2", "text", None, None, "обеспечить переоформление", paragraph="2"),
        ]
        segs = _group_segments(records)
        assert len(segs) == 1
        # Все три записи в одном сегменте
        assert len(segs[0]["body"]) == 3
        # Заголовок сегмента — Пункт 2 (первый paragraph в теле)
        assert segs[0]["title"] == "Пункт 2"
class TestGroupSegmentsWithAppendix:
    """Test 3: Appendix не смешивается с основным документом."""

    def test_appendix_separated_from_main(self):
        """документ → paragraph → appendix → section → paragraph → не смешиваются."""
        records = [
            _rec("n0", "text", None, None, "В соответствии с законом № 79-ФЗ"),
            _rec("n1", "paragraph", "1", None, "1. Утвердить форму", paragraph="1"),
            _rec("n2", "appendix", None, None, "УТВЕРЖДЕНА", appendix="1"),
            _rec("n3", "text", None, None, "ПРИМЕРНАЯ ФОРМА", appendix="1"),
            _rec("n4", "paragraph", "1", None, "1. По настоящему контракту", appendix="1",
                 section="I", paragraph="1"),
            _rec("n5", "paragraph", "2", None, "2. Гражданский служащий обязуется", appendix="1",
                 section="I", paragraph="2"),
        ]
        segs = _group_segments(records)
        assert len(segs) >= 2, f"Expected at least 2 segments, got {len(segs)}"

        # Main document segment
        main_seg = segs[0]
        main_ids = {b["node_id"] for b in main_seg["body"]}
        assert "n0" in main_ids, "Pre-article text should be in main segment"
        assert "n1" in main_ids, "Paragraph 1 should be in main segment"
        assert "n2" not in main_ids, "Appendix marker should NOT be in main segment"

        # Appendix segment
        app_seg = segs[1]
        assert app_seg["type"] == "appendix", f"Expected appendix type, got {app_seg['type']!r}"
        app_ids = {b["node_id"] for b in app_seg["body"]}
        assert "n3" in app_ids, "Appendix text should be in appendix segment"
        assert "n1" not in app_ids, "Main paragraph should NOT be in appendix segment"

    def test_section_detection_within_appendix(self):
        """section внутри appendix → отдельный сегмент для каждого раздела."""
        records = [
            _rec("n0", "appendix", None, None, "УТВЕРЖДЕНА", appendix="1"),
            _rec("na", "text", None, None, "ПРИМЕРНАЯ ФОРМА", appendix="1"),
            _rec("n1", "paragraph", "1", None, "1. Первый пункт", appendix="1",
                 section="I", paragraph="1"),
            _rec("n2", "paragraph", "2", None, "2. Второй пункт", appendix="1",
                 section="I", paragraph="2"),
            _rec("n3", "paragraph", "5", None, "5. Права и обязанности", appendix="1",
                 section="II", paragraph="5"),
            _rec("n4", "text", None, None, "Гражданский служащий имеет право", appendix="1",
                 section="II", paragraph="5"),
        ]
        segs = _group_segments(records)
        # Должно быть: appendix + section I + section II = 3 segments
        assert len(segs) == 3, f"Expected 3 segments, got {len(segs)}: {[(s['type'], s['title']) for s in segs]}"

        # Appendix segment (marker heading + body text)
        app = segs[0]
        assert app["type"] == "appendix", f"Expected appendix, got {app['type']!r}"
        app_ids = {b["node_id"] for b in app["body"]}
        assert "na" in app_ids, "Appendix body text should be in appendix"
        assert "n1" not in app_ids, "Section paragraph should NOT be in appendix"

        # Section I
        sec1 = segs[1]
        assert sec1["type"] == "section", f"Expected section, got {sec1['type']!r}"
        assert sec1["title"] == "Раздел I", f"Expected Раздел I, got {sec1['title']!r}"
        sec1_ids = {b["node_id"] for b in sec1["body"]}
        assert "n1" in sec1_ids, "Paragraph 1 should be in section I"
        assert "n2" in sec1_ids, "Paragraph 2 should be in section I"

        # Section II
        sec2 = segs[2]
        assert sec2["type"] == "section", f"Expected section, got {sec2['type']!r}"
        assert sec2["title"] == "Раздел II", f"Expected Раздел II, got {sec2['title']!r}"
        sec2_ids = {b["node_id"] for b in sec2["body"]}
        assert "n3" in sec2_ids, "Paragraph 5 should be in section II"
        assert "n4" in sec2_ids, "Text should be in section II"


class TestGroupSegments79FZ:
    """Test 4: 79-FZ — существующее поведение не изменилось."""

    def test_articles_still_work(self):
        """article-записи по-прежнему создают отдельные сегменты."""
        records = [
            _rec("n0", "text", None, None, "Федеральный закон № 79-ФЗ"),
            _rec("n2", "article", "1", "1", "## Статья 1. Термины", chapter="1"),
            _rec("n3", "item", "1", "1", "1) термин один", chapter="1"),
            _rec("n4", "article", "2", "2", "## Статья 2. Предмет", chapter="1"),
            _rec("n5", "paragraph", "1", "2", "1. Предмет регулирования", chapter="1"),
        ]
        segs = _group_segments(records)
        assert [s["type"] for s in segs] == ["preamble", "article", "article"]
class TestBuildChunksForSegmentDynamicTitle:
    """Test 1-2: Динамический заголовок для paragraph-контекста."""

    def test_preamble_segment_gets_preamble_title(self):
        """Сегмент без paragraph-контекста → title = Преамбула."""
        records = [
            _rec("n0", "text", None, None, "В соответствии с законом № 79-ФЗ"),
        ]
        segs = _group_segments(records)
        assert len(segs) == 1
        assert segs[0]["title"] == "Преамбула"

    def test_paragraph_context_used_for_title(self):
        """Первый paragraph в сегменте определяет заголовок."""
        records = [
            _rec("n0", "text", None, None, "В соответствии с законом"),
            _rec("n1", "paragraph", "1", None, "1. Утвердить форму", paragraph="1"),
            _rec("n2", "paragraph", "2", None, "2. Руководителям обеспечить", paragraph="2"),
        ]
        segs = _group_segments(records)
        assert segs[0]["title"] == "Пункт 1", f"Expected Пункт 1, got {segs[0]['title']!r}"


class TestGroupSegmentsUkaz159:
    """Test 5: ukaz-159 — полная структурная проверка."""

    def test_ukaz159_structure(self):
        """Проверить, что ukaz-159 правильно разбивается на сегменты."""
        from app.ingestion.markdown_structure_parser import (
            build_context_recursive, build_linear, build_records,
            build_tree, parse_blocks, read_markdown,
        )
        md_path = PROJECT_ROOT / "markdown" / "ukaz-159.md"
        if not md_path.exists():
            import pytest
            pytest.skip("ukaz-159.md не найден")
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)
        segs = _group_segments(records)
        # Должно быть: main body + appendix + 10 sections = 12 segments
        assert len(segs) == 12, f"Expected 12 segments, got {len(segs)}"
        # Seg 0: main body (pre-paragraph text + paragraphs 1-3)
        s0 = segs[0]
        assert s0["type"] in ("preamble",), f"Expected preamble, got {s0['type']!r}"
        s0_ids = {b["node_id"] for b in s0["body"]}
        assert "n0007" in s0_ids, "Main paragraph 1 should be in seg 0"
        assert "n0008" in s0_ids, "Main paragraph 2 should be in seg 0"
        assert "n0012" in s0_ids, "Main paragraph 3 should be in seg 0"
        # Seg 1: appendix
        s1 = segs[1]
        assert s1["type"] == "appendix", f"Expected appendix, got {s1['type']!r}"
        assert s1["appendix"] == "1", "Should have appendix=1"
        s1_ids = {b["node_id"] for b in s1["body"]}
        assert "n0018" in s1_ids, "ПРИМЕРНАЯ ФОРМА should be in appendix"
        assert "n0007" not in s1_ids, "Main paragraph should NOT be in appendix"
        # Seg 2-11: sections I-X
        roman_list = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        for i, seg in enumerate(segs[2:], 1):
            roman = roman_list[i - 1]
            assert seg["type"] == "section", f"Section {i} should be type=section, got {seg['type']!r}"
            assert seg["title"] == f"Раздел {roman}", f"Expected Раздел {roman}, got {seg['title']!r}"
            assert seg["appendix"] == "1", f"Section should have appendix=1"
            first = seg["body"][0]
            cf = first["structure"]["context_flat"]
            assert cf.get("section") == roman, f"First body record should have section={roman}, got {cf.get('section')!r}"

    def test_ukaz159_appendix_isolated(self):
        """Appendix в ukaz-159 не смешивается с основным документом."""
        from app.ingestion.markdown_structure_parser import (
            build_context_recursive, build_linear, build_records,
            build_tree, parse_blocks, read_markdown,
        )
        md_path = PROJECT_ROOT / "markdown" / "ukaz-159.md"
        if not md_path.exists():
            import pytest
            pytest.skip("ukaz-159.md не найден")
        lines = read_markdown(md_path)
        blocks = parse_blocks(lines)
        linear = build_linear(blocks)
        tree = build_tree(linear)
        build_context_recursive(tree)
        records = build_records(tree)
        segs = _group_segments(records)
        main_ids = {b["node_id"] for b in segs[0]["body"]}
        appendix_records = {r["node_id"] for r in records
                           if r["structure"]["context_flat"].get("appendix")}
        appendix_records.discard("n0017")
        overlapping = main_ids & appendix_records
        assert not overlapping, f"Appendix records in main segment: {overlapping}"
        section_ids = set()
        for seg in segs[2:]:
            section_ids.update(b["node_id"] for b in seg["body"])
        overlapping_sec = main_ids & section_ids
        assert not overlapping_sec, f"Section records in main segment: {overlapping_sec}"
# ============================================================================
# Интеграционный тест на реальном structure 79-ФЗ (если файл присутствует)
# ============================================================================

class TestConvert79FZ:
    def test_convert_and_validate(self, tmp_path):
        if not STRUCTURE_79FZ.exists():
            import pytest
            pytest.skip("structure/79-FZ.json отсутствует")
        out = convert("79-FZ.json", out_dir=tmp_path)
        assert out.exists()
        errors, count, tokens = validate_jsonl(out)
        assert errors == []
        assert count > 0
        assert max(tokens) <= 400

        chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        # url == raw/79-FZ.pdf во всех чанках
        assert all(c["url"] == "raw/79-FZ.pdf" for c in chunks)

    def test_articles_not_mixed(self, tmp_path):
        if not STRUCTURE_79FZ.exists():
            import pytest
            pytest.skip("structure/79-FZ.json отсутствует")
        out = convert("79-FZ.json", out_dir=tmp_path)
        chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        # каждый чанк-статья содержит ровно свой заголовок статьи в префиксе
        for c in chunks:
            title = c["title"]
            if title.startswith("Статья "):
                assert c["text"].startswith("[79-ФЗ] [" + title + "]")

    def test_every_chunk_text_within_400(self, tmp_path):
        """Финальный text каждого чанка (ровно как в JSONL) <= 400 токенов FRIDA."""
        if not STRUCTURE_79FZ.exists():
            import pytest
            pytest.skip("structure/79-FZ.json отсутствует")
        out = convert("79-FZ.json", out_dir=tmp_path)
        chunks = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
        for c in chunks:
            n = count_tokens(c["text"])
            # поле text уже содержит префикс [79-ФЗ] [Статья N. …]; считаем его целиком,
            # НЕ добавляя префикс повторно
            assert n <= 400, f"chunk {c['id']}: {n} токенов > 400"
# ============================================================================
# Тесты пакетного режима (batch_convert / process_markdown_file)
# ============================================================================

STRUCTURE_DIR = PROJECT_ROOT / "structure"


class TestBatchConvert:
    def test_multiple_markdowns(self, tmp_path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "79-FZ.md").touch()
        (md_dir / "58-FZ.md").touch()

        out_dir = tmp_path / "chunks"

        results = batch_convert(
            markdown_dir=md_dir,
            structure_dir=STRUCTURE_DIR,
            out_dir=out_dir,
        )

        assert len(results) == 2
        assert (out_dir / "79-FZ.jsonl").exists()
        assert (out_dir / "58-FZ.jsonl").exists()

        # Каждый JSONL непустой и содержит правильные поля
        for r in results:
            errors, count, _ = validate_jsonl(r)
            assert errors == []
            assert count > 0

    def test_existing_jsonl_replaced(self, tmp_path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "79-FZ.md").touch()

        out_dir = tmp_path / "chunks"
        out_dir.mkdir()
        old = out_dir / "79-FZ.jsonl"
        old.write_text('{"old":"data"}\n')

        results = batch_convert(
            markdown_dir=md_dir,
            structure_dir=STRUCTURE_DIR,
            out_dir=out_dir,
        )

        assert len(results) == 1
        content = old.read_text(encoding="utf-8")
        assert '"old"' not in content
        assert '"id"' in content

    def test_one_error_does_not_stop_all(self, tmp_path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "79-FZ.md").touch()
        (md_dir / "nonexistent.md").touch()

        out_dir = tmp_path / "chunks"

        results = batch_convert(
            markdown_dir=md_dir,
            structure_dir=STRUCTURE_DIR,
            out_dir=out_dir,
        )

        # 79-FZ должен обработаться, nonexistent — пропущен с ошибкой
        assert len(results) >= 1
        assert (out_dir / "79-FZ.jsonl").exists()

    def test_empty_markdown_dir(self, tmp_path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        out_dir = tmp_path / "chunks"

        results = batch_convert(
            markdown_dir=md_dir,
            structure_dir=STRUCTURE_DIR,
            out_dir=out_dir,
        )

        assert results == []

    def test_token_limit_preserved(self, tmp_path):
        md_dir = tmp_path / "markdown"
        md_dir.mkdir()
        (md_dir / "79-FZ.md").touch()

        out_dir = tmp_path / "chunks"

        batch_convert(
            markdown_dir=md_dir,
            structure_dir=STRUCTURE_DIR,
            out_dir=out_dir,
        )

        errors, count, tokens = validate_jsonl(out_dir / "79-FZ.jsonl")
        assert errors == []
        assert max(tokens) <= 400

    def test_process_markdown_file_ok(self, tmp_path):
        md_path = tmp_path / "79-FZ.md"
        md_path.touch()
        out_dir = tmp_path / "chunks"

        stats = process_markdown_file(
            md_path,
            structure_dir=STRUCTURE_DIR,
            out_dir=out_dir,
        )

        assert stats["file"] == "79-FZ.md"
        assert stats["chunks"] > 0
        assert stats["max_tokens"] <= 400
        assert stats["out_path"].exists()

    def test_process_markdown_file_missing_structure(self, tmp_path):
        md_path = tmp_path / "no_such_file.md"
        md_path.touch()

        import pytest
        with pytest.raises(FileNotFoundError):
            process_markdown_file(md_path, structure_dir=STRUCTURE_DIR)
