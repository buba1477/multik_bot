"""Tests for retrieval utility functions (app.rag.retrieval_utils).

Standalone — no heavy dependencies (Qdrant, Ollama, models).
"""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.rag.retrieval_utils import (
    multi_part_boost,
    reconstruct_article_context,
    extract_article_prefix,
    is_enumeration_query,
    part_number,
    get_document_id,
    MAX_SEQUENTIAL_PARTS,
    DEFAULT_SCORE,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _node(node_id, text="content", metadata=None, score=0.05):
    meta = metadata or {}
    node = TextNode(text=text, id_=node_id, metadata=meta)
    return NodeWithScore(node=node, score=score)


def _make_node_map(prefix="117-fz_st219", count=10):
    node_map = {}
    for i in range(1, count + 1):
        n = TextNode(text=f"chunk p{i}", id_=f"{prefix}_p{i}",
                     metadata={"document_id": "None"})
        node_map[f"{prefix}_p{i}"] = n
    return node_map


# ===========================================================================
# extract_article_prefix
# ===========================================================================

class TestExtractArticlePrefix:
    def test_basic(self):
        assert extract_article_prefix("117-fz_st219_p1") == "117-fz_st219_"

    def test_multipart_id(self):
        assert extract_article_prefix("117-fz_st219_1_p1") == "117-fz_st219_1_"

    def test_large_part_number(self):
        assert extract_article_prefix("117-fz_st171_p13") == "117-fz_st171_"

    def test_no_match(self):
        assert extract_article_prefix("some_random_id") is None

    def test_no_match_without_part(self):
        assert extract_article_prefix("117-fz_st219") is None

    def test_separate_articles(self):
        p219 = extract_article_prefix("117-fz_st219_p1")
        p219_1 = extract_article_prefix("117-fz_st219_1_p1")
        assert p219 == "117-fz_st219_"
        assert p219_1 == "117-fz_st219_1_"
        assert p219 != p219_1


# ===========================================================================
# is_enumeration_query
# ===========================================================================

class TestIsEnumerationQuery:
    def test_all_patterns(self):
        for q in ["\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438",
                   "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438 \u0432\u0441\u0435",
                   "\u043a\u0430\u043a\u0438\u0435 \u0432\u0438\u0434\u044b",
                   "\u043a\u0430\u043a\u0438\u0435 \u0431\u044b\u0432\u0430\u044e\u0442",
                   "\u043d\u0430\u0437\u043e\u0432\u0438 \u0432\u0441\u0435",
                   "\u0443\u043a\u0430\u0436\u0438 \u0432\u0441\u0435",
                   "\u043f\u043e\u043b\u043d\u044b\u0439 \u043f\u0435\u0440\u0435\u0447\u0435\u043d\u044c"]:
            assert is_enumeration_query(q), f"Should detect: {q!r}"

    def test_negative(self):
        for q in ["\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435",
                   "\u043a\u0430\u043a \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c",
                   "\u0440\u0430\u0437\u043c\u0435\u0440 \u0432\u044b\u0447\u0435\u0442\u0430",
                   "\u043c\u0430\u043a\u0441\u0438\u043c\u0430\u043b\u044c\u043d\u0430\u044f \u0441\u0443\u043c\u043c\u0430"]:
            assert not is_enumeration_query(q), f"Should NOT detect: {q!r}"

    def test_with_extra_text(self):
        assert is_enumeration_query("\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438 \u0441\u043e\u0446\u0438\u0430\u043b\u044c\u043d\u044b\u0435 \u043d\u0430\u043b\u043e\u0433\u043e\u0432\u044b\u0435 \u0432\u044b\u0447\u0435\u0442\u044b")

    def test_case_insensitive(self):
        assert is_enumeration_query("\u041f\u0415\u0420\u0415\u0427\u0418\u0421\u041b\u0418")


# ===========================================================================
# part_number
# ===========================================================================

class TestPartNumber:
    def test_simple(self):
        assert part_number("117-fz_st219_p3") == 3

    def test_double_digit(self):
        assert part_number("117-fz_st219_p10") == 10

    def test_no_match(self):
        assert part_number("random_id") == 0


# ===========================================================================
# get_document_id
# ===========================================================================

class TestGetDocumentId:
    def test_none_string(self):
        assert get_document_id({"document_id": "None"}) == ""

    def test_empty(self):
        assert get_document_id({}) == ""

    def test_valid(self):
        assert get_document_id({"document_id": "doc1"}) == "doc1"


# ===========================================================================
# multi_part_boost -- fallback (legacy chunks without document_id)
# ===========================================================================

class TestMultiPartBoostFallback:
    def test_boost_applies_to_group(self):
        nodes = [
            _node("117-fz_st219_p1", score=0.04, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.04, metadata={"document_id": "None"}),
            _node("117-fz_st219_p3", score=0.50, metadata={"document_id": "None"}),
        ]
        result = multi_part_boost(nodes, top_n=10)
        p1 = next(r for r in result if r.node.node_id == "117-fz_st219_p1")
        p2 = next(r for r in result if r.node.node_id == "117-fz_st219_p2")
        assert p1.score == pytest.approx(0.04 + 0.03, rel=1e-5)
        assert p2.score == pytest.approx(0.04 + 0.03, rel=1e-5)

    def test_no_boost_for_single_chunk(self):
        node = _node("117-fz_st219_p3", score=0.5, metadata={"document_id": "None"})
        result = multi_part_boost([node])
        assert result[0].score == 0.5

    def test_boost_only_for_top_n_group(self):
        """Группа не в топ-N — буст не применяется (top_n=1, все 3 части с низким скором)."""
        nodes = [
            _node("117-fz_st219_p1", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st219_p3", score=0.01, metadata={"document_id": "None"}),
        ]
        # top_n=1: только верхний чанк в топ-1; все три имеют score=0.01,
        # ни один не попадает в топ-1 (первый попавшийся — 0.01).
        # На самом деле все три имеют одинаковый score, первый в списке
        # будет в top_n_ids. Но если узлов больше, то проверка сработает.
        # Сделаем более реалистичный тест: много узлов, один чанк группы
        # с нормальным скором, но не в топ-N.
        nodes = [
            # Высоко-скоровые узлы другой группы ПЕРВЫМИ в списке,
            # т.к. top_ids = nodes[:top_n] берутся из исходного порядка
            *[_node(f"other_p{i}", score=0.5, metadata={"document_id": "doc1", "point": "p", "total_parts": 2})
              for i in range(10)],
            _node("117-fz_st219_p1", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st219_p3", score=0.01, metadata={"document_id": "None"}),
        ]
        # top_n=5: первые 5 в списке — other_p0..p4 (score=0.5)
        # st219 группа (score=0.01, в конце списка) не в топ-5, буст не применяется
        result = multi_part_boost(nodes, top_n=5)
        for r in result:
            if r.node.node_id.startswith("117-fz_st219"):
                assert r.score == 0.01, f"{r.node.node_id} should NOT be boosted: {r.score}"

    def test_different_articles_separate(self):
        nodes = [
            _node("117-fz_st219_p1", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st171_p13", score=0.50, metadata={"document_id": "None"}),
        ]
        result = multi_part_boost(nodes, top_n=10)
        st171 = next(r for r in result if r.node.node_id == "117-fz_st171_p13")
        assert st171.score == 0.50

    def test_219_1_separate_from_219(self):
        nodes = [
            _node("117-fz_st219_p1", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.01, metadata={"document_id": "None"}),
            _node("117-fz_st219_1_p1", score=0.50, metadata={"document_id": "None"}),
        ]
        result = multi_part_boost(nodes, top_n=10)
        st219_1 = next(r for r in result if r.node.node_id == "117-fz_st219_1_p1")
        assert st219_1.score == 0.50


# ===========================================================================
# multi_part_boost -- primary (with document_id)
# ===========================================================================

class TestMultiPartBoostPrimary:
    def test_primary_boost(self):
        nodes = [
            _node("chunk_p1", score=0.04, metadata={
                "document_id": "doc1", "point": "point1", "total_parts": 3}),
            _node("chunk_p2", score=0.04, metadata={
                "document_id": "doc1", "point": "point1", "total_parts": 3}),
            _node("chunk_p3", score=0.50, metadata={
                "document_id": "doc1", "point": "point1", "total_parts": 3}),
        ]
        result = multi_part_boost(nodes, top_n=10)
        p1 = next(r for r in result if r.node.node_id == "chunk_p1")
        p2 = next(r for r in result if r.node.node_id == "chunk_p2")
        assert p1.score == pytest.approx(0.04 + 0.03, rel=1e-5)
        assert p2.score == pytest.approx(0.04 + 0.03, rel=1e-5)


# ===========================================================================
# reconstruct_article_context
# ===========================================================================

class TestReconstructArticleContext:
    def test_enumeration_adds_sequential_chunks(self):
        node_map = _make_node_map("117-fz_st219", 23)
        final = [_node("117-fz_st219_p3", score=0.3, metadata={"document_id": "None"})]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438 \u0441\u043e\u0446\u0438\u0430\u043b\u044c\u043d\u044b\u0435 \u0432\u044b\u0447\u0435\u0442\u044b",
            final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st219_p4" not in ids
        assert "117-fz_st219_p10" not in ids

    def test_non_enumeration_no_change(self):
        node_map = _make_node_map("117-fz_st219", 23)
        final = [_node("117-fz_st219_p3", score=0.3, metadata={"document_id": "None"})]
        result = reconstruct_article_context("\u043a\u0430\u043a \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c \u0432\u044b\u0447\u0435\u0442", final, node_map)
        assert len(result) == len(final)

    def test_empty_final_nodes(self):
        result = reconstruct_article_context("\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", [], {})
        assert result == []

    def test_low_score_article_skipped(self):
        node_map = _make_node_map("117-fz_st219", 5)
        final = [_node("117-fz_st219_p3", score=0.04, metadata={"document_id": "None"})]
        result = reconstruct_article_context("\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        assert len(result) == len(final)

    def test_multiple_articles(self):
        node_map = {}
        for prefix, count in [("117-fz_st219", 5), ("117-fz_st171", 14)]:
            for i in range(1, count + 1):
                n = TextNode(text=f"chunk p{i}", id_=f"{prefix}_p{i}",
                             metadata={"document_id": "None"})
                node_map[f"{prefix}_p{i}"] = n
        final = [
            _node("117-fz_st219_p3", score=0.3, metadata={"document_id": "None"}),
            _node("117-fz_st171_p13", score=0.4, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438 \u0432\u0441\u0435 \u0432\u0438\u0434\u044b",
            final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st171_p1" in ids
        assert "117-fz_st171_p2" in ids
        assert "117-fz_st171_p13" in ids
        assert "117-fz_st171_p4" not in ids

    def test_default_score_for_added_chunks(self):
        node_map = _make_node_map("117-fz_st219", 5)
        final = [_node("117-fz_st219_p3", score=0.3, metadata={"document_id": "None"})]
        result = reconstruct_article_context("\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        for nws in result:
            if nws.node.node_id != "117-fz_st219_p3":
                assert nws.score == DEFAULT_SCORE

    def test_219_1_separate_from_219(self):
        node_map = {}
        for prefix, count in [("117-fz_st219", 5), ("117-fz_st219_1", 3)]:
            for i in range(1, count + 1):
                n = TextNode(text=f"chunk p{i}", id_=f"{prefix}_p{i}",
                             metadata={"document_id": "None"})
                node_map[f"{prefix}_p{i}"] = n
        final = [
            _node("117-fz_st219_p3", score=0.5, metadata={"document_id": "None"}),
            _node("117-fz_st219_1_p1", score=0.5, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context("\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_1_p1" in ids
        assert "117-fz_st219_1_p2" in ids
        assert "117-fz_st219_1_p3" in ids
        assert "117-fz_st219_p4" not in ids


# ===========================================================================
    def test_enumeration_only_p1_present_adds_p2_p3_not_p4(self):
        # final содержит только p1 -> результат: p1, p2, p3, но НЕ p4
        node_map = _make_node_map("117-fz_st219", 10)
        final = [_node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st219_p4" not in ids
        assert len(result) == 3

    def test_enumeration_p1_and_p3_present_adds_p2_not_p4(self):
        # final содержит p1 и p3, отсутствует p2 -> добавить только p2
        node_map = _make_node_map("117-fz_st219", 10)
        final = [
            _node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"}),
            _node("117-fz_st219_p3", score=0.4, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st219_p4" not in ids
        assert len(result) == 3

    def test_enumeration_p3_present_adds_p1_p2_not_p4(self):
        # final содержит только p3 -> reconstruction добавляет p1, p2 (не p4)
        node_map = _make_node_map("117-fz_st219", 10)
        final = [_node("117-fz_st219_p3", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        ids = [n.node.node_id for n in result]
        assert ids[0] == "117-fz_st219_p3"  # оригинальный чанк на месте
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p4" not in ids
        assert len(ids) == 3

    def test_no_duplicates_in_reconstruction(self):
        # Если p1/p2/p3 уже в final_nodes, дублей быть не должно
        node_map = _make_node_map("117-fz_st219", 6)
        final = [
            _node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.5, metadata={"document_id": "None"}),
            _node("117-fz_st219_p3", score=0.5, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        ids = [n.node.node_id for n in result]
        assert len(ids) == 3  # ничего не добавлено, дублей нет
        assert len(set(ids)) == len(ids)
        assert "117-fz_st219_p4" not in ids

    def test_non_enumeration_unchanged_regression(self):
        # Обычный (не enumeration) запрос не должен меняться
        node_map = _make_node_map("117-fz_st219", 10)
        final = [
            _node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.4, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context(
            "\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435", final, node_map)
        assert [n.node.node_id for n in result] ==             ["117-fz_st219_p1", "117-fz_st219_p2"]

# ===========================================================================
# Edge cases
# ===========================================================================

class TestRetrievalUtilsEdgeCases:
    def test_empty_nodes_for_boost(self):
        assert multi_part_boost([]) == []

    def test_single_node_boost(self):
        n = _node("117-fz_st219_p1", score=0.5, metadata={"document_id": "doc1"})
        result = multi_part_boost([n])
        assert result[0].score == 0.5

    def test_node_map_without_matching_prefix(self):
        node_map = {"other_doc_p1": TextNode(text="x", id_="other_doc_p1")}
        final = [_node("117-fz_st219_p3", score=0.3, metadata={"document_id": "None"})]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        assert len(result) == len(final)

    def test_boost_does_not_remove_low_score(self):
        nodes = [
            _node("117-fz_st219_p1", score=0.02, metadata={"document_id": "None"}),
            _node("117-fz_st219_p2", score=0.02, metadata={"document_id": "None"}),
            _node("117-fz_st219_p3", score=0.5, metadata={"document_id": "None"}),
        ]
        result = multi_part_boost(nodes, top_n=10)
        assert len(result) == 3

    def test_sequential_chunk_selection_start_from_1(self):
        node_map = _make_node_map("117-fz_st219", 10)
        final = [
            _node("117-fz_st219_p5", score=0.5, metadata={"document_id": "None"}),
            _node("117-fz_st219_p8", score=0.4, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context(
            "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st219_p4" not in ids
        assert len(result) == 5
# ===========================================================================
# Cross-document & cross-article tests (regression — no mixing)
# ===========================================================================

class TestReconstructCrossDocument:
    """Проверка, что reconstruction НЕ смешивает разные документы/статьи."""

    def test_79_fz_st39_reconstruction(self):
        """79-ФЗ статья 39: p1 → добавляются p2, p3, НЕ p4."""
        node_map = {}
        for i in range(1, 6):
            node_map[f"79-fz_st39_p{i}"] = TextNode(
                text=f"79-fz st39 p{i}", id_=f"79-fz_st39_p{i}",
                metadata={"document_id": "None"})
        final = [_node("79-fz_st39_p1", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert "79-fz_st39_p1" in ids
        assert "79-fz_st39_p2" in ids
        assert "79-fz_st39_p3" in ids
        assert "79-fz_st39_p4" not in ids
        assert len(ids) == 3

    def test_ukaz_app1_reconstruction(self):
        """Указ №112 приложение 1: p1 → добавляются p2, p3."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"ukaz-112_app1_p{i}"] = TextNode(
                text=f"ukaz app1 p{i}", id_=f"ukaz-112_app1_p{i}",
                metadata={"document_id": "None"})
        final = [_node("ukaz-112_app1_p1", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert "ukaz-112_app1_p1" in ids
        assert "ukaz-112_app1_p2" in ids
        assert "ukaz-112_app1_p3" in ids
        assert "ukaz-112_app1_p4" not in ids
        assert len(ids) == 3

    def test_postanovlenie_pre_reconstruction(self):
        """Постановление преамбула: p3 найден → добавляются p1, p2."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"postanovlenie-1000_pre_p{i}"] = TextNode(
                text=f"post pre p{i}", id_=f"postanovlenie-1000_pre_p{i}",
                metadata={"document_id": "None"})
        final = [_node("postanovlenie-1000_pre_p3", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert "postanovlenie-1000_pre_p1" in ids
        assert "postanovlenie-1000_pre_p2" in ids
        assert "postanovlenie-1000_pre_p3" in ids
        assert "postanovlenie-1000_pre_p4" not in ids
        assert len(ids) == 3

    def test_no_mixing_117_and_79(self):
        """Разные документы: 117-ФЗ и 79-ФЗ не смешиваются."""
        node_map = {}
    def test_no_mixing_219_and_219_1(self):
        """Статья 219 и подстатья 219.1: reconstruction для обеих независим."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"219 p{i}", id_=f"117-fz_st219_p{i}",
                metadata={"document_id": "None"})
            node_map[f"117-fz_st219_1_p{i}"] = TextNode(
                text=f"219.1 p{i}", id_=f"117-fz_st219_1_p{i}",
                metadata={"document_id": "None"})
        final = [
            _node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"}),
            _node("117-fz_st219_1_p1", score=0.5, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st219_1_p1" in ids
        assert "117-fz_st219_1_p2" in ids
        assert "117-fz_st219_1_p3" in ids
        assert "117-fz_st219_p4" not in ids
        assert "117-fz_st219_1_p4" not in ids

    def test_no_mixing_219_and_219_2(self):
        """Статья 219 и подстатья 219.2: разные префиксы, не смешиваются."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"219 p{i}", id_=f"117-fz_st219_p{i}",
                metadata={"document_id": "None"})
            node_map[f"117-fz_st219_2_p{i}"] = TextNode(
                text=f"219.2 p{i}", id_=f"117-fz_st219_2_p{i}",
                metadata={"document_id": "None"})
        final = [_node("117-fz_st219_2_p1", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_2_p1" in ids
        assert "117-fz_st219_2_p2" in ids
        assert "117-fz_st219_2_p3" in ids
        assert "117-fz_st219_p1" not in ids
        assert "117-fz_st219_p2" not in ids
        assert len(ids) == 3

    def test_empty_metadata_fallback_on_chunk_id(self):
        """Пустая metadata → fallback на chunk_id (reconstruction работает)."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"219 p{i}", id_=f"117-fz_st219_p{i}", metadata={})
        final = [_node("117-fz_st219_p1", score=0.5, metadata={})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert len(ids) == 3

    def test_document_id_none_still_works(self):
        """document_id='None' — reconstruction всё равно работает по chunk_id."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"79-fz_st39_p{i}"] = TextNode(
                text=f"79 p{i}", id_=f"79-fz_st39_p{i}",
                metadata={"document_id": "None"})
        final = [_node("79-fz_st39_p1", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert "79-fz_st39_p1" in ids
        assert "79-fz_st39_p2" in ids
        assert "79-fz_st39_p3" in ids
        assert len(ids) == 3
        for i in range(1, 5):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"117 p{i}", id_=f"117-fz_st219_p{i}",
                metadata={"document_id": "None"})
            node_map[f"79-fz_st39_p{i}"] = TextNode(
                text=f"79 p{i}", id_=f"79-fz_st39_p{i}",
                metadata={"document_id": "None"})
        final = [_node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "79-fz_st39_p1" not in ids
        assert "79-fz_st39_p2" not in ids
        assert len(ids) == 3

    def test_no_mixing_219_and_220(self):
        """Разные статьи одного документа: 219 и 220 не смешиваются."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"219 p{i}", id_=f"117-fz_st219_p{i}",
                metadata={"document_id": "None"})
            node_map[f"117-fz_st220_p{i}"] = TextNode(
                text=f"220 p{i}", id_=f"117-fz_st220_p{i}",
                metadata={"document_id": "None"})
        final = [_node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st220_p1" not in ids
        assert "117-fz_st220_p2" not in ids
        assert len(ids) == 3
    def test_p21_found_adds_p1_p2_p3(self):
        """Если найден p21, добавляются p1,p2,p3 (начало той же статьи)."""
        node_map = {}
        for i in range(1, 25):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"219 p{i}", id_=f"117-fz_st219_p{i}",
                metadata={"document_id": "None"})
        final = [_node("117-fz_st219_p21", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "117-fz_st219_p21" in ids
        assert "117-fz_st219_p4" not in ids
        assert len(ids) == 4

    def test_79_fz_st39_p3_adds_p1_p2(self):
        """79-ФЗ ст39: найден p3 → добавляются p1,p2, НЕ p4."""
        node_map = {}
        for i in range(1, 8):
            node_map[f"79-fz_st39_p{i}"] = TextNode(
                text=f"79 p{i}", id_=f"79-fz_st39_p{i}",
                metadata={"document_id": "None"})
        final = [_node("79-fz_st39_p3", score=0.5, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert "79-fz_st39_p1" in ids
        assert "79-fz_st39_p2" in ids
        assert "79-fz_st39_p3" in ids
        assert "79-fz_st39_p4" not in ids
        assert len(ids) == 3

    def test_multiple_articles_both_reconstructed(self):
        """Две статьи одновременно: обе получают p1,p2,p3."""
        node_map = {}
        for i in range(1, 6):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"219 p{i}", id_=f"117-fz_st219_p{i}",
                metadata={"document_id": "None"})
            node_map[f"79-fz_st39_p{i}"] = TextNode(
                text=f"39 p{i}", id_=f"79-fz_st39_p{i}",
                metadata={"document_id": "None"})
        final = [
            _node("117-fz_st219_p1", score=0.5, metadata={"document_id": "None"}),
            _node("79-fz_st39_p1", score=0.5, metadata={"document_id": "None"}),
        ]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = {n.node.node_id for n in result}
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" in ids
        assert "117-fz_st219_p3" in ids
        assert "79-fz_st39_p1" in ids
        assert "79-fz_st39_p2" in ids
        assert "79-fz_st39_p3" in ids
        assert len(ids) == 6

    def test_low_score_article_not_reconstructed(self):
        """Низкорелевантная статья (score < 0.05) не запускает reconstruction."""
        node_map = {}
        for i in range(1, 5):
            node_map[f"117-fz_st219_p{i}"] = TextNode(
                text=f"219 p{i}", id_=f"117-fz_st219_p{i}",
                metadata={"document_id": "None"})
        final = [_node("117-fz_st219_p1", score=0.02, metadata={"document_id": "None"})]
        result = reconstruct_article_context("перечисли", final, node_map)
        ids = [n.node.node_id for n in result]
        assert len(ids) == 1
        assert "117-fz_st219_p1" in ids
        assert "117-fz_st219_p2" not in ids
