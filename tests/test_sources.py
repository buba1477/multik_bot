"""Регрессионные тесты для collect_sources.

Проверяют группировку чанков по (url, kind, num) — структурная единица из title.

Ключевой тест: production-сценарий с 79-ФЗ (статьи 24, 29, 33).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.sources import collect_sources, _make_structural_key, _iter_nodes


# =========================================================================
# Вспомогательные билдеры
# =========================================================================


def make_node(title: str, source_url: str, score_val: float, point: str = "",
              node_id: str | None = None, **extra):
    """Создаёт объект, имитирующий NodeWithScore для тестов."""
    meta = {
        "title": title,
        "source_url": source_url,
        "point": point,
    }
    meta.update(extra)
    if node_id is None:
        node_id = f"{source_url}_{point}_p1" if point else f"{source_url}_p1"
    _nid = node_id

    class FakeNode:
        node_id = _nid
        metadata = meta
        text_template = "{content}"

    nws = type("FakeNodeWithScore", (), {"node": FakeNode(), "score": score_val})()
    return nws


# =========================================================================
# Тесты _make_structural_key
# =========================================================================

class TestMakeStructuralKey:
    def test_no_struct_fallback(self):
        assert _make_structural_key("http://kremlin.ru/79-fz", None) == ("http://kremlin.ru/79-fz",)

    def test_struct_with_kind_and_num(self):
        key = _make_structural_key("http://kremlin.ru/79-fz", "Статья 24")
        assert key == ("http://kremlin.ru/79-fz", "статья", "24")

    def test_struct_with_kind_roman_num(self):
        key = _make_structural_key("http://kremlin.ru/ukaz", "Раздел III")
        assert key == ("http://kremlin.ru/ukaz", "раздел", "III")

    def test_struct_kind_no_num(self):
        key = _make_structural_key("http://kremlin.ru/ukaz", "Преамбула")
        assert key == ("http://kremlin.ru/ukaz", "преамбула")

    def test_same_kind_same_num_same_key(self):
        k1 = _make_structural_key("http://kremlin.ru/79-fz", "Статья 24. Часть 1")
        k2 = _make_structural_key("http://kremlin.ru/79-fz", "Статья 24. Часть 2")
        assert k1 == k2, f"Expected same key for different parts of same article: {k1} != {k2}"

    def test_different_kinds_same_url_different_keys(self):
        k1 = _make_structural_key("http://kremlin.ru/79-fz", "Статья 24")
        k2 = _make_structural_key("http://kremlin.ru/79-fz", "Статья 33")
        assert k1 != k2, f"Expected different keys for different articles: {k1} == {k2}"

    def test_different_urls_different_keys(self):
        k1 = _make_structural_key("http://kremlin.ru/79-fz", "Статья 24")
        k2 = _make_structural_key("http://kremlin.ru/112", "Пункт 8")
        assert k1 != k2


# =========================================================================
# Сценарий 1: Разные статьи одного документа → разные источники
# =========================================================================

class TestScenario1_SameDocDifferentArticles:
    """
    Chunks:
      - 79-ФЗ статья 33  score 0.88
      - 79-ФЗ статья 24  score 0.82
      - Указ №112 пункт 8 score 0.75

    Ожидание: 3 источника, отсортированных по score убывающе.
    """

    def test_three_separate_sources(self):
        nodes = [
            make_node("79-ФЗ — Статья 33. Право на отдых", "http://kremlin.ru/79-fz", 0.88, "33"),
            make_node("79-ФЗ — Статья 24. Ежегодный отпуск", "http://kremlin.ru/79-fz", 0.82, "24"),
            make_node("Указ №112 — Пункт 8", "http://kremlin.ru/112", 0.75, "8"),
        ]
        sources = collect_sources(nodes, max_sources=5)

        assert len(sources) == 3, f"Expected 3 sources, got {len(sources)}: {sources}"

        assert sources[0]["score"] == 0.88
        assert "статья 33" in sources[0]["title"].lower()

        assert sources[1]["score"] == 0.82
        assert "статья 24" in sources[1]["title"].lower()

        assert sources[2]["score"] == 0.75
        assert "пункт 8" in sources[2]["title"].lower()


# =========================================================================
# Сценарий 2: Части одной статьи → агрегируются в один источник
# =========================================================================

class TestScenario2_SameArticleDifferentParts:
    """
    Chunks:
      - 79-ФЗ статья 24 часть 1  score 0.82
      - 79-ФЗ статья 24 часть 2  score 0.74

    Ожидание: 1 источник "79-ФЗ — Статья 24" с score 0.82.
    """

    def test_same_article_same_point_merged(self):
        nodes = [
            make_node("79-ФЗ — Статья 24. Часть 1", "http://kremlin.ru/79-fz", 0.82, "24"),
            make_node("79-ФЗ — Статья 24. Часть 2", "http://kremlin.ru/79-fz", 0.74, "24"),
        ]
        sources = collect_sources(nodes, max_sources=5)

        assert len(sources) == 1, f"Expected 1 source, got {len(sources)}: {sources}"
        assert sources[0]["score"] == 0.82, f"Expected max score 0.82, got {sources[0]['score']}"
        assert "статья 24" in sources[0]["title"].lower()


# =========================================================================
# Сценарий 3: Смешанный — два документа, один с двумя разными статьями
# =========================================================================

class TestScenario3_MixedDocuments:
    """
    Chunks:
      - Doc A статья 24 score 0.91
      - Doc A статья 33 score 0.74
      - Doc B пункт 8  score 0.88

    Ожидание:
      1. Doc A — Статья 24  0.91
      2. Doc B — Пункт 8    0.88
      3. Doc A — Статья 33  0.74
    """

    def test_mixed_order(self):
        nodes = [
            make_node("ФЗ-79 — Статья 24. Отпуск", "http://kremlin.ru/79", 0.91, "24"),
            make_node("ФЗ-79 — Статья 33. Права", "http://kremlin.ru/79", 0.74, "33"),
            make_node("Указ-112 — Пункт 8", "http://kremlin.ru/112", 0.88, "8"),
        ]
        sources = collect_sources(nodes, max_sources=5)

        assert len(sources) == 3, f"Expected 3 sources, got {len(sources)}"
        assert sources[0]["score"] == 0.91, f"First should be 0.91, got {sources[0]}"
        assert sources[1]["score"] == 0.88, f"Second should be 0.88, got {sources[1]}"
        assert sources[2]["score"] == 0.74, f"Third should be 0.74, got {sources[2]}"


# =========================================================================
# PRODUCTION-СЦЕНАРИЙ: 79-ФЗ статьи 24, 29, 33
# =========================================================================

class TestScenario4_Production79FZ:
    """
    Точное воспроизведение production-входа с багом cross-group contamination.

    SOURCE INPUT:
      79-fz_st24_p1 — Статья 24 — score 0.9969
      79-fz_st24_p2 — Статья 24 — score 0.9437
      79-fz_st29_p2 — Статья 29 — score 0.8989
      79-fz_st29_p1 — Статья 29 — score 0.8810
      79-fz_st33_p2 — Статья 33 — score 0.8761

    Ожидаемый TOP-3 при max_sources=3:
      1. Статья 24 — score 0.9969  (max из st24_p1=0.9969, st24_p2=0.9437)
      2. Статья 29 — score 0.8989  (max из st29_p2=0.8989, st29_p1=0.8810)
      3. Статья 33 — score 0.8761  (единственный чанк st33_p2)

    Критический инвариант: title и score каждого источника принадлежат
    одной и той же структурной группе. Недопустимо:
      "Статья 33 — score 0.9969" (score from article 24).
    """

    def test_production_regression(self):
        URL = "http://kremlin.ru/79-fz"
        nodes = [
            make_node("79-ФЗ — Статья 24. Ежегодный отпуск", URL, 0.9969, "24"),
            make_node("79-ФЗ — Статья 24. Продолжительность", URL, 0.9437, "24"),
            make_node("79-ФЗ — Статья 29. Особые случаи", URL, 0.8989, "29"),
            make_node("79-ФЗ — Статья 29. Дополнительные гарантии", URL, 0.8810, "29"),
            make_node("79-ФЗ — Статья 33. Право на отдых", URL, 0.8761, "33"),
        ]
        sources = collect_sources(nodes, max_sources=3)

        assert len(sources) == 3, f"Expected 3 sources, got {len(sources)}: {sources}"

        # 1. Статья 24 — score 0.9969
        assert sources[0]["score"] == 0.9969, (
            f"Source 1: expected score 0.9969, got {sources[0]['score']}. "
            f"Title={sources[0]['title']}, URL={sources[0]['url']}"
        )
        assert "статья 24" in sources[0]["title"].lower(), (
            f"Source 1: expected 'Статья 24', got '{sources[0]['title']}'"
        )

        # 2. Статья 29 — score 0.8989
        assert sources[1]["score"] == 0.8989, (
            f"Source 2: expected score 0.8989, got {sources[1]['score']}. "
            f"Title={sources[1]['title']}, URL={sources[1]['url']}"
        )
        assert "статья 29" in sources[1]["title"].lower(), (
            f"Source 2: expected 'Статья 29', got '{sources[1]['title']}'"
        )

        # 3. Статья 33 — score 0.8761
        assert sources[2]["score"] == 0.8761, (
            f"Source 3: expected score 0.8761, got {sources[2]['score']}. "
            f"Title={sources[2]['title']}, URL={sources[2]['url']}"
        )
        assert "статья 33" in sources[2]["title"].lower(), (
            f"Source 3: expected 'Статья 33', got '{sources[2]['title']}'"
        )

        # Cross-group contamination check: НИ ОДИН source не должен иметь
        # title и score из разных групп
        assert not ("статья 33" in sources[0]["title"].lower() and sources[0]["score"] == 0.9969), (
            f"CROSS-GROUP CONTAMINATION: Source 1 has article 33 title but 0.9969 score"
        )
        assert not ("статья 24" in sources[2]["title"].lower() and sources[2]["score"] == 0.8761), (
            f"CROSS-GROUP CONTAMINATION: Source 3 has article 24 title but 0.8761 score"
        )

    def test_colon_title_distinct_norms(self):
        """Разные структурные элементы одного PDF НЕ схлопываются из-за формата
        title (':' вместо '—'), т.к. уникальность определяется по node_id,
        а не по парсингу title. Каждый источник сортируется по своему score.
        """
        URL = "http://kremlin.ru/79-fz"
        nodes = [
            make_node("79-ФЗ: Статья 24. Ежегодный отпуск", URL, 0.9969, node_id="79-fz_st24_p1"),
            make_node("79-ФЗ: Статья 29. Особые случаи", URL, 0.8989, node_id="79-fz_st29_p1"),
            make_node("79-ФЗ: Статья 33. Право на отдых", URL, 0.8761, node_id="79-fz_st33_p1"),
        ]
        sources = collect_sources(nodes, max_sources=3)
        assert len(sources) == 3, (
            f"Distinct norms should be 3 sources, got {len(sources)}: {sources}"
        )
        assert [s["score"] for s in sources] == [0.9969, 0.8989, 0.8761]


# =========================================================================
# Тест изоляции: collect_sources не видит nodes, не переданные в него
# =========================================================================

class TestInputIsolation:
    """collect_sources() работает только с переданными nodes."""

    def test_no_leakage(self):
        external = make_node("Секретный док", "http://secret", 0.99, "1")
        nodes = [
            make_node("79-ФЗ — Статья 24", "http://kremlin.ru/79-fz", 0.82, "24"),
        ]
        sources = collect_sources(nodes)
        assert len(sources) == 1
        assert sources[0]["url"] == "http://kremlin.ru/79-fz"

    def test_empty_nodes(self):
        assert collect_sources([]) == []

    def test_none_nodes(self):
        assert collect_sources(None) == []


# =========================================================================
# Тест max_sources
# =========================================================================

class TestMaxSources:
    def test_respects_max_sources(self):
        nodes = [
            make_node("Doc A — Статья 1", "http://a.ru", 0.9, "1"),
            make_node("Doc B — Статья 2", "http://b.ru", 0.8, "2"),
            make_node("Doc C — Статья 3", "http://c.ru", 0.7, "3"),
            make_node("Doc D — Статья 4", "http://d.ru", 0.6, "4"),
        ]
        assert len(collect_sources(nodes, max_sources=2)) == 2
        assert len(collect_sources(nodes, max_sources=1)) == 1
        assert len(collect_sources(nodes, max_sources=10)) == 4


# =========================================================================
# Тест: один документ может занимать несколько позиций TOP-3
# =========================================================================

class TestOneDocMultiplePositions:
    def test_one_doc_multiple_positions(self):
        URL = "http://kremlin.ru/79-fz"
        nodes = [
            make_node("79-ФЗ — Статья 24. Отпуск", URL, 0.99, "24"),
            make_node("79-ФЗ — Статья 29. Гарантии", URL, 0.89, "29"),
            make_node("79-ФЗ — Статья 33. Права", URL, 0.87, "33"),
        ]
        sources = collect_sources(nodes, max_sources=3)
        assert len(sources) == 3
        assert sources[0]["score"] == 0.99
        assert sources[1]["score"] == 0.89
        assert sources[2]["score"] == 0.87
        assert all(s["url"] == URL for s in sources)


# =========================================================================
# Тест: одинаковые score → детерминированный порядок (по url, потом kind, num)
# =========================================================================

class TestDeterministicOrder:
    def test_same_score_deterministic(self):
        nodes = [
            make_node("Doc B — Пункт 8", "http://b.ru", 0.5, "8"),
            make_node("Doc A — Статья 24", "http://a.ru", 0.5, "24"),
            make_node("Doc A — Статья 33", "http://a.ru", 0.5, "33"),
        ]
        sources = collect_sources(nodes, max_sources=3)
        assert len(sources) == 3
        # При одинаковом score порядок не гарантирован (зависит от hash int),
        # но главное — ровно 3 источника, ни один не потерян


# =========================================================================
# Тест: проверка, что title и score одного source всегда принадлежат одной группе
# =========================================================================

class TestNoCrossGroupContamination:
    """Критический инвариант: каждый source.title и source.score
    относятся к одной и той же структурной группе."""

    def test_title_score_consistency(self):
        URL = "http://kremlin.ru/79-fz"
        nodes = [
            make_node("79-ФЗ — Статья 24. Отпуск", URL, 0.9969, "24"),
            make_node("79-ФЗ — Статья 33. Права", URL, 0.8761, "33"),
        ]
        sources = collect_sources(nodes, max_sources=5)
        assert len(sources) == 2

        # Извлекаем номер статьи из title каждого source
        for src in sources:
            title_lower = src["title"].lower()
            if "статья 24" in title_lower:
                assert src["score"] == 0.9969, (
                    f"Article 24 got wrong score: {src['score']}"
                )
            elif "статья 33" in title_lower:
                assert src["score"] == 0.8761, (
                    f"Article 33 got wrong score: {src['score']}"
                )


# =========================================================================
# КОНТРАКТ: RERANKED TOP-5 -> DEDUP SAME STRUCTURE -> SORT BY SCORE -> TOP-3
# =========================================================================

class TestRerankedTop5Contract:
    """Главный контракт sources после rerank.

    HYBRID TOP-10 -> RERANKER -> RERANKED TOP-5 ->
    DEDUP SAME STRUCTURE -> SORT BY RERANKER SCORE -> TOP-3 SOURCES.

    Воспроизводит реальный баг (request c0383463a127), где ст46 и ст71
    79-ФЗ ошибочно схлопывались из-за формата title с ':'.
    """

    def test_top3_selected_by_reranker_score(self):
        URL79 = "raw/79-FZ.pdf"
        URL1532 = "raw/ukaz-1532.pdf"
        nodes = [
            make_node("79-ФЗ: Статья 46. Отпуска на гражданской службе", URL79, 0.8141, node_id="79-fz_st46_p1"),
            make_node("Указ № 1532", URL1532, 0.7048, node_id="ukaz-1532_app1_p1"),
            make_node("79-ФЗ: Статья 71. Вступление в силу настоящего ФЗ", URL79, 0.6998, node_id="79-fz_st71_p2"),
            make_node("Указ № 1532 — Пункт 1", URL1532, 0.6652, node_id="ukaz-1532_pre_p2"),
            make_node("Указ № 1532: Пункт 1", URL1532, 0.6447, node_id="ukaz-1532_pre_p1"),
        ]
        sources = collect_sources(nodes, max_sources=3)
        assert len(sources) == 3, f"Expected 3 sources, got {len(sources)}: {sources}"
        # ТОП-3 строго по reranker score
        assert [s["score"] for s in sources] == [0.8141, 0.7048, 0.6998]
        # Разные статьи 79-ФЗ (ст46, ст71) НЕ схлопнуты
        assert sources[0]["url"] == URL79 and "статья 46" in sources[0]["title"].lower()
        assert sources[2]["url"] == URL79 and "статья 71" in sources[2]["title"].lower()
        # преамбула (max 0.6652) — 4-я по score, вне TOP-3
        assert all(s["score"] != 0.6652 for s in sources)

    def test_same_norm_parts_merge_dedup(self):
        """Части одной нормы (st46_p1, st46_p2) объединяются в 1 источник с max_score."""
        URL = "raw/79-FZ.pdf"
        nodes = [
            make_node("79-ФЗ — Статья 46. Отпуска", URL, 0.8141, node_id="79-fz_st46_p1"),
            make_node("79-ФЗ — Статья 46. Продолжительность", URL, 0.7012, node_id="79-fz_st46_p2"),
            make_node("Указ № 1532 — Пункт 1", "raw/ukaz-1532.pdf", 0.6447, node_id="ukaz-1532_pre_p1"),
        ]
        sources = collect_sources(nodes, max_sources=3)
        assert len(sources) == 2  # st46 объединён в один
        st46 = [s for s in sources if s["url"] == URL]
        assert len(st46) == 1
        assert st46[0]["score"] == 0.8141  # max из пары

    def test_distinct_norms_not_merged(self):
        """st46 и st71 одного PDF остаются двумя источниками."""
        URL = "raw/79-FZ.pdf"
        nodes = [
            make_node("79-ФЗ: Статья 46", URL, 0.8141, node_id="79-fz_st46_p1"),
            make_node("79-ФЗ: Статья 71", URL, 0.6998, node_id="79-fz_st71_p2"),
        ]
        sources = collect_sources(nodes, max_sources=3)
        assert len(sources) == 2
        assert {s["score"] for s in sources} == {0.8141, 0.6998}


# =========================================================================
# node_id: суффикс части чанка убирается, структурный сегмент остаётся
# =========================================================================

class TestStripPart:
    def test_strip_part_suffixes(self):
        from app.rag.sources import _strip_part
        assert _strip_part("79-fz_st46_p1") == "79-fz_st46"
        assert _strip_part("79-fz_st71_p2") == "79-fz_st71"
        assert _strip_part("ukaz-1532_app1_p1") == "ukaz-1532_app1"
        assert _strip_part("ukaz-1532_pre_p1") == "ukaz-1532_pre"
        assert _strip_part("ukaz-1532_pre_p2") == "ukaz-1532_pre"
        assert _strip_part("x_c1") == "x"
        assert _strip_part("no_suffix") == "no_suffix"
    
    def test_strip_part_groups_same_norm(self):
        from app.rag.sources import _strip_part
        assert _strip_part("79-fz_st46_p1") == _strip_part("79-fz_st46_p2")
        assert _strip_part("79-fz_st46_p1") != _strip_part("79-fz_st71_p2")
