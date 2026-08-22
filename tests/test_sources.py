"""Тесты формирования источников ответа (app.rag.sources).

Standalone — не требуют Qdrant, Ollama, RAG. Проверяют универсальную
логику источников без привязки к конкретным документам.
"""

from app.rag.sources import collect_sources, _split_title, _struct_kind_number


def _node(title, url, score=None):
    """Нода виде dict-а (как умеет _iter_nodes)."""
    return {"metadata": {"title": title, "source_url": url}, "score": score}


# ---------------------------------------------------------------------------
# Базовые инварианты
# ---------------------------------------------------------------------------

class TestCollectSourcesBasics:
    def test_same_url_only_once(self):
        nodes = [
            _node("Указ № 112 \u2014 Пункт 8", "raw/ukaz-112.pdf", 0.9),
            _node("Указ № 112 \u2014 Пункт 9", "raw/ukaz-112.pdf", 0.7),
        ]
        res = collect_sources(nodes)
        assert len(res) == 1
        assert res[0]["url"] == "raw/ukaz-112.pdf"
        assert res[0]["title"].startswith("Указ № 112")

    def test_different_documents_separate(self):
        nodes = [
            _node("Указ № 112 \u2014 Пункт 8", "http://a/112.pdf", 0.8),
            _node("Указ № 96 \u2014 Раздел III", "http://b/96.pdf", 0.6),
        ]
        res = collect_sources(nodes)
        assert len(res) == 2
        urls = {r["url"] for r in res}
        assert urls == {"http://a/112.pdf", "http://b/96.pdf"}

    def test_no_technical_chunk_id_in_title(self):
        nodes = [_node("Указ № 112 \u2014 Пункт 8", "u.pdf", 0.5)]
        res = collect_sources(nodes)
        assert "p1" not in res[0]["title"]
        assert "ukaz-112" not in res[0]["title"]
        assert "_" not in res[0]["title"]

    def test_max_score_taken_for_doc(self):
        nodes = [
            _node("Указ № 112 \u2014 Пункт 8", "u.pdf", 0.7),
            _node("Указ № 112 \u2014 Пункт 9", "u.pdf", 0.95),
            _node("Указ № 112 \u2014 Пункт 10", "u.pdf", 0.8),
        ]
        res = collect_sources(nodes)
        assert res[0]["score"] == 0.95

    def test_empty_url_skipped(self):
        nodes = [_node("Указ № 112 \u2014 Пункт 8", "", 0.9)]
        assert collect_sources(nodes) == []

    def test_max_sources_limit(self):
        nodes = [
            _node("A \u2014 Пункт 1", "u1", 0.9),
            _node("B \u2014 Пункт 1", "u2", 0.8),
            _node("C \u2014 Пункт 1", "u3", 0.7),
            _node("D \u2014 Пункт 1", "u4", 0.6),
        ]
        assert len(collect_sources(nodes, max_sources=3)) == 3


# ---------------------------------------------------------------------------
# Объединение структурных элементов
# ---------------------------------------------------------------------------

class TestStructMerging:
    def test_consecutive_paras_range(self):
        nodes = [
            _node("Указ № 112 \u2014 Пункт 8", "u.pdf"),
            _node("Указ № 112 \u2014 Пункт 9", "u.pdf"),
            _node("Указ № 112 \u2014 Пункт 10", "u.pdf"),
        ]
        res = collect_sources(nodes)
        assert "пункты 8\u201310" in res[0]["title"]

    def test_non_consecutive_paras_enumeration(self):
        nodes = [
            _node("Указ № 112 \u2014 Пункт 8", "u.pdf"),
            _node("Указ № 112 \u2014 Пункт 10", "u.pdf"),
            _node("Указ № 112 \u2014 Пункт 17", "u.pdf"),
        ]
        res = collect_sources(nodes)
        assert "пункты 8, 10, 17" in res[0]["title"]

    def test_roman_section(self):
        nodes = [_node("Указ № 96 \u2014 Раздел III", "v.pdf")]
        res = collect_sources(nodes)
        assert "раздел III" in res[0]["title"]


# ---------------------------------------------------------------------------
# Парсер title (generic)
# ---------------------------------------------------------------------------


    def test_duplicate_structs_only_once(self):
        """Несколько чанков одного структурного элемента (3 чанка Раздел III)."""
        nodes = [
            {"metadata": {"title": "Указ № 96 — Раздел III", "source_url": "u.pdf"}, "score": 0.9},
            {"metadata": {"title": "Указ № 96 — Раздел III", "source_url": "u.pdf"}, "score": 0.8},
        ]
        res = collect_sources(nodes)
        title = res[0]["title"]
        # должен быть один "раздел III", а не трижды
        assert title.count("III") == 1, f"ожидалось 1xIII, получено {title}"
class TestTitleParser:
    def test_split(self):
        assert _split_title("Указ № 112 \u2014 Пункт 8") == ("Указ № 112", "Пункт 8")

    def test_split_no_struct(self):
        assert _split_title("Указ № 112") == ("Указ № 112", None)

    def test_kind_number_article(self):
        k, n = _struct_kind_number("Статья 22. Порядок")
        assert (k, n) == ("статья", "22")

    def test_kind_number_roman(self):
        k, n = _struct_kind_number("Раздел III")
        assert (k, n) == ("раздел", "III")

    def test_preamble_no_number(self):
        k, n = _struct_kind_number("Преамбула")
        # у преамбулы нет номера; это неструктурированный элемент
        assert n is None
        # распознаётся как нечисловой вид
        assert k == "преамбула"

    def test_other_doc_types(self):
        # 79-ФЗ -> статья 22
        k, n = _struct_kind_number("Статья 22. Основные термины")
        assert k == "статья" and n == "22"
        # Постановление -> раздел III
        k2, n2 = _struct_kind_number("Раздел III")
        assert k2 == "раздел" and n2 == "III"


# ---------------------------------------------------------------------------
# Составные структурные элементы (Приложение N, пункт M)
# ---------------------------------------------------------------------------

class TestCompoundStruct:
    def test_appendix_and_paragraphs(self):
        nodes = [
            {"metadata": {"title": "Указ № 112 — Приложение 1, пункт 4", "source_url": "u.pdf"}, "score": 0.99},
            {"metadata": {"title": "Указ № 112 — Приложение 1, пункт 5", "source_url": "u.pdf"}, "score": 0.95},
        ]
        res = collect_sources(nodes)
        assert "Приложение 1, пункты 4–5" in res[0]["title"]

    def test_separate_appendix_preserved(self):
        """Разные приложения не смешиваются."""
        nodes = [
            {"metadata": {"title": "Указ № 112 — Приложение 1, пункт 4", "source_url": "u.pdf"}, "score": 0.9},
            {"metadata": {"title": "Указ № 112 — Приложение 2, пункт 1", "source_url": "u.pdf"}, "score": 0.8},
        ]
        res = collect_sources(nodes)
        t = res[0]["title"]
        # оба приложения должны быть перечислены или разделены
        assert "Приложение 1" in t
        assert "Приложение 2" in t

    def test_last_struct_part_compound(self):
        from app.rag.sources import _last_struct_part
        p, k, n = _last_struct_part("Приложение 1, пункт 4")
        assert (p, k, n) == ("Приложение 1, ", "пункт", "4")
