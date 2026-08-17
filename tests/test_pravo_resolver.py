"""Тесты для app/pravo_resolver.py.

Юнит-тесты работают на сохранённых HTML-фикстурах (без сети).
Интеграционные тесты (marker ``integration``) делают реальные запросы
к pravo.gov.ru и пропускаются при недоступной сети.
"""
import json
from pathlib import Path

import pytest

from app.pravo_resolver import (
    DocumentNotFoundError,
    _parse_latest_rdk,
    cached_resolve,
    find_latest_revision,
    parse_search_results,
    pick_document,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pravo"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_bytes().decode("windows-1251", "replace")


# ============================================================================
# Парсинг списка документов
# ============================================================================
class TestParseSearchResults:
    def test_79fz_list(self):
        docs = parse_search_results(read_fixture("search_79fz.html"))
        assert len(docs) >= 15
        target = next(d for d in docs if d["nd"] == "102088054")
        assert target["date"] == "27.07.2004"
        assert "79-ФЗ" in target["title"]
        assert "гражданской службе" in target["name"]

    def test_58fz_list(self):
        docs = parse_search_results(read_fixture("search_58fz.html"))
        assert len(docs) >= 15
        target = next(d for d in docs if d["nd"] == "102081744")
        assert target["date"] == "27.05.2003"
        assert "58-ФЗ" in target["title"]
        assert "системе государственной службы" in target["name"]

    def test_title_filtered_list_single(self):
        docs = parse_search_results(read_fixture("search_58fz_title.html"))
        assert len(docs) == 1
        assert docs[0]["nd"] == "102081744"


# ============================================================================
# Выбор документа из кандидатов
# ============================================================================
CANDIDATES_79 = parse_search_results(read_fixture("search_79fz.html"))
CANDIDATES_58 = parse_search_results(read_fixture("search_58fz.html"))
CANDIDATES_667 = parse_search_results(read_fixture("search_667r.html"))


class TestPickDocument:
    def test_79fz_by_number_date_title(self):
        best = pick_document(
            CANDIDATES_79,
            number="79-ФЗ",
            date="27.07.2004",
            title="О государственной гражданской службе Российской Федерации",
        )
        assert best["nd"] == "102088054"

    def test_79fz_by_number_date_only(self):
        # Даже без названия номер+дата однозначно определяют документ
        best = pick_document(CANDIDATES_79, number="79-ФЗ", date="27.07.2004")
        assert best["nd"] == "102088054"

    def test_58fz_by_number_date_title(self):
        best = pick_document(
            CANDIDATES_58,
            number="58-ФЗ",
            date="27.05.2003",
            title="О системе государственной службы Российской Федерации",
        )
        assert best["nd"] == "102081744"

    def test_667r_empty_name_by_number_date(self):
        """Распоряжение 667-р (26.05.2005): у целевого документа пустое name.

        Номер+дата однозначно определяют документ, даже без совпадения по названию.
        """
        best = pick_document(
            CANDIDATES_667,
            number="667-р",
            date="26.05.2005",
            title="Об утверждении формы анкеты для участия в конкурсе "
            "на замещение вакантной должности государственной гражданской "
            "службы Российской Федерации",
        )
        assert best["nd"] == "102092603"

    def test_667r_other_date_with_title_raises(self):
        """Тот же номер, другая дата + несовпадающее название — fail-closed."""
        with pytest.raises(DocumentNotFoundError):
            pick_document(
                CANDIDATES_667,
                number="667-р",
                date="03.04.1992",
                title="Об утверждении формы анкеты",
            )

    def test_667r_other_date_without_title_ok(self):
        """Тот же номер, другая дата, без названия — возвращает правильный документ."""
        best = pick_document(CANDIDATES_667, number="667-р", date="03.04.1992")
        assert best["nd"] == "102015587"

    def test_wrong_date_raises(self):
        with pytest.raises(DocumentNotFoundError):
            pick_document(CANDIDATES_79, number="79-ФЗ", date="01.01.1990")

    def test_wrong_number_raises(self):
        with pytest.raises(DocumentNotFoundError):
            pick_document(CANDIDATES_79, number="999-ФЗ", date="27.07.2004")

    def test_empty_candidates_raises(self):
        with pytest.raises(DocumentNotFoundError):
            pick_document([], number="79-ФЗ", date="27.07.2004")

    def test_wrong_title_raises(self):
        with pytest.raises(DocumentNotFoundError):
            pick_document(
                CANDIDATES_58,
                number="58-ФЗ",
                date="27.05.2003",
                title="О бюджете Пенсионного фонда",
            )


# ============================================================================
# resolve_document: поиск от более специфичного к менее (фрагменты названия)
# ============================================================================
class TestResolveDocument:
    def test_full_title_ok(self, monkeypatch):
        from app import pravo_resolver as pr

        monkeypatch.setattr(
            pr, "search_documents", lambda number, title=None: CANDIDATES_79
        )
        best = pr.resolve_document(
            "79-ФЗ", "О государственной гражданской службе РФ", "27.07.2004"
        )
        assert best["nd"] == "102088054"

    def test_fragments_used_when_full_title_misses(self, monkeypatch):
        """Полный title не дал кандидатов -> идём по коротким фрагментам."""
        from app import pravo_resolver as pr

        calls: list[str | None] = []

        def fake_search(number, title=None):
            calls.append(title)
            if title == "государственной гражданской службе":
                return CANDIDATES_79
            return []

        monkeypatch.setattr(pr, "search_documents", fake_search)
        best = pr.resolve_document(
            "79-ФЗ", "О государственной гражданской службе РФ", "27.07.2004"
        )
        assert best["nd"] == "102088054"
        assert calls[0] == "О государственной гражданской службе РФ"  # полный title первым
        assert "государственной гражданской службе" in calls  # фрагмент сработал

    def test_all_queries_fail_raises(self, monkeypatch):
        from app import pravo_resolver as pr

        monkeypatch.setattr(pr, "search_documents", lambda number, title=None: [])
        with pytest.raises(DocumentNotFoundError):
            pr.resolve_document(
                "79-ФЗ", "О государственной гражданской службе РФ", "27.07.2004"
            )

    def test_no_title_searches_by_number_only(self, monkeypatch):
        from app import pravo_resolver as pr

        calls: list[str | None] = []

        def fake_search(number, title=None):
            calls.append(title)
            return CANDIDATES_79

        monkeypatch.setattr(pr, "search_documents", fake_search)
        best = pr.resolve_document("79-ФЗ", None, "27.07.2004")
        assert best["nd"] == "102088054"
        assert calls == [None]  # только поиск по номеру, без названия


# ============================================================================
# Редакции документа
# ============================================================================
class TestParseLatestRdk:
    def test_58fz_latest_edition(self):
        """Обычная действующая последняя редакция выбирается как max rdk."""
        best = _parse_latest_rdk(read_fixture("card_58fz.html"))
        assert best is not None
        rdk, label = best
        assert rdk == 18
        assert "365-ФЗ" in label

    def test_79fz_latest_edition(self):
        """Обычная действующая последняя редакция выбирается как max rdk."""
        best = _parse_latest_rdk(read_fixture("card_79fz.html"))
        assert best is not None
        rdk, label = best
        assert rdk >= 90
        assert "от" in label

    def test_no_editions(self):
        """Пустой HTML → None."""
        assert _parse_latest_rdk("<html><body>no select</body></html>") is None

    def test_rasporyazhenie_667r_skips_inactive_edition(self):
        """rasporyazhenie-667-r: rdk=7 (не действ.) пропущен, выбран rdk=6."""
        html = (
            '<select name="doc_editions">'
            '<option value="6,102092603" >6 - от 22.04.2022 № 986-р (изм.)</option>'
            '<option value="7,102092603" >7 - от 28.11.2024 № 1664 (не действ.)</option>'
            '</select>'
        )
        best = _parse_latest_rdk(html)
        assert best is not None
        rdk, label = best
        assert rdk == 6
        assert "986-р" in label

    def test_ukaz_159_skips_inactive_edition(self):
        """ukaz-159: rdk=1 (не действ.) пропущен, выбран rdk=0 (исходная)."""
        html = (
            '<select name="doc_editions">'
            '<option value="0,102091086" >Исходная редакция</option>'
            '<option value="1,102091086" >1 - от 10.10.2024 № 871 (не действ.)</option>'
            '</select>'
        )
        best = _parse_latest_rdk(html)
        assert best is not None
        rdk, label = best
        assert rdk == 0
        assert "Исходная" in label

    def test_skips_not_ready_edition(self):
        """Редакция с (не готова) пропускается."""
        html = (
            '<select name="doc_editions">'
            '<option value="10,102088054" >10 - от 25.11.2009 № 269-ФЗ (изм.)</option>'
            '<option value="11,102088054" >11 - от 17.12.2009 (не готова)</option>'
            '<option value="12,102088054" >12 - от 17.12.2009 № 322-ФЗ (изм.)(не готова)</option>'
            '</select>'
        )
        best = _parse_latest_rdk(html)
        assert best is not None
        rdk, label = best
        assert rdk == 10
        assert "269-ФЗ" in label

    def test_all_editions_inactive_returns_none(self):
        """Если все редакции недействующие → None."""
        html = (
            '<select name="doc_editions">'
            '<option value="1,102091086" >1 - от 10.10.2024 № 871 (не действ.)</option>'
            '<option value="2,102091086" >2 - от 01.01.2025 № 999 (не действ.)</option>'
            '</select>'
        )
        assert _parse_latest_rdk(html) is None

    def test_skips_n_value_disabled(self):
        """Опция со значением 'n' пропускается (недоступная редакция)."""
        html = (
            '<select name="doc_editions">'
            '<option value="3,102088054" >3 - от 12.04.2007 № 48-ФЗ (изм.)</option>'
            '<option disabled value="n">от 17.12.2009 (не готова)</option>'
            '</select>'
        )
        best = _parse_latest_rdk(html)
        assert best is not None
        rdk, label = best
        assert rdk == 3


# ============================================================================
# Кэш найденных nd
# ============================================================================
class TestCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        from app.pravo_resolver import load_cache, save_cache

        cache_file = tmp_path / "resolved.json"
        save_cache(
            {"79-FZ": {"nd": "102088054", "resolved_at": "x", "source": "pravo.gov.ru"}},
            cache_file,
        )
        loaded = load_cache(cache_file)
        assert loaded["79-FZ"]["nd"] == "102088054"

    def test_load_missing_file(self, tmp_path):
        from app.pravo_resolver import load_cache

        assert load_cache(tmp_path / "absent.json") == {}

    def test_load_corrupted_file(self, tmp_path):
        from app.pravo_resolver import load_cache

        bad = tmp_path / "bad.json"
        bad.write_text("{broken", encoding="utf-8")
        assert load_cache(bad) == {}

    def test_cached_resolve_uses_cache_on_second_call(self, tmp_path, monkeypatch):
        """Повторный вызов не должен ходить в сеть (resolve вызывается один раз)."""
        calls = {"n": 0}

        def fake_resolve(number, title=None, date=None):
            calls["n"] += 1
            return {"nd": "102088054", "title": "x", "name": "y", "date": date}

        monkeypatch.setattr("app.pravo_resolver.resolve_document", fake_resolve)
        cache_file = tmp_path / "resolved.json"

        nd1, _ = cached_resolve(
            "79-FZ", "79-ФЗ", "title", "27.07.2004", cache_path=cache_file
        )
        nd2, _ = cached_resolve(
            "79-FZ", "79-ФЗ", "title", "27.07.2004", cache_path=cache_file
        )
        assert nd1 == nd2 == "102088054"
        assert calls["n"] == 1
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert data["79-FZ"]["nd"] == "102088054"
        assert data["79-FZ"]["source"] == "pravo.gov.ru"

    def test_cached_resolve_force_reruns(self, tmp_path, monkeypatch):
        """При force=True поиск выполняется заново, несмотря на кэш."""
        calls = {"n": 0}

        def fake_resolve(number, title=None, date=None):
            calls["n"] += 1
            return {"nd": "102088054", "title": "x", "name": "y", "date": date}

        monkeypatch.setattr("app.pravo_resolver.resolve_document", fake_resolve)
        cache_file = tmp_path / "resolved.json"
        cached_resolve("79-FZ", "79-ФЗ", "t", "27.07.2004", cache_path=cache_file)
        cached_resolve(
            "79-FZ", "79-ФЗ", "t", "27.07.2004", cache_path=cache_file, force=True
        )
        assert calls["n"] == 2


# ============================================================================
# Интеграционные тесты (реальные запросы к pravo.gov.ru)
# ============================================================================
def _network_available() -> bool:
    import socket

    try:
        socket.create_connection(("pravo.gov.ru", 80), timeout=5).close()
        return True
    except OSError:
        return False


class TestIntegration:
    def setup_method(self):
        if not _network_available():
            pytest.skip("pravo.gov.ru недоступен")

    def test_resolve_79fz_returns_known_nd(self):
        best = pick_document(
            CANDIDATES_79,
            number="79-ФЗ",
            date="27.07.2004",
            title="О государственной гражданской службе Российской Федерации",
        )
        assert best["nd"] == "102088054"
        rev = find_latest_revision(best["nd"])
        assert rev is not None and rev["rdk"] >= 90

    def test_resolve_58fz_automatically(self, tmp_path):
        nd, entry = cached_resolve(
            "58-FZ",
            "58-ФЗ",
            "О системе государственной службы Российской Федерации",
            "27.05.2003",
            cache_path=tmp_path / "resolved.json",
        )
        assert nd == "102081744"
        assert entry["source"] == "pravo.gov.ru"

