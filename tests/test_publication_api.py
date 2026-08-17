"""Тесты официального API publication.pravo.gov.ru.

Unit-тесты работают на сохранённых JSON/PDF фикстурах и моках HTTP-слоя
(без сети). Integration-тесты делают реальные запросы и auto-skip при
отсутствии сети.
"""
import json
from pathlib import Path

import pytest

import app.publication_api as pub

FIX = Path(__file__).resolve().parent / "fixtures" / "publication"


def fixture(name: str):
    return FIX / name


def load_json(name: str):
    return json.loads(fixture(name).read_text(encoding="utf-8"))


SEARCH_79 = load_json("documents_79fz.json")["items"]
SEARCH_58 = load_json("documents_58fz.json")["items"]
DETAIL_2026 = load_json("document_detail_79fz_2026.json")

# Реквизиты современного 79-ФЗ (есть в базе) и старого 79-ФЗ (нет в базе)
REC_MODERN = {
    "id": "79-FZ-2026", "type": "Федеральный закон", "number": "79-ФЗ",
    "date": "09.04.2026",
    "title": "О внесении изменений в Гражданский процессуальный кодекс РФ",
}
REC_OLD_79 = {
    "id": "79-FZ", "type": "Федеральный закон", "number": "79-ФЗ",
    "date": "27.07.2004", "title": "О государственной гражданской службе РФ",
}

# Реквизиты 667-р (распоряжение, есть в базе)
REC_667R = {
    "id": "667-r", "type": "Распоряжение Правительства Российской Федерации", "number": "667-р",
    "date": "26.05.2005",
    "title": "Об утверждении формы анкеты для участия в конкурсе "
             "на замещение вакантной должности государственной гражданской "
             "службы Российской Федерации",
}


class TestSearchAndDocs:
    def test_search_documents_parses_items(self, monkeypatch):
        monkeypatch.setattr(pub, "_get_json", lambda url: {"items": SEARCH_79})
        items = pub.search_documents(number="79-ФЗ")
        assert len(items) == 15
        assert all(it["number"] == "79-ФЗ" for it in items)

    def test_get_document_returns_detail(self, monkeypatch):
        monkeypatch.setattr(pub, "_get_json", lambda url: DETAIL_2026)
        doc = pub.get_document("0001202604090006")
        assert doc["eoNumber"] == "0001202604090006"
        assert doc["documentType"]["name"] == "Федеральный закон"

    def test_pdf_url(self):
        assert "eoNumber=00012026" in pub.pdf_url("00012026")


class TestCanonicalType:
    """Канонизация типов из двух полей: documentType.name + signatoryAuthorities[0].name."""

    def test_ukaz_prezident(self):
        doc = {
            "documentType": {"name": "Указ"},
            "signatoryAuthorities": [{"name": "Президент Российской Федерации"}],
        }
        assert pub._canonical_type(doc) == "указ президента российской федерации"

    def test_rasporyazhenie_pravitelstvo(self):
        doc = {
            "documentType": {"name": "Распоряжение"},
            "signatoryAuthorities": [{"name": "Правительство Российской Федерации"}],
        }
        assert pub._canonical_type(doc) == "распоряжение правительства российской федерации"

    def test_postanovlenie_pravitelstvo(self):
        doc = {
            "documentType": {"name": "Постановление"},
            "signatoryAuthorities": [{"name": "Правительство Российской Федерации"}],
        }
        assert pub._canonical_type(doc) == "постановление правительства российской федерации"

    def test_federal_law_unchanged(self):
        """Федеральный закон + Президент — не в whitelist, возвращается doc_type."""
        doc = {
            "documentType": {"name": "Федеральный закон"},
            "signatoryAuthorities": [{"name": "Президент Российской Федерации"}],
        }
        assert pub._canonical_type(doc) == "федеральный закон"

    def test_prikaz_unchanged(self):
        """Приказ + Минюст — не в whitelist, возвращается doc_type."""
        doc = {
            "documentType": {"name": "Приказ"},
            "signatoryAuthorities": [{"name": "Министерство юстиции Российской Федерации"}],
        }
        assert pub._canonical_type(doc) == "приказ"

    def test_missing_signatory_authorities(self):
        """Пустой/отсутствующий signatoryAuthorities не падает."""
        doc = {"documentType": {"name": "Указ"}}
        assert pub._canonical_type(doc) == "указ"

    def test_missing_type_returns_empty(self):
        assert pub._canonical_type({}) == ""


class TestSearchFragments:
    """build_search_fragments: от более специфичного к менее."""

    def test_builds_fragments_short(self):
        from app.search_fragments import build_search_fragments

        frags = build_search_fragments(
            "О внесении изменений в Гражданский процессуальный кодекс РФ"
        )
        assert frags
        assert frags[0] == "внесении изменений гражданский процессуальный"
        assert all(len(f) <= 60 for f in frags)

    def test_empty_for_no_title(self):
        from app.search_fragments import build_search_fragments

        assert build_search_fragments(None) == []
        assert build_search_fragments("") == []
        assert build_search_fragments("о и в на к по") == []  # только стоп-слова

    def test_fragments_progressive(self):
        from app.search_fragments import build_search_fragments

        title = "Об утверждении формы анкеты для участия в конкурсе"
        frags = build_search_fragments(title)
        # От более специфичного (4 слова) к менее (3, 2)
        assert len(frags) >= 2
        assert frags[0] == "утверждении формы анкеты участия"
        assert len(frags[0].split()) >= len(frags[1].split())


class TestPickUnique:
    """_pick_unique: None при неоднозначности/поисковой невозможности."""

    def test_empty_candidates(self):
        assert pub._pick_unique([], REC_MODERN) is None

    def test_unique_pick_modern(self, monkeypatch):
        monkeypatch.setattr(pub, "_get_json", lambda url: DETAIL_2026)
        doc = pub._pick_unique(SEARCH_79, REC_MODERN)
        assert doc is not None
        assert doc["eoNumber"] == "0001202604090006"

    def test_ambiguity_returns_none(self):
        """Два кандидата с одинаковым номером+датой -> None (неоднозначность)."""
        dup = [
            {**SEARCH_79[0], "eoNumber": "0001"},
            {**SEARCH_79[0], "eoNumber": "0002"},
        ]
        assert pub._pick_unique(dup, REC_MODERN) is None

    def test_wrong_number_none(self):
        """API вернул документ с другим номером -> None (поисковая невозможность)."""
        item = {**SEARCH_79[0], "number": "88-ФЗ"}
        assert pub._pick_unique([item], REC_MODERN) is None


class TestResolveExact:
    def _patch(self, monkeypatch, search_items=None, detail=None):
        def fake_get_json(url):
            if "Documents" in url:
                return {"items": search_items if search_items is not None else SEARCH_79}
            if "Document?" in url:
                return detail if detail is not None else DETAIL_2026
            raise AssertionError(f"unexpected url: {url}")

        monkeypatch.setattr(pub, "_get_json", fake_get_json)

    def test_exact_match_modern(self, monkeypatch):
        self._patch(monkeypatch)
        doc = pub.resolve_exact(REC_MODERN)
        assert doc["eoNumber"] == "0001202604090006"
        assert doc["number"] == "79-ФЗ"
        assert doc["documentType"]["name"] == "Федеральный закон"

    def test_not_found_old_law_2004(self, monkeypatch):
        # 79-ФЗ 2004 отсутствует в базе (по дате 27.07.2004 совпадений нет)
        self._patch(monkeypatch)
        with pytest.raises(pub.DocumentNotFoundError):
            pub.resolve_exact(REC_OLD_79)

    def test_wrong_number_not_found(self, monkeypatch):
        # В ответе на number=999-ФЗ пусто
        self._patch(monkeypatch, search_items=[])
        with pytest.raises(pub.DocumentNotFoundError):
            pub.resolve_exact({**REC_MODERN, "number": "999-ФЗ"})

    def test_ambiguity_now_not_found(self, monkeypatch):
        # Два кандидата с одинаковым номером И датой -> неоднозначность.
        # _pick_unique возвращает None -> второй этап (name) тоже неоднозначен ->
        # итог DocumentNotFoundError (поисковая невозможность, fallback в legacy).
        dup = [
            {**SEARCH_79[0], "eoNumber": "0001"},
            {**SEARCH_79[0], "eoNumber": "0002"},
        ]
        self._patch(monkeypatch, search_items=dup)
        with pytest.raises(pub.DocumentNotFoundError):
            pub.resolve_exact(REC_MODERN)

    def test_mismatch_number_not_found(self, monkeypatch):
        # API вернул документ с другим номером при той же дате
        item = {**SEARCH_79[0], "number": "88-ФЗ"}
        self._patch(monkeypatch, search_items=[item])
        with pytest.raises(pub.DocumentNotFoundError):
            pub.resolve_exact(REC_MODERN)


class TestResolveExactWithFragments:
    """resolve_exact с фрагментами: когда номер+фрагмент названия дают результат."""

    def _patch(self, monkeypatch, number_search=None, name_search=None, detail=None):
        """Mock _get_json с разными ответами для number-only и number+name."""

        def fake_get_json(url):
            if "Document?" in url:
                return detail if detail is not None else DETAIL_2026
            if "name=" not in url:
                return {"items": number_search if number_search is not None else SEARCH_79}
            return {"items": name_search if name_search is not None else []}

        monkeypatch.setattr(pub, "_get_json", fake_get_json)

    def test_name_search_helps_667r(self, monkeypatch):
        """667-р: поиск по номеру не даёт совпадения, а номер+имя находит ровно один."""
        SEARCH_667R = [{
            "eoNumber": "0001201701030034", "number": "667-р",
            "documentDate": "2005-05-26T00:00:00",
        }]
        DETAIL_667R = {
            "eoNumber": "0001201701030034", "number": "667-р",
            "documentDate": "2005-05-26T00:00:00",
            "documentType": {"name": "Распоряжение"},
            "signatoryAuthorities": [{"name": "Правительство Российской Федерации"}],
        }
        self._patch(monkeypatch, number_search=[], name_search=SEARCH_667R, detail=DETAIL_667R)
        doc = pub.resolve_exact(REC_667R)
        assert doc["eoNumber"] == "0001201701030034"
        assert doc["number"] == "667-р"

    def test_name_search_found_ukaz(self, monkeypatch):
        """Указ-16: поиск по номеру не даёт совпадения, а номер+имя находит."""
        SEARCH_UKAZ = [{
            "eoNumber": "0001201701160034", "number": "16",
            "documentDate": "2017-01-16T00:00:00",
        }]
        DETAIL_UKAZ = {
            "eoNumber": "0001201701160034", "number": "16",
            "documentDate": "2017-01-16T00:00:00",
            "documentType": {"name": "Указ"},
            "signatoryAuthorities": [{"name": "Президент Российской Федерации"}],
        }
        REC_UKAZ = {
            "id": "ukaz-16", "type": "Указ Президента Российской Федерации", "number": "16",
            "date": "16.01.2017",
            "title": "О внесении изменений в некоторые акты Президента Российской Федерации",
        }
        self._patch(monkeypatch, number_search=[], name_search=SEARCH_UKAZ, detail=DETAIL_UKAZ)
        doc = pub.resolve_exact(REC_UKAZ)
        assert doc["eoNumber"] == "0001201701160034"


class TestValidateAgainstRecord:
    def test_mismatch_number(self):
        with pytest.raises(pub.DocumentMismatchError):
            pub._validate_against_record(
                {"number": "88-ФЗ", "documentDate": "2026-04-09T00:00:00",
                 "documentType": {"name": "Федеральный закон"}}, REC_MODERN)

    def test_mismatch_date(self):
        with pytest.raises(pub.DocumentMismatchError):
            pub._validate_against_record(
                {"number": "79-ФЗ", "documentDate": "2025-04-21T00:00:00",
                 "documentType": {"name": "Федеральный закон"}}, REC_MODERN)

    def test_mismatch_type(self):
        with pytest.raises(pub.DocumentMismatchError):
            pub._validate_against_record(
                {"number": "79-ФЗ", "documentDate": "2026-04-09T00:00:00",
                 "documentType": {"name": "Постановление"}}, REC_MODERN)

    def test_ok(self):
        pub._validate_against_record(DETAIL_2026, REC_MODERN)

    def test_ukaz_type_canonical(self):
        """Указ + Президент РФ канонизируется в 'Указ Президента Российской Федерации'."""
        doc = {
            "number": "16", "documentDate": "2017-01-16T00:00:00",
            "documentType": {"name": "Указ"},
            "signatoryAuthorities": [{"name": "Президент Российской Федерации"}],
        }
        rec = {"number": "16", "type": "Указ Президента Российской Федерации",
               "date": "16.01.2017"}
        pub._validate_against_record(doc, rec)

    def test_rasporyazhenie_type_canonical(self):
        """Распоряжение + Правительство РФ канонизируется в полную запись."""
        doc = {
            "number": "2867-р", "documentDate": "2016-12-28T00:00:00",
            "documentType": {"name": "Распоряжение"},
            "signatoryAuthorities": [{"name": "Правительство Российской Федерации"}],
        }
        rec = {"number": "2867-р", "type": "Распоряжение Правительства Российской Федерации",
               "date": "28.12.2016"}
        pub._validate_against_record(doc, rec)


class TestDownloadPdf:
    def test_download_valid(self, tmp_path, monkeypatch):
        good = b"%PDF-1.4\n%test\n" * 10
        monkeypatch.setattr(pub, "_get_bytes", lambda url: good)
        dest = tmp_path / "out.pdf"
        size = pub.download_pdf("0001202604090006", dest)
        assert dest.read_bytes() == good
        assert size == len(good)

    def test_download_invalid_pdf_stops(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pub, "_get_bytes", lambda url: b"not a pdf")
        dest = tmp_path / "out.pdf"
        with pytest.raises(pub.PublicAPIError):
            pub.download_pdf("0001202604090006", dest)
        assert not dest.exists()


# ============================================================================
# Integration (реальные запросы; auto-skip без сети)
# ============================================================================
def _network_available() -> bool:
    import socket
    try:
        socket.create_connection(("publication.pravo.gov.ru", 80), timeout=5).close()
        return True
    except OSError:
        return False


class TestIntegration:
    def setup_method(self):
        if not _network_available():
            pytest.skip("publication.pravo.gov.ru недоступен")

    def test_modern_exact_match_real(self):
        doc = pub.resolve_exact(REC_MODERN)
        assert doc["eoNumber"] == "0001202604090006"

    def test_old_79fz_not_found_real(self):
        with pytest.raises(pub.DocumentNotFoundError):
            pub.resolve_exact(REC_OLD_79)

    def test_download_real_pdf(self, tmp_path):
        dest = tmp_path / "real.pdf"
        size = pub.download_pdf("0001202604090006", dest)
        assert size == 2550273
        assert dest.read_bytes().startswith(b"%PDF-")
