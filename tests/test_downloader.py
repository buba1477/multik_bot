"""Тесты оркестрации download_documents.py (гибрид + инкрементальное скачивание).

Проверяются fail-closed ветки, разделение кэша и инкрементальная логика
(first / unchanged / revision-changed / ошибки), без реальной сети
(сетевые функции и PDF-конвертация мокингются).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import urllib.request

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import app.publication_api as pub  # noqa: E402
import app.pravo_resolver as legacy  # noqa: E402

_dl_path = PROJECT / "scripts" / "download_documents.py"
_spec = importlib.util.spec_from_file_location("download_documents", _dl_path)
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

FIX = Path(__file__).resolve().parent / "fixtures" / "publication"
DETAIL_2026 = json.loads((FIX / "document_detail_79fz_2026.json").read_text(encoding="utf-8"))

REC_PUB = {
    "id": "demo-modern", "type": "Федеральный закон", "number": "79-ФЗ",
    "date": "09.04.2026", "title": "О внесении изменений в ГПК РФ",
}
REC_LEG = {
    "id": "79-FZ", "type": "Федеральный закон", "number": "79-ФЗ",
    "date": "27.07.2004", "title": "О государственной гражданской службе РФ",
}

LEG_REV = {"rdk": 98, "label": "98 - от 08.03.2026 № 52-ФЗ (изм.)", "date": "08.03.2026"}
LEG_REV_NEW = {"rdk": 99, "label": "99 - от 01.04.2026 № 10-ФЗ (изм.)", "date": "01.04.2026"}


def _setup_pub_download(monkeypatch, pdf_bytes=b"%PDF-1.4\n%demo", pages=5):
    monkeypatch.setattr(pub, "resolve_exact", lambda rec: DETAIL_2026)
    monkeypatch.setattr(pub, "get_document", lambda eo: DETAIL_2026)
    monkeypatch.setattr(pub, "download_pdf",
                        lambda eo, dest: Path(dest).write_bytes(pdf_bytes) or len(pdf_bytes))
    monkeypatch.setattr(dd, "validate_pdf", lambda p: pages)
    monkeypatch.setattr(legacy, "find_latest_rdk", _no_legacy_call())


def _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\n%leg", pages=5,
                           valid=True):
    def not_found(rec):
        raise pub.DocumentNotFoundError("нет")
    monkeypatch.setattr(pub, "resolve_exact", not_found)
    monkeypatch.setattr(legacy, "resolve_document",
                        lambda num, title=None, date=None: {"nd": "102088054"})
    monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: rev)
    monkeypatch.setattr(legacy, "find_latest_rdk", lambda nd: (rev["rdk"], rev["label"]))
    monkeypatch.setattr(legacy, "print_url", lambda nd, rdk: "http://x/?print")
    monkeypatch.setattr(dd, "get_bytes",
                        lambda url: ("Федеральный закон от 27.07.2004 № 79-ФЗ "
                                     "О государственной гражданской службы Российской Федерации")
                        .encode("windows-1251"))
    if valid:
        def _mock_convert(html_path, pdf_path):
            pdf_path.write_bytes(pdf_bytes)
            return {"title": "mock", "pages": pages, "pdf_size": len(pdf_bytes), "html_len": 1000}
        monkeypatch.setattr(dd, "convert_html_to_pdf", _mock_convert)
        monkeypatch.setattr(dd, "validate_pdf", lambda p: pages)
    else:
        # невалидный новый PDF: конвертация кладёт мусор, validate_pdf -> ошибка
        def _mock_convert_invalid(html_path, pdf_path):
            pdf_path.write_bytes(b"not a pdf")
            return {"title": "mock", "pages": pages, "pdf_size": 9, "html_len": 1000}
        monkeypatch.setattr(dd, "convert_html_to_pdf", _mock_convert_invalid)
        def _bad_validate(_path):
            raise dd.ResolutionError("невалидный PDF")
        monkeypatch.setattr(dd, "validate_pdf", _bad_validate)


def _no_legacy_call():
    def _boom(*a, **k):
        raise AssertionError("legacy /proxy/ips/ НЕ должен вызываться")
    return _boom


def _read_entry(cache_path, doc_id):
    return json.loads(cache_path.read_text(encoding="utf-8"))[doc_id]


# ============================================================================
# Разрешение метода (fail-closed)
# ============================================================================
class TestResolveDoc:
    def test_api_exact_match_publication_no_legacy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pub, "resolve_exact", lambda rec: DETAIL_2026)
        monkeypatch.setattr(legacy, "resolve_document", _no_legacy_call())
        entry = dd.resolve_doc(REC_PUB, cache_path=tmp_path / "c.json")
        assert entry["method"] == "publication"
        assert entry["detail"]["eoNumber"] == "0001202604090006"

    def test_api_not_found_legacy_fallback(self, tmp_path, monkeypatch):
        def not_found(rec):
            raise pub.DocumentNotFoundError("документа нет на портале")
        monkeypatch.setattr(pub, "resolve_exact", not_found)
        monkeypatch.setattr(legacy, "resolve_document",
                            lambda num, title=None, date=None: {"nd": "102088054"})
        entry = dd.resolve_doc(REC_LEG, cache_path=tmp_path / "c.json")
        assert entry["method"] == "legacy"
        assert entry["detail"]["nd"] == "102088054"

    def test_api_error_stop_no_legacy(self, tmp_path, monkeypatch):
        def net_error(rec):
            raise pub.PublicAPIError("API недоступен")
        monkeypatch.setattr(pub, "resolve_exact", net_error)
        monkeypatch.setattr(legacy, "resolve_document", _no_legacy_call())
        with pytest.raises(pub.PublicAPIError):
            dd.resolve_doc(REC_LEG, cache_path=tmp_path / "c.json")

    def test_api_ambiguity_stop_no_legacy(self, tmp_path, monkeypatch):
        def ambiguous(rec):
            raise pub.DocumentMismatchError("Неоднозначный результат")
        monkeypatch.setattr(pub, "resolve_exact", ambiguous)
        monkeypatch.setattr(legacy, "resolve_document", _no_legacy_call())
        with pytest.raises(pub.DocumentMismatchError):
            dd.resolve_doc(REC_PUB, cache_path=tmp_path / "c.json")


# ============================================================================
# Кэш метода (publication / legacy)
# ============================================================================
class TestCacheResolveDoc:
    def test_publication_entry_cached(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        def resolve(rec):
            calls["n"] += 1
            return DETAIL_2026
        monkeypatch.setattr(pub, "resolve_exact", resolve)
        cpath = tmp_path / "c.json"
        dd.resolve_doc(REC_PUB, cache_path=cpath)
        dd.resolve_doc(REC_PUB, cache_path=cpath)
        assert calls["n"] == 1
        data = json.loads(cpath.read_text(encoding="utf-8"))
        assert data["demo-modern"]["method"] == "publication"

    def test_legacy_entry_cached(self, tmp_path, monkeypatch):
        def not_found(rec):
            raise pub.DocumentNotFoundError("нет")
        monkeypatch.setattr(pub, "resolve_exact", not_found)
        calls = {"n": 0}
        def lresolve(num, title=None, date=None):
            calls["n"] += 1
            return {"nd": "102088054"}
        monkeypatch.setattr(legacy, "resolve_document", lresolve)
        cpath = tmp_path / "c.json"
        dd.resolve_doc(REC_LEG, cache_path=cpath)
        dd.resolve_doc(REC_LEG, cache_path=cpath)
        assert calls["n"] == 1
        data = json.loads(cpath.read_text(encoding="utf-8"))
        assert data["79-FZ"]["method"] == "legacy"
        assert data["79-FZ"]["detail"]["nd"] == "102088054"


# ============================================================================
# Базовые сценарии скачивания (первый запуск по методу)
# ============================================================================
class TestFirstDownload:
    def test_publication_writes_pdf_without_legacy(self, tmp_path, monkeypatch):
        _setup_pub_download(monkeypatch)
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_PUB, cache_path=cpath)
        out = tmp_path / "demo-modern.pdf"
        assert out.exists()
        entry = _read_entry(cpath, "demo-modern")
        assert entry["method"] == "publication"
        assert entry["revision"]["id"] == "0001202604090006"
        assert entry["pdf_path"] == str(out)
        assert entry["pdf_pages"] == 5

    def test_legacy_writes_pdf(self, tmp_path, monkeypatch):
        _setup_legacy_download(monkeypatch)
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        out = tmp_path / "79-FZ.pdf"
        assert out.exists()
        entry = _read_entry(cpath, "79-FZ")
        assert entry["method"] == "legacy"
        assert entry["revision"]["id"] == 98
        assert entry["downloaded_at"]
        assert entry["pdf_path"] == str(out)
        assert entry["pdf_size"] == len(b"%PDF-1.4\n%leg")
        assert entry["pdf_pages"] == 5
        assert entry.get("pdf_source") == "playwright"
        assert entry.get("html_path")


# ============================================================================
# Инкрементальная логика
# ============================================================================
class TestIncremental:
    def test_unchanged_does_not_download(self, tmp_path, monkeypatch):
        _setup_legacy_download(monkeypatch)
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        first_stat = (tmp_path / "79-FZ.pdf").stat().st_mtime_ns

        # повторный запуск с той же редакцией -> скачивание не выполняется
        dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").stat().st_mtime_ns == first_stat
        assert not (tmp_path / "79-FZ.new.pdf").exists()

    def test_revision_changed_downloads_and_replaces(self, tmp_path, monkeypatch):
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nOLD")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == b"%PDF-1.4\nOLD"

        # редакция изменилась -> новый PDF, атомарная замена, кэш обновлён
        monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: LEG_REV_NEW)
        monkeypatch.setattr(legacy, "find_latest_rdk", lambda nd: (LEG_REV_NEW["rdk"], LEG_REV_NEW["label"]))
        monkeypatch.setattr(dd, "convert_html_to_pdf",
                            lambda html, out: (
                                out.write_bytes(b"%PDF-1.4\nNEW"),
                                {"title": "m", "pages": 5, "pdf_size": 14, "html_len": 100}
                            )[1])
        dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == b"%PDF-1.4\nNEW"
        assert not (tmp_path / "79-FZ.new.pdf").exists()
        entry = _read_entry(cpath, "79-FZ")
        assert entry["revision"]["id"] == 99

    def test_invalid_new_pdf_keeps_old(self, tmp_path, monkeypatch):
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nOLD")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)   # рабочий первый заход
        old_bytes = (tmp_path / "79-FZ.pdf").read_bytes()

        # теперь новая редакция, но новый PDF невалидный
        monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: LEG_REV_NEW)
        monkeypatch.setattr(legacy, "find_latest_rdk", lambda nd: (LEG_REV_NEW["rdk"], LEG_REV_NEW["label"]))
        _setup_legacy_download(monkeypatch, rev=LEG_REV_NEW, valid=False)
        with pytest.raises(dd.ResolutionError):
            dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == old_bytes   # старый цел
        assert not (tmp_path / "79-FZ.new.pdf").exists()
        assert _read_entry(cpath, "79-FZ")["revision"]["id"] == 98  # кэш не тронут

    def test_network_error_keeps_old(self, tmp_path, monkeypatch):
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nOLD")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        old_bytes = (tmp_path / "79-FZ.pdf").read_bytes()

        def net_error(url):
            raise OSError("сеть недоступна")
        monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: LEG_REV_NEW)
        monkeypatch.setattr(dd, "get_bytes", net_error)
        with pytest.raises(OSError):
            dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == old_bytes
        assert _read_entry(cpath, "79-FZ")["revision"]["id"] == 98

    def test_cache_corruption_redownloads(self, tmp_path, monkeypatch):
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nNEW")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        cpath.write_text("{broken json", encoding="utf-8")
        dd.download_one(REC_LEG, cache_path=cpath)   # воспринимается как первый запуск
        assert (tmp_path / "79-FZ.pdf").exists()
        # кэш теперь валидный и с revision
        entry = _read_entry(cpath, "79-FZ")
        assert entry["revision"]["id"] == 98

    def test_missing_local_pdf_redownloads(self, tmp_path, monkeypatch):
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nNEW")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        (tmp_path / "79-FZ.pdf").unlink()          # PDF пропал, хотя кэш есть

        monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: LEG_REV)
        monkeypatch.setattr(dd, "convert_html_to_pdf",
                            lambda html, out: (
                                out.write_bytes(b"%PDF-1.4\nAGAIN"),
                                {"title": "m", "pages": 5, "pdf_size": 16, "html_len": 100}
                            )[1])
        dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == b"%PDF-1.4\nAGAIN"
        assert _read_entry(cpath, "79-FZ")["downloaded_at"]


class TestNetworkErrorsAtMainLevel:
    @pytest.mark.parametrize(
        ("net_exc", "expected_text"),
        [
            (TimeoutError("timed out"), "timeout при обращении к источнику"),
            (ConnectionError("connection refused"), "ошибка соединения"),
        ],
    )
    def test_network_error_one_doc_does_not_stop_others(
        self, tmp_path, monkeypatch, capsys, net_exc, expected_text
    ):
        """Timeout/connection error одного документа не останавливает остальных."""
        _setup_pub_download(monkeypatch)
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)

        def resolve_exact(rec):
            if rec["id"] == REC_LEG["id"]:
                raise pub.DocumentNotFoundError("нет в базе официального опубликования")
            return DETAIL_2026

        monkeypatch.setattr(pub, "resolve_exact", resolve_exact)

        def net_error(num, title=None, date=None):
            raise net_exc

        monkeypatch.setattr(legacy, "resolve_document", net_error)

        registry = tmp_path / "documents.json"
        registry.write_text(
            json.dumps({"documents": [REC_LEG, REC_PUB]}, ensure_ascii=False),
            encoding="utf-8",
        )
        monkeypatch.setattr(dd, "REGISTRY", registry)
        cpath = tmp_path / "resolved.json"
        monkeypatch.setattr(dd, "CACHE_PATH", cpath)

        dd.main()

        out = capsys.readouterr().out
        assert f"!! ОШИБКА СЕТИ: {REC_LEG['id']}: {expected_text}" in out
        assert "== document: demo-modern ==" in out
        assert (tmp_path / "demo-modern.pdf").exists()      # второй документ обработан
        assert not (tmp_path / "79-FZ.pdf").exists()        # упавший документ не скачан
        assert not (tmp_path / "79-FZ.new.pdf").exists()    # временных файлов нет
        # требование: запись об успешно разрешённом документе не создаётся,
        # если legacy resolve не завершился успешно
        cache = json.loads(cpath.read_text(encoding="utf-8"))
        assert "79-FZ" not in cache
        assert cache["demo-modern"]["method"] == "publication"

    def test_document_mismatch_does_not_stop_others(
        self, tmp_path, monkeypatch, capsys,
    ):
        """DocumentMismatchError (реальное несовпадение) не прерывает остальные."""
        _setup_pub_download(monkeypatch)
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)

        def resolve_exact(rec):
            if rec["id"] == REC_LEG["id"]:
                raise pub.DocumentMismatchError("несовпадение реквизитов: тестовый документ")
            return DETAIL_2026

        monkeypatch.setattr(pub, "resolve_exact", resolve_exact)
        # DocumentMismatchError НЕ должен приводить к вызову legacy
        monkeypatch.setattr(legacy, "resolve_document", _no_legacy_call())

        registry = tmp_path / "documents.json"
        registry.write_text(
            json.dumps({"documents": [REC_LEG, REC_PUB]}, ensure_ascii=False),
            encoding="utf-8",
        )
        monkeypatch.setattr(dd, "REGISTRY", registry)
        cpath = tmp_path / "resolved.json"
        monkeypatch.setattr(dd, "CACHE_PATH", cpath)

        dd.main()

        out = capsys.readouterr().out
        assert "!! ОСТАНОВ" in out or "!! ОШИБКА" in out
        assert "== document: demo-modern ==" in out
        assert (tmp_path / "demo-modern.pdf").exists()  # второй документ обработан
        # Документ с MismatchError не должен быть добавлен в кэш
        cache = json.loads(cpath.read_text(encoding="utf-8"))
        assert "79-FZ" not in cache
        assert cache["demo-modern"]["method"] == "publication"


class TestDataIntegrity:
    """SHA-256, Content-Length, гарантированная очистка временных файлов."""

    def test_get_bytes_checks_content_length(self, monkeypatch):
        """get_bytes проверяет Content-Length и бросает ResolutionError при несовпадении."""
        class _MockResp:
            headers = {"Content-Length": "100"}

            def read(self):
                return b"short"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout=120: _MockResp())
        with pytest.raises(dd.ResolutionError, match="Content-Length mismatch"):
            dd.get_bytes("http://example.com")

    def test_content_length_mismatch_keeps_old(self, tmp_path, monkeypatch):
        """Content-Length mismatch не удаляет старый PDF и не обновляет кэш."""
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nOLD")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        old_bytes = (tmp_path / "79-FZ.pdf").read_bytes()

        monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: LEG_REV_NEW)
        monkeypatch.setattr(legacy, "find_latest_rdk",
                            lambda nd: (LEG_REV_NEW["rdk"], LEG_REV_NEW["label"]))
        monkeypatch.setattr(dd, "get_bytes", lambda url: (_ for _ in ()).throw(
            dd.ResolutionError("Content-Length mismatch: ожидалось 1000 байт, получено 500")))
        with pytest.raises(dd.ResolutionError, match="Content-Length mismatch"):
            dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == old_bytes
        assert not (tmp_path / "79-FZ.new.pdf").exists()
        assert not (tmp_path / "79-FZ.new.html").exists()
        assert _read_entry(cpath, "79-FZ")["revision"]["id"] == 98

    def test_connection_abort_keeps_old(self, tmp_path, monkeypatch):
        """Обрыв соединения (ConnectionResetError) не удаляет старый PDF."""
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nOLD")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        monkeypatch.setattr(dd, "RAW_HTML_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        old_bytes = (tmp_path / "79-FZ.pdf").read_bytes()

        monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: LEG_REV_NEW)
        monkeypatch.setattr(legacy, "find_latest_rdk",
                            lambda nd: (LEG_REV_NEW["rdk"], LEG_REV_NEW["label"]))
        monkeypatch.setattr(dd, "get_bytes", lambda url: (_ for _ in ()).throw(
            ConnectionResetError("Connection aborted by peer")))
        with pytest.raises(ConnectionResetError):
            dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == old_bytes
        assert not (tmp_path / "79-FZ.new.pdf").exists()
        assert _read_entry(cpath, "79-FZ")["revision"]["id"] == 98

    def test_successful_download_stores_sha256(self, tmp_path, monkeypatch):
        """После успешного скачивания в кэше появляется pdf_sha256."""
        _setup_pub_download(monkeypatch, pdf_bytes=b"%PDF-1.4\nhello\nworld\nend", pages=3)
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_PUB, cache_path=cpath)
        entry = _read_entry(cpath, "demo-modern")
        assert "pdf_sha256" in entry
        expected = hashlib.sha256(b"%PDF-1.4\nhello\nworld\nend").hexdigest()
        assert entry["pdf_sha256"] == expected

    def test_successful_replacement_stores_new_sha256(self, tmp_path, monkeypatch):
        """Обновление редакции сохраняет новый sha256, отличный от старого."""
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nOLD", pages=2)
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        old_entry = _read_entry(cpath, "79-FZ")
        old_sha256 = old_entry.get("pdf_sha256")
        assert old_sha256

        monkeypatch.setattr(legacy, "find_latest_revision", lambda nd: LEG_REV_NEW)
        monkeypatch.setattr(legacy, "find_latest_rdk",
                            lambda nd: (LEG_REV_NEW["rdk"], LEG_REV_NEW["label"]))
        monkeypatch.setattr(dd, "convert_html_to_pdf",
                            lambda html, out: (
                                out.write_bytes(b"%PDF-1.4\nNEW"),
                                {"title": "m", "pages": 3, "pdf_size": 14, "html_len": 100}
                            )[1])
        monkeypatch.setattr(dd, "validate_pdf", lambda p: 3)
        dd.download_one(REC_LEG, cache_path=cpath)
        new_entry = _read_entry(cpath, "79-FZ")
        new_sha256 = new_entry.get("pdf_sha256")
        assert new_sha256 != old_sha256
        expected_new = hashlib.sha256(b"%PDF-1.4\nNEW").hexdigest()
        assert new_sha256 == expected_new

    def test_manual_pdf_modification_detected_redownloads(self, tmp_path, monkeypatch):
        """Локальный PDF изменён вручную → SHA-256 mismatch → скачать заново."""
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nORIG")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == b"%PDF-1.4\nORIG"

        # вручную изменяем содержимое локального PDF (правка/повреждение)
        (tmp_path / "79-FZ.pdf").write_bytes(b"%PDF-1.4\nTAMPERED")

        # та же редакция, но SHA-256 не совпадает → перезагрузка
        monkeypatch.setattr(dd, "convert_html_to_pdf",
                            lambda html, out: (
                                out.write_bytes(b"%PDF-1.4\nRESTORED"),
                                {"title": "m", "pages": 5, "pdf_size": 19, "html_len": 100}
                            )[1])
        dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == b"%PDF-1.4\nRESTORED"
        assert not (tmp_path / "79-FZ.new.pdf").exists()
        entry = _read_entry(cpath, "79-FZ")
        assert entry["pdf_sha256"] == hashlib.sha256(b"%PDF-1.4\nRESTORED").hexdigest()

    def test_truncated_pdf_detected_redownloads(self, tmp_path, monkeypatch):
        """Локальный PDF обрезан/повреждён → SHA-256 mismatch → скачать заново."""
        _setup_legacy_download(monkeypatch, rev=LEG_REV, pdf_bytes=b"%PDF-1.4\nFULL")
        monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
        cpath = tmp_path / "c.json"
        dd.download_one(REC_LEG, cache_path=cpath)

        # обрезаем PDF (потеря данных в конце файла)
        pdf = tmp_path / "79-FZ.pdf"
        pdf.write_bytes(pdf.read_bytes()[:5])

        monkeypatch.setattr(dd, "convert_html_to_pdf",
                            lambda html, out: (
                                out.write_bytes(b"%PDF-1.4\nRECOVERED"),
                                {"title": "m", "pages": 5, "pdf_size": 20, "html_len": 100}
                            )[1])
        dd.download_one(REC_LEG, cache_path=cpath)
        assert (tmp_path / "79-FZ.pdf").read_bytes() == b"%PDF-1.4\nRECOVERED"
        entry = _read_entry(cpath, "79-FZ")
        assert entry["pdf_sha256"] == hashlib.sha256(b"%PDF-1.4\nRECOVERED").hexdigest()



class TestDocumentLostForce:
    """_document_lost_force(): обнаружение полной утраты силы в legacy-HTML."""

    def test_active_667r_passes(self):
        """Синтетическая действующая редакция 667-р → False (не утратила силу)."""
        html = (
            "<html><body>"
            "Распоряжение Правительства Российской Федерации "
            "от 26.05.2005 г. № 667-р "
            "Изменения на 22.04.2022 г. См. последующие изменения"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is False

    def test_lost_force_in_header_detected(self):
        """'Утратило силу - Постановление...' в заголовке → True."""
        html = (
            "<html><body>"
            "Распоряжение Правительства Российской Федерации "
            "от 26.05.2005 г. № 667-р г. Москва "
            "Утратило силу - Постановление Правительства Российской Федерации "
            "от 28.11.2024 № 1664"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is True

    def test_lost_force_masculine_detected(self):
        """'Утратил силу - Указ...' в заголовке → True."""
        html = (
            "<html><body>"
            "УКАЗ ПРЕЗИДЕНТА РОССИЙСКОЙ ФЕДЕРАЦИИ "
            "Утратил силу - Указ Президента Российской Федерации "
            "от 10.10.2024 № 871"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is True

    def test_not_acts_detected(self):
        """'Не действует' в заголовке → True."""
        html = (
            "<html><body>"
            "Распоряжение Правительства Российской Федерации "
            "от 26.05.2005 г. № 667-р г. Москва "
            "Не действует"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is True

    def test_partial_article_lost_not_blocked(self):
        """'Статья утратила силу' → False (частичная утрата)."""
        html = (
            "<html><body>"
            "Федеральный закон от 27.07.2004 № 79-ФЗ "
            "О государственной гражданской службе Российской Федерации"
            "<p>Статья 1. Предмет регулирования</p>"
            "<p>Статья 2 утратила силу - Федеральный закон от 01.01.2020 № 1-ФЗ</p>"
            "<p>Статья 3. Основные понятия</p>"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is False

    def test_partial_paragraph_lost_not_blocked(self):
        """'Пункт утратил силу' → False (частичная утрата)."""
        html = (
            "<html><body>"
            "Постановление Правительства Российской Федерации "
            "от 01.01.2020 № 1 "
            "<p>1. Утвердить...</p>"
            "<p>Пункт 2 утратил силу - Постановление от 01.01.2021 № 2</p>"
            "<p>3. Контроль...</p>"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is False

    def test_partial_chapter_lost_not_blocked(self):
        """'Глава утратила силу' → False (частичная утрата)."""
        html = (
            "<html><body>"
            "Федеральный закон от 27.07.2004 № 79-ФЗ "
            "<p>Глава 1 утратила силу - Федеральный закон от 01.01.2020 № 1-ФЗ</p>"
            "<p>Глава 2. Должности гражданской службы</p>"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is False

    def test_mixed_partial_and_active_not_blocked(self):
        """Смесь частичных утрат и активных норм → False."""
        html = (
            "<html><body>"
            "Федеральный закон от 27.07.2004 № 79-ФЗ "
            "<p>Статья 1. Предмет</p>"
            "<p>Статья 2 утратила силу</p>"
            "<p>Пункт 3 утратил силу</p>"
            "<p>Часть 4 утратила силу</p>"
            "<p>Раздел II. Особенная часть</p>"
            "</body></html>"
        )
        assert dd._document_lost_force(html) is False

    def test_empty_html_returns_false(self):
        """Пустой/нерелевантный HTML → False."""
        assert dd._document_lost_force("<html><body>нет данных</body></html>") is False


class TestRegistry:
    def test_documents_json_has_no_nd(self):
        reg = json.loads((dd.PROJECT / "documents.json").read_text(encoding="utf-8"))
        for doc in reg["documents"]:
            assert "nd" not in doc, f"{doc.get('id')} не должен содержать nd"
            assert doc["number"] and doc["date"] and doc.get("title")


PRAVO_FIX = Path(__file__).resolve().parent / "fixtures" / "pravo"

NUM_667R = "667-р"
DATE_667R = "26.05.2005"
TITLE_667R = (
    "Об утверждении формы анкеты для участия в конкурсе на замещение вакантной "
    "должности государственной гражданской службы Российской Федерации"
)


class TestVerifyDocument:
    """verify_document(): number + date строго (fail-closed), title — fail-open."""

    @pytest.fixture()
    def html_667r(self) -> str:
        raw = (PRAVO_FIX / "print_667r.html").read_bytes()
        return raw.decode("windows-1251", "replace")

    def test_667r_real_html_passes(self, html_667r):
        """Правильные number + date + title → True."""
        assert dd.verify_document(html_667r, NUM_667R, DATE_667R, TITLE_667R) is True

    def test_title_mismatch_warns_but_passes(self, html_667r, caplog):
        """Правильные number + date, но title не совпадает → True + warning."""
        other_title = "Об утверждении правил пожарной безопасности на объектах"
        with caplog.at_level("WARNING", logger=dd.logger.name):
            assert dd.verify_document(html_667r, NUM_667R, DATE_667R, other_title) is True
        assert any("title не найден" in r.getMessage() for r in caplog.records)

    def test_wrong_number_rejected(self, html_667r):
        """Неправильный number → False (fail-closed)."""
        assert dd.verify_document(html_667r, "999-р", DATE_667R, TITLE_667R) is False

    def test_wrong_date_rejected(self, html_667r):
        """Неправильная date → False (fail-closed)."""
        assert dd.verify_document(html_667r, NUM_667R, "01.01.2000", TITLE_667R) is False

    def test_79fz_number_date_passes(self):
        """79-ФЗ: правильные number + date → True (название из «шапочных» слов не мешает)."""
        # Шапка реального print_url-представления nd=102088054
        html = (
            "<html><body><span>Федеральный закон от 27.07.2004 г. № 79-ФЗ "
            '("Парламентская газета" от 31.07.2004 г.; Собрание законодательства '
            "Российской Федерации, 02.08.2004, № 31, ст. 3215)</span>"
            "<p>О государственной гражданской службе Российской Федерации</p>"
            "</body></html>"
        )
        assert dd.verify_document(
            html,
            "79-ФЗ",
            "27.07.2004",
            "О государственной гражданской службе Российской Федерации",
        ) is True
