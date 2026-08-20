"""Загрузчик НПА: официальный API publication.pravo.gov.ru + legacy-резерв (инкрементальный).

Основной путь — официальный API ``publication.pravo.gov.ru``:

    documents.json -> /api/Documents (поиск) -> eoNumber
        -> /file/pdf?eoNumber=... -> raw/<id>.pdf

Разрешение только по стабильным критериям: number + date (+ type),
fail-closed: при неоднозначности/несовпадении — ОСТАНОВ, без угадывания.

Legacy-резерв (app/pravo_resolver.py, /proxy/ips/) используется ТОЛЬКО когда
API подтверждённо не находит документ (DocumentNotFoundError) — т.е. акт
отсутствует в базе официального опубликования (старые законы).

ИНКРЕМЕНТАЛЬНОЕ СКАЧИВАНИЕ:
  В app/resolved_documents.json для каждого документа хранится состояние
  последней успешно скачанной редакции (revision + downloaded_at + pdf_path).
  Если текущая последняя редакция совпадает с сохранённой и локальный PDF на
  месте -> скачивание пропускается ("unchanged → skip download").
  Если отличается / PDF отсутствует / кэш повреждён -> новый PDF скачивается
  во временный файл, проверяется, и только после успешной проверки атомарно
  заменяет старый (os.replace). При ошибке старый PDF не удаляется.

Кэш app/resolved_documents.json разделён по способу разрешения
(publication / legacy), при необходимости — принудительный re-resolve.
"""

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.pravo_resolver as legacy  # noqa: E402  (резервный механизм)
import app.publication_api as pub  # noqa: E402  (основной официальный API)
from app.ingestion.html_to_pdf import convert_html_to_pdf  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
REGISTRY = PROJECT / "documents.json"
RAW_DIR = PROJECT / "raw"
RAW_HTML_DIR = PROJECT / "raw_html"
CACHE_PATH = PROJECT / "app" / "resolved_documents.json"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ============================================================================
# Диагностическое логирование IPS lookup
# ============================================================================
def _log_ips_lookup(doc_id: str, method: str, nd: str | None,
                     ips_status: int | str, ips_text_len: int,
                     ips_img_count: int, ips_is_textual: bool,
                     source_selected: str) -> None:
    """Единообразный лог для каждого IPS lookup."""
    print(f"  ips_debug | {doc_id} | method={method} | nd={nd or 'N/A'} | "
          f"status={ips_status} | text_len={ips_text_len} | "
          f"img={ips_img_count} | is_textual={ips_is_textual} | "
          f"source={source_selected}")


# ============================================================================
# HTTP/вспомогательные (общие для обоих путей)
# ============================================================================
def get_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def decode_html(data: bytes) -> str:
    return data.decode("windows-1251", "replace")


class ResolutionError(RuntimeError):
    """Документ не удалось однозначно определить ни одним способом."""


# ============================================================================
# Кэш (единая схема, разделение по method)
# ============================================================================
def load_cache(path: Path | None = None) -> dict:
    cache_file = path or CACHE_PATH
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict, path: Path | None = None) -> None:
    cache_file = path or CACHE_PATH
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(cache_file)


def persist_entry(doc_id: str, entry: dict, cache_path: Path | None = None) -> None:
    """Обновить entry конкретного документа в кэше (атомарно)."""
    cache = load_cache(cache_path)
    cache[doc_id] = entry
    save_cache(cache, cache_path)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================================
# Разрешение документа (метод + detail) — определяется один раз и кэшируется
# ============================================================================
def resolve_doc(doc: dict, force: bool = False, cache_path: Path | None = None) -> dict:
    """Определить документ: основной путь — официальный API, резерв — legacy.

    Возвращает entry кэша: {method: 'publication'|'legacy', resolved_at, detail}.
    """
    cache = load_cache(cache_path)
    doc_id = doc["id"]
    if not force and cache.get(doc_id, {}).get("method"):
        return cache[doc_id]

    # --- ОСНОВНОЙ путь: официальный API (fail-closed) ---
    try:
        matched = pub.resolve_exact(doc)
    except pub.DocumentNotFoundError:
        # Документа нет в базе официального опубликования (старый акт) -> резерв
        # Используем nd из documents.json, если он уже указан
        nd_from_doc = doc.get("nd")
        if nd_from_doc:
            nd_val = str(nd_from_doc)
        else:
            try:
                legacy_rec = legacy.resolve_document(
                    doc["number"], doc.get("title"), doc.get("date")
                )
            except legacy.DocumentNotFoundError as exc:
                raise ResolutionError(
                    f"{doc_id}: не найден ни в API публикации, ни в legacy-резерве: {exc}"
                ) from exc
            nd_val = str(legacy_rec["nd"])
        entry = {
            "method": "legacy",
            "resolved_at": iso_now(),
            "detail": {"nd": nd_val},
        }
        cache[doc_id] = entry
        save_cache(cache, cache_path)
        return entry

    entry = {
        "method": "publication",
        "resolved_at": iso_now(),
        "detail": {
            "eoNumber": matched["eoNumber"],
            "number": matched.get("number"),
            "documentDate": matched.get("documentDate"),
        },
    }
    cache[doc_id] = entry
    save_cache(cache, cache_path)
    return entry


# ============================================================================
# Проверка/конвертация PDF
# ============================================================================
def verify_document(html: str, number: str, title: str) -> bool:
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    words = title.split()
    if not words:
        return number in text
    core = (
        " ".join(words[1:3]) if words[0].lower() in ("о", "об") else " ".join(words[:3])
    )
    return (number in text) and (core.lower() in text.lower())



def pdf_pages(path: Path) -> int:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return len(reader.pages)
    except Exception:
        raw = path.read_bytes()
        return raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")


def validate_pdf(path: Path) -> int:
    """Проверить, что это валидный непустой PDF; вернуть число страниц (>0)."""
    if not path.exists() or path.stat().st_size == 0:
        raise ResolutionError(f"PDF файл пуст/отсутствует: {path}")
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ResolutionError(f"Ответ не является PDF: {path}")
    pages = pdf_pages(path)
    if pages < 1:
        raise ResolutionError(f"PDF имеет 0 страниц: {path}")
    return pages


# ============================================================================
# Текущая редакция документа (идентификатор для сравнения)
# ============================================================================
def _revision_current(doc: dict, entry: dict) -> dict:
    """Актуальное состояние последней редакции для метода разрешения.

    legacy:      {id: rdk, label, date}
    publication: {id: eoNumber, fingerprint: eo|docDate|publishDate|pdfLen|pages, label}
    Сравнение идёт по содержимому этих словарей (а не по времени запуска).
    """
    method = entry["method"]
    if method == "publication":
        eo = entry["detail"]["eoNumber"]
        m = pub.get_document(eo)
        return {
            "id": eo,
            "fingerprint": "|".join(
                str(x)
                for x in [
                    eo,
                    m.get("documentDate"),
                    m.get("publishDateShort"),
                    m.get("pdfFileLength"),
                    m.get("pagesCount"),
                ]
            ),
            "label": (m.get("complexName") or m.get("title") or eo),
            "publishDateShort": m.get("publishDateShort"),
            "pagesCount": m.get("pagesCount"),
        }
    rev = legacy.find_latest_revision(entry["detail"]["nd"])
    if rev is None:
        raise ResolutionError(f"{doc['id']}: нет доступных редакций (legacy)")
    return {"id": rev["rdk"], "label": rev["label"], "date": rev["date"]}


def _revision_unchanged(entry: dict, current_rev: dict, out_pdf: Path) -> bool:
    """True если кэшированная редакция == текущей и необходимые файлы существуют.

    Для legacy: проверяет наличие HTML и PDF.
    Для publication: проверяет наличие PDF (и nd для IPS lookup).
    """
    cached = entry.get("revision")
    if not cached:
        return False
    if cached != current_rev:
        return False

    method = entry.get("method")
    if method == "legacy":
        # Для legacy необходимы оба файла: HTML (источник RAG) и PDF (архив)
        doc_id = out_pdf.stem
        html_path = RAW_HTML_DIR / f"{doc_id}.html"
        if not html_path.exists() or not out_pdf.exists():
            return False
        return True

    # publication
    if not out_pdf.exists():
        return False
    # Если нет nd, пытаемся заново (IPS lookup)
    if not entry.get("detail", {}).get("nd"):
        return False
    return True


# ============================================================================
# Скачивание документа: IPS HTML + (опционально) оригинальный PDF
# ============================================================================
def _download_to_tmp(doc: dict, entry: dict) -> tuple[Path | None, int, int, str | None]:
    """Загрузить документ. Возвращает (tmp_pdf, size, pages, html_sha256).

    * html_sha256 - SHA256 скачанного HTML; None если HTML не сохранялся.
    * tmp_pdf     - временный путь к PDF, если PDF получен; None при ошибке PDF.
    * size/pages  - 0 если PDF не получен.

    LEGACY-ветка:
      IPS HTML является основным источником для RAG.
      Сохраняется в raw_html/<id>.html.
      Из этого же HTML через Playwright + bundled Chromium создаётся архивный PDF.
      PDF сохраняется в raw/<id>.pdf.

    PUBLICATION-ветка (основной сценарий):
      Оригинальный PDF из API сохраняется.
      После этого IPS HTML как дополнительный источник для RAG.

    PUBLICATION-ветка (fallback):
      Если оригинальный PDF недоступен, но IPS HTML найден и является textual,
      PDF создаётся из IPS HTML через Playwright.

    На ошибке выбрасывает исключение, временные файлы подчищает,
    при этом целевые RAW_DIR/<id>.pdf / RAW_HTML_DIR/<id>.html не трогает.
    """
    doc_id = doc["id"]
    RAW_DIR.mkdir(exist_ok=True)
    RAW_HTML_DIR.mkdir(exist_ok=True)
    tmp_pdf = RAW_DIR / f"{doc_id}.new.pdf"
    html_sha256: str | None = None
    pdf_source: str | None = None

    try:
        if entry["method"] == "publication":
            eo = entry["detail"]["eoNumber"]
            # 1. Оригинальный PDF из официального API
            original_pdf_ok = False
            try:
                size = pub.download_pdf(eo, tmp_pdf)
                pages = validate_pdf(tmp_pdf)
                original_pdf_ok = True
                pdf_source = "original"
            except Exception as exc:
                print(f"  ! original PDF недоступен: {exc}. Пробую fallback через IPS HTML + Playwright.")
                _silent_unlink(tmp_pdf)

            # 2. IPS HTML как дополнительный источник
            # Приоритет: doc["nd"] > entry["detail"]["nd"] > find_nd_by_record()
            nd = doc.get("nd") or entry["detail"].get("nd")
            if not nd:
                nd_info = legacy.find_nd_by_record(doc)
                if nd_info:
                    nd = nd_info["nd"]

            if nd:
                ips_data, ips_meta = legacy.get_ips_print_html(nd)
                if ips_data and ips_meta.get("is_textual"):
                    entry["detail"]["nd"] = nd
                    raw_html_dest = RAW_HTML_DIR / f"{doc_id}.html"
                    raw_html_dest.write_bytes(ips_data)
                    html_sha256 = hashlib.sha256(ips_data).hexdigest()
                    _log_ips_lookup(
                        doc_id, "publication", nd,
                        ips_meta["status"], ips_meta["text_length"],
                        ips_meta["img_count"], ips_meta["is_textual"],
                        "IPS_HTML"
                    )
                    # Fallback: если оригинальный PDF недоступен — генерируем из HTML
                    if not original_pdf_ok:
                        print(f"  генерация PDF из IPS HTML через Playwright (fallback)")
                        try:
                            metrics = convert_html_to_pdf(raw_html_dest, tmp_pdf)
                            size = metrics["pdf_size"]
                            pages = validate_pdf(tmp_pdf)
                            pdf_source = "playwright"
                            print(f"  playwright PDF успешно создан: {size} байт, {pages} стр.")
                        except Exception as pw_exc:
                            print(f"  !! Playwright fallback тоже не удался: {pw_exc}")
                            _silent_unlink(tmp_pdf)
                            raise ResolutionError(
                                f"{doc_id}: оригинальный PDF недоступен и "
                                f"Playwright fallback не удался"
                            ) from pw_exc
                else:
                    _log_ips_lookup(
                        doc_id, "publication", nd,
                        ips_meta.get("status", "ERR"), ips_meta.get("text_length", 0),
                        ips_meta.get("img_count", 0), ips_meta.get("is_textual", False),
                        f"IPS_FAILED: {ips_meta.get('error', 'not_textual')}"
                    )
            else:
                _log_ips_lookup(
                    doc_id, "publication", None,
                    "N/A", 0, 0, False, "ND_NOT_FOUND"
                )

            if not original_pdf_ok and pdf_source is None:
                raise ResolutionError(
                    f"{doc_id}: оригинальный PDF недоступен, "
                    f"IPS HTML не textual/не найден"
                )

            if pdf_source:
                entry["pdf_source"] = pdf_source
            else:
                entry.pop("pdf_source", None)

            return tmp_pdf, size, pages, html_sha256

        # --- legacy ---
        nd = entry["detail"]["nd"]
        latest = legacy.find_latest_rdk(nd)
        if latest is None:
            raise ResolutionError(
                f"{doc_id}: нет доступных редакций (legacy) для nd={nd}"
            )
        rdk, edition_label = latest
        data = get_bytes(legacy.print_url(nd, rdk))
        html = decode_html(data)
        if not verify_document(html, doc["number"], doc["title"]):
            raise ResolutionError(
                f"{doc_id}: legacy-HTML не соответствует реквизитам (number/дата/title)"
            )

        # IPS HTML = основной источник для RAG
        raw_html_dest = RAW_HTML_DIR / f"{doc_id}.html"
        raw_html_dest.write_bytes(data)
        html_sha256 = hashlib.sha256(data).hexdigest()

        _log_ips_lookup(
            doc_id, "legacy", nd,
            200, len(data), data.decode('windows-1251', 'replace').lower().count("<img"),
            True, "LEGACY_HTML"
        )

        # Создаём архивный PDF из IPS HTML через Playwright + bundled Chromium
        print(f"  генерация PDF из IPS HTML через Playwright (legacy)")
        try:
            metrics = convert_html_to_pdf(raw_html_dest, tmp_pdf)
            size = metrics["pdf_size"]
            pages = validate_pdf(tmp_pdf)
            entry["pdf_source"] = "playwright"
            print(f"  playwright PDF успешно создан: {size} байт, {pages} стр.")
        except Exception as pw_exc:
            print(f"  !! Playwright HTML→PDF не удался: {pw_exc}")
            _silent_unlink(tmp_pdf)
            # Поднимаем исключение — старый PDF и кэш не должны быть тронуты
            raise ResolutionError(
                f"{doc_id}: не удалось создать PDF из IPS HTML: {pw_exc}"
            ) from pw_exc

        return tmp_pdf, size, pages, html_sha256

    except Exception:
        _silent_unlink(tmp_pdf)
        raise



def _silent_unlink(p: Path) -> None:
    try:
        if p and Path(p).exists():
            Path(p).unlink()
    except OSError:
        pass


# ============================================================================
# Оркестрация одного документа (инкрементально)
# ============================================================================
def download_one(doc: dict, cache_path: Path | None = None) -> None:
    print(f"== document: {doc['id']} ==")
    entry = resolve_doc(doc, cache_path=cache_path)
    method = entry["method"]
    print(f"  method      : {method} (публикационный API / legacy-резерв)")
    print(f"  detail      : {entry['detail']}")

    out_pdf = RAW_DIR / f"{doc['id']}.pdf"
    current_rev = _revision_current(doc, entry)

    if _revision_unchanged(entry, current_rev, out_pdf):
        print("  unchanged → skip download")
        if method == "legacy":
            html_path = RAW_HTML_DIR / f"{doc['id']}.html"
            print(f"  html exists: {html_path} ({html_path.stat().st_size} байт)")
        else:
            print(f"  output exists: {out_pdf} ({out_pdf.stat().st_size} байт)")
            html_path = RAW_HTML_DIR / f"{doc['id']}.html"
            if html_path.exists():
                print(f"  html exists: {html_path} ({html_path.stat().st_size} байт)")
        return

    print("  revision: первая загрузка / изменилась / нет файлов → скачивание")
    print(f"  current revision: {current_rev.get('id')} | {current_rev.get('label')}")

    tmp_pdf, size, pages, html_sha256 = _download_to_tmp(doc, entry)

    if tmp_pdf is not None:
        # publication: атомарно заменяем PDF
        try:
            os.replace(tmp_pdf, out_pdf)
        finally:
            _silent_unlink(tmp_pdf)

    entry["revision"] = current_rev
    entry["downloaded_at"] = iso_now()
    if html_sha256:
        entry["html_path"] = str(RAW_HTML_DIR / f"{doc['id']}.html")
        entry["html_sha256"] = html_sha256
    if tmp_pdf is not None:
        entry["pdf_path"] = str(out_pdf)
        entry["pdf_size"] = size
        entry["pdf_pages"] = pages
    elif "pdf_path" in entry:
        # legacy: удаляем старый pdf_path из кэша (PDF больше нет)
        del entry["pdf_path"]
        entry.pop("pdf_size", None)
        entry.pop("pdf_pages", None)

    persist_entry(doc["id"], entry, cache_path)

    print(f"  source: {'legacy (IPS HTML)' if method == 'legacy' else 'publication (PDF + IPS HTML)'}")
    if tmp_pdf is not None:
        print(f"  pdf_size    : {size} байт")
        print(f"  pdf_pages   : {pages}")
        print(f"  pdf_path    : {out_pdf}")
    if html_sha256:
        html_path = RAW_HTML_DIR / f"{doc['id']}.html"
        print(f"  html_path   : {html_path}")
        print(f"  html_sha256 : {html_sha256}")
    if method == "legacy":
        print(f"  nd          : {entry['detail'].get('nd')}")
    else:
        print(f"  nd          : {entry['detail'].get('nd', 'N/A')}")


# ============================================================================
# Точка входа
# ============================================================================
def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for doc in registry["documents"]:
        if not doc.get("enabled", True):
            continue
        try:
            download_one(doc)
        except pub.DocumentMismatchError as exc:
            print(f"  !! ОСТАНОВ(неоднозначность/несовпадение): {exc}")
        except pub.PublicAPIError as exc:
            print(f"  !! ОСТАНОВ(API недоступен): {exc}")
        except ResolutionError as exc:
            print(f"  !! ОШИБКА: {exc}")


if __name__ == "__main__":
    main()
