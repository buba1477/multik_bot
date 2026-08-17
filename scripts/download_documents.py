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
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

import app.pravo_resolver as legacy  # noqa: E402  (резервный механизм)
import app.publication_api as pub    # noqa: E402  (основной официальный API)

PROJECT = Path(__file__).resolve().parent.parent
REGISTRY = PROJECT / "documents.json"
RAW_DIR = PROJECT / "raw"
CACHE_PATH = PROJECT / "app" / "resolved_documents.json"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


# ============================================================================
# HTTP/вспомогательные (общие для обоих путей)
# ============================================================================
def get_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        cl = resp.headers.get("Content-Length")
    if cl is not None:
        expected = int(cl)
        if len(data) != expected:
            raise ResolutionError(
                f"Content-Length mismatch: ожидалось {expected} байт, получено {len(data)}"
            )
    return data


def decode_html(data: bytes) -> str:
    return data.decode("windows-1251", "replace")


def sha256_file(path: Path) -> str:
    """SHA-256 хеш содержимого файла (hex)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
        try:
            legacy_rec = legacy.resolve_document(
                doc["number"], doc.get("title"), doc.get("date")
            )
        except legacy.DocumentNotFoundError as exc:
            raise ResolutionError(
                f"{doc_id}: не найден ни в API публикации, ни в legacy-резерве: {exc}"
            ) from exc
        entry = {
            "method": "legacy",
            "resolved_at": iso_now(),
            "detail": {"nd": str(legacy_rec["nd"])},
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
def verify_document(html: str, number: str, date: str, title: str) -> bool:
    """Верификация legacy-документа по HTML актуальной редакции.

    Обязательные строгие реквизиты идентификации нормативного акта (fail-closed):
      - number присутствует в HTML;
      - date присутствует в HTML.
    При несовпадении любого из них — False.

    Проверка title — дополнительная (fail-open): при несовпадении логируется
    warning, но загрузка продолжается (number + date совпали).

    Ответственность за выбор актуальной редакции несёт find_latest_rdk(nd),
    а не эта функция.
    """
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)

    # 1. Номер — строгая проверка (fail-closed)
    if number not in text:
        return False

    # 2. Дата — строгая проверка (fail-closed)
    if date not in text:
        return False

    # 3. Title — дополнительная неблокирующая проверка
    if title and title not in text:
        logger.warning(
            "verify_document: title не найден в HTML (number=%r, date=%r) — "
            "продолжаем, обязательные реквизиты совпали",
            number,
            date,
        )

    return True


def convert_html_to_pdf(tmp_html: Path, out_pdf: Path) -> None:
    RAW_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["soffice", "--headless", "--convert-to", "pdf",
         "--outdir", str(RAW_DIR), str(tmp_html)],
        check=True,
        capture_output=True,
    )
    assert out_pdf.exists() and out_pdf.stat().st_size > 0, f"PDF не создан: {out_pdf}"


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
                str(x) for x in [
                    eo, m.get("documentDate"), m.get("publishDateShort"),
                    m.get("pdfFileLength"), m.get("pagesCount"),
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
    """True если кэшированная редакция == текущей и локальный PDF (с правильным SHA-256) существует.

    Если revision совпадает, PDF существует, но SHA-256 не совпадает с кэшированным
    — локальный PDF повреждён/изменён, возвращается False (скачать заново).
    """
    cached = entry.get("revision")
    if not cached:
        return False
    if cached != current_rev:
        return False
    if not out_pdf.exists():
        return False
    cached_sha = entry.get("pdf_sha256")
    if cached_sha is None:
        return False  # нет хеша — перезагружаем для безопасности
    return sha256_file(out_pdf) == cached_sha


# ============================================================================
# Скачивание нового PDF во временный файл (до атомарной замены)
# ============================================================================
def _download_to_tmp(doc: dict, entry: dict) -> tuple[Path, int, int]:
    """Скачать PDF в RAW_DIR/<id>.new.pdf. Возвращает (tmp_pdf, size, pages).

    На ошибке/невалидности — выбрасывает исключение, временные файлы подчищает,
    при этом целевой RAW_DIR/<id>.pdf не трогает.
    """
    doc_id = doc["id"]
    RAW_DIR.mkdir(exist_ok=True)
    tmp_pdf = RAW_DIR / f"{doc_id}.new.pdf"
    tmp_html = RAW_DIR / f"{doc_id}.new.html"

    try:
        if entry["method"] == "publication":
            eo = entry["detail"]["eoNumber"]
            size = pub.download_pdf(eo, tmp_pdf)   # пишет tmp_pdf, валидирует %PDF-
        else:
            nd = entry["detail"]["nd"]
            latest = legacy.find_latest_rdk(nd)
            if latest is None:
                raise ResolutionError(f"{doc_id}: нет доступных редакций (legacy) для nd={nd}")
            rdk, edition_label = latest
            data = get_bytes(legacy.print_url(nd, rdk))
            html = decode_html(data)
            if not verify_document(html, doc["number"], doc["date"], doc["title"]):
                raise ResolutionError(
                    f"{doc_id}: legacy-HTML не соответствует реквизитам (number/дата/title)"
                )
            tmp_html.write_bytes(data)
            convert_html_to_pdf(tmp_html, tmp_pdf)
            size = tmp_pdf.stat().st_size

        pages = validate_pdf(tmp_pdf)
        return tmp_pdf, size, pages
    except Exception:
        _silent_unlink(tmp_pdf)
        _silent_unlink(tmp_html)
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
        print(f"  output exists: {out_pdf} ({out_pdf.stat().st_size} байт)")
        return

    print("  revision: первая загрузка / изменилась / нет PDF → скачивание")
    print(f"  current revision: {current_rev.get('id')} | {current_rev.get('label')}")

    tmp_pdf, size, pages = _download_to_tmp(doc, entry)
    pdf_sha256 = sha256_file(tmp_pdf)
    try:
        os.replace(tmp_pdf, out_pdf)          # атомарная замена после успешной проверки
    finally:
        _silent_unlink(tmp_pdf)
        _silent_unlink(RAW_DIR / f"{doc['id']}.new.html")

    entry["revision"] = current_rev
    entry["downloaded_at"] = iso_now()
    entry["pdf_path"] = str(out_pdf)
    entry["pdf_size"] = size
    entry["pdf_pages"] = pages
    entry["pdf_sha256"] = pdf_sha256
    persist_entry(doc["id"], entry, cache_path)

    print(f"  pdf_size    : {size} байт")
    print(f"  pdf_pages   : {pages}")
    print(f"  pdf_sha256  : {pdf_sha256}")
    print(f"  output      : {out_pdf}")


# ============================================================================
# Точка входа
# ============================================================================
def _network_error_text(exc: BaseException) -> str:
    """Короткое описание сетевой ошибки (timeout/соединение) для лога."""
    if isinstance(exc, TimeoutError):
        return "timeout при обращении к источнику"
    if isinstance(exc, urllib.error.URLError):
        return f"сеть недоступна: {getattr(exc, 'reason', exc)}"
    if isinstance(exc, ConnectionError):
        return f"ошибка соединения: {exc}"
    return str(exc)


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
        except (
            TimeoutError,
            ConnectionError,
            urllib.error.URLError,
            legacy.PravoError,
        ) as exc:
            print(f"  !! ОШИБКА СЕТИ: {doc['id']}: {_network_error_text(exc)}")
        except ResolutionError as exc:
            print(f"  !! ОШИБКА: {exc}")


if __name__ == "__main__":
    main()
