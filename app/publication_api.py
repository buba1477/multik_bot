"""Работа с официальным API портала «Официальное опубликование правовых актов».

Документация: http://publication.pravo.gov.ru/help

Используемые официальные эндпоинты (исследованы реальными HTTP-запросами):

* Поиск документов:   GET /api/Documents?number=<номер>
    Ответ — JSON ``{items, itemsTotalCount, ...}``. Параметры ``date``/поиска/
    пагинация/сортировка в текущей реализации API игнорируются; надёжно
    фильтрует только ``number``. Каждый ``item`` содержит ``eoNumber``,
    ``number``, ``documentDate``, ``documentType``, ``title``, ``name``,
    ``complexName``, ``pagesCount``, ``pdfFileLength``.

* Детали документа:   GET /api/Document?eoNumber=<eoNumber>
    Полный объект документа (те же поля + ``id`` GUID).

* Файл PDF:           GET /file/pdf?eoNumber=<eoNumber>
    Бинарный PDF документа (заголовок ``%PDF-...``).

Стабильный идентификатор портала — ``eoNumber``.

ВАЖНОЕ ОГРАНИЧЕНИЕ (зафиксировано реальными запросами): портал содержит
акты, официально опубликованные на нём (примерно с 2011–2012 гг.). Старые
федеральные законы (например 79-ФЗ от 27.07.2004, 58-ФЗ от 27.05.2003) в
этой базе ОТСУТСТВУЮТ. Для таких документов ``resolve_exact`` вернёт
``DocumentNotFoundError``, и downloader переходит на legacy-резерв.

Принцип exact/fail-closed: документ выбирается ТОЛЬКО по стабильным
критериям — номер + дата (+ тип). Никакого подбора по похожести названия.
При неоднозначности — исключение, а не выбор "самого похожего".
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from app.search_fragments import build_search_fragments

BASE_URL = "http://publication.pravo.gov.ru"
API_URL = BASE_URL + "/api"
PDF_URL = BASE_URL + "/file/pdf"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_RU_DATE_RE = re.compile(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})")


class PublicAPIError(RuntimeError):
    """Базовая ошибка взаимодействия с publication.pravo.gov.ru."""


class DocumentNotFoundError(PublicAPIError):
    """Документ отсутствует в базе официального опубликования."""


class DocumentMismatchError(PublicAPIError):
    """API вернул неоднозначный результат / несовпадающие реквизиты."""


# ============================================================================
# Канонизация типов актов
# ============================================================================
# Портал публикует сложные вложенные названия типов ("Указ Президента
# Российской Федерации"). В documents.json типы заданы коротко ("Указ").
# Канонизируем ТОЛЬКО известные федеральные комбинации, чтобы не спорить
# с порталом о неизвестных вариантах (для них остаётся строгое сравнение).
_CANONICAL_TYPES = {
    ("указ", "президент российской федерации"): "указ президента российской федерации",
    ("распоряжение", "правительство российской федерации"): "распоряжение правительства российской федерации",
    ("постановление", "правительство российской федерации"): "постановление правительства российской федерации",
}


def _canonical_type(doc: dict) -> str:
    """Каноническое имя типа документа API (lower, без лишних пробелов).

    Строится из двух полей ответа API:
      - documentType.name — короткий тип ("Указ", "Распоряжение");
      - signatoryAuthorities[0].name — издающий орган
        ("Президент Российской Федерации").
    Если пара (documentType.name, орган) присутствует в whitelist
    ``_CANONICAL_TYPES`` — возвращается каноническая запись.
    Иначе — исходный documentType.name (приведённый к нижнему регистру).
    """
    doc_type = (
        ((doc.get("documentType") or {}).get("name") or "")
        .strip()
        .lower()
    )

    authorities = doc.get("signatoryAuthorities") or []
    authority = (
        (authorities[0].get("name") if authorities else "") or ""
    ).strip().lower()

    return _CANONICAL_TYPES.get(
        (doc_type, authority),
        doc_type,
    )


# ============================================================================
# HTTP-слой
# ============================================================================
def _get_json(url: str, timeout: int = 90):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise PublicAPIError(f"publication API HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise PublicAPIError(f"publication API недоступен ({url}): {exc.reason}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PublicAPIError(f"Некорректный JSON от publication API: {url}") from exc


def _get_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise PublicAPIError(f"publication API HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise PublicAPIError(f"publication API недоступен ({url}): {exc.reason}") from exc


# ============================================================================
# Поиск / детали
# ============================================================================
def search_documents(number: str | None = None, name: str | None = None) -> list[dict]:
    """Поиск документов через официальный API."""
    params: dict[str, str] = {}
    if number:
        params["number"] = number
    if name:
        params["name"] = name
    if not params:
        raise PublicAPIError("Не задано ни одного критерия поиска")
    url = f"{API_URL}/Documents?{urllib.parse.urlencode(params)}"
    data = _get_json(url)
    if not isinstance(data, dict) or "items" not in data:
        raise PublicAPIError("Неожиданный формат ответа /api/Documents")
    return list(data.get("items") or [])


def get_document(eo_number: str) -> dict:
    """Детали документа по eoNumber (GET /api/Document)."""
    url = f"{API_URL}/Document?eoNumber={urllib.parse.quote(eo_number)}"
    data = _get_json(url)
    if not isinstance(data, dict):
        raise PublicAPIError("Неожиданный формат ответа /api/Document")
    return data


def pdf_url(eo_number: str) -> str:
    """URL файла PDF документа."""
    return f"{PDF_URL}?eoNumber={urllib.parse.quote(eo_number)}"


# ============================================================================
# Нормализация и строгая проверка реквизитов
# ============================================================================
def _norm_number(number: str) -> str:
    return (number or "").replace(" ", "").upper()


def _date_to_iso(date: str) -> str | None:
    """Дата реестра (ДД.ММ.ГГГГ) -> ISO YYYY-MM-DD."""
    if not date:
        return None
    m = _RU_DATE_RE.search(date)
    if not m:
        return None
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"


def _doc_date_iso(doc: dict) -> str | None:
    """documentDate из API ('2004-04-09T00:00:00') -> '2004-04-09'."""
    raw = doc.get("documentDate") or ""
    m = _ISO_DATE_RE.search(raw)
    return m.group(0) if m else None


def _validate_against_record(doc: dict, record: dict) -> None:
    """Проверить соответствие документа реквизитам реестра. Mismatch -> ошибка."""
    exp_number = _norm_number(record.get("number"))
    if not exp_number:
        raise DocumentMismatchError(
            "В documents.json не задан 'number' — нельзя однозначно определить документ"
        )
    if _norm_number(doc.get("number")) != exp_number:
        raise DocumentMismatchError(
            f"Номер не совпадает: ожидалось '{record.get('number')}', API вернул '{doc.get('number')}'"
        )

    exp_date = _date_to_iso(record.get("date"))
    if exp_date:
        got_date = _doc_date_iso(doc)
        if got_date != exp_date:
            raise DocumentMismatchError(
                f"Дата не совпадает: ожидалось {record.get('date')} ({exp_date}), "
                f"API вернул {got_date or '—'}"
            )

    exp_type = (record.get("type") or "").strip().lower()
    if exp_type:
        got_type = _canonical_type(doc)
        if got_type != exp_type:
            raise DocumentMismatchError(
                f"Тип не совпадает: ожидалось '{record.get('type')}', API вернул '{got_type}'"
            )


def _pick_unique(candidates: list[dict], record: dict) -> dict | None:
    """Выбрать единственного кандидата, точно совпадающего с реквизитами.

    * 0 кандидатов с нужным номером/датой -> ``None`` (поисковая невозможность);
    * >1 кандидатов с одинаковыми номером+датой -> ``None`` (неоднозначность);
    * ровно 1 -> полный объект /api/Document после строгой валидации
      (``DocumentMismatchError`` при реальном несовпадении реквизитов).

    ``None`` означает «не удалось однозначно определить этим поиском» —
    вызывающий код может попробовать следующий (менее специфичный) запрос
    или перейти к legacy-резерву. Реальное несовпадение (вариант с другим
    номером/датой) всегда ошибочно и не допускает альтернатив.
    """
    if not candidates:
        return None

    exp_number = _norm_number(record.get("number"))
    by_number = [c for c in candidates if _norm_number(c.get("number")) == exp_number]
    if not by_number:
        return None

    pool = by_number
    exp_date = _date_to_iso(record.get("date"))
    if exp_date:
        by_date = [c for c in by_number if _doc_date_iso(c) == exp_date]
        if not by_date:
            return None
        pool = by_date

    if len(pool) > 1:
        # Неоднозначность (несколько редакций одного акта в базе) — не
        # выбираем «первый попавшийся», пробуем другой критерий поиска.
        return None

    matched = pool[0]
    # В списке /api/Documents нет вложенного documentType (только documentTypeId),
    # поэтому финальную валидацию (включая тип) делаем по полному объекту /api/Document.
    full = get_document(matched["eoNumber"])
    _validate_against_record(full, record)
    return full


def resolve_exact(record: dict) -> dict:
    """Найти ЕДИНСТВЕННЫЙ официальный документ, точно совпадающий с реквизитами.

    Двухэтапный поиск (fail-closed, без угадывания):

    1. ``number`` -> кандидаты, затем ``_pick_unique``;
    2. если неоднозначно/не найдено и в реестре есть ``title`` —
       короткие фрагменты названия (от более специфичного к менее)
       в комбинации ``number + name`` (search_fragments.build_search_fragments).

    Итоговое отсутствие результата -> ``DocumentNotFoundError`` (поисковая
    невозможность; downloader переходит на legacy-резерв). Реальное
    несовпадение реквизитов -> ``DocumentMismatchError`` (останов, без резерва).

    Возвращает полный объект документа API (содержит ``eoNumber``).
    """
    number = record.get("number")

    # Этап 1: строго по номеру.
    picked = _pick_unique(search_documents(number=number), record)
    if picked is not None:
        return picked

    # Этап 2: номер + короткие фрагменты названия (уточнение поиска API).
    fragments = build_search_fragments(record.get("title"))
    for fragment in fragments:
        picked = _pick_unique(
            search_documents(number=number, name=fragment), record
        )
        if picked is not None:
            return picked

    raise DocumentNotFoundError(
        f"В базе публикации не удалось однозначно найти документ "
        f"'{number}' от {record.get('date') or '—'} (число и название). "
        f"Возможно, акт отсутствует на портале официального опубликования"
    )


# ============================================================================
# Скачивание PDF
# ============================================================================
def download_pdf(eo_number: str, dest: Path | str) -> int:
    """Скачать официальный PDF по eoNumber и сохранить в dest. Возвращает размер."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = _get_bytes(pdf_url(eo_number))
    if not data.startswith(b"%PDF-"):
        raise PublicAPIError(f"Ответ по {eo_number} не является PDF")
    dest.write_bytes(data)
    if not dest.stat().st_size:
        raise PublicAPIError(f"Скачан пустой файл для {eo_number}")
    return len(data)


def pages_count(path: Path) -> int:
    """Количество страниц PDF."""
    path = Path(path)
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)
    except Exception:
        raw = path.read_bytes()
        return raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
