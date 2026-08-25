# ============================================================================
# DEPRECATED / legacy RESERVE
# Этот модуль поиска по старой HTML-системе /proxy/ips/ используется ТОЛЬКО как
# резерв (fallback) для актов, отсутствующих в базе официального публикационного
# API publication.pravo.gov.ru. Основной путь разрешения реквизитов ->
# app/publication_api.py. Не использовать title_score/pick_document в основном потоке.
# ============================================================================

"""Поиск внутренних идентификаторов (nd) документов на pravo.gov.ru.

pravo.gov.ru не предоставляет официального JSON-API: поиск выполняется
через HTML-эндпоинты устаревшей фреймовой системы ``/proxy/ips/``.
Модуль использует реальные HTTP-эндпоинты, исследованные прямыми
запросами из Python:

* Поиск документа по номеру/названию:
    GET /proxy/ips/?list_itself=&a8=<номер>&a8type=2&a1=<название>&page=first
  Ответ — HTML-список документов. Каждый элемент содержит внутренний
  идентификатор ``nd`` и реквизиты (дата, номер, название, статус).

* Карточка документа и список редакций:
    GET /proxy/ips/?docbody=&nd=<nd>
  Ответ — HTML с ``<select name="doc_editions">``, опции вида
  ``<option value="<номер>,<nd>">N - от ДД.ММ.ГГГГ № ... (изм.)</option>``.
  Номер опции и есть ``rdk``; последняя доступная редакция — ``max(rdk)``.

* Печатное представление редакции:
    GET /proxy/ips/?docview&page=1&print=1&nd=<nd>&rdk=<rdk>&empire=
  HTML конкретной редакции для конвертации в PDF.

Модуль содержит чистые функции парсинга (тестируются на фикстурах без
сети) и HTTP-обёртки, а также локальный кэш найденных ``nd``.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from app.search_fragments import build_search_fragments

PROJECT = Path(__file__).resolve().parent.parent
BASE_URL = "http://pravo.gov.ru/proxy/ips/"
CACHE_PATH = PROJECT / "app" / "resolved_documents.json"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_DATE_RE = re.compile(r"от\s+(\d{2}\.\d{2}\.\d{4})")
_WORD_RE = re.compile(r"[а-яёa-z0-9-]+")


# ============================================================================
# HTTP-слой
# ============================================================================
class PravoError(RuntimeError):
    """Базовая ошибка взаимодействия с pravo.gov.ru."""


class DocumentNotFoundError(PravoError):
    """Документ не найден / найденный документ не совпал с реквизитами."""


def get_bytes(url: str, timeout: int = 60) -> bytes:
    """Выполнить GET и вернуть исходные байты ответа."""
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise PravoError(f"HTTP {exc.code} для {url}") from exc
    except urllib.error.URLError as exc:
        raise PravoError(f"Сеть недоступна для {url}: {exc.reason}") from exc


def get_html(url: str, timeout: int = 60) -> str:
    """GET и декодирование ответа из windows-1251 (только для парсинга)."""
    return get_bytes(url, timeout=timeout).decode("windows-1251", "replace")


# ============================================================================
# Поиск документа (nd) по реквизитам
# ============================================================================
def search_documents(number: str, title: str | None = None) -> list[dict]:
    """Поиск документов на pravo.gov.ru по номеру и (опц.) названию.

    ``number`` — полный номер документа, например ``"58-ФЗ"``, ``"79-ФЗ"``.
    ``title`` — текст названия (параметр ``a1``), сужает поиск.

    Если ``a1`` не даёт результатов (у некоторых документов пустое описание),
    поиск автоматически повторяется без ``a1``.

    Возвращает список кандидатов::

        [{"nd": "102081744", "title": "Федеральный закон от 27.05.2003 № 58-ФЗ",
          "name": "О системе государственной службы Российской Федерации",
          "status": "Действует c изменениями", "date": "27.05.2003"}, ...]
    """
    params: dict[str, object] = {
        "list_itself": "",
        "a8": number,
        "a8type": 2,  # «Точно» — точное совпадение номера
        "page": "first",
    }
    if title:
        params["a1"] = title
    query = urllib.parse.urlencode(params, encoding="windows-1251")
    html = get_html(f"{BASE_URL}?{query}")
    results = parse_search_results(html)
    if not results and title:
        # a1 не дал результатов (например, название пустое в выборке сервера)
        # → пробуем без названия, фильтрация по дате/номеру в pick_document
        params.pop("a1", None)
        query = urllib.parse.urlencode(params, encoding="windows-1251")
        html = get_html(f"{BASE_URL}?{query}")
        results = parse_search_results(html)
    return results


def parse_search_results(html: str) -> list[dict]:
    """Разобрать HTML-список ``list_itself`` в список кандидатов."""
    results: list[dict] = []
    for block in re.finditer(r'<table class="list_elem[^"]*".*?</table>', html, re.S):
        item = block.group(0)
        nd_m = re.search(r"nd=(\d+)", item)
        if not nd_m:
            continue
        link_m = re.search(r'<a id="link_\d+"[^>]*>\s*(.*?)\s*</a>', item, re.S)
        name_m = re.search(r'<span class="bold">\s*(.*?)\s*</span>', item, re.S)
        status_m = re.search(r'<span class="tiny_italic_bold">\s*(.*?)\s*</span>', item, re.S)

        title = _collapse(link_m.group(1)) if link_m else ""
        date_m = _DATE_RE.search(title)
        results.append(
            {
                "nd": nd_m.group(1),
                "title": title,
                "name": _collapse(name_m.group(1)) if name_m else "",
                "status": _collapse(status_m.group(1)) if status_m else "",
                "date": date_m.group(1) if date_m else None,
            }
        )
    return results


def _collapse(text: str) -> str:
    """Убрать лишние пробелы/переводы строк внутри фрагмента HTML."""
    return re.sub(r"\s+", " ", text).strip()


def _norm_number(number: str) -> str:
    """Нормализация номера для сравнения: без пробелов, верхний регистр."""
    return re.sub(r"\s+", "", number or "").upper()


def _norm_date(date: str) -> str | None:
    """Привести дату к ДД.ММ.ГГГГ, если она передана в любом внятном виде."""
    if not date:
        return None
    m = re.search(r"(\d{2})[.\-/](\d{2})[.\-/](\d{4})", date)
    if not m:
        return None
    return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"


def _title_score(candidate_name: str, reference_title: str) -> float:
    """Доля значимых слов искомого названия, встречающихся в названии кандидата.

    Если ``candidate_name`` пуст (сервер не вернул описание документа),
    возвращается 1.0 — номер+дата уже подтверждены, а названия нет для сравнения.
    """
    if not candidate_name:
        return 1.0  # нет информации о названии — не можем проверить, номер+дата совпали
    ref_words = {w for w in _WORD_RE.findall((reference_title or "").lower()) if len(w) >= 3}
    if not ref_words:
        return 0.0
    cand_words = {w for w in _WORD_RE.findall((candidate_name or "").lower()) if len(w) >= 3}
    if not cand_words:
        return 0.0
    matched = sum(1 for w in ref_words if any(w in cw or cw in w for cw in cand_words))
    return matched / len(ref_words)


def pick_document(
    candidates: list[dict], number: str, title: str | None = None, date: str | None = None
) -> dict:
    """Выбрать из кандидатов документ, соответствующий реквизитам.

    Проверки по убыванию приоритета:
      1. номер присутствует в ``title`` кандидата (обязательно);
      2. дата (если задана) присутствует в ``title`` кандидата (обязательно);
      3. максимальное совпадение по словам названия.

    При пустом/неоднозначном результате поднимает ``DocumentNotFoundError``.
    """
    norm_number = _norm_number(number)
    norm_date = _norm_date(date)
    eligible: list[tuple[float, int, dict]] = []
    for idx, cand in enumerate(candidates):
        cand_title = _norm_number(cand.get("title") or "")
        if norm_number not in cand_title:
            continue
        if norm_date and norm_date not in (cand_title + (cand.get("date") or "")):
            continue
        score = _title_score(cand.get("name"), title)
        eligible.append((score, idx, cand))

    if not eligible:
        raise DocumentNotFoundError(
            f"По реквизитам (номер={number!r}, дата={date!r}) не найден ни один документ"
        )

    # Максимум по score, при равенстве — первый (стабильный порядок).
    best_score, _, best = max(eligible, key=lambda item: (item[0], -item[1]))
    if title and best_score < 0.2:
        raise DocumentNotFoundError(
            f"Найденные документы не совпадают по названию с {title!r} "
            f"(лучшее совпадение {best_score:.2f})"
        )
    return best


def resolve_document(
    number: str, title: str | None = None, date: str | None = None
) -> dict:
    """Найти nd документа на pravo.gov.ru по номеру/дате/названию.

    Возвращает словарь с ключами ``nd``, ``title``, ``name``, ``status``, ``date``.
    Не имеет хардкода конкретных документов: работает для любых реквизитов.

    Поиск идёт от более специфичного запроса к менее специфичному:
    сначала полный title (если задан), затем короткие фрагменты названия
    (search_fragments.build_search_fragments) — каждый кандидат проходит
    строгую проверку ``pick_document`` (номер + дата).
    """
    queries: list[str | None] = [title]
    if title:
        queries.extend(build_search_fragments(title))
    queries = [q for q in dict.fromkeys(queries)]  # убрать дубликаты, сохранить порядок

    for query in queries:
        candidates = search_documents(number, query)
        try:
            return pick_document(candidates, number, title, date)
        except DocumentNotFoundError:
            continue  # этот запрос не дал однозначного результата — пробуем следующий

    raise DocumentNotFoundError(
        f"По реквизитам (номер={number!r}, дата={date!r}, название={title!r}) "
        f"не найден ни один документ на pravo.gov.ru"
    )


# ============================================================================
# Редакции документа (rdk)
# ============================================================================
def find_latest_rdk(nd: str) -> tuple[int, str] | None:
    """Вернуть ``(rdk, label)`` последней доступной редакции документа.

    Редакции перечисляются в ``<select name="doc_editions">`` карточки
    документа; ``rdk`` — максимальный номер редакции в этом списке.
    Опции со значением ``"n"`` (недоступная редакция) пропускаются.
    """
    url = f"{BASE_URL}?docbody=&nd={nd}"
    nav = get_html(url)
    return _parse_latest_rdk(nav)


def _parse_latest_rdk(html: str) -> tuple[int, str] | None:
    """Вернуть ``(rdk, label)`` последней ДЕЙСТВУЮЩЕЙ редакции документа.

    Пропускаются:
      - опции со значением ``"n"`` (недоступная редакция);
      - опции, чья метка (label) содержит ``(не действ.)`` или ``(не готова)``
        (регистр не учитывается).
    Из оставшихся выбирается максимальный rdk.
    Если после фильтрации редакций не осталось — возвращается ``None``.
    """
    best: tuple[int, str] | None = None
    for value, label in re.findall(
        r'<option[^>]*value="([^"]*)"[^>]*>(.*?)</option>', html, re.S
    ):
        value = value.strip()
        if not value or value.lower() == "n":
            continue
        num = value.split(",")[0].strip()
        if not num.isdigit():
            continue
        rdk = int(num)
        label_clean = re.sub(r"<[^>]+>", "", label).strip()
        label_lower = label_clean.lower()
        # Пропускаем недействующие и неготовые редакции
        if "(не действ.)" in label_lower or "(не готова)" in label_lower:
            continue
        if best is None or rdk > best[0]:
            best = (rdk, label_clean)
    return best


def find_latest_revision(nd: str) -> dict | None:
    """Обёртка над ``find_latest_rdk``: ``{rdk, label, date}``.

    ``date`` извлекается из метки редакции вида
    ``"18 - от 29.09.2025 № 365-ФЗ (изм.)"`` — дата редакции.
    """
    best = find_latest_rdk(nd)
    if best is None:
        return None
    rdk, label = best
    date_m = _DATE_RE.search(label)
    return {"rdk": rdk, "label": label, "date": date_m.group(1) if date_m else None}


def print_url(nd: str, rdk: int) -> str:
    """URL печатного представления редакции документа."""
    return f"{BASE_URL}?docview&page=1&print=1&nd={nd}&rdk={rdk}&empire="


# ============================================================================
# Кэш найденных nd (resolved_documents.json)
# ============================================================================
def load_cache(path: Path | None = None) -> dict:
    """Прочитать кэш ``{doc_id: {"nd": ..., "resolved_at": ..., "source": ...}}``."""
    cache_file = path or CACHE_PATH
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict, path: Path | None = None) -> None:
    """Атомарно записать кэш (сначала во временный файл, затем replace)."""
    cache_file = path or CACHE_PATH
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(cache_file)


def cached_resolve(
    doc_id: str,
    number: str,
    title: str | None = None,
    date: str | None = None,
    force: bool = False,
    cache_path: Path | None = None,
) -> tuple[str, dict]:
    """Получить nd документа с использованием кэша.

    * если в кэше есть рабочий nd — вернуть его без сетевых запросов;
    * cache miss (или ``force=True``) — выполнить поиск на pravo.gov.ru
      и сохранить результат в кэш;
    * если поиск не дал совпадения — поднимается ``DocumentNotFoundError``.

    Возвращает ``(nd, entry)``.
    """
    cache = load_cache(cache_path)
    entry = cache.get(doc_id)
    if entry and entry.get("nd") and not force:
        return str(entry["nd"]), entry

    resolved = resolve_document(number, title, date)
    nd = str(resolved["nd"])
    entry = {
        "nd": nd,
        "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "pravo.gov.ru",
    }
    cache[doc_id] = entry
    save_cache(cache, cache_path)
    return nd, entry



# ============================================================================
# Novye funkcii: poisk nd po rekvizitam + poluchenie IPS-teksta
# Ispolzuyutsya dlya publication-dokumentov, u kotoryh net sohranyonnogo nd.
# ============================================================================


def _search_ips_by_number(number, date=None, title=None):
    try:
        candidates = search_documents(number, title)
    except (PravoError, OSError):
        return []
    results = []
    for c in candidates:
        results.append({
            "nd": c["nd"],
            "title": c["title"],
            "number_in_text": number,
            "date_in_text": c.get("date"),
            "status": c.get("status", "unknown"),
        })
    return results


def _verify_nd_against_record(nd, record, candidate_title=None):
    title_text = candidate_title
    if not title_text:
        url = f"{BASE_URL}?docbody=&nd={nd}"
        try:
            html = get_html(url)
        except (PravoError, OSError):
            return None
        opt1 = re.findall(r"<option[^>]*value='([^']*)'[^>]*>(.*?)</option>", html, re.S)
        opt2 = re.findall('<option[^>]*value="([^"]*)"[^>]*>(.*?)</option>', html, re.S)
        for v, lbl in opt1 + opt2:
            clean = re.sub(r'<[^>]+>', '', lbl).strip()
            if clean and clean != 'n':
                title_text = clean
                break
    if not title_text:
        return None
    rn = record.get("number", "")
    if rn and rn not in title_text:
        return None
    rd = record.get("date", "")
    if rd:
        parts = rd.split(".")
        if len(parts) == 3:
            sd = f"от {parts[0]}.{parts[1]}.{parts[2]}"
            if sd not in title_text:
                return None
    rt = record.get("type", "").lower()
    if rt:
        tkw = rt.split()
        if not any(kw.lower() in title_text.lower() for kw in tkw):
            return None
    return {"nd": nd, "title": title_text}


def _parse_rdk_from_html(card_html):
    rdk = None
    label = None
    opt1 = re.findall(r"<option[^>]*value='([^']*)'[^>]*>(.*?)</option>", card_html, re.S)
    opt2 = re.findall('<option[^>]*value="([^"]*)"[^>]*>(.*?)</option>', card_html, re.S)
    for val_str, lbl in opt1 + opt2:
        parts = val_str.split(",")
        if len(parts) == 2 and parts[0].isdigit():
            o_rdk = int(parts[0])
            clean_lbl = re.sub(r'<[^>]+>', '', lbl).strip()
            if rdk is None or o_rdk > rdk:
                rdk = o_rdk
                label = clean_lbl
    if rdk is None:
        rdk = 1
    return rdk, label


def find_nd_by_record(record):
    num = record.get("number", "")
    date = record.get("date", "")
    title = record.get("title", "")
    if not num:
        return None
    candidates = _search_ips_by_number(num, date, title)
    if not candidates:
        return None
    verified = {}
    for c in candidates:
        vi = _verify_nd_against_record(c["nd"], record, c.get("title"))
        if vi:
            verified[vi["nd"]] = vi
    if len(verified) == 1:
        return next(iter(verified.values()))
    if len(verified) > 1:
        return None
    if title:
        from app.search_fragments import build_search_fragments
        for frag in build_search_fragments(title):
            try:
                cs = search_documents(num, frag)
            except (DocumentNotFoundError, PravoError, OSError):
                continue
            for c in cs:
                if c["nd"] in verified:
                    continue
                vi = _verify_nd_against_record(c["nd"], record, c.get("title"))
                if vi:
                    verified[vi["nd"]] = vi
        if len(verified) == 1:
            return next(iter(verified.values()))
    return None


def get_ips_print_html(nd):
    result = {
        "status": None, "size": 0, "text_length": 0,
        "img_count": 0, "is_textual": False,
        "rdk": None, "edition_label": None, "error": None,
    }
    try:
        card_html = get_html(f"{BASE_URL}?docbody=&nd={nd}")
    except (PravoError, OSError) as e:
        result["error"] = f"card: {e}"
        return None, result
    rdk, label = _parse_rdk_from_html(card_html)
    result["rdk"] = rdk
    result["edition_label"] = label
    try:
        data = get_bytes(f"{BASE_URL}?docview&page=1&print=1&nd={nd}&rdk={rdk}&empire=")
        html_text = data.decode("windows-1251", "replace")
    except (PravoError, OSError) as e:
        result["error"] = f"print: {e}"
        return None, result
    body = re.search(r'<body[^>]*>(.*?)</body>', html_text, re.S | re.I)
    text = re.sub(r'<[^>]+>', '', body.group(1) if body else html_text).strip()
    img = html_text.lower().count("<img")
    # Текстовым считаем HTML с достаточным объёмом текста;
    # небольшое количество изображений — диагностический признак, не причина отказа.
    is_textual = len(text) > 2000 or (len(text) > 100 and img == 0)
    result.update({
        "status": 200, "size": len(data),
        "text_length": len(text), "img_count": img,
        "is_textual": is_textual,
    })
    return data, result


def check_ips_html_is_textual(html_text, record):
    body = re.search(r'<body[^>]*>(.*?)</body>', html_text, re.S | re.I)
    text = re.sub(r'<[^>]+>', '', body.group(1) if body else html_text).strip()
    if len(text) <= 100:
        return False
    # Если текста достаточно (>2000 символов), изображения не блокируют принятие
    if len(text) > 2000:
        pass  # textual regardless of images
    elif html_text.lower().count("<img") > 0:
        return False
    num = record.get("number", "")
    if num and num not in text:
        return False
    dt = record.get("type", "")
    if dt:
        tkw = dt.lower().split()
        if not any(kw.lower() in text.lower() for kw in tkw):
            return False
    return True
