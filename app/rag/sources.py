"""Формирование источников ответа из найденных чанков.

Единственная точка, где источники собираются для фронтенда.

Логика работает ТОЛЬКО с metadata найденных узлов:
  - title      - "<название документа> \u2014 <структурный элемент>"
  - source_url - ссылка на исходный документ (одна на документ)

Свойства:
  - один источник на один source_url (дедупликация по URL);
  - структурные части одного документа объединяются
    ("пункт 8" + "пункт 9" + "пункт 10" -> "пункты 8\u201310");
  - без hardcode под конкретные документы (ukaz-112 и т.п.);
  - НЕ показывает технические chunk-id.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_STRUCT_KEYWORDS = (
    "Пункт", "Статья", "Раздел", "Глава", "Подпункт",
    "Пункты", "Статьи", "Разделы", "Главы",
    "Приложение", "Преамбула",
)

_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}

_PLURAL = {
    "пункт": "пункты",
    "статья": "статьи",
    "раздел": "разделы",
    "глава": "главы",
    "подпункт": "подпункты",
}

_SINGULAR = {v: k for k, v in _PLURAL.items()}


# ---------------------------------------------------------------------------
# Парсинг title
# ---------------------------------------------------------------------------

def _last_struct_part(struct: str, *, _kw_lower: set | None = None) -> tuple[str, str | None, str | None]:
    """Разобрать составной структурный элемент на (префикс, kind, num).

    >>> _last_struct_part("Пункт 8")
    ("", "пункт", "8")
    >>> _last_struct_part("Приложение 1, пункт 4")
    ("Приложение 1, ", "пункт", "4")
    >>> _last_struct_part("Раздел III")
    ("", "раздел", "III")
    >>> _last_struct_part("Преамбула")
    ("", "преамбула", None)
    >>> _last_struct_part("")
    ("", None, None)
    """
    if not struct:
        return "", None, None
    if _kw_lower is None:
        _kw_lower = {kw.lower() for kw in _STRUCT_KEYWORDS}
    # Идём справа налево, ищем последний структурный сегмент
    # Разделители: запятая, точка с запятой
    segments = re.split(r"[,;]\s*", struct.strip())
    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i].strip()
        if not seg:
            continue
        first = seg.split()[0].lower() if seg.split() else ""
        if first in _kw_lower:
            prefix = ", ".join(segments[:i]).strip()
            if prefix:
                prefix += ", "
            kind, num = _struct_kind_number(seg)
            return prefix, kind, num
    return "", None, None


def _split_title(title: str) -> tuple[str, str | None]:
    """Разбить title на (название, структурный элемент)."""
    if not title:
        return "", None
    t = title.strip()
    if "\u2014" in t:
        left, _, right = t.partition("\u2014")
        struct = right.strip()
        return left.strip(), (struct or None)
    for sep in (" - ", "\u2013"):
        if sep in t:
            left, _, right = t.partition(sep)
            struct = right.strip()
            return left.strip(), (struct or None)
    return t, None


def _struct_kind(struct: str | None) -> str | None:
    if not struct:
        return None
    parts = struct.split()
    first = parts[0] if parts else ""
    for kw in _STRUCT_KEYWORDS:
        if first.lower().startswith(kw.lower()):
            return _SINGULAR.get(kw.lower(), kw.lower())
    return None


def _extract_number(struct: str | None) -> str | None:
    if not struct:
        return None
    parts = struct.split()
    for tok in parts[1:]:
        tok_clean = tok.strip(",.;:")
        if not tok_clean:
            continue
        if tok_clean.replace(".", "", 1).isdigit():
            return tok_clean
        if tok_clean in _ROMAN:
            return tok_clean
        break
    return None


def _struct_kind_number(struct: str | None) -> tuple[str | None, str | None]:
    kind = _struct_kind(struct)
    num = _extract_number(struct) if kind else None
    return kind, num


def _struct_description(struct: str | None) -> str:
    kind, num = _struct_kind_number(struct)
    if kind and num:
        return f"{kind} {num}"
    return struct or ""


# ---------------------------------------------------------------------------
# Объединение структур в диапазоны / перечисления
# ---------------------------------------------------------------------------

def _num_or_roman(x: str, is_roman: bool) -> int:
    if is_roman:
        return _ROMAN[x]
    return int(float(x))


def _sort_nums(nums, is_roman: bool):
    def _key(x):
        if is_roman:
            return _ROMAN[x]
        return (0, float(x.replace(",", "."))) if x.replace(",", ".").replace(".", "", 1).isdigit() else (1, 0)
    return sorted(nums, key=_key)


def _format_kind_nums(kind: str, nums: list) -> str:
    if not nums:
        return ""
    is_roman = all(n in _ROMAN for n in nums)
    try:
        all_numeric = all(n.replace(".", "", 1).isdigit() for n in nums)
    except AttributeError:
        all_numeric = False

    if all_numeric or is_roman:
        srt = _sort_nums(nums, is_roman)
        ranges: list = []
        start = prev = srt[0]
        for n in srt[1:]:
            if _num_or_roman(n, is_roman) == _num_or_roman(prev, is_roman) + 1:
                prev = n
            else:
                ranges.append((start, prev))
                start = prev = n
        ranges.append((start, prev))
        items = [a if a == b else f"{a}\u2013{b}" for a, b in ranges]
        num_str = ", ".join(items)
    else:
        num_str = ", ".join(sorted(nums))

    label = _PLURAL.get(kind, kind) if len(nums) > 1 else kind
    return f"{label} {num_str}"


# ---------------------------------------------------------------------------
# Группировка и публичная точка
# ---------------------------------------------------------------------------

class _SourceGroup:
    """Группа источников одного URL.

    Вместо неструктурированного набора строк хранит разобранные
    структурные части: (prefix, kind) → list[numbers].
    """

    __slots__ = ("url", "doc", "_subs", "max_score")

    def __init__(self, url: str, doc: str):
        self.url = url
        self.doc = doc
        self._subs: dict[tuple[str, str], list[str]] = {}
        self.max_score = None

    def add(self, struct, score) -> None:
        if struct:
            prefix, kind, num = _last_struct_part(struct)
            if kind and num is not None:
                self._subs.setdefault((prefix, kind), []).append(num)
            elif kind:
                # kind без номера (например, "Преамбула")
                key = (prefix, kind)
                self._subs.setdefault(key, [])
            else:
                # неструктурированный (не подходит ни под один kind)
                key = ("", struct)
                self._subs.setdefault(key, [])
        if score is not None:
            val = float(score)
            if self.max_score is None or val > self.max_score:
                self.max_score = val

    @staticmethod
    def _dedup_nums(nums: list[str]) -> list[str]:
        """Убрать дубликаты номеров, сохранив порядок первого вхождения."""
        seen: set[str] = set()
        result: list[str] = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                result.append(n)
        return result

    def format(self) -> str:
        """Собрать читаемую строку структурных элементов."""
        parts: list[str] = []
        other: list[str] = []
        for (prefix, kind), nums in sorted(self._subs.items()):
            if kind and nums:
                # Группа с номерами: группируем в диапазоны
                nums = self._dedup_nums(nums)
                num_str = _format_kind_nums(kind, nums)
                parts.append(f"{prefix}{num_str}")
            elif kind:
                parts.append(f"{prefix}{kind}")
            else:
                other.append(f"{prefix}{kind}" if prefix else kind)
        for s in sorted(other):
            if s:
                parts.append(s)
        return ", ".join(parts)


def _iter_nodes(nodes):
    for n in nodes if nodes else []:
        if hasattr(n, "node"):
            meta = getattr(n.node, "metadata", {}) or {}
            score = getattr(n, "score", None)
            yield meta.get("title", ""), meta.get("source_url", ""), score
        elif isinstance(n, dict):
            meta = n.get("metadata", n)
            yield (
                meta.get("title", n.get("title", "")),
                meta.get("source_url", n.get("url", "")),
                n.get("score"),
            )
        else:
            yield getattr(n, "title", ""), getattr(n, "url", ""), getattr(n, "score", None)


def collect_sources(nodes, max_sources: int = 3) -> list[dict]:
    """Собрать и дедуплицировать источники из нод."""
    groups: dict[str, _SourceGroup] = {}
    for title, url, score in _iter_nodes(nodes):
        if not url:
            continue
        g = groups.setdefault(url, _SourceGroup(url, ""))
        doc, struct = _split_title(title)
        if doc:
            g.doc = doc
        g.add(struct, score)

    scored = sorted(
        groups.values(),
        key=lambda g: (g.max_score if g.max_score is not None else float("-inf")),
        reverse=True,
    )
    result: list[dict] = []
    for g in scored[:max_sources]:
        struct_str = g.format()
        title = g.doc
        if struct_str:
            title = f"{g.doc} \u2014 {struct_str}"
        result.append({
            "url": g.url,
            "title": (title or g.url)[:120],
            "score": g.max_score,
        })
    return result
