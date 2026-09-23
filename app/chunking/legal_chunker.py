"""Legal chunker: markdown/*.md -> structure JSON -> chunks JSONL.

Вход — markdown/<document>.md, для каждого ищется соответствующий
structure/<document>.json (результат markdown_structure_parser).
Выход — chunks/<document>.jsonl (по одному JSON-объекту на строку).

Модуль не знает ни о PDF, ни об HTML, ни о Docling, ни о Pandoc — он работает
ТОЛЬКО поверх records-слоя structure JSON.

Схема каждой строки JSONL (ровно 5 полей, без дополнительных):
  id, title, text, local_img, url

Логика нарезки:
  - Статья — логическая граница. Заголовок статьи присутствует в каждом её чанке.
  - Разные статьи не смешиваются в одном чанке.
  - Большие статьи делятся по paragraph/item/subparagraph-блокам.
  - Отдельный элемент, превышающий MAX_TOKENS, режется по предложениям, затем по словам.
  - Мелкие последовательные части одной статьи объединяются до TARGET_TOKENS.
  - Ни один чанк не превышает MAX_TOKENS.

Редакционные блоки преамбулы ("(В редакции федеральных законов от ...)",
"(С учетом ...)") НЕ дробятся на множество поисковых чанков: они остаются
единым блоком, а при превышении лимита обрезаются в пределах одного чанка.
"""
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
STRUCTURE_DIR = PROJECT_DIR / "structure"
MARKDOWN_DIR = PROJECT_DIR / "markdown"
CHUNKS_DIR = PROJECT_DIR / "chunks"
RESOLVED_CACHE = PROJECT_DIR / "app" / "resolved_documents.json"

# Лимиты (в токенах FRIDA).
TARGET_TOKENS = 450
MAX_TOKENS = 512
# Хвостовые обрывки короче этого порога объединяются с предыдущим чанком.
MIN_CHUNK_TOKENS = 40

_MD_HEADING_RE = re.compile(r"^#{1,6}\s*")
_WS_RE = re.compile(r"\s+")
_SENT_RE = re.compile(r"(?<=[.;!?])\s+(?=[А-ЯA-Z0-9])")
# Редакционные блоки преамбулы (списки редакций) — единый, не дробятся.
_EDITORIAL_RE = re.compile(
    r"^\s*\((?:В редакции|В ред\.|С учетом|С учётом)", re.IGNORECASE)

_TOKENIZER = None  # None=не загружен, False=недоступен, иначе объект tokenizer


def _load_tokenizer():
    """Загрузить FRIDA-токенизатор в offline-режиме (кэш в _TOKENIZER).

    Если модель недоступна — возвращает None, и count_tokens() переходит
    на эвристику (~4 символа на токен).
    """
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER or None
    model_dir = PROJECT_DIR / "hf_cache" / "FRIDA"
    if not model_dir.exists():
        _TOKENIZER = False
        return None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        _TOKENIZER = tok
        return tok
    except Exception:
        _TOKENIZER = False
        return None


@lru_cache(maxsize=1024)
def _tokenize_cached(text: str):
    """Точные token IDs текста (tuple), либо None, если токенизатор недоступен.

    Кэш: один и тот же длинный текст (гигантский editorial/preamb-блок) не
    токенизируется повторно — и в count_tokens, и в нарезке используются
    одни и те же ids. Возвращаем tuple (неизменяемо, безопасно резать).
    """
    tok = _load_tokenizer()
    if tok is None:
        return None
    try:
        return tuple(tok.encode(text, add_special_tokens=False))
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Число токенов. Точный подсчёт токенами FRIDA (через кэш ids).

    Эвристика ~4 символа на токен используется ТОЛЬКО как fallback,
    если токенизатор недоступен или упал.
    """
    if not text:
        return 0
    ids = _tokenize_cached(text)
    if ids is not None:
        return len(ids)
    return max(1, (len(text) + 3) // 4)


def _norm_text(s: str) -> str:
    """Убрать markdown-заголовки и схлопнуть пробелы."""
    s = _MD_HEADING_RE.sub("", s)
    return _WS_RE.sub(" ", s).strip()


def _clean_header(text: str) -> str:
    """Очищенный заголовок (например, 'Статья 1. Основные термины')."""
    return _norm_text(text)


def _is_editorial(text: str) -> bool:
    """True, если блок — редакционная пометка/список редакций преамбулы."""
    return bool(_EDITORIAL_RE.match(text))


def _truncate_to_limit(text: str, limit: int) -> str:
    """Обрезать текст до limit токенов по границе слова.

    НИКОГДА не разрывает слово и не добавляет " …".
    """
    if count_tokens(text) <= limit:
        return text

    # 1. Обрезаем по предложениям
    sents = _SENT_RE.split(text)
    if len(sents) <= 1:
        sents = [text]

    for n in range(len(sents) - 1, 0, -1):
        candidate = "".join(sents[:n]).strip()
        if not candidate:
            continue
        if count_tokens(candidate) <= limit:
            return candidate

    # 2. Обрезаем по словам
    words = text.split()
    for n in range(len(words) - 1, 0, -1):
        candidate = " ".join(words[:n])
        if not candidate:
            continue
        if count_tokens(candidate) <= limit:
            return candidate

    # 3. Крайний случай — первое слово (если и оно не влезает, то пустая строка)
    if words and count_tokens(words[0]) <= limit:
        return words[0]
    return ""


def _fit_full_text(prefix: str, part: str, max_tokens: int) -> str:
    """Гарантировать count_tokens(prefix + '\\n' + part) <= max_tokens.

    Если полный текст не влезает, обрезает part по естественным границам
    (предложение → слово). НИКОГДА не разрывает слово и не добавляет " …".
    """
    full = prefix + "\n" + part
    if count_tokens(full) <= max_tokens:
        return full

    # 1. Обрезаем по предложениям
    sents = _SENT_RE.split(part)
    if len(sents) <= 1:
        sents = [part]

    for n in range(len(sents) - 1, 0, -1):
        candidate_body = "".join(sents[:n]).strip()
        if not candidate_body:
            continue
        candidate = prefix + "\n" + candidate_body
        if count_tokens(candidate) <= max_tokens:
            return candidate

    # 2. Если не влезает даже одно предложение — режем по словам
    words = part.split()
    for n in range(len(words) - 1, 0, -1):
        candidate_body = " ".join(words[:n])
        if not candidate_body:
            continue
        candidate = prefix + "\n" + candidate_body
        if count_tokens(candidate) <= max_tokens:
            return candidate

    # 3. Крайний случай — только префикс (тело пустое)
    return prefix



def _split_oversized(para: str, limit: int, prefix: str | None = None) -> list[str]:
    """Разрезать слишком длинный элемент: по предложениям, затем по словам.

    Каждая возвращаемая часть гарантированно <= limit токенов.
    Ни одна часть не содержит разорванных слов.
    """
    result: list[str] = []

    def _cut_by_tokens(text: str) -> list[str] | None:
        """Нарезать текст на части <= limit токенов, не разрывая слова.

        Разбиение идёт по словам (split()), группировка по limit токенов.
        Гарантирует, что ни одно слово не будет разрезано.
        Возвращает None, если токенизатор недоступен (тогда вызывающий
        использует запасной путь _push_words).
        """
        # Проверка доступности токенизатора через кэш
        ids = _tokenize_cached(text)
        if ids is None:
            return None
        words = text.split()
        parts: list[str] = []
        cur: list[str] = []
        ct = 0
        for w in words:
            tw = count_tokens(w)
            if ct + tw > limit and cur:
                parts.append(" ".join(cur))
                cur = []
                ct = 0
            cur.append(w)
            ct += tw
        if cur:
            parts.append(" ".join(cur))
        return parts

    def _push_words(words: list[str]) -> None:
        """Запасной путь (без токенизатора): per-word count_tokens (эвристика)."""
        cur: list[str] = []
        ct = 0
        for w in words:
            tw = count_tokens(w)
            if ct + tw > limit and cur:
                result.append(" ".join(cur))
                cur = []
                ct = 0
            cur.append(w)
            ct += tw
        if cur:
            result.append(" ".join(cur))

    for sent in _SENT_RE.split(para):
        sent = sent.strip()
        if not sent:
            continue
        if count_tokens(sent) <= limit:
            result.append(sent)
            continue
        parts = _cut_by_tokens(sent)
        if parts is not None:
            result.extend(parts)
        else:
            _push_words(sent.split())

    # Финальная гарантия: любой кусок, всё ещё превышающий лимит, режем по словам.
    fitted: list[str] = []
    for p in result:
        if count_tokens(p) > limit:
            parts = _cut_by_tokens(p)
            if parts is not None:
                fitted.extend(parts)
            else:
                _push_words(p.split())
        else:
            fitted.append(p)
    return [p for p in fitted if p.strip()]


def _pack_blocks(
    blocks: list[tuple[str, bool, str | None]],
    prefix: str,
    max_tokens: int,
    target_tokens: int,
    min_tokens: int,
    split_block_indices: set[int] | None = None,
    paragraph_is_boundary: bool = False,
) -> list[tuple[str, list[int]]]:
    """Упаковать блоки (текст, is_editorial, paragraph_num) в чанки.

    Правила:
      - Никогда не превышать max_tokens.
      - Стремиться к target_tokens.
      - Не разрывать блоки; редакционные блоки не дробить (обрезать одним чанком).
      - Мелкие хвостовые части объединять с предыдущим чанком.
      - Если paragraph_is_boundary=True, смена paragraph вызывает flush()
        (для сегментов, где paragraph — основная структурная единица).
      - Если paragraph_is_boundary=False, paragraph используется только
        для метаданных, но не как граница чанка.

    Возвращает:
        list[tuple[str, list[int]]] — (текст части, индексы исходных блоков).
    """
    prefix_tokens = count_tokens(prefix)
    # Запас в 1 токен на "\n" между префиксом и телом.
    # Для editorial-блоков и split_oversized используется limit.
    # Для обычных блоков решение принимается по полному тексту через count_tokens.
    limit = max(max_tokens - prefix_tokens - 1, 1)
    chunks: list[tuple[str, list[int]]] = []
    current: list[str] = []
    current_indices: list[int] = []
    current_para: str | None = None

    def _full_tokens(body_parts: list[str]) -> int:
        """Токены полного текста чанка: prefix + '\\n' + body."""
        if not body_parts:
            return prefix_tokens
        return count_tokens(prefix + "\n" + "\n".join(body_parts))

    def flush() -> None:
        nonlocal current, current_indices
        if current:
            chunks.append(("\n".join(current), current_indices))
            current = []
            current_indices = []

    for bi, (text, is_editorial, para) in enumerate(blocks):

        # Force split before this block if in split_block_indices
        if split_block_indices and bi in split_block_indices:
            flush()

        # Граница параграфа: новый пункт — новый chunk (только для сегментов,
        # где paragraph — основная структурная единица, например preamble).
        # Для article/appendix/section параграфы объединяются по токенам.
        # Continuation (para=None) наследует текущий paragraph и НЕ вызывает flush.
        if current and para is not None and current_para is not None and para != current_para and paragraph_is_boundary:
            flush()

        if is_editorial:
            # Редакционный блок — единый, НЕ дробить на много чанков.
            b_t = count_tokens(text)
            if b_t > limit:
                flush()
                chunks.append((_truncate_to_limit(text, limit), [bi]))
                continue
            if current and _full_tokens(current + [text]) > max_tokens:
                flush()
            current.append(text)
            current_indices.append(bi)
            continue

        # Обычный блок: если один элемент превышает лимит — дробить.
        b_t = count_tokens(text)
        if b_t > limit:
            flush()
            for fragment in _split_oversized(text, limit, prefix):
                chunks.append((fragment, [bi]))
            continue

        # Проверка по полному тексту: влезает ли блок в текущий чанк?
        candidate_t = _full_tokens(current + [text])
        if current and candidate_t > max_tokens:
            flush()
        if para is not None:
            current_para = para
        current.append(text)
        current_indices.append(bi)

    flush()

    # Объединение мелких хвостовых частей.
    # При paragraph_is_boundary=True НЕ объединять чанки из разных параграфов.
    # При paragraph_is_boundary=False разные параграфы могут объединяться.
    if len(chunks) >= 2:
        merged: list[tuple[str, list[int]]] = []
        for ch_text, ch_indices in chunks:
            if merged:
                prev_text, prev_indices = merged[-1]
                # Определяем параграф для каждого чанка (по первому блоку в нём).
                prev_para = blocks[prev_indices[0]][2] if prev_indices else None
                curr_para = blocks[ch_indices[0]][2] if ch_indices else None
                # Разные известные параграфы — не объединять (только для boundary-режима).
                if paragraph_is_boundary and prev_para is not None and curr_para is not None and prev_para != curr_para:
                    merged.append((ch_text, ch_indices))
                    continue
                prev_t = count_tokens(prev_text)
                if prev_t < min_tokens and _full_tokens([prev_text, ch_text]) <= max_tokens:
                    merged[-1] = (prev_text + "\n" + ch_text, prev_indices + ch_indices)
                    continue
            merged.append((ch_text, ch_indices))
        chunks = merged

    return chunks



def _get_paragraph_ctx(rec: dict) -> str | None:
    """Извлечь номер пункта из context_flat."""
    return (rec.get("structure", {}).get("context_flat") or {}).get("paragraph") or None


def _group_segments(records: list[dict]) -> list[dict]:
    """Сгруппировать records в сегменты.

    Границы сегментов определяются по context_flat НА СТРУКТУРНОМ уровне:

      - chapter:    изменение context_flat.chapter
      - section:    изменение context_flat.section
      - subsection: изменение context_flat.subsection
      - article:    запись с type='article' (или смена context_flat.article)
      - appendix:   запись с type='appendix' (или смена context_flat.appendix)

    Paragraph/subparagraph/item НЕ создают границ сегментов: дочерние
    структурные элементы объединяются по token-бюджету внутри РОДИТЕЛЬСКОГО
    структурного сегмента. Так один и тот же универсальный алгоритм работает
    для федеральных законов (статьи), указов/положений (разделы/главы) и
    приказов (приложения/пункты) без специальных условий про конкретный
    тип документа.

    Каждый сегмент:
      {type, number, title, article, appendix, chapter, section, body(list[dict])}.
    """
    segments: list[dict] = []
    current: dict | None = None
    prev_ctx: dict = {}

    # Структурные ключи в порядке приоритета: если меняется несколько сразу,
    # выбираем самый верхний уровень иерархии.
    STRUCTURAL = ("chapter", "section", "subsection", "article", "appendix")
    TYPE_BY_KEY = {
        "chapter": "chapter",
        "section": "section",
        "subsection": "subsection",
        "article": "article",
        "appendix": "appendix",
    }

    def flush() -> None:
        nonlocal current
        if current and current["body"]:
            segments.append(current)
        current = None

    for rec in records:
        st = rec["structure"]
        rtype = st["type"]
        ctx = st.get("context_flat") or {}

        # Определяем изменившийся структурный ключ (самый приоритетный).
        change_key = None
        for key in STRUCTURAL:
            cur_val = ctx.get(key)
            prev_val = prev_ctx.get(key)
            # «cur_val truthy» — защита от ложного сегмента при выходе
            # из структурного элемента (номер уходит в None после последней
            # статьи/раздела), когда это не начало нового элемента.
            if cur_val and cur_val != prev_val:
                change_key = key
                break

        # Явные heading-записи (article/appendix) — всегда граница,
        # даже если context_flat не изменился (например, статьи-подпункты
        # «Статья 59», «Статья 59 1», «Статья 59 2» имеют одинаковый
        # context_flat.article=59, но это самостоятельные статьи).
        if rtype == "article":
            change_key = "article"
        elif rtype == "appendix" and ctx.get("appendix"):
            change_key = "appendix"

        if change_key is not None:
            flush()
            seg_type = TYPE_BY_KEY[change_key]
            seg_number = ctx.get(change_key) or st.get("number")

            if seg_type == "section":
                section_num = ctx.get("section") or st.get("number") or ""
                title = f"Раздел {section_num}" if section_num else ""
            elif seg_type == "chapter":
                chapter_num = ctx.get("chapter") or st.get("number") or ""
                title = f"Глава {chapter_num}" if chapter_num else ""
            elif seg_type == "subsection":
                sub_num = ctx.get("subsection") or st.get("number") or ""
                title = f"Подраздел {sub_num}" if sub_num else ""
            else:
                # article/appendix — используем текст записи (в нём уже есть
                # номер + полное название, например:
                # «Статья 59 3. Порядок применения взысканий...»).
                title = _clean_header(rec["text"])

            current = {
                "type": seg_type,
                "number": seg_number,
                "title": title,
                "article": ctx.get("article"),
                "appendix": ctx.get("appendix"),
                "chapter": ctx.get("chapter"),
                "section": ctx.get("section"),
                "body": [],
            }
            # Для section/chapter/subsection триггер-запись — content-запись
            # (paragraph/text), её нужно добавить в тело сегмента.
            if seg_type in ("section", "chapter", "subsection", "appendix"):
                current["body"].append(rec)
            prev_ctx = ctx
            continue

        if current is None:
            current = {
                "type": "preamble", "number": None,
                "title": "Преамбула", "article": None,
                "appendix": None, "chapter": None,
                "section": None, "body": [],
            }

        current["body"].append(rec)
        prev_ctx = ctx

    flush()

    # Финальный проход: уточняем заголовки для сегментов без фиксированного
    # структурного title (preamble/прочее). Если в таком сегменте пункт —
    # верхний значимый структурный элемент без более релевантного родителя,
    # title = «Пункт N». Иначе — «Преамбула».
    for seg in segments:
        if seg["type"] not in ("article", "appendix", "section", "chapter", "subsection"):
            body = seg["body"]
            if body:
                first_para = _get_paragraph_ctx(body[0]) if body else None
                seg["title"] = f"Пункт {first_para}" if first_para else "Преамбула"

    return segments


def _make_doc_display_name(safe_doc_id: str, doc_display: str) -> str:
    """Преобразовать идентификатор документа в читаемое название.

    Examples:
        '79-fz', '79-ФЗ'           -> '79-ФЗ'
        'ukaz-557', '557'            -> 'Указ № 557'
        'postanovlenie-1000', '1000'  -> 'Постановление № 1000'
        'rasporyazhenie-2867-r', '2867-р' -> 'Распоряжение № 2867-р'
    """
    if safe_doc_id.lower().endswith('-fz'):
        # Для ФЗ doc_display уже в формате '79-ФЗ'
        return doc_display
    m = re.match(r'^(ukaz|postanovlenie|rasporyazhenie)-(.+)', safe_doc_id)
    if m:
        prefix_map = {
            'ukaz': 'Указ',
            'postanovlenie': 'Постановление',
            'rasporyazhenie': 'Распоряжение',
        }
        type_label = prefix_map[m.group(1)]
        return f'{type_label} № {doc_display}'
    # Fallback — doc_display как есть
    return doc_display


def _safe_truncate(text: str, max_len: int = 120) -> str:
    """Обрезать до max_len по границе слова, добавить '…' если обрезано."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_space = truncated.rstrip().rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + '…'


# ---------------------------------------------------------------------------
# Детерминированный semantic title (rule-based, без LLM/ML)
# ---------------------------------------------------------------------------


def _make_semantic_title(text: str, doc_display_name: str, point_key: str | None) -> str:
    """Структурный title чанка на основе заголовка сегмента документа.

    Title формируется ТОЛЬКО из реальной структуры документа
    (статья/раздел/глава/приложение/пункт/преамбула).
    Никаких семантических догадок, keyword-категорий или извлечения
    темы из текста чанка.

    Правила (по приоритету):

    0. point_key со структурным префиксом
       (Статья/Раздел/Глава/Приложение/Пункт/Преамбула)
       → ``"{doc}: {point_key}"``.
    1. point_key — чистое число → ``"{doc} — Пункт {num}"``.
    2. point_key is None, текст начинается со структурного заголовка
       (Статья/Раздел/Глава/Приложение N. Content) → полный заголовок.
    3. point_key is None, текст с номера в начале
       → ``"{doc} — Пункт {num}"``.
    Fallback — только ``doc_display_name``.

    Args:
        text: Тело чанка (без префикса ``[doc] [title]``).
        doc_display_name: Отображаемое имя документа.
        point_key: Структурный заголовок сегмента, номер пункта или None.

    Returns:
        Структурный заголовок чанка.
    """
    # ==================================================================
    # Priority 0: Структурный заголовок сегмента
    # Если point_key — это полный заголовок статьи/раздела/главы/
    # приложения/пункта/преамбулы, используем его как есть.
    # ==================================================================
    if point_key and not re.match(r'^\d+(?:[.]\d+)*$', point_key):
        return f"{doc_display_name}: {point_key}"

    # ==================================================================
    # Priority 1: Чистый номер пункта (point_key — число)
    # ==================================================================
    if point_key and re.match(r"^\d+(?:[.]\d+)*$", point_key):
        return f"{doc_display_name} — Пункт {point_key}"

    # ==================================================================
    # Priority 2: Структурный заголовок из текста (point_key is None)
    # Если текст начинается со "Статья/Раздел/Глава/Приложение N.",
    # извлекаем полный заголовок.
    # ==================================================================
    if text and point_key is None:
        m = re.match(
            r"^\s*(Статья|Раздел|Глава|Приложение)\s+"
            r"(\d+(?:[\s.]\d+)*)[.)]?\s*(.*)",
            text,
            re.IGNORECASE,
        )
        if m:
            kind = m.group(1)
            num = m.group(2).strip()
            content = m.group(3).strip().rstrip(".")
            title_part = f"{kind} {num}"
            if content:
                title_part += f". {content}"
            return f"{doc_display_name}: {title_part}"

        # Если текст начинается с числа → это пункт
        m = re.match(r"^\s*(\d+(?:[\s.]\d+)*)\.[ 	]+", text)
        if m:
            num = re.sub(r"\s+", ".", m.group(1).strip())
            return f"{doc_display_name} — Пункт {num}"

    # ==================================================================
    # Fallback — только имя документа
    # ==================================================================
    return doc_display_name

def _build_chunks_for_segment(
    seg: dict,
    doc_id: str,
    doc_display: str,
    doc_display_name: str,
    source_url: str,
    max_tokens: int,
    target_tokens: int,
    min_tokens: int,
    seg_idx: int = 0,
) -> list[dict]:
    """Собрать чанки из одного сегмента (статьи/преамбулы/приложения)."""
    seg_type = seg["type"]

    # Собираем блоки + отслеживаем, какие body-рекорды в какой блок попали
    # Каждый блок хранит (text, is_editorial, paragraph_num) —
    # paragraph_num наследуется от последнего явного paragraph/пункта для text-блоков,
    # что позволяет корректно определять границы между разными пунктами.
    blocks: list[tuple[str, bool, str | None]] = []
    block_to_record: list[int] = []  # индекс body-рекорда для каждого блока
    current_para: str | None = None
    for bi, r in enumerate(seg["body"]):
        txt = _norm_text(r["text"])
        if not txt:
            continue
        para = _get_paragraph_ctx(r)
        if para is not None:
            current_para = para  # явный номер пункта — обновляем контекст
        # text-блоки без номера наследуют номер последнего явного пункта
        blocks.append((txt, _is_editorial(txt), current_para))
        block_to_record.append(bi)

    if not blocks:
        return []

    # Пробуем сегментный префикс; для динамических заголовков используем
    # первый попавшийся paragraph в теле
    prefix_title = seg["title"]
    if seg_type not in ("article", "appendix", "section", "chapter", "subsection"):
        # Пробуем найти paragraph-контекст для префикса
        pn = _get_paragraph_ctx(seg["body"][0]) if seg["body"] else None
        if pn:
            prefix_title = f"Пункт {pn}"
        else:
            prefix_title = "Преамбула"

    prefix = f"[{doc_display}] [{prefix_title}]"
    prefix_tokens = count_tokens(prefix)

    # Detect blocks with premiums «maximum size is not limited» -
    # they should not be merged with bonus blocks into one chunk.
    split_block_indices: set[int] = set()
    for bi_p, (txt_p, _, _) in enumerate(blocks):
        if "премии" in txt_p.lower() and "не ограничивается" in txt_p.lower():
            split_block_indices.add(bi_p)

    parts = _pack_blocks(blocks, prefix, max_tokens, target_tokens, min_tokens,
                         split_block_indices=split_block_indices,
                         paragraph_is_boundary=False)

    chunks: list[dict] = []
    for i, (part, part_block_indices) in enumerate(parts, 1):
        part = part.strip()
        if not part:
            continue

        # Определяем заголовок для этого чанка из paragraph первого блока.
        # blocks[][2] уже содержит наследованный paragraph для continuation.
        first_block_in_part = part_block_indices[0] if part_block_indices else 0
        pn_block = blocks[first_block_in_part][2]  # paragraph из блока (уже наследован)
        if pn_block and seg_type not in ("article", "appendix", "section", "chapter", "subsection"):
            point_key = pn_block
            chunk_title = f"Пункт {pn_block}"
        else:
            point_key = prefix_title
            chunk_title = prefix_title

        cid = _make_chunk_id(seg, doc_id, i, seg_type, seg_idx=seg_idx)

        # Строим text с корректным префиксом
        chunk_prefix = f"[{doc_display}] [{chunk_title}]"
        chunks.append({
            "id": cid,
            "title": _safe_truncate(_make_semantic_title(part, doc_display_name, point_key)),
            "text": _fit_full_text(chunk_prefix, part, max_tokens),
            "local_img": "",
            "url": source_url,
        })

    return chunks


def _extract_full_article_number(title: str) -> str | None:
    """Извлечь полный номер статьи из заголовка, включая подстатьи.

    'Статья 12. Поступление...'             -> '12'
    'Статья 12 1. Порядок...'              -> '12_1'
    'Статья 12.1. Порядок...'              -> '12_1'
    'Статья 20 2. Представление...'         -> '20_2'
    'Статья 18 1. Военная служба...'        -> '18_1'

    Если не удалось извлечь — возвращает None.
    """
    m = re.match(
        r'(?:Статья|Раздел|Глава|Пункт)\s+'
        r'(\d+(?:[\s.]+\d+)*)',
        title,
    )
    if m:
        num = m.group(1)
        # Нормализация разделителей: точка -> '_', множественные пробелы -> '_'
        num = num.replace('.', '_')
        num = '_'.join(num.split())
        return num
    return None


def _make_chunk_id(seg: dict, doc_id: str, part_index: int, seg_type: str,
                   seg_idx: int = 0) -> str:
    """Сгенерировать id для чанка на основе сегмента.

    Для article дополнительно извлекает полный номер из заголовка,
    чтобы «Статья 12» и «Статья 12 1» (12.1) получали разные ID.
    Для section/chapter/subsection с отсутствующим номером используется seg_idx,
    чтобы разные секции не получали одинаковые ID.
    """
    if seg_type == "article":
        full_num = _extract_full_article_number(seg.get("title") or "")
        if full_num:
            safe_num = full_num
        else:
            safe_num = str(seg.get("number") or seg.get("article") or "0").replace(".", "_")
        return f"{doc_id}_st{safe_num}_p{part_index}"
    elif seg_type == "appendix":
        safe_num = str(seg.get("number") or seg.get("appendix") or "app").replace(".", "_")
        return f"{doc_id}_app{safe_num}_p{part_index}"
    elif seg_type == "section":
        raw_num = seg.get("number")
        if raw_num:
            safe_num = str(raw_num).replace(".", "_")
        else:
            safe_num = f"s{seg_idx}"
        return f"{doc_id}_sec{safe_num}_p{part_index}"
    elif seg_type == "chapter":
        raw_num = seg.get("number")
        if raw_num:
            safe_num = str(raw_num).replace(".", "_")
        else:
            safe_num = f"ch{seg_idx}"
        return f"{doc_id}_ch{safe_num}_p{part_index}"
    elif seg_type == "subsection":
        raw_num = seg.get("number")
        if raw_num:
            safe_num = str(raw_num).replace(".", "_")
        else:
            safe_num = f"sub{seg_idx}"
        return f"{doc_id}_sub{safe_num}_p{part_index}"
    else:  # preamble или body
        return f"{doc_id}_pre_p{part_index}"

def _load_resolved_meta() -> dict:
    """Метаданные документов из app/resolved_documents.json (содержат pdf_path)."""
    if not RESOLVED_CACHE.exists():
        return {}
    try:
        return json.loads(RESOLVED_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_source_url(structure_fname: str, doc: dict | None = None) -> str:
    """Проектно-относительная ссылка на исходный PDF документа.

    Берётся pdf_path из app/resolved_documents.json и приводится к виду
    относительно корня проекта (например, 'raw/79-FZ.pdf').

    Если метаданные отсутствуют — безопасный fallback 'raw/<stem>.pdf'.
    """
    stem = Path(structure_fname).stem
    meta = _load_resolved_meta()

    candidates: list[str] = []
    if doc:
        for key in ("id", "number", "title"):
            val = doc.get(key)
            if val:
                candidates.append(str(val))
    candidates.append(stem)

    entry = None
    for cand in candidates:
        if cand in meta:
            entry = meta[cand]
            break

    pdf_path = (entry or {}).get("pdf_path") if entry else None
    if not pdf_path:
        return f"raw/{stem}.pdf"

    p = Path(str(pdf_path))
    try:
        rel = p.relative_to(PROJECT_DIR)
    except ValueError:
        rel = Path(p.name)  # вне проекта — оставляем только имя файла
    return rel.as_posix()


def convert(
    structure_fname: str,
    structure_dir: Path | None = None,
    out_dir: Path | None = None,
    source_url: str | None = None,
) -> Path:
    """Собрать чанки из одного structure JSON в chunks JSONL.

    Args:
        structure_fname: Имя файла в structure_dir (например, "79-FZ.json").
        structure_dir: Директория со structure JSON (по умолчанию structure/).
        out_dir: Директория для chunks (по умолчанию chunks/).
        source_url: Явная ссылка на PDF. Если не задана — автоматически
            берётся из метаданных (pdf_path в app/resolved_documents.json).

    Returns:
        Путь к созданному .jsonl файлу.
    """
    structure_dir = structure_dir or STRUCTURE_DIR
    out_dir = out_dir or CHUNKS_DIR
    src = structure_dir / structure_fname
    if not src.exists():
        raise FileNotFoundError(f"structure JSON ne najden: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    doc = data.get("doc") or {}
    records = data.get("records") or []

    # Используем имя файла как базовый doc_id (детерминированно, без коллизий).
    # doc.id/number из JSON могут совпадать у разных документов (например, "79-ФЗ").
    safe_doc_id = re.sub(r"[^\w\-.]", "_", src.stem).lower()
    doc_display = doc.get("number") or doc.get("title") or safe_doc_id

    if not source_url:
        source_url = resolve_source_url(structure_fname, doc=doc)

    doc_display_name = _make_doc_display_name(safe_doc_id, doc_display)
    segments = _group_segments(records)
    chunks: list[dict] = []
    for seg_idx, seg in enumerate(segments, 1):
        chunks.extend(_build_chunks_for_segment(
            seg, safe_doc_id, doc_display, doc_display_name, source_url,
            MAX_TOKENS, TARGET_TOKENS, MIN_CHUNK_TOKENS,
            seg_idx=seg_idx))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + ".jsonl")
    # Атомарная запись: сначала во временный файл, затем os.replace().
    tmp = out.with_suffix(".jsonl.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    print(f"done: {src.name} -> {out.relative_to(out_dir)} ({len(chunks)} chankov)")
    return out


def process_markdown_file(
    md_path: Path,
    structure_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Обработать один Markdown-файл через пайплайн чанкера.

    Находит соответствующий structure JSON и запускает нарезку.

    Args:
        md_path: Путь к исходному .md файлу.
        structure_dir: Директория со structure JSON (по умолчанию STRUCTURE_DIR).
        out_dir: Директория для выходных chunks JSONL (по умолчанию CHUNKS_DIR).

    Returns:
        Словарь со статистикой: file, out_path, chunks, min_tokens, max_tokens,
        avg_tokens, over_400.

    Raises:
        FileNotFoundError: если нет соответствующего structure JSON.
    """
    structure_dir = structure_dir or STRUCTURE_DIR
    out_dir = out_dir or CHUNKS_DIR

    stem = md_path.stem
    structure_path = structure_dir / f"{stem}.json"

    if not structure_path.exists():
        raise FileNotFoundError(
            f"Для {md_path.name} не найден structure JSON: {structure_path}"
        )

    out_path = convert(
        structure_path.name,
        structure_dir=structure_dir,
        out_dir=out_dir,
    )

    _, count, token_counts = validate_jsonl(out_path)

    return {
        "file": md_path.name,
        "out_path": out_path,
        "chunks": count,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "over_400": sum(1 for t in token_counts if t > 512),
    }


def batch_convert(
    markdown_dir: Path | None = None,
    structure_dir: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Пакетная обработка всех Markdown-файлов.

    Для каждого .md файла находит соответствующий structure JSON,
    выполняет нарезку и сохраняет результат в chunks/.
    Идемпотентно: существующие JSONL полностью заменяются.

    Args:
        markdown_dir: Директория с .md файлами (по умолчанию MARKDOWN_DIR).
        structure_dir: Директория со structure JSON (по умолчанию STRUCTURE_DIR).
        out_dir: Директория для выходных JSONL (по умолчанию CHUNKS_DIR).

    Returns:
        Список созданных .jsonl файлов.
    """
    markdown_dir = markdown_dir or MARKDOWN_DIR
    structure_dir = structure_dir or STRUCTURE_DIR
    out_dir = out_dir or CHUNKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(markdown_dir.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {markdown_dir}")
        return []

    results: list[Path] = []
    errors: list[dict] = []
    all_stats: list[dict] = []

    for md_path in md_files:
        try:
            stats = process_markdown_file(md_path, structure_dir=structure_dir, out_dir=out_dir)
            all_stats.append(stats)
            results.append(stats["out_path"])

            print(f"\n=== {md_path.name} ===")
            print(f"Чанков: {stats['chunks']}")
            print(f"Min: {stats['min_tokens']}")
            print(f"Max: {stats['max_tokens']}")
            print(f"Avg: {stats['avg_tokens']}")
            print(f">400: {stats['over_400']}")
        except Exception as exc:
            errors.append({"file": md_path.name, "error": str(exc)})
            print(f"\nFAIL: {md_path.name}: {exc}")

    # Summary
    total_chunks = sum(s["chunks"] for s in all_stats)
    print(f"\n{'=' * 60}")
    print(f"ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'=' * 60}")
    print(f"Markdown найдено: {len(md_files)}")
    print(f"Успешно обработано: {len(results)}")
    print(f"Ошибок: {len(errors)}")

    if errors:
        print("\nДокументы с ошибками:")
        for e in errors:
            print(f"  - {e['file']}: {e['error']}")

    if all_stats:
        max_overall = max(s["max_tokens"] for s in all_stats)
        over_400_total = sum(s["over_400"] for s in all_stats)
        print(f"Всего чанков: {total_chunks}")
        print(f"Максимальный chunk: {max_overall}")
        print(f">400: {over_400_total}")

    return results


def validate_jsonl(filepath: str | Path) -> tuple[list[str], int, list[int]]:
    """Проверить JSONL: парсинг, ровно требуемые ключи, непустой text.

    Returns:
        (errors, count, token_counts).
    """
    required = {"id", "title", "text", "local_img", "url"}
    errors: list[str] = []
    count = 0
    token_counts: list[int] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"stroka {i}: ne JSON")
                continue
            if set(obj.keys()) != required:
                errors.append(f"stroka {i}: klyuchi {sorted(obj.keys())} != {sorted(required)}")
            missing = required - set(obj.keys())
            if missing:
                errors.append(f"stroka {i}: net klyuchey {sorted(missing)}")
            if not str(obj.get("text", "")).strip():
                errors.append(f"stroka {i}: pustoy text")
            token_counts.append(count_tokens(str(obj.get("text", ""))))
            count += 1
    return errors, count, token_counts


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or "--all" in args:
        batch_convert()
    else:
        for a in args:
            if a == "--all":
                continue
            convert(a)
