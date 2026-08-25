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
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
STRUCTURE_DIR = PROJECT_DIR / "structure"
MARKDOWN_DIR = PROJECT_DIR / "markdown"
CHUNKS_DIR = PROJECT_DIR / "chunks"
RESOLVED_CACHE = PROJECT_DIR / "app" / "resolved_documents.json"

# Лимиты (в токенах FRIDA).
TARGET_TOKENS = 350
MAX_TOKENS = 400
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


def count_tokens(text: str) -> int:
    """Число токенов. Считает токенами FRIDA; при их недоступности — эвристика."""
    if not text:
        return 0
    tok = _load_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
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
    """Обрезать текст до limit токенов (единым блоком), сохраняя начало + ' …'.

    Использует запас в 1 токен для компенсации расхождений decode/re-encode.
    """
    tok = _load_tokenizer()
    if tok is not None:
        try:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) <= limit:
                return text
            safe_limit = max(1, limit - 1)
            cut = tok.decode(ids[:safe_limit], skip_special_tokens=True)
            result = cut.rstrip() + " …"
            # Дополнительная страховка: если после decode/re-encode всё ещё
            # превышает limit, уменьшаем ещё на 1.
            while count_tokens(result) > limit and safe_limit > 1:
                safe_limit -= 1
                cut = tok.decode(ids[:safe_limit], skip_special_tokens=True)
                result = cut.rstrip() + " …"
            return result
        except Exception:
            pass
    safe_chars = max(1, limit * 4 - 4)
    if len(text) <= safe_chars:
        return text
    return text[:safe_chars].rstrip() + " …"


def _fit_full_text(prefix: str, part: str, max_tokens: int) -> str:
    """Гарантировать count_tokens(prefix + '\\n' + part) <= max_tokens.

    Финализирующая страховка на уровне полного текста чанка. Учитывает,
    что декод/пере-кодирование токенов может дать +1-2 токена к границе.
    """
    full = prefix + "\n" + part
    if count_tokens(full) <= max_tokens:
        return full
    tok = _load_tokenizer()
    if tok is not None:
        try:
            ids = tok.encode(full, add_special_tokens=False)
            # используем запас в 1 токен для компенсации расхождений
            safe_n = len(ids) - 1
            while safe_n > 0:
                candidate = tok.decode(ids[:safe_n], skip_special_tokens=True).rstrip() + " …"
                if count_tokens(candidate) <= max_tokens:
                    return candidate
                safe_n -= 1
            return full[:1]
        except Exception:
            pass
    safe_chars = max(1, max_tokens * 4 - 4)
    return full[:safe_chars].rstrip() + " …"



def _split_oversized(para: str, limit: int) -> list[str]:
    """Разрезать слишком длинный элемент: по предложениям, затем по токенам.

    Каждая возвращаемая часть гарантированно <= limit токенов.
    """
    result: list[str] = []

    def _cut_by_tokens(text: str) -> list[str] | None:
        """Нарезать текст на части <= limit токенов за один encode/decode.

        Возвращает None, если токенизатор недоступен или упал (тогда
        вызывающий использует запасной путь по словам).
        """
        tok = _load_tokenizer()
        if tok is None:
            return None
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:
            return None
        parts: list[str] = []
        for i in range(0, len(ids), limit):
            chunk = tok.decode(ids[i:i + limit], skip_special_tokens=True).strip()
            if chunk:
                parts.append(chunk)
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

    # Финальная гарантия: любой кусок, всё ещё превышающий лимит, режем по токенам.
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
    blocks: list[tuple[str, bool]],
    prefix_tokens: int,
    max_tokens: int,
    target_tokens: int,
    min_tokens: int,
    split_block_indices: set[int] | None = None,
) -> list[tuple[str, list[int]]]:
    """Упаковать блоки (текст, is_editorial) в чанки.

    Правила:
      - Никогда не превышать max_tokens (с учётом фиксированного префикса).
      - Стремиться к target_tokens.
      - Не разрывать блоки; редакционные блоки не дробить (обрезать одним чанком).
      - Мелкие хвостовые части объединять с предыдущим чанком.

    Возвращает:
        list[tuple[str, list[int]]] — (текст части, индексы исходных блоков).
    """
    limit = max(max_tokens - prefix_tokens, 1)
    chunks: list[tuple[str, list[int]]] = []
    current: list[str] = []
    current_indices: list[int] = []
    current_t = 0

    def flush() -> None:
        nonlocal current, current_indices, current_t
        if current:
            chunks.append(("\n".join(current), current_indices))
            current = []
            current_indices = []
            current_t = 0

    for bi, (text, is_editorial) in enumerate(blocks):
        b_t = count_tokens(text)

        # Force split before this block if in split_block_indices
        if split_block_indices and bi in split_block_indices:
            flush()

        if is_editorial:
            # Редакционный блок — единый, НЕ дробить на много чанков.
            if b_t > limit:
                flush()
                chunks.append((_truncate_to_limit(text, limit), [bi]))
                continue
            if current and current_t + b_t > limit:
                flush()
            current.append(text)
            current_indices.append(bi)
            current_t += b_t
            continue

        # Обычный блок: если один элемент превышает лимит — дробить.
        if b_t > limit:
            flush()
            for fragment in _split_oversized(text, limit):
                chunks.append((fragment, [bi]))
            continue

        if current and current_t + b_t > limit:
            flush()
        elif current and current_t >= target_tokens and current_t + b_t > target_tokens:
            flush()
        current.append(text)
        current_indices.append(bi)
        current_t += b_t

    flush()

    # Объединение мелких хвостовых частей (в пределах одной статьи/преамбулы).
    if len(chunks) >= 2:
        merged: list[tuple[str, list[int]]] = []
        for ch_text, ch_indices in chunks:
            if merged:
                prev_text, prev_indices = merged[-1]
                prev_t = count_tokens(prev_text)
                if prev_t < min_tokens and count_tokens(prev_text + "\n" + ch_text) <= limit:
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

    Границы сегментов определяются по context_flat:
      - article: record type = article
      - appendix: record type = appendix (с context_flat.appendix)
      - section внутри appendix: изменение context_flat.section

    Paragraph-контекст НЕ создаёт нового сегмента, но отслеживается
    в теле через context_flat.paragraph для динамического заголовка чанков.

    Каждый сегмент: {type, number, title, article, appendix, body(list[dict])}.
    """
    segments: list[dict] = []
    current: dict | None = None
    last_appendix: str | None = None
    last_section: str | None = None

    def flush() -> None:
        nonlocal current
        if current and current["body"]:
            segments.append(current)
        current = None

    for rec in records:
        st = rec["structure"]
        rtype = st["type"]
        ctx = st.get("context_flat") or {}
        article = ctx.get("article")
        appendix = ctx.get("appendix")
        section = ctx.get("section")

        # Определяем, нужно ли создать новый сегмент
        is_new = False
        seg_type = None

        if rtype == "article":
            is_new = True
            seg_type = "article"
        elif rtype == "appendix" and appendix:
            # Приложение: только при переходе контекста
            if current is None or current.get("appendix") != appendix:
                is_new = True
                seg_type = "appendix"
                last_appendix = appendix
                last_section = None  # сбрасываем при входе в appendix
        elif appendix and section and section != last_section:
            # Смена раздела внутри приложения
            is_new = True
            seg_type = "section"
            last_section = section

        if is_new:
            flush()
            if seg_type == "section":
                # Заголовок раздела строится синтетически, т.к. section-узлы
                # не попадают в records (structural nodes with children).
                section_num = section or st.get("number") or ""
                title = f"Раздел {section_num}" if section_num else ""
            else:
                title = _clean_header(rec["text"])
            current = {
                "type": seg_type,
                "number": st.get("number"),
                "title": title,
                "article": article,
                "appendix": appendix,
                "body": [],
            }
            if seg_type == "section":
                # Для section триггер-запись — content (paragraph/text),
                # и её нужно добавить в тело сегмента.
                current["body"].append(rec)
            continue

        if current is None:
            current = {
                "type": "preamble", "number": None,
                "title": "Преамбула", "article": None,
                "appendix": None, "body": [],
            }

        current["body"].append(rec)

    flush()

    # Финальный проход: уточняем заголовки для сегментов без фиксированного title
    for seg in segments:
        if seg["type"] not in ("article", "appendix", "section"):
            body = seg["body"]
            if body:
                first_para = None
                for br in body:
                    pn = _get_paragraph_ctx(br)
                    if pn:
                        first_para = pn
                        break
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
    if point_key and re.match(
        r'^(?:Статья|Раздел|Глава|Приложение|Пункт|Преамбула)',
        point_key,
    ):
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
    blocks: list[tuple[str, bool]] = []
    block_to_record: list[int] = []  # индекс body-рекорда для каждого блока
    for bi, r in enumerate(seg["body"]):
        txt = _norm_text(r["text"])
        if not txt:
            continue
        blocks.append((txt, _is_editorial(txt)))
        block_to_record.append(bi)

    if not blocks:
        return []

    # Пробуем сегментный префикс; для динамических заголовков используем
    # первый попавшийся paragraph в теле
    prefix_title = seg["title"]
    if seg_type not in ("article", "appendix", "section"):
        # Пробуем найти paragraph-контекст для префикса
        pn = None
        for br in seg["body"]:
            pn = _get_paragraph_ctx(br)
            if pn:
                prefix_title = f"Пункт {pn}"
                break
        if pn is None:
            prefix_title = "Преамбула"

    prefix = f"[{doc_display}] [{prefix_title}]"
    prefix_tokens = count_tokens(prefix)

    # Detect blocks with premiums «maximum size is not limited» -
    # they should not be merged with bonus blocks into one chunk.
    split_block_indices: set[int] = set()
    for bi_p, (txt_p, _) in enumerate(blocks):
        if "премии" in txt_p.lower() and "не ограничивается" in txt_p.lower():
            split_block_indices.add(bi_p)

    parts = _pack_blocks(blocks, prefix_tokens, max_tokens, target_tokens, min_tokens,
                         split_block_indices=split_block_indices)

    chunks: list[dict] = []
    for i, (part, part_block_indices) in enumerate(parts, 1):
        part = part.strip()
        if not part:
            continue

        # Определяем заголовок для этого чанка
        if seg_type not in ("article", "appendix", "section"):
            # Находим body-рекорд для первого блока в part
            first_block_in_part = part_block_indices[0] if part_block_indices else 0
            record_idx = block_to_record[first_block_in_part]
            pn = _get_paragraph_ctx(seg["body"][record_idx])
            if pn:
                point_key = pn
                chunk_title = f"Пункт {pn}"
            else:
                point_key = prefix_title
                chunk_title = prefix_title
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
    Для section с отсутствующим номером используется seg_idx,
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
        "over_400": sum(1 for t in token_counts if t > 400),
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
