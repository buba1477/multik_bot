"""Pure utility functions for retrieval — multi-part boost and article reconstruction.

Testable without heavy dependencies (no Qdrant, Ollama, models).
Uses only llama_index.core.schema for NodeWithScore/TextNode.
"""

import re
from typing import Optional, List, Dict, Any
from collections import defaultdict

from llama_index.core.schema import NodeWithScore, MetadataMode

from ..logger import logger as app_logger
logger = app_logger

# ============================================================
# CONSTANTS
# ============================================================

# Регулярное выражение для парсинга ID чанка: документ_статья_часть
ARTICLE_CHUNK_ID_RE = re.compile(r'^(.+)_p(\d+)$')

# Паттерны запросов на перечень/полноту
ENUMERATION_QUERY_PATTERNS = [
    "перечисли",
    "перечисли все",
    "какие виды",
    "какие бывают",
    "назови все",
    "укажи все",
    "полный перечень",
]

# Для reconstruction: минимальный последовательный блок (начало статьи)
MAX_SEQUENTIAL_PARTS = 3

# Score для добавленных чанков (чуть выше порога 0.05)
DEFAULT_SCORE = 0.051

# ============================================================
# HELPERS
# ============================================================

def extract_article_prefix(chunk_id: str) -> Optional[str]:
    """Извлекает префикс статьи из ID чанка.

    117-fz_st219_p1    -> '117-fz_st219_'
    117-fz_st219_1_p1  -> '117-fz_st219_1_' (другая статья: 219.1)
    117-fz_st171_p13   -> '117-fz_st171_'
    """
    m = ARTICLE_CHUNK_ID_RE.match(chunk_id)
    if m:
        return m.group(1) + "_"
    return None

def is_enumeration_query(query_text: str) -> bool:
    """Определяет, является ли запрос запросом на перечень/полноту."""
    q = query_text.lower().strip()
    return any(p in q for p in ENUMERATION_QUERY_PATTERNS)

def part_number(chunk_id: str) -> int:
    """Извлекает номер части из ID чанка (117-fz_st219_p3 -> 3)."""
    m = ARTICLE_CHUNK_ID_RE.match(chunk_id)
    return int(m.group(2)) if m else 0

def get_document_id(meta: dict) -> str:
    """Безопасное получение document_id, обрабатывая 'None' и пустые значения."""
    doc_id = meta.get("document_id", "")
    if not doc_id or str(doc_id) == "None":
        return ""
    return doc_id

# ============================================================
# MULTI-PART BOOST (core logic)
# ============================================================

def multi_part_boost(
    nodes: List[NodeWithScore],
    top_n: int = 10,
    bonus: float = 0.03,
) -> List[NodeWithScore]:
    """
    Мягкий буст для многочастных блоков.

    Двухуровневая группировка:
    1. Primary: по (document_id, point) с total_parts > 1
    2. Fallback: по префиксу статьи, когда document_id отсутствует (= 'None')

    Если хотя бы одна часть блока находится в топ-N (top_n),
    все части этого блока получают +bonus к скору.
    """
    if not nodes:
        return nodes

    top_ids = {nws.node.node_id for nws in nodes[:top_n]}

    # Primary groups: по (document_id, point)
    primary_groups: Dict[tuple, List[NodeWithScore]] = defaultdict(list)
    for nws in nodes:
        meta = nws.node.metadata
        doc_id = get_document_id(meta)
        point = meta.get("point", "")
        total_parts = meta.get("total_parts", 1)
        if doc_id and point and total_parts > 1:
            primary_groups[(doc_id, point)].append(nws)

    # Fallback groups: по префиксу статьи (legacy chunks без document_id)
    fallback_groups: Dict[str, List[NodeWithScore]] = defaultdict(list)
    for nws in nodes:
        meta = nws.node.metadata
        doc_id = get_document_id(meta)
        if not doc_id:
            prefix = extract_article_prefix(nws.node.node_id)
            if prefix:
                fallback_groups[prefix].append(nws)

    boosted = set()
    primary_activated = 0
    fallback_activated = 0

    # Apply primary groups
    for key, group in primary_groups.items():
        if any(nws.node.node_id in top_ids for nws in group):
            for nws in group:
                if nws.node.node_id not in boosted:
                    nws.score += bonus
                    boosted.add(nws.node.node_id)
            primary_activated += 1

    # Apply fallback groups (только группы с >= 2 частями)
    for prefix, group in fallback_groups.items():
        if len(group) < 2:
            continue
        if any(nws.node.node_id in top_ids for nws in group):
            for nws in group:
                if nws.node.node_id not in boosted:
                    nws.score += bonus
                    boosted.add(nws.node.node_id)
            fallback_activated += 1

    if boosted:
        logger.info(
            f"🔗 Multi-part boost applied to {len(boosted)} nodes "
            f"({primary_activated} primary groups, {fallback_activated} fallback groups)"
        )
        nodes = sorted(nodes, key=lambda x: x.score, reverse=True)

    return nodes

# ============================================================
# STRUCTURE-AWARE RECONSTRUCTION (core logic)
# ============================================================

def reconstruct_article_context(
    query_text: str,
    final_nodes: List[NodeWithScore],
    node_map: Dict[str, Any],
    max_sequential_parts: int = MAX_SEQUENTIAL_PARTS,
) -> List[NodeWithScore]:
    """
    Для запросов на перечень/полноту: восстанавливает минимальный
    последовательный блок чанков статьи (первые N частей).

    Отличается от старой реализации тем, что добавляет ТОЛЬКО первые
    max_sequential_parts (3) части статьи, а не все найденные чанки.

    ВНИМАНИЕ:
    - Не содержит собственного ограничения token budget — это ответственность
      вызывающего кода (engine_rag.py).
    - Добавляет только минимальный последовательный блок, необходимый для
      полноты ответа (первые 3 части статьи).

    Возвращает обновлённый список final_nodes (оригинальные + добавленные).
    """

    if not final_nodes:
        return final_nodes
    if not is_enumeration_query(query_text):
        return final_nodes

    # 1. Извлекаем префиксы статей из финальных чанков + макс. score
    article_prefixes: Dict[str, float] = {}
    for nws in final_nodes:
        prefix = extract_article_prefix(nws.node.node_id)
        if prefix:
            if prefix not in article_prefixes or nws.score > article_prefixes[prefix]:
                article_prefixes[prefix] = nws.score

    if not article_prefixes:
        logger.info("ℹ️ Structure-aware: no article chunks found, skipping")
        return final_nodes

    # 2. Отбираем статьи с минимальным score (чтобы не тащить нерелевантные)
    # Используем порог из основного pipeline (0.05), чтобы не отсекать
    # чанки, которые прошли reranker.
    MIN_ARTICLE_SCORE = 0.05
    active_prefixes = {
        pfx for pfx, sc in article_prefixes.items()
        if sc >= MIN_ARTICLE_SCORE
    }

    skipped = set(article_prefixes.keys()) - active_prefixes
    if skipped:
        logger.info(
            f"ℹ️ Structure-aware: skipping low-score articles: "
            f"{', '.join(sorted(skipped))}"
        )
    if not active_prefixes:
        return final_nodes

    # 3. Собираем недостающие чанки: ТОЛЬКО первые max_sequential_parts частей
    #    статьи (с начала), которых ещё нет в final_nodes.
    existing_ids = {nws.node.node_id for nws in final_nodes}
    added_chunks: List[NodeWithScore] = []

    for prefix in sorted(active_prefixes):
        # Все части статьи из node_map (включая уже присутствующие)
        all_ids = [
            nid for nid in node_map
            if nid.startswith(prefix)
            and nid[len(prefix)].isalpha()
        ]
        if not all_ids:
            continue

        # Сортируем по номеру части
        all_ids.sort(key=part_number)

        # Берём первые max_sequential_parts частей статьи (p1, p2, p3)
        first_parts = all_ids[:max_sequential_parts]

        # Добавляем только те, которых ещё нет в final_nodes
        selected = [nid for nid in first_parts if nid not in existing_ids]
        if not selected:
            continue

        for node_id in selected:
            node = node_map[node_id]
            added_chunks.append(NodeWithScore(node=node, score=DEFAULT_SCORE))
            existing_ids.add(node_id)

        logger.info(
            f"  🔧 Structure-aware: added {len(selected)} of first "
            f"{max_sequential_parts} sequential chunks for prefix '{prefix}' "
            f"(IDs: {', '.join(selected)})"
        )

    if not added_chunks:
        return final_nodes

    result = list(final_nodes) + added_chunks
    logger.info(
        f"🔧 Structure-aware reconstruction: added {len(added_chunks)} chunks "
        f"for {len(active_prefixes)} article(s): "
        f"{', '.join(sorted(active_prefixes))}"
    )
    return result
