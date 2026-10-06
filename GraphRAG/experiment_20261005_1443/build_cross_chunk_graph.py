#!/usr/bin/env python3
"""Build cross-chunk graph from graph_nodes.jsonl.

For each entity (normalized), collect all chunks where it appears.
Filter:
  - entities in <2 chunks → too rare (no cross-chunk link)
  - entities in >=20 chunks → too generic (noise)
  - Keep entities in 2-19 chunks (sweet spot)

Build chunk↔chunk graph: edge weight = shared entity count.
Limit neighbors per chunk to top-20 by weight.

Output: cross_chunk_graph.json with chunk_entities, entity_chunks, chunk_graph.
"""

import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MIN_CHUNKS = 2
MAX_CHUNKS = 19
MAX_NEIGHBORS = 20


def load_nodes(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def build_cross_chunk_graph(nodes: list[dict]) -> dict:
    # Step 1: build raw entity→chunks map from graph_data
    raw_entity_chunks: dict[str, set[str]] = {}
    chunk_entities_all: dict[str, set[str]] = {}
    no_graph = 0

    for node in nodes:
        chunk_id = node.get("id", "")
        graph_data = node.get("graph_data")
        if not graph_data or "entities" not in graph_data:
            no_graph += 1
            continue

        entities = graph_data["entities"]
        names: list[str] = []
        for ent in entities:
            name = ent.get("normalized", ent.get("name", "")).strip().lower()
            if name:
                names.append(name)

        if not names:
            no_graph += 1
            continue

        chunk_entities_all[chunk_id] = set(names)
        for e_name in names:
            raw_entity_chunks.setdefault(e_name, set()).add(chunk_id)

    log.info(f"Всего nodes: {len(nodes)}")
    log.info(f"С граф-данными: {len(chunk_entities_all)}")
    log.info(f"Без граф-данных: {no_graph}")
    log.info(f"Всего уникальных сущностей (raw): {len(raw_entity_chunks)}")

    # Step 2: filter entities by chunk frequency
    rare = 0      # <2 chunks
    too_common = 0  # >=20 chunks
    kept = 0
    entity_chunks: dict[str, set[str]] = {}
    for e_name, cids in raw_entity_chunks.items():
        cnt = len(cids)
        if cnt < MIN_CHUNKS:
            rare += 1
        elif cnt >= MAX_CHUNKS:
            too_common += 1
        else:
            entity_chunks[e_name] = cids
            kept += 1

    log.info(f"\nФильтрация сущностей по частоте в чанках:")
    log.info(f"  Сущностей в <{MIN_CHUNKS} чанках (редкие, отсечено): {rare}")
    log.info(f"  Сущностей в >={MAX_CHUNKS} чанках (общие, отсечено): {too_common}")
    log.info(f"  Сущностей в {MIN_CHUNKS}-{MAX_CHUNKS-1} чанках (рабочие): {kept}")

    # Step 3: prune chunk_entities to only kept entities
    chunk_entities: dict[str, set[str]] = {}
    for cid, enames in chunk_entities_all.items():
        pruned = {e for e in enames if e in entity_chunks}
        if pruned:
            chunk_entities[cid] = pruned

    log.info(f"  Чанков с хотя бы 1 рабочей сущностью: {len(chunk_entities)}")

    # Step 4: build chunk↔chunk graph with weights
    chunk_graph_raw: dict[str, dict[str, int]] = {}
    for e_name, cids in entity_chunks.items():
        cid_list = list(cids)
        for i, cid_a in enumerate(cid_list):
            for cid_b in cid_list[i + 1 :]:
                if cid_a not in chunk_graph_raw:
                    chunk_graph_raw[cid_a] = {}
                chunk_graph_raw[cid_a][cid_b] = (
                    chunk_graph_raw[cid_a].get(cid_b, 0) + 1
                )
                if cid_b not in chunk_graph_raw:
                    chunk_graph_raw[cid_b] = {}
                chunk_graph_raw[cid_b][cid_a] = (
                    chunk_graph_raw[cid_b].get(cid_a, 0) + 1
                )

    total_edges_raw = sum(len(v) for v in chunk_graph_raw.values())
    log.info(f"\nГраф (до ограничения соседей):")
    log.info(f"  Чанков с рёбрами: {len(chunk_graph_raw)}")
    log.info(f"  Всего рёбер: {total_edges_raw // 2}")

    # Step 5: limit neighbors per chunk to top MAX_NEIGHBORS by weight
    chunk_graph: dict[str, dict[str, int]] = {}
    for cid, neighbors in chunk_graph_raw.items():
        sorted_n = sorted(neighbors.items(), key=lambda x: -x[1])
        chunk_graph[cid] = dict(sorted_n[:MAX_NEIGHBORS])

    total_edges = sum(len(v) for v in chunk_graph.values())
    log.info(f"\nГраф (после ограничения top-{MAX_NEIGHBORS} соседей):")
    log.info(f"  Чанков с рёбрами: {len(chunk_graph)}")
    log.info(f"  Всего рёбер: {total_edges // 2}")

    # Step 6: statistics
    neighbor_counts = [len(v) for v in chunk_graph.values()]
    edge_weights = []
    for cid, neighbors in chunk_graph.items():
        edge_weights.extend(neighbors.values())

    if neighbor_counts:
        log.info(f"\nСтатистика по соседям на чанк:")
        log.info(f"  Медиана: {statistics.median(neighbor_counts):.0f}")
        log.info(
            f"  Среднее: {statistics.fmean(neighbor_counts):.1f}"
        )
        log.info(f"  Минимум: {min(neighbor_counts)}")
        log.info(f"  Максимум: {max(neighbor_counts)}")
        # isolated chunks (in chunk_entities but 0 neighbors after limiting)
        all_chunks_with_entities = set(chunk_entities.keys())
        chunks_with_edges = set(chunk_graph.keys())
        isolated = all_chunks_with_entities - chunks_with_edges
        log.info(
            f"  Изолированные (0 соседей после лимита): {len(isolated)} "
            f"({len(isolated) / len(all_chunks_with_entities) * 100:.0f}%)"
        )
    else:
        log.info("  ⚠️ Граф пуст — нет ни одного ребра!")

    if edge_weights:
        log.info(f"\nСтатистика по весу ребра:")
        log.info(f"  Средний вес: {statistics.fmean(edge_weights):.2f}")
        log.info(f"  Медиана: {statistics.median(edge_weights):.0f}")
        log.info(f"  Максимум: {max(edge_weights)}")

    # Convert sets to sorted lists for JSON
    return {
        "chunk_entities": {k: sorted(v) for k, v in chunk_entities.items()},
        "entity_chunks": {k: sorted(v) for k, v in entity_chunks.items()},
        "chunk_graph": chunk_graph,
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    nodes_path = base / "graph_nodes.jsonl"
    out_path = base / "cross_chunk_graph.json"

    if not nodes_path.exists():
        log.error(f"❌ Не найден: {nodes_path}")
        sys.exit(1)

    log.info("📂 Загрузка graph_nodes.jsonl...")
    nodes = load_nodes(nodes_path)
    log.info(f"  Загружено {len(nodes)} nodes")

    log.info("🏗  Построение cross-chunk графа...")
    graph = build_cross_chunk_graph(nodes)

    out_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        f"✅ Сохранён: {out_path} "
        f"({out_path.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    main()
