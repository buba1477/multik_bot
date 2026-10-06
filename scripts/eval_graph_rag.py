#!/usr/bin/env python3
"""Офлайн-eval Graph-RAG индекса (LlamaIndex persist) с FRIDA.
Сравнение с прод-dense retrieval на одинаковых qrels.

Не импортирует app/rag/engine_rag.py, eval_retrieval.py, main_api.py.

Метрики (золото может содержать НЕСКОЛЬКО правильных чанков):
  * recall@k   — (найденные релевантные в топ-k) / (всего релевантных);
  * map@k      — Mean Average Precision;
  * mrr@k      — 1/ранг ПЕРВОГО релевантного;
  * ndcg@k     — бинарный DCG@k / IDCG@k.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_DIR = Path(__file__).resolve().parent.parent

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_NLTK_DATA = PROJECT_DIR / "nltk_data"
if _NLTK_DATA.exists():
    os.environ["NLTK_DATA"] = str(_NLTK_DATA) + os.pathsep + os.environ.get("NLTK_DATA", "")

EMB_QUERY_PREFIX = "search_query: "
EMB_DOC_PREFIX = "search_document: "

_QUERY_ALIASES = ("query", "question", "q", "вопрос", "запрос", "text", "prompt")
_RELEVANT_ALIASES = (
    "relevant", "relevant_ids", "ground_truth", "ground_truth_ids", "ground_truths",
    "positive_ids", "positives", "source_id", "source_ids", "answer_chunk_id",
    "answer_chunk_ids", "chunk_ids", "chunk_id", "doc_ids", "doc_id",
    "relevant_chunk_ids", "relevant_chunks", "ids", "id",
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================
# ЭМБЕДДЕР FRIDA (как в create_index_graph_sber.py)
# ============================================================================
class FRIDAEmbedding:
    """Кастомный эмбеддер для index loading и encode запросов."""
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer, models

        word = models.Transformer(model_path)
        self.dim = word.get_word_embedding_dimension()
        pooling = models.Pooling(self.dim, pooling_mode="cls")
        self.model = SentenceTransformer(modules=[word, pooling], device=device)

    def encode_query(self, q: str) -> list[float]:
        return self.model.encode(EMB_QUERY_PREFIX + q).tolist()

    def encode_doc(self, text: str) -> list[float]:
        return self.model.encode(EMB_DOC_PREFIX + text).tolist()


# ============================================================================
# ЗАГРУЗКА GOLDEN SET (копия из eval_retrieval.py)
# ============================================================================
def _pick(rec: dict, names: tuple[str, ...]) -> tuple[str | None, Any]:
    for n in names:
        if n in rec and rec[n] not in (None, "", [], {}):
            return n, rec[n]
    return None, None


def _norm_relevant(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [str(k) for k in value.keys()]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for v in value:
            if isinstance(v, dict):
                for key in ("id", "chunk_id", "doc_id", "source_id"):
                    if key in v:
                        out.append(str(v[key]))
                        break
            elif v is not None:
                out.append(str(v))
        return out
    return [str(value)]


def _iter_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() in (".csv", ".tsv"):
        import csv
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter=delim))
    if text[0] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
            if isinstance(data, dict):
                for key in ("items", "queries", "data", "qrels", "golden"):
                    if isinstance(data.get(key), list):
                        return [r for r in data[key] if isinstance(r, dict)]
                return [data]
        except json.JSONDecodeError:
            pass
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def load_qrels(path: Path, query_field: str | None, relevant_field: str | None) -> list[dict]:
    records = _iter_records(path)
    items: list[dict] = []
    skipped = 0
    for rec in records:
        if query_field:
            qname, q = query_field, rec.get(query_field)
        else:
            qname, q = _pick(rec, _QUERY_ALIASES)
        if relevant_field:
            rname, rval = relevant_field, rec.get(relevant_field)
        else:
            rname, rval = _pick(rec, _RELEVANT_ALIASES)
        q = (q or "").strip() if isinstance(q, str) else ""
        relevant = _norm_relevant(rval)
        if not q or not relevant:
            skipped += 1
            continue
        items.append({"query": q, "relevant": relevant, "meta": {"qfield": qname, "rfield": rname}})
    if skipped:
        log(f"   ⚠️  Пропущено записей без query/relevant: {skipped}")
    return items


def log_gold_stats(qrels: list[dict]) -> None:
    counts = [len(item["relevant"]) for item in qrels]
    if not counts:
        return
    hist: dict[int, int] = {}
    for c in counts:
        hist[c] = hist.get(c, 0) + 1
    hist_str = ", ".join(f"{n} шт.: {hist[n]}" for n in sorted(hist))
    log(f"📊 Релевантных на запрос: mean {statistics.fmean(counts):.2f} | min {min(counts)} | "
        f"max {max(counts)} | распределение ({hist_str})")


# ============================================================================
# ЗАГРУЗКА LlamaIndex-ИНДЕКСА
# ============================================================================
def load_graph_index(index_dir: Path, embed_model: FRIDAEmbedding):
    """Загружает LlamaIndex persist, извлекает nodes + векторы.
    Возвращает объект с .corpus, .corpus_by_id, .matrix, .ids, .encoder.
    """
    import numpy as np
    from llama_index.core import Settings, StorageContext, load_index_from_storage
    from llama_index.core.embeddings import BaseEmbedding
    from typing import List

    class _LiEmbed(BaseEmbedding):
        def __init__(self, frida: FRIDAEmbedding, **kwargs):
            super().__init__(**kwargs)
            self._frida = frida
        def _get_query_embedding(self, query: str) -> List[float]:
            return self._frida.encode_query(query)
        def _get_text_embedding(self, text: str) -> List[float]:
            return self._frida.encode_doc(text)
        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)
        async def _aget_text_embedding(self, text: str) -> List[float]:
            return self._get_text_embedding(text)

    li_embed = _LiEmbed(embed_model)
    Settings.embed_model = li_embed
    Settings.llm = None
    Settings.node_parser = None

    log(f"📦 Загрузка индекса из {index_dir}...")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)

    docstore = index.docstore.docs
    log(f"   Найдено {len(docstore)} nodes в docstore")

    corpus: list[dict] = []
    node_to_idx: dict[str, int] = {}

    for node_id, node in docstore.items():
        if not hasattr(node, 'text') or not node.text:
            continue
        metadata = node.metadata or {}
        chunk_id = metadata.get("id", node_id)
        title = metadata.get("title", "")
        corpus.append({
            "id": chunk_id,
            "text": node.text,
            "title": title,
            "_node_id": node_id,
        })
        node_to_idx[node_id] = len(corpus) - 1

    log(f"   Извлечено {len(corpus)} документов из графового индекса")

    # Извлекаем векторы
    import json as _json
    vectors: dict[str, list[float]] = {}

    # Пробуем через API
    vs = storage_context.vector_store
    if hasattr(vs, '_data') and hasattr(vs._data, 'embedding_dict'):
        emb_dict = vs._data.embedding_dict
        for nid, vec in emb_dict.items():
            vectors[str(nid)] = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
    elif hasattr(vs, 'data') and hasattr(vs.data, 'embedding_dict'):
        emb_dict = vs.data.embedding_dict
        for nid, vec in emb_dict.items():
            vectors[str(nid)] = vec.tolist() if hasattr(vec, 'tolist') else list(vec)

    if not vectors:
        # Fallback: читаем из файла
        vec_path = index_dir / "default__vector_store.json"
        if vec_path.exists():
            log(f"   ⚠️ Загружаем векторы напрямую из {vec_path}")
            raw = _json.loads(vec_path.read_text(encoding="utf-8"))
            emb_dict = raw.get("embedding_dict", raw)
            for nid, vec_list in emb_dict.items():
                vectors[str(nid)] = list(vec_list)

    log(f"   Загружено {len(vectors)} векторов")

    # Строим матрицу
    matrix_rows: list[np.ndarray] = []
    valid_ids: list[str] = []
    for node_id, idx in node_to_idx.items():
        if node_id in vectors:
            matrix_rows.append(np.asarray(vectors[node_id], dtype="float32"))
            valid_ids.append(corpus[idx]["id"])

    if not matrix_rows:
        log("❌ Нет векторов для построения матрицы поиска!")
        sys.exit(1)

    matrix = np.stack(matrix_rows)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    log(f"   Матрица поиска: {matrix.shape[0]} векторов x {matrix.shape[1]} dim")

    # Фильтруем corpus
    id_set = set(valid_ids)
    corpus = [d for d in corpus if d["id"] in id_set]

    class IndexData:
        pass

    data = IndexData()
    data.corpus = corpus
    data.corpus_by_id = {d["id"]: d for d in corpus}
    data.matrix = matrix
    data.ids = valid_ids
    data.encoder = embed_model

    return data


# ============================================================================
# МЕТРИКИ (копия из eval_retrieval.py)
# ============================================================================
def _segment_of(chunk_id: str) -> str:
    return re.sub(r"_p\d+$", "", chunk_id)


def _document_of(chunk_id: str) -> str:
    m = re.match(r"^([a-zA-Z0-9_]+?)(?:_st\d+|_app\d+|_pre|_sec|_p\d)", chunk_id)
    if m:
        return m.group(1).rstrip("_")
    return chunk_id.split("_p")[0] if "_p" in chunk_id else chunk_id


def _key_of(chunk_id: str, level: str) -> str:
    if level == "strict":
        return chunk_id
    if level == "document":
        return _document_of(chunk_id)
    return _segment_of(chunk_id)


def relevant_keys(relevant: list[str], level: str) -> set[str]:
    return {_key_of(r, level) for r in relevant}


def hit_ranks(ranked_ids: list[str], rel_keys: set[str], level: str, max_k: int) -> list[int]:
    ranks: list[int] = []
    seen: set[str] = set()
    for i, cid in enumerate(ranked_ids[:max_k], 1):
        key = _key_of(cid, level)
        if key in rel_keys and key not in seen:
            seen.add(key)
            ranks.append(i)
    return ranks


def average_precision(ranks: list[int], k: int, n_rel: int) -> float:
    if n_rel <= 0:
        return 0.0
    hits = [r for r in ranks if r <= k]
    if not hits:
        return 0.0
    ap = sum(idx / r for idx, r in enumerate(hits, 1))
    return ap / min(n_rel, k)


def ndcg_at_k(ranks: list[int], k: int, n_rel: int) -> float:
    if n_rel <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_rel, k) + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def query_metrics(ranked_ids: list[str], rel_keys: set[str], level: str, ks: list[int]) -> dict:
    max_k = max(ks)
    ranks = hit_ranks(ranked_ids, rel_keys, level, max_k)
    n_rel = len(rel_keys)
    out: dict[str, float] = {}
    for k in ks:
        found = sum(1 for r in ranks if r <= k)
        out[f"recall@{k}"] = (found / n_rel) if n_rel else 0.0
        out[f"map@{k}"] = average_precision(ranks, k, n_rel)
    out[f"mrr@{max_k}"] = (1.0 / ranks[0]) if ranks else 0.0
    out[f"ndcg@{max_k}"] = ndcg_at_k(ranks, max_k, n_rel)
    out["n_relevant"] = float(n_rel)
    out[f"n_hits@{max_k}"] = float(len(ranks))
    out["first_hit_rank"] = float(ranks[0]) if ranks else 0.0
    return out


# ============================================================================
# ФИЛЬТРАЦИЯ QRELS по 79-ФЗ
# ============================================================================
def is_79fz_id(chunk_id: str) -> bool:
    return chunk_id.startswith("79-fz_")


def filter_qrels_79fz(qrels: list[dict], mode: str = "wide") -> list[dict]:
    filtered: list[dict] = []
    skipped_wide = 0
    skipped_strict = 0

    for item in qrels:
        relevant = item["relevant"]
        n_79fz = sum(1 for r in relevant if is_79fz_id(r))

        if n_79fz == 0:
            skipped_wide += 1
            continue

        if mode == "strict" and n_79fz < len(relevant):
            skipped_strict += 1
            continue

        filtered.append(item)

    if mode == "wide":
        log(f"   📊 wide: {len(filtered)} запросов (пропущено {skipped_wide} без GT из 79-ФЗ)")
    else:
        log(f"   📊 strict: {len(filtered)} запросов "
            f"(пропущено {skipped_wide} без 79-ФЗ + {skipped_strict} смешанных)")

    return filtered


def build_79fz_qrels(input_path: Path, output_path: Path, mode: str) -> int:
    qrels = load_qrels(input_path, query_field=None, relevant_field=None)
    filtered = filter_qrels_79fz(qrels, mode=mode)
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in filtered),
        encoding="utf-8"
    )
    return len(filtered)


# ============================================================================
# DENSE RETRIEVAL ПО ГРАФОВОМУ ИНДЕКСУ
# ============================================================================
class GraphDenseRetriever:
    """In-memory dense retrieval по графовому индексу."""
    def __init__(self, index_data):
        self.corpus = index_data.corpus
        self.corpus_by_id = index_data.corpus_by_id
        self.matrix = index_data.matrix
        self.ids = index_data.ids
        self.encoder = index_data.encoder

    def top_ids(self, query: str, limit: int) -> list[str]:
        import numpy as np
        qv = np.asarray(self.encoder.encode_query(query), dtype="float32")
        n = float(np.linalg.norm(qv)) or 1.0
        sims = self.matrix @ (qv / n)
        idx = np.argsort(sims)[::-1][:limit]
        return [self.ids[i] for i in idx]


# ============================================================================
# CROSS-CHUNK GRAPH EXPAND RETRIEVER
# ============================================================================
class GraphExpandRetriever:
    """Dense retrieval + cross-chunk graph expansion.

    Steps:
      1. Dense retrieval top-K (K=20 by default)
      2. Take top-N from dense (N=5 by default)
      3. Find neighbor chunks via shared entities (from cross_chunk_graph)
      4. Union: dense top-K + neighbors (cap at 20)
      5. Rerank by cosine similarity (FRIDA query vs doc vectors)
      6. Return top-5

    Tracks expansion_stats per-query for diagnostics.
    """

    def __init__(self, index_data, cross_chunk_graph,
                 dense_k: int = 20, expand_from: int = 5,
                 graph_alpha: float = 0.0):
        self.index_data = index_data
        self.id_to_midx = {cid: i for i, cid in enumerate(index_data.ids)}
        self.chunk_entities = cross_chunk_graph.get("chunk_entities", {})
        self.entity_chunks = cross_chunk_graph.get("entity_chunks", {})
        self.chunk_graph = cross_chunk_graph.get("chunk_graph", {})
        self.dense_k = dense_k
        self.expand_from = expand_from
        self.graph_alpha = graph_alpha  # 0.0 = pure cosine, >0 = hybrid

    def top_ids(self, query: str, limit: int = 5) -> list[str]:
        import numpy as np

        # 1. Dense retrieval — all scores
        qv = np.asarray(self.index_data.encoder.encode_query(query),
                        dtype="float32")
        qn = float(np.linalg.norm(qv)) or 1.0
        all_sims = self.index_data.matrix @ (qv / qn)

        sorted_midxs = np.argsort(all_sims)[::-1]
        top_k_midxs = sorted_midxs[:self.dense_k]

        # 2. Dense top-K IDs
        dense_top_ids = [self.index_data.ids[midx] for midx in top_k_midxs]

        # 3. Expand from top-N chunks
        expand_ids = dense_top_ids[:self.expand_from]
        neighbor_set: set[str] = set()
        for cid in expand_ids:
            for neighbor_id in self.chunk_graph.get(cid, {}):
                if neighbor_id not in dense_top_ids:
                    neighbor_set.add(neighbor_id)

        # 4. Union: dense + neighbors (rerank will pick top-5)
        candidates = list(dense_top_ids)
        n_added = 0
        for nid in neighbor_set:
            if nid not in candidates:
                candidates.append(nid)
                n_added += 1

        # 5. Rerank by cosine similarity (+ optional graph boost)
        # Pre-compute max graph weight for normalization
        max_gw = 1
        if self.graph_alpha > 0 and n_added > 0:
            all_weights = [w for nbrs in self.chunk_graph.values() for w in nbrs.values()]
            if all_weights:
                max_gw = max(all_weights)

        scored: list[tuple[float, str]] = []
        for cid in candidates:
            midx = self.id_to_midx.get(cid)
            if midx is not None:
                cos_score = float(all_sims[midx])
                if self.graph_alpha > 0 and cid in self.chunk_graph:
                    # Find max edge weight to any of the expand_ids
                    gw = 0
                    for eid in expand_ids:
                        w = self.chunk_graph.get(eid, {}).get(cid, 0)
                        if w > gw:
                            gw = w
                    hybrid = cos_score + self.graph_alpha * (gw / max_gw)
                else:
                    hybrid = cos_score
                scored.append((hybrid, cid))

        scored.sort(key=lambda x: x[0], reverse=True)

        top5 = [cid for _, cid in scored[:limit]]
        n_neighbors_in_top5 = sum(
            1 for cid in top5 if cid in neighbor_set
        )
        n_dense_only = sum(
            1 for cid in top5 if cid in dense_top_ids
        )

        # Store stats on the instance for collection after eval
        self._last_stats = {
            "n_dense_top": len(dense_top_ids),
            "n_neighbors_added": n_added,
            "n_neighbors_in_top5": n_neighbors_in_top5,
            "n_dense_in_top5": n_dense_only,
        }

        return top5


# ============================================================================
# ПРОГОН
# ============================================================================
def run_eval(qrels: list[dict], retriever: object,
             ks: list[int], level: str) -> dict:
    max_k = max(ks)
    metric_names = (
        [f"recall@{k}" for k in ks]
        + [f"map@{k}" for k in ks]
        + [f"mrr@{max_k}", f"ndcg@{max_k}"]
        + [f"n_hits@{max_k}"]
    )

    acc: dict[str, list[float]] = {m: [] for m in metric_names}
    latencies: list[float] = []
    expand_n_added: list[int] = []
    expand_n_in_top5: list[int] = []

    for item in qrels:
        q = item["query"]
        rel_keys = relevant_keys(item["relevant"], level)

        t0 = time.perf_counter()
        ranked = retriever.top_ids(q, max_k)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        m = query_metrics(ranked, rel_keys, level, ks)
        for name in metric_names:
            acc[name].append(m[name])

        # Collect expansion per-query stats
        if hasattr(retriever, "_last_stats"):
            stats = retriever._last_stats
            expand_n_added.append(stats.get("n_neighbors_added", 0))
            expand_n_in_top5.append(stats.get("n_neighbors_in_top5", 0))

    result: dict = {
        "n_queries": len(qrels),
        "metrics": {m: (statistics.fmean(v) if v else 0.0) for m, v in acc.items()},
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
            "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0,
        },
    }

    if expand_n_added:
        result["expand_stats"] = {
            "n_added": expand_n_added,
            "n_in_top5": expand_n_in_top5,
            "mean_added": statistics.fmean(expand_n_added),
            "mean_in_top5": statistics.fmean(expand_n_in_top5),
        }

    return result


def print_results(results: list[dict], ks: list[int]) -> None:
    max_k = max(ks)
    header = (["config", "subset"] + [f"R@{k}" for k in ks]
              + [f"MAP@{max_k}", f"MRR@{max_k}", f"nDCG@{max_k}", "lat_mean_ms"])
    log("  " + " | ".join(f"{h:>11}" for h in header))
    log("  " + "-" * (13 * len(header)))
    for r in results:
        if "metrics" not in r:
            continue
        m = r["metrics"]
        cells = [f"{r['config']:>11}", f"{r.get('subset', 'all'):>8}"]
        cells += [f"{m[f'recall@{k}']:.3f}" for k in ks]
        cells += [f"{m[f'map@{max_k}']:.3f}", f"{m[f'mrr@{max_k}']:.3f}", f"{m[f'ndcg@{max_k}']:.3f}"]
        cells += [f"{r['latency_ms']['mean']:.1f}"]
        log("  " + " | ".join(cells))


def save_report(out_path: Path, results: list[dict], meta: dict, ks: list[int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_k = max(ks)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    lines = [
        "# Eval Graph-RAG — отчёт", "",
        f"- Дата: {meta['timestamp']}",
        f"- Golden set: `{meta['qrels']}` ({meta['n_queries']} запросов)",
        f"- Индекс: `{meta['graph_index']}` ({meta['corpus_size']} чанков)",
        f"- Match-level: `{meta['match_level']}`",
        f"- Устройство: `{meta.get('device', 'cpu')}`", "",
        "| config | subset | " + " | ".join(f"R@{k}" for k in ks) +
        f" | MAP@{max_k} | MRR@{max_k} | nDCG@{max_k} | lat_mean_ms |",
        "|" + "---|" * (len(ks) + 6),
    ]
    for r in results:
        if "metrics" not in r:
            continue
        m = r["metrics"]
        row = [r["config"], r.get("subset", "all")]
        row += [f"{m[f'recall@{k}']:.3f}" for k in ks]
        row += [f"{m[f'map@{max_k}']:.3f}", f"{m[f'mrr@{max_k}']:.3f}", f"{m[f'ndcg@{max_k}']:.3f}"]
        row += [f"{r['latency_ms']['mean']:.1f}"]
        lines.append("| " + " | ".join(row) + " |")

        # Add expansion diagnostics if present
        if "expand_stats" in r:
            es = r["expand_stats"]
            lines.append("")
            lines.append(f"**Expansion stats**:")
            lines.append(f"- Среднее добавлено соседей: {es['mean_added']:.1f}")
            lines.append(f"- Среднее соседей в top-5: {es['mean_in_top5']:.1f}")
            lines.append(f"- Всего добавлено (сумма): {sum(es['n_added'])}")
            lines.append(f"- Всего соседей в top-5 (сумма): {sum(es['n_in_top5'])}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"💾 Отчёт: {out_path}")
    log(f"💾 JSON:  {json_path}")


# ============================================================================
# MAIN
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Офлайн-eval Graph-RAG индекса (LlamaIndex persist)")
    p.add_argument("--qrels", required=True)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--query-field", default=None)
    p.add_argument("--relevant-field", default=None)
    p.add_argument("--match-level", choices=("segment", "document", "strict"),
                   default="segment")
    p.add_argument("--top-k", default="1,3,5,10")
    p.add_argument("--out", default="/tmp/eval_graph_rag_report.md")
    p.add_argument("--device", default="cpu")
    p.add_argument("--filter-79fz", choices=("no", "wide", "strict", "both"),
                   default="no")
    p.add_argument("--gen-filtered-qrels", default=None,
                   help="wide:path или strict:path — создать отфильтрованный qrels и выйти")
    p.add_argument("--graph-expand", action="store_true",
                   help="Включить cross-chunk graph expansion поверх dense-retrieval")
    p.add_argument("--dense-k", type=int, default=20,
                   help="Сколько чанков брать dense-retrieval'ом перед expansion (default: 20)")
    p.add_argument("--expand-from", type=int, default=5,
                   help="Из скольких top dense-чанков искать соседей (default: 5)")
    p.add_argument("--cross-chunk-graph", default=None,
                   help="Путь к cross_chunk_graph.json (default: index_dir/../cross_chunk_graph.json)")
    p.add_argument("--graph-alpha", type=float, default=0.0,
                   help="Вес графа в гибридном score (0.0 = pure cosine, 0.2 = лёгкий graph bump)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ks = [int(k.strip()) for k in args.top_k.split(",") if k.strip()]
    if not ks:
        ks = [1, 3, 5, 10]

    qrels_path = Path(args.qrels)
    index_dir = Path(args.index_dir)

    if not index_dir.exists():
        log(f"❌ Индекс не найден: {index_dir}")
        sys.exit(1)

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = PROJECT_DIR / "hf_cache" / "FRIDA"
    if not model_path.exists():
        log(f"❌ FRIDA модель не найдена: {model_path}")
        sys.exit(1)

    # Генерация отфильтрованных qrels (если запрошено) и выход
    if args.gen_filtered_qrels:
        mode, out_path_str = args.gen_filtered_qrels.split(":", 1)
        out_path = Path(out_path_str)
        n = build_79fz_qrels(qrels_path, out_path, mode)
        log(f"✅ Создан {mode}-qrels: {out_path} ({n} запросов)")
        return

    qrels = load_qrels(qrels_path, args.query_field, args.relevant_field)
    if not qrels:
        log("❌ Не удалось прочитать ни одного запроса из golden set.")
        sys.exit(1)
    log(f"📄 Golden set: {len(qrels)} запросов  ({qrels_path})")
    log_gold_stats(qrels)

    subsets: list[tuple[str, list[dict]]] = [("all", qrels)]
    if args.filter_79fz == "wide":
        subsets = [("wide (>=1 GT 79-FZ)", filter_qrels_79fz(qrels, mode="wide"))]
    elif args.filter_79fz == "strict":
        subsets = [("strict (all GT 79-FZ)", filter_qrels_79fz(qrels, mode="strict"))]
    elif args.filter_79fz == "both":
        wide = filter_qrels_79fz(qrels, mode="wide")
        strict = filter_qrels_79fz(qrels, mode="strict")
        subsets = [("wide (>=1 GT 79-FZ)", wide), ("strict (all GT 79-FZ)", strict)]

    log(f"\n🧠 Загрузка FRIDA из {model_path}...")
    embedder = FRIDAEmbedding(str(model_path), device=args.device)
    log(f"✅ FRIDA загружена (dim={embedder.dim})")

    log(f"\n📦 Загрузка графового индекса из {index_dir}...")
    index_data = load_graph_index(index_dir, embedder)
    corpus_size = len(index_data.corpus)
    log(f"✅ Индекс загружен: {corpus_size} чанков")

    # ================================================================
    # Cross-chunk graph expansion
    # ================================================================
    if args.graph_expand:
        if args.cross_chunk_graph:
            ccg_path = Path(args.cross_chunk_graph)
        else:
            ccg_path = index_dir.parent / "cross_chunk_graph.json"
        if not ccg_path.exists():
            log(f"❌ Cross-chunk graph не найден: {ccg_path}")
            log("   Сначала запусти build_cross_chunk_graph.py")
            sys.exit(1)
        log(f"\n📦 Загрузка cross-chunk графа из {ccg_path}...")
        cross_chunk_graph = json.loads(
            ccg_path.read_text(encoding="utf-8")
        )
        n_entities = len(cross_chunk_graph.get("entity_chunks", {}))
        n_chunks = len(cross_chunk_graph.get("chunk_graph", {}))
        log(f"✅ Cross-chunk граф: {n_entities} сущностей, "
            f"{n_chunks} чанков с рёбрами")
        retriever = GraphExpandRetriever(
            index_data, cross_chunk_graph,
            dense_k=args.dense_k, expand_from=args.expand_from,
            graph_alpha=args.graph_alpha,
        )
        config_label = f"graph_expand(k={args.dense_k},from={args.expand_from},α={args.graph_alpha})"
    else:
        retriever = GraphDenseRetriever(index_data)
        config_label = "graph_dense"

    results: list[dict] = []
    for label, subset_qrels in subsets:
        if not subset_qrels:
            log(f"   ⚠️ {label}: нет запросов, пропускаем")
            continue
        log(f"\n▶️  {label} ({len(subset_qrels)} запросов)")
        r = run_eval(subset_qrels, retriever, ks, args.match_level)
        r["config"] = config_label
        r["subset"] = label
        results.append(r)

        m = r["metrics"]
        max_k = max(ks)
        parts = [f"   recall@{k}={m[f'recall@{k}']:.3f}" for k in ks]
        parts += [f"MAP@{max_k}={m[f'map@{max_k}']:.3f}"]
        parts += [f"MRR@{max_k}={m[f'mrr@{max_k}']:.3f}"]
        parts += [f"nDCG@{max_k}={m[f'ndcg@{max_k}']:.3f}"]
        log(" | ".join(parts))
        log(f"   latency_mean={r['latency_ms']['mean']:.1f}ms")

        # expansion diagnostics
        if args.graph_expand and hasattr(retriever, "_last_stats"):
            # Collect per-query stats
            expand_stats = r.get("expand_stats", {})
            if expand_stats:
                avg_added = statistics.fmean(expand_stats["n_added"])
                avg_n_top5 = statistics.fmean(expand_stats["n_in_top5"])
                log(f"   📊 expansion: в среднем {avg_added:.1f} соседей добавлено, "
                    f"{avg_n_top5:.1f} из них в top-5")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "qrels": str(qrels_path),
        "n_queries": len(qrels),
        "graph_index": str(index_dir),
        "corpus_size": corpus_size,
        "match_level": args.match_level,
        "device": args.device,
        "filter_79fz": args.filter_79fz,
    }

    if args.graph_expand:
        meta["graph_expand"] = True
        meta["dense_k"] = args.dense_k
        meta["expand_from"] = args.expand_from
        meta["graph_alpha"] = args.graph_alpha

    config_title = "Graph-Expand" if args.graph_expand else "Graph-RAG Dense"
    log("\n" + "=" * 78)
    log(f"РЕЗУЛЬТАТЫ ({config_title})")
    log("=" * 78)
    print_results(results, ks)

    out_path = Path(args.out)
    save_report(out_path, results, meta, ks)


if __name__ == "__main__":
    main()
