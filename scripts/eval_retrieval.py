#!/usr/bin/env python3
"""Офлайн-eval retrieval/rerank для RAG ФНС: recall@k, MAP@k, MRR, nDCG@k + ablation.

Метрики (gold может содержать НЕСКОЛЬКО правильных чанков на вопрос):
  * recall@k   — (число найденных релевантных в топ-k) / (всего релевантных), НЕ бинарно;
  * map@k      — Mean Average Precision (для каждого k; в таблице — MAP@max_k);
  * mrr@k      — 1/ранг ПЕРВОГО релевантного (по определению учитывает одно попадание);
  * ndcg@k     — бинарный DCG@k / IDCG@k, где IDCG@k = sum 1/log2(i+1) по min(n_rel, k).
  Попадание считается по ключу match-level: несколько чанков одного segment/document
  не дублируют попадание (recall не превышает 1.0).
  При одном релевантном на вопрос все метрики совпадают с прежними (бинарными).

Не импортирует app/rag/engine_rag.py (чтобы не тянуть Ollama/LlamaIndex),
но ТОЧНО повторяет его формулы и параметры (сверено с app/rag/engine_rag.py):

  * эмбеддер FRIDA: models.Transformer + models.Pooling(cls), БЕЗ Normalize;
  * префиксы: 'search_query: ' (запрос) / 'search_document: ' (документ);
  * RRF: score += weight / (k + rank + 1); веса VECTOR_WEIGHT/BM25_WEIGHT, k=30;
    веса можно переопределить флагами --vector-weight/--bm25-weight или спецификацией
    конфига hybrid@0.9/0.1 (тюнинг; дефолт синхронизируем с app/rag/engine_rag.py);
  * vector top-K = initial_top_k = 30; BM25_TOP_K = 30;
  * BM25-корпус = node.get_content(MetadataMode.LLM) -> 'РАЗДЕЛ: title: <t>' + 'ТЕКСТ:' + body;
  * токенизация = RerankedEngine._tokenize (SnowballStemmer russian, 'не' сохраняется);
  * реранк: cross-encoder по top-10 -> top-5 (SCORE_THRESHOLD в проде отключён).

Конфигурации ablation:
  * dense            — dense top-30;
  * dense_rerank     — dense top-30 -> cross-encoder по top-10 -> top-5 (без RRF и BM25);
  * bm25             — BM25 top-30;
  * hybrid           — RRF(dense top-30, BM25 top-30) с весами VECTOR_WEIGHT/BM25_WEIGHT;
  * hybrid@V/B       — то же, но с явными весами (пример: hybrid@0.9/0.1, hybrid_rerank@0.65/0.35);
  * hybrid_rerank    — hybrid -> cross-encoder по top-10 -> top-5;
  * adaptive         — веса RRF по типу запроса (точные реквизиты / user-style);
  * adaptive_rerank  — adaptive -> cross-encoder (прод-подобный контур).

Golden set (--qrels) — авто-детект формата:
  * JSONL/JSON: {"query": "...", "relevant_ids": ["<chunk_id>", ...]}
  * aliases запроса:  query|question|q|вопрос|запрос
  * aliases релевантного: relevant|relevant_ids|ground_truth|ground_truth_ids|
    positive_ids|positives|source_id|answer_chunk_id|chunk_ids|doc_ids|ids
  * CSV/TSV с --query-field/--relevant-field (по умолчанию query / relevant_ids).

Запуск:
  # базовый ablation
  venv/bin/python scripts/eval_retrieval.py --qrels eval/qrels.jsonl --configs dense,bm25,hybrid,hybrid_rerank --top-k 1,3,5,10 --match-level segment --out eval/

  # тюнинг весов RRF: одна таблица, один прогон (реранкер грузится один раз)
  venv/bin/python scripts/eval_retrieval.py --qrels eval/qrels.jsonl \
      --configs dense,dense_rerank,hybrid@1.0/0.0,hybrid@0.9/0.1,hybrid@0.65/0.35,hybrid@0.5/0.5 \
      --top-k 1,3,5,10 --out eval/weights_gold137/
"""
from __future__ import annotations

import argparse
import csv
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

# СТРОГИЙ ОФФЛАЙН (как в engine_rag.py / чанкере)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_NLTK_DATA = PROJECT_DIR / "nltk_data"
if _NLTK_DATA.exists():
    os.environ["NLTK_DATA"] = str(_NLTK_DATA) + os.pathsep + os.environ.get("NLTK_DATA", "")

# ============================================================================
# ПАРАМЕТРЫ — КОПИЯ ИЗ app/rag/engine_rag.py (RerankedEngine)
# Менять здесь ТОЛЬКО вместе с движком, иначе метрики разойдутся с продом.
# ============================================================================
VECTOR_WEIGHT = 0.9   # синхронно с app/rag/engine_rag.py::RerankedEngine
BM25_WEIGHT = 0.1
RRF_K = 30
INITIAL_TOP_K = 30
BM25_TOP_K = 30
RERANK_POOL = 10
RERANK_BATCH = 32  # батч cross-encoder'а (как в sentence-transformers по умолчанию)
FINAL_TOP_K = 5
EMB_QUERY_PREFIX = "search_query: "
EMB_DOC_PREFIX = "search_document: "

QUERY_REPLACEMENTS = {
    "госслужащему": "государственному гражданскому служащему",
    "госслужащий": "гражданский служащий",
    "госслужба": "гражданская служба",
    "госслужащих": "государственных гражданских служащих",
    "на госслужбе": "на гражданской службе",
    "иноагент": "иностранный агент",
    "инагент": "иностранный агент",
    "работать": "проходить гражданскую службу",
    "увольнение": "прекращение служебного контракта",
    "начальник": "представитель нанимателя",
    "зарплата": "денежное содержание",
    "взятка": "коррупционное правонарушение",
    "отпуск": "ежегодный оплачиваемый отпуск",
    "коррупция": "коррупционное правонарушение",
    "коррупционный": "коррупционное правонарушение",
    "цкп": "цифровая кадровая платформа",
    "ё": "е",
}

# EXACT-веса (синхронно с app/rag/classifier.py)
EXACT_VECTOR_WEIGHT = 0.5
EXACT_BM25_WEIGHT = 0.5


def _are_exact_weights(w: tuple[float, float]) -> bool:
    """True если веса соответствуют EXACT-режиму."""
    return abs(w[0] - EXACT_VECTOR_WEIGHT) < 0.01 and abs(w[1] - EXACT_BM25_WEIGHT) < 0.01


# --- adaptive: веса по типу запроса (делегировано LLM-классификатору app.rag.classifier) ---
def adaptive_weights(query: str) -> tuple[float, float]:
    """Веса RRF по типу запроса: EXACT -> (0.5, 0.5), USER -> (0.9, 0.1)."""
    sys.path.insert(0, str(PROJECT_DIR))
    try:
        from app.rag.classifier import classify_query
    except ImportError:
        log("⚠️ app.rag.classifier не найден — fallback USER для всех запросов")
        return (VECTOR_WEIGHT, BM25_WEIGHT)
    return classify_query(normalize_query(query))


def parse_config_spec(spec: str) -> tuple[str, tuple[float, float] | None]:
    """'hybrid@0.9/0.1' -> ('hybrid', (0.9, 0.1)); 'dense' -> ('dense', None)."""
    name, sep, weights = spec.strip().partition("@")
    if not sep:
        return name, None
    v, _, b = weights.partition("/")
    if not v or not b:
        raise ValueError(f"Неверная спецификация весов: {spec!r} (ожидается config@vector/bm25)")
    return name, (float(v), float(b))


ALL_CONFIGS = ("dense", "dense_rerank", "bm25", "hybrid", "hybrid_rerank",
               "adaptive", "adaptive_rerank")
RERANK_CONFIGS = frozenset({"dense_rerank", "hybrid_rerank", "adaptive_rerank"})


def log(msg: str) -> None:
    print(msg, flush=True)


def _segment_of(chunk_id: str) -> str:
    s = re.sub(r"_p\d+$", "", chunk_id or "")
    s = re.sub(r"_s\d+$", "", s)
    return s


def _document_of(chunk_id: str) -> str:
    return (chunk_id or "").split("_")[0]


def normalize_query(text: str) -> str:
    out = text.lower()
    for k in sorted(QUERY_REPLACEMENTS, key=len, reverse=True):
        out = out.replace(k, QUERY_REPLACEMENTS[k])
    return out


_STEMMER = None


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import SnowballStemmer
        _STEMMER = SnowballStemmer("russian")
    return _STEMMER


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    clean = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text.lower())
    st = _stemmer()
    out: list[str] = []
    for w in clean.split():
        if w == "не":
            out.append(w)
        elif len(w) >= 2:
            out.append(st.stem(w))
    return out


def bm25_doc_text(doc: dict) -> str:
    title = doc.get("title", "")
    body = doc.get("text", "")
    meta_str = f"title: {title}" if title else ""
    return f"РАЗДЕЛ: {meta_str}\nТЕКСТ:\n{body}"

# ============================================================================
# ЗАГРУЗКА GOLDEN SET (мультиформат)
# ============================================================================
_QUERY_ALIASES = ("query", "question", "q", "вопрос", "запрос", "text", "prompt")
_RELEVANT_ALIASES = (
    "relevant", "relevant_ids", "ground_truth", "ground_truth_ids", "ground_truths",
    "positive_ids", "positives", "source_id", "source_ids", "answer_chunk_id",
    "answer_chunk_ids", "chunk_ids", "chunk_id", "doc_ids", "doc_id",
    "relevant_chunk_ids", "relevant_chunks", "ids", "id",
)


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


# ============================================================================
# КОРПУС
# ============================================================================
def load_corpus_qdrant(host: str, port: int, collection: str) -> list[dict]:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=host, port=port)
    docs: list[dict] = []
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=1000, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            pay = p.payload or {}
            cid = str(pay.get("id") or p.id)
            if cid in seen:
                continue
            seen.add(cid)
            docs.append({
                "id": cid,
                "title": pay.get("title", ""),
                "text": pay.get("text", ""),
                "url": pay.get("source_url", ""),
            })
        if offset is None:
            break
    client.close()
    return docs


def load_corpus_local(chunks_dir: Path) -> list[dict]:
    docs: list[dict] = []
    seen: set[str] = set()
    for f in sorted(chunks_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = str(rec.get("id", ""))
            if not cid or cid in seen:
                continue
            seen.add(cid)
            docs.append({
                "id": cid,
                "title": rec.get("title", ""),
                "text": rec.get("text", ""),
                "url": rec.get("url", ""),
            })
    return docs


# ============================================================================
# ЭМБЕДДЕР FRIDA (как в engine_rag / индексаторе)
# ============================================================================
class FridaEncoder:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer, models

        word = models.Transformer(str(model_path))
        pooling = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        self.dim = word.get_word_embedding_dimension()
        self.model = SentenceTransformer(modules=[word, pooling], device=device)

    def encode_query(self, q: str) -> list[float]:
        return self.model.encode(EMB_QUERY_PREFIX + q).tolist()

    def encode_docs(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        pref = [EMB_DOC_PREFIX + t for t in texts]
        return self.model.encode(pref, batch_size=batch_size, show_progress_bar=True).tolist()

# ============================================================================
# RETRIEVERS
# ============================================================================
class DenseRetriever:
    def __init__(self, mode: str, corpus: list[dict], encoder: FridaEncoder,
                 qdrant_args: dict | None = None) -> None:
        self.mode = mode
        self.encoder = encoder
        self.client = None
        self.collection = None
        self.corpus = corpus
        self.ids = [d["id"] for d in corpus]
        self.matrix = None
        if mode == "qdrant":
            from qdrant_client import QdrantClient

            qdrant_args = qdrant_args or {}
            self.client = QdrantClient(host=qdrant_args["host"], port=qdrant_args["port"])
            self.collection = qdrant_args["collection"]
        else:
            import numpy as np

            vecs = np.asarray(self.encoder.encode_docs([d["text"] for d in corpus]), dtype="float32")
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = vecs / norms

    def top_ids(self, query: str, limit: int) -> list[str]:
        vec = self.encoder.encode_query(query)
        if self.mode == "qdrant":
            return self._qdrant_search(vec, limit)
        import numpy as np

        qv = np.asarray(vec, dtype="float32")
        n = float(np.linalg.norm(qv)) or 1.0
        sims = self.matrix @ (qv / n)
        idx = np.argsort(sims)[::-1][:limit]
        return [self.ids[i] for i in idx]

    def _qdrant_search(self, vec: list[float], limit: int) -> list[str]:
        try:
            resp = self.client.query_points(
                collection_name=self.collection, query=vec, limit=limit, with_payload=True
            )
            pts = resp.points
        except Exception:
            pts = self.client.search(
                collection_name=self.collection, query_vector=vec, limit=limit, with_payload=True
            )
        out: list[str] = []
        for p in pts:
            pay = p.payload or {}
            out.append(str(pay.get("id") or p.id))
        return out


def build_bm25(corpus: list[dict]):
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(bm25_doc_text(d)) for d in corpus]
    return BM25Okapi(tokenized)


def bm25_top_ids(bm25, query: str, corpus: list[dict], top_k: int = BM25_TOP_K) -> list[str]:
    import numpy as np

    scores = bm25.get_scores(tokenize(normalize_query(query)))
    order = np.argsort(scores)[::-1][:top_k]
    return [corpus[i]["id"] for i in order if scores[i] > 0]


def rrf_rank(vector_ids: list[str], bm25_ids: list[str],
             weights: tuple[float, float] | None = None) -> list[str]:
    """Weighted RRF: score += w / (k + rank + 1). weights=None -> модульные константы."""
    vw, bw = weights if weights else (VECTOR_WEIGHT, BM25_WEIGHT)
    scores: dict[str, float] = {}
    for rank, nid in enumerate(vector_ids):
        scores[nid] = scores.get(nid, 0.0) + vw / (RRF_K + rank + 1)
    for rank, nid in enumerate(bm25_ids):
        scores[nid] = scores.get(nid, 0.0) + bw / (RRF_K + rank + 1)
    return [nid for nid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def rerank_rank(query: str, candidate_ids: list[str], corpus_by_id: dict, ce,
                pool: int = RERANK_POOL) -> list[str]:
    head = [c for c in candidate_ids[:pool] if c in corpus_by_id]
    if not head:
        return candidate_ids
    pairs = [(query, corpus_by_id[c]["text"]) for c in head]
    scores = ce.predict(pairs, batch_size=RERANK_BATCH)
    order = sorted(range(len(head)), key=lambda i: float(scores[i]), reverse=True)
    reranked = [head[i] for i in order]
    head_set = set(head)
    tail = [c for c in candidate_ids if c not in head_set]
    return reranked + tail


# ============================================================================
# МЕТРИКИ
# ============================================================================
def _key_of(chunk_id: str, level: str) -> str:
    if level == "strict":
        return chunk_id
    if level == "document":
        return _document_of(chunk_id)
    return _segment_of(chunk_id)


def relevant_keys(relevant: list[str], level: str) -> set[str]:
    return {_key_of(r, level) for r in relevant}


def first_hit_rank(ranked_ids: list[str], rel_keys: set[str], level: str) -> int | None:
    for i, cid in enumerate(ranked_ids, 1):
        if _key_of(cid, level) in rel_keys:
            return i
    return None


def hit_ranks(ranked_ids: list[str], rel_keys: set[str], level: str, max_k: int) -> list[int]:
    """Позиции (1-based) попаданий релевантных ключей в топ-max_k.

    Ключ учитывается один раз: несколько чанков одного segment/document не дублируют
    попадание (иначе recall мог бы превысить 1.0).
    """
    ranks: list[int] = []
    seen: set[str] = set()
    for i, cid in enumerate(ranked_ids[:max_k], 1):
        key = _key_of(cid, level)
        if key in rel_keys and key not in seen:
            seen.add(key)
            ranks.append(i)
    return ranks


def average_precision(ranks: list[int], k: int, n_rel: int) -> float:
    """AP@k при бинарной релевантности: сумма Precision на позициях попаданий / min(n_rel, k)."""
    if n_rel <= 0:
        return 0.0
    hits = [r for r in ranks if r <= k]
    if not hits:
        return 0.0
    ap = sum(idx / r for idx, r in enumerate(hits, 1))
    return ap / min(n_rel, k)


def ndcg_at_k(ranks: list[int], k: int, n_rel: int) -> float:
    """nDCG@k (бинарная релевантность): DCG@k / IDCG@k, IDCG@k — по min(n_rel, k) позициям."""
    if n_rel <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_rel, k) + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def query_metrics(ranked_ids: list[str], rel_keys: set[str], level: str, ks: list[int]) -> dict:
    """Метрики одного запроса.

    recall@k и map@k — по доле найденных релевантных (gold может быть многозначным);
    mrr — только первое попадание; ndcg@k учитывает несколько релевантных.
    """
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


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]

# ============================================================================
# ПРОГОН КОНФИГУРАЦИИ
# ============================================================================
def run_config(cfg: str, qrels: list[dict], corpus: list[dict],
               corpus_by_id: dict, bm25, dense: DenseRetriever, ce,
               ks: list[int], level: str, keep_per_query: bool = False,
               weights: tuple[float, float] | None = None,
               label: str | None = None) -> dict:
    """weights — явные веса RRF (hybrid/hybrid_rerank); label — подпись в отчёте (hybrid@0.9/0.1)."""
    if cfg in RERANK_CONFIGS and ce is None:
        return {"config": label or cfg, "skipped": "reranker не загружен"}

    max_k = max(ks)
    limit = max(INITIAL_TOP_K, max_k)
    metric_names = ([f"recall@{k}" for k in ks] + [f"map@{k}" for k in ks]
                    + [f"mrr@{max_k}", f"ndcg@{max_k}"])
    acc: dict[str, list[float]] = {m: [] for m in metric_names}
    latencies: list[float] = []
    per_query: list[dict] = []
    adaptive_exact = 0  # для adaptive: сколько запросов ушло в exact-ветку весов

    for item in qrels:
        q = item["query"]
        norm_q = normalize_query(q)
        rel_keys = relevant_keys(item["relevant"], level)

        t0 = time.perf_counter()
        query_weights = weights
        if cfg in ("adaptive", "adaptive_rerank"):
            query_weights = adaptive_weights(q)
            if _are_exact_weights(query_weights):
                adaptive_exact += 1
        dense_ids = None
        if cfg in ("dense", "dense_rerank", "hybrid", "hybrid_rerank", "adaptive", "adaptive_rerank"):
            dense_ids = dense.top_ids(norm_q, limit if cfg == "dense" else INITIAL_TOP_K)
        if cfg == "dense":
            ranked = dense_ids
        elif cfg == "dense_rerank":
            ranked = rerank_rank(q, dense_ids, corpus_by_id, ce)
        elif cfg == "bm25":
            ranked = bm25_top_ids(bm25, q, corpus, top_k=BM25_TOP_K)
        elif cfg in ("hybrid", "adaptive"):
            ranked = rrf_rank(dense_ids, bm25_top_ids(bm25, q, corpus), query_weights)
        elif cfg == "hybrid_rerank":
            fused = rrf_rank(dense_ids, bm25_top_ids(bm25, q, corpus), query_weights)
            ranked = rerank_rank(q, fused, corpus_by_id, ce)
        elif cfg == "adaptive_rerank":
            fused = rrf_rank(dense_ids, bm25_top_ids(bm25, q, corpus), query_weights)
            if _are_exact_weights(query_weights):
                ranked = rerank_rank(q, fused, corpus_by_id, ce)
            else:
                ranked = fused
        else:
            raise ValueError(f"Неизвестная конфигурация: {cfg}")
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        m = query_metrics(ranked, rel_keys, level, ks)
        for name in metric_names:
            acc[name].append(m[name])
        if keep_per_query:
            per_query.append({
                "query": q, "relevant": item["relevant"],
                "n_relevant": int(m["n_relevant"]),
                "n_hits": int(m[f"n_hits@{max_k}"]),
                "first_hit_rank": int(m["first_hit_rank"]),
                "top5": ranked[:5], "latency_ms": round(dt, 1),
            })

    if weights:
        weights_label = f"{weights[0]:.2f}/{weights[1]:.2f}"
    elif cfg in ("adaptive", "adaptive_rerank"):
        weights_label = "adaptive"
    else:
        weights_label = "default"
    result = {
        "config": label or cfg,
        "weights": weights_label,
        "n_queries": len(qrels),
        "metrics": {m: (statistics.fmean(v) if v else 0.0) for m, v in acc.items()},
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _pct(latencies, 50),
            "p95": _pct(latencies, 95),
        },
    }
    if keep_per_query:
        result["per_query"] = per_query
    if cfg in ("adaptive", "adaptive_rerank"):
        result["adaptive_exact_queries"] = adaptive_exact
    return result


# ============================================================================
# RAGAS (заглушка, air-gapped)
# ============================================================================
def ragas_status() -> dict:
    try:
        import ragas  # noqa: F401
        return {
            "status": "skipped",
            "reason": "ragas установлен, но оценка требует генерации ответов полным RAG-контуром "
                      "(Qdrant + Ollama) и локального judge-LLM — в этом harness не считается.",
        }
    except ImportError:
        return {
            "status": "skipped",
            "reason": "air-gapped: ragas не установлен; внешние judge-API (OpenAI) не используются.",
        }


# ============================================================================
# ОТЧЁТ
# ============================================================================
def log_gold_stats(qrels: list[dict]) -> None:
    """Сколько релевантных на запрос: mean/min/max + распределение (gold бывает многозначным)."""
    counts = [len(item["relevant"]) for item in qrels]
    if not counts:
        return
    hist: dict[int, int] = {}
    for c in counts:
        hist[c] = hist.get(c, 0) + 1
    hist_str = ", ".join(f"{n} шт.: {hist[n]}" for n in sorted(hist))
    log(f"📊 Релевантных на запрос: mean {statistics.fmean(counts):.2f} | min {min(counts)} | "
        f"max {max(counts)} | распределение ({hist_str})")


def print_table(results: list[dict], ks: list[int], baseline: str = "dense") -> None:
    max_k = max(ks)
    header = (["config", "weights"] + [f"R@{k}" for k in ks]
              + [f"MAP@{max_k}", f"MRR@{max_k}", f"nDCG@{max_k}", "lat_mean", "lat_p95"])
    log("  " + " | ".join(f"{h:>9}" for h in header))
    log("  " + "-" * (11 * len(header)))
    base = next((r for r in results if r.get("config") == baseline and "metrics" in r), None)
    for r in results:
        if "metrics" not in r:
            log(f"  {r['config']:>9} | SKIPPED: {r.get('skipped', '')}")
            continue
        m = r["metrics"]
        cells = [f"{r['config']:>9}", f"{r.get('weights', 'default'):>9}"]
        cells += [f"{m[f'recall@{k}']:>9.3f}" for k in ks]
        cells.append(f"{m[f'map@{max_k}']:>9.3f}")
        cells.append(f"{m[f'mrr@{max_k}']:>9.3f}")
        cells.append(f"{m[f'ndcg@{max_k}']:>9.3f}")
        cells.append(f"{r['latency_ms']['mean']:>8.1f}m")
        cells.append(f"{r['latency_ms']['p95']:>8.1f}m")
        line = " | ".join(cells)
        if base and r is not base and base.get("metrics"):
            lift = m[f"recall@{max_k}"] - base["metrics"][f"recall@{max_k}"]
            line += f"   (deltaR@{max_k} vs {baseline}: {lift:+.3f})"
        log("  " + line)


def save_reports(out_dir: Path, results: list[dict], meta: dict, ks: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_report.json").write_text(
        json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    max_k = max(ks)
    lines = [
        "# Eval retrieval/rerank — отчёт", "",
        f"- Дата: {meta['timestamp']}",
        f"- Golden set: `{meta['qrels']}` ({meta['n_queries']} запросов)",
        f"- Корпус: `{meta['corpus_mode']}` ({meta['corpus_size']} чанков)",
        f"- Match-level: `{meta['match_level']}`",
        f"- Устройство: `{meta.get('device', 'cpu')}`",
        f"- Параметры (дефолт для конфигов без @весов): vector_w={meta['params']['vector_weight']}, "
        f"bm25_w={meta['params']['bm25_weight']}, rrf_k={meta['params']['rrf_k']}, "
        f"top_k={meta['params']['initial_top_k']}, bm25_top_k={meta['params']['bm25_top_k']}",
        f"- RAGAS: {meta['ragas']['status']} — {meta['ragas']['reason']}", "",
        "| config | weights | " + " | ".join(f"R@{k}" for k in ks) +
        f" | MAP@{max_k} | MRR@{max_k} | nDCG@{max_k} | lat_mean_ms | lat_p95_ms |",
        "|" + "---|" * (len(ks) + 7),
    ]
    for r in results:
        if "metrics" not in r:
            lines.append(f"| {r['config']} | — | skipped | " + " | " * (len(ks) + 4))
            continue
        m = r["metrics"]
        row = [r["config"], str(r.get("weights", "default"))]
        row += [f"{m[f'recall@{k}']:.3f}" for k in ks]
        row += [f"{m[f'map@{max_k}']:.3f}", f"{m[f'mrr@{max_k}']:.3f}", f"{m[f'ndcg@{max_k}']:.3f}"]
        row += [f"{r['latency_ms']['mean']:.1f}", f"{r['latency_ms']['p95']:.1f}"]
        if "adaptive_exact_queries" in r:
            row.append(f"adaptive exact-запросов: {r['adaptive_exact_queries']}")
        lines.append("| " + " | ".join(row) + " |")
    (out_dir / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# MAIN
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Офлайн-eval retrieval/rerank для RAG ФНС")
    p.add_argument("--qrels", required=True)
    p.add_argument("--query-field", default=None)
    p.add_argument("--relevant-field", default=None)
    p.add_argument("--configs", default="dense,bm25,hybrid,hybrid_rerank",
                   help="Список конфигов; веса можно задать спецификацией: "
                        "hybrid@0.9/0.1,hybrid_rerank@0.65/0.35")
    p.add_argument("--vector-weight", type=float, default=None,
                   help="RRF-вес dense: переопределяет VECTOR_WEIGHT (дефолт синхронен с движком)")
    p.add_argument("--bm25-weight", type=float, default=None,
                   help="RRF-вес BM25: переопределяет BM25_WEIGHT (дефолт синхронен с движком)")
    p.add_argument("--top-k", default="1,3,5,10")
    p.add_argument("--match-level", choices=["strict", "segment", "document"], default="segment")
    p.add_argument("--corpus", choices=["qdrant", "local"], default="qdrant")
    p.add_argument("--chunks-dir", default=str(PROJECT_DIR / "chunks"))
    p.add_argument("--qdrant-host", default=os.getenv("QDRANT_HOST", "localhost"))
    p.add_argument("--qdrant-port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "fns_collection"))
    p.add_argument("--model-path", default=str(PROJECT_DIR / "hf_cache" / "FRIDA"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--reranker-path", default=str(PROJECT_DIR / "reranker"))
    p.add_argument("--rerank-batch", type=int, default=None,
                   help="Батч cross-encoder'а (для GPU-прогонов на малой VRAM; None = дефолт ST)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default=str(PROJECT_DIR / "eval"))
    p.add_argument("--ragas", action="store_true")
    p.add_argument("--per-query", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qrels_path = Path(args.qrels)
    if not qrels_path.exists():
        log(f"❌ Golden set не найден: {qrels_path}")
        sys.exit(1)

    global VECTOR_WEIGHT, BM25_WEIGHT, RERANK_BATCH

    ks = sorted({int(x) for x in args.top_k.split(",") if x.strip()})
    if args.vector_weight is not None or args.bm25_weight is not None:
        if args.vector_weight is not None:
            VECTOR_WEIGHT = args.vector_weight
        if args.bm25_weight is not None:
            BM25_WEIGHT = args.bm25_weight
        log(f"⚖️  Веса RRF переопределены: vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT} "
            f"(у конфигов со своим @весом приоритет)")
    if args.rerank_batch is not None:
        RERANK_BATCH = args.rerank_batch
        log(f"⚙️  Батч cross-encoder'а: {RERANK_BATCH} (для GPU на малой VRAM)")
    try:
        specs = [parse_config_spec(c) for c in args.configs.split(",") if c.strip()]
    except ValueError as e:  # noqa: BLE001
        log(f"❌ {e}")
        sys.exit(1)
    for name, _w in specs:
        if name not in ALL_CONFIGS:
            log(f"❌ Неизвестная конфигурация: {name}. Доступны: {', '.join(ALL_CONFIGS)}")
            sys.exit(1)
    configs = [name for name, _w in specs]
    labels = [f"{name}@{w[0]}/{w[1]}" if w else name for name, w in specs]

    log("=" * 78)
    log("EVAL RETRIEVAL / RERANK (offline, ablation)")
    log("=" * 78)

    qrels = load_qrels(qrels_path, args.query_field, args.relevant_field)
    if args.limit:
        qrels = qrels[: args.limit]
    if not qrels:
        log("❌ Не удалось прочитать ни одного запроса из golden set.")
        sys.exit(1)
    log(f"📄 Golden set: {len(qrels)} запросов  ({qrels_path})")
    log_gold_stats(qrels)
    log(f"🎯 match-level: {args.match_level} | top-k: {ks} | configs: {', '.join(labels)}")
    log(f"🖥  device: {args.device}")

    log("\n📦 Загрузка корпуса...")
    if args.corpus == "qdrant":
        corpus = load_corpus_qdrant(args.qdrant_host, args.qdrant_port, args.collection)
    else:
        corpus = load_corpus_local(Path(args.chunks_dir))
    if not corpus:
        log("❌ Корпус пуст.")
        sys.exit(1)
    corpus_by_id = {d["id"]: d for d in corpus}
    log(f"✅ Корпус: {len(corpus)} чанков (уникальных id)")

    log("\n🧠 Загрузка FRIDA...")
    encoder = FridaEncoder(Path(args.model_path), device=args.device)
    log(f"✅ FRIDA загружена (dim={encoder.dim})")

    log("🔎 Построение BM25...")
    bm25 = build_bm25(corpus)
    log("✅ BM25 готов")

    dense = DenseRetriever(
        args.corpus, corpus, encoder,
        qdrant_args={"host": args.qdrant_host, "port": args.qdrant_port, "collection": args.collection},
    )

    ce = None
    if RERANK_CONFIGS & set(configs):
        try:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(args.reranker_path, device=args.device)
            log(f"✅ Reranker загружен: {args.reranker_path}")
        except Exception as e:  # noqa: BLE001
            log(f"⚠️ Reranker не загрузился ({e}); hybrid_rerank будет пропущен")

    if args.warmup and qrels:
        for _ in range(args.warmup):
            _ = dense.top_ids(normalize_query(qrels[0]["query"]), INITIAL_TOP_K)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "qrels": str(qrels_path),
        "n_queries": len(qrels),
        "corpus_mode": args.corpus,
        "corpus_size": len(corpus),
        "match_level": args.match_level,
        "device": args.device,
        "config_weights": {label: (list(w) if w else "default")
                           for (_name, w), label in zip(specs, labels, strict=True)},
        "params": {
            "vector_weight": VECTOR_WEIGHT, "bm25_weight": BM25_WEIGHT, "rrf_k": RRF_K,
            "initial_top_k": INITIAL_TOP_K, "bm25_top_k": BM25_TOP_K,
            "rerank_pool": RERANK_POOL, "rerank_batch": RERANK_BATCH, "final_top_k": FINAL_TOP_K,
        },
        "ragas": ragas_status() if args.ragas else {"status": "not_requested", "reason": ""},
    }

    results: list[dict] = []
    for (cfg, cfg_weights), label in zip(specs, labels, strict=True):
        extra = f"  (vector={cfg_weights[0]}, bm25={cfg_weights[1]})" if cfg_weights else ""
        log(f"\n▶️  Конфигурация: {label}{extra}")
        results.append(run_config(cfg, qrels, corpus, corpus_by_id, bm25, dense, ce,
                                  ks, args.match_level, keep_per_query=args.per_query,
                                  weights=cfg_weights, label=label))
        save_reports(Path(args.out), results, meta, ks)  # инкрементально: не терять длинный прогон

    log("\n" + "=" * 78)
    log("РЕЗУЛЬТАТЫ (ablation)")
    log("=" * 78)
    print_table(results, ks)
    save_reports(Path(args.out), results, meta, ks)
    log(f"\n💾 Отчёты: {args.out}/eval_report.json, {args.out}/eval_report.md")
    if meta["ragas"]["status"] == "skipped":
        log(f"ℹ️  RAGAS: {meta['ragas']['reason']}")


if __name__ == "__main__":
    main()
