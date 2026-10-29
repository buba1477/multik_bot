# Eval retrieval/rerank — отчёт

- Дата: 2026-10-28T10:21:38
- Golden set: `eval/manual/qrels_exact_clean.jsonl` (145 запросов)
- Корпус: `qdrant` (5446 чанков)
- Match-level: `segment`
- Устройство: `cuda`
- Параметры (дефолт для конфигов без @весов): vector_w=0.9, bm25_w=0.1, rrf_k=30, top_k=30, bm25_top_k=30
- RAGAS: not_requested — 

| config | weights | R@1 | R@3 | R@5 | R@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MAP@10 | MRR@10 | nDCG@10 | lat_mean_ms | lat_p95_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rerank@0.5/0.5 | 0.50/0.50 | 0.259 | 0.533 | 0.640 | 0.797 | 0.324 | 0.621 | 0.738 | 0.876 | 0.446 | 0.501 | 0.549 | 5825.5 | 5935.0 |
