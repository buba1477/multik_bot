# Eval retrieval/rerank — отчёт

- Дата: 2026-10-28T10:36:42
- Golden set: `eval/manual/qrels_exact_clean.jsonl` (145 запросов)
- Корпус: `qdrant` (5446 чанков)
- Match-level: `segment`
- Устройство: `cuda`
- Параметры (дефолт для конфигов без @весов): vector_w=0.9, bm25_w=0.1, rrf_k=30, top_k=30, bm25_top_k=30
- RAGAS: not_requested — 

| config | weights | R@1 | R@3 | R@5 | R@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MAP@10 | MRR@10 | nDCG@10 | lat_mean_ms | lat_p95_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rerank@0.5/0.5 | 0.50/0.50 | 0.356 | 0.518 | 0.621 | 0.741 | 0.455 | 0.628 | 0.717 | 0.814 | 0.490 | 0.568 | 0.572 | 1760.7 | 1839.8 |
