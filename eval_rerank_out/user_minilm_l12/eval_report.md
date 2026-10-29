# Eval retrieval/rerank — отчёт

- Дата: 2026-10-28T09:58:47
- Golden set: `eval/manual/qrels_user_clean.jsonl` (76 запросов)
- Корпус: `qdrant` (5446 чанков)
- Match-level: `segment`
- Устройство: `cuda`
- Параметры (дефолт для конфигов без @весов): vector_w=0.9, bm25_w=0.1, rrf_k=30, top_k=30, bm25_top_k=30
- RAGAS: not_requested — 

| config | weights | R@1 | R@3 | R@5 | R@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MAP@10 | MRR@10 | nDCG@10 | lat_mean_ms | lat_p95_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_rerank@0.65/0.35 | 0.65/0.35 | 0.333 | 0.590 | 0.684 | 0.783 | 0.513 | 0.776 | 0.842 | 0.882 | 0.545 | 0.646 | 0.636 | 2031.9 | 2305.7 |
