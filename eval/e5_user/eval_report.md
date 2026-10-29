# Eval retrieval/rerank — отчёт

- Дата: 2026-10-28T15:23:26
- Golden set: `/home/amlin04/multik_bot/eval/manual/qrels_user_clean.jsonl` (76 запросов)
- Корпус: `qdrant` (5446 чанков)
- Match-level: `segment`
- Устройство: `cpu`
- Параметры (дефолт для конфигов без @весов): vector_w=0.9, bm25_w=0.1, rrf_k=30, top_k=30, bm25_top_k=30
- RAGAS: not_requested — 

| config | weights | R@1 | R@3 | R@5 | R@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MAP@10 | MRR@10 | nDCG@10 | lat_mean_ms | lat_p95_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | default | 0.357 | 0.549 | 0.651 | 0.765 | 0.592 | 0.763 | 0.816 | 0.868 | 0.547 | 0.688 | 0.639 | 151.2 | 170.5 |
