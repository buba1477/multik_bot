# eval/ — офлайн-оценка retrieval/rerank

Директория с harness'ом для измерения качества поиска RAG-контура ФНС
**без внешних API** (air-gapped): метрики + ablation + латентность.

## Файлы

| Файл | Назначение |
|------|------------|
| `../scripts/eval_retrieval.py` | Harness: recall@k, MRR, nDCG@k + ablation (dense/bm25/hybrid/hybrid_rerank) + латентность |
| `../scripts/build_eval_set.py` | Сборка golden set: черновой (`llm-gen`) или из своих вопросов (`from-questions` → `finalize`) |
| `questions.txt` | Вопросы «из головы» (по одному на строку) для режима `from-questions` |
| `qrels.jsonl` | Golden set (генерируется; НЕ коммитится) |
| `eval_report.json` / `.md` | Отчёты прогона (генерируются; НЕ коммитятся) |

## Быстрый старт (черновой gold)

```bash
# запускать из КОРНЯ проекта
venv/bin/python scripts/build_eval_set.py --mode llm-gen --n 100 --out eval/qrels.jsonl
venv/bin/python scripts/eval_retrieval.py --qrels eval/qrels.jsonl \
    --configs dense,bm25,hybrid,hybrid_rerank --top-k 1,3,5,10 \
    --match-level segment --out eval/ --ragas --per-query
```

## Честный gold (рекомендуется)

1. Заполните `eval/questions.txt` своими вопросами.
2. `--mode from-questions --auto-judge` → `eval/qrels_review.jsonl` (кандидаты + авто-выбор).
3. Глазами подтвердите поле `relevant`, поставьте `"needs_review": false`.
4. `--mode finalize` → `eval/qrels.jsonl`; `--mode validate` — проверка id в корпусе.
5. Запустите `eval_retrieval.py`.

## Как считаются метрики

- **recall@k** — попал ли релевантный чанк в top-k (бинарно, усреднение по запросам).
- **MRR@K** — 1/(ранг первого релевантного), где K = max(top-k).
- **nDCG@K** — бинарный DCG/IDCG (одна релевантная цель ⇒ IDCG=1).
- **match-level**: `segment` (любая часть той же статьи, по умолчанию) | `strict` (точный chunk_id) | `document`.
- **latency**: средний/p50/p95 на весь retrieval-шаг конфигурации (CPU, как в проде).

## Важные ограничения (честно)

- **Слабый gold**: вопросы из `llm-gen` сгенерированы из текста чанка и могут частично
  перекрываться с ответом ⇒ метрики завышены. Для честных цифр — `from-questions` + ручная проверка.
- **RAGAS**: в air-gapped контуре внешний judge недоступен; в отчёт пишется `status: skipped`
  с причиной, без фейковых чисел. Подключение возможно позже через локальную Ollama.
- **dense** в режиме `--corpus qdrant` использует реальный поиск Qdrant по продовым векторам —
  это самый близкий к прода вариант. `--corpus local` считает эмбеддинги на лету (дольше).
- Параметры (0.65/0.45, k=30, top-30, rerank top-10→5) сверены с `app/rag/engine_rag.py`;
  при изменении движка обновлять константы в `eval_retrieval.py`.
