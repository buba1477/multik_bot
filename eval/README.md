# eval/ — офлайн-оценка retrieval/rerank

Директория с harness'ом для измерения качества поиска RAG-контура ФНС
**без внешних API** (air-gapped): метрики + ablation + латентность.
Исключение — сборка golden set (`build_eval_set.py`): разметка (генерация вопросов и
судья релевантности) идёт через **GigaChat-2-Max** (облако, Sber API); сам eval
запускается офлайн по уже готовому `qrels.jsonl`.

## Файлы

| Файл | Назначение |
|------|------------|
| `../scripts/eval_retrieval.py` | Harness: recall@k, MAP@k, MRR, nDCG@k + ablation (dense / dense_rerank / bm25 / hybrid / hybrid_rerank / adaptive) + тюнинг весов RRF (`hybrid@0.9/0.1`, `--vector-weight/--bm25-weight`) + латентность |
| `../scripts/build_eval_set.py` | Сборка golden set: черновой (`llm-gen`) или из своих вопросов (`from-questions` → `finalize`). LLM-разметка — **GigaChat-2-Max** (Sber API) |
| `questions.txt` | User-style вопросы «из головы» (137 шт.) для режима `from-questions` |
| `questions_exact.txt` | Вопросы с точными реквизитами (30 шт.: «117-ФЗ статья 217», «пункт 4 статьи 346.13», «Указ 112 приложение 1») |
| `qrels.jsonl` / `qrels_exact.jsonl` | Golden set'ы (генерируются; НЕ коммитятся) |
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

1. Заполните `eval/questions.txt` своими вопросами (user-style формулировки).
2. Ключ `API_KEY_GIGACHAT` в `.env` (или флаг `--gigachat-key`) — судья разметки: `GigaChat-2-Max`.
3. `--mode from-questions --auto-judge --candidates 10` → `eval/qrels_review.jsonl` (кандидаты + выбор судьи);
   в конце прогона печатается сводка: «судья сработал N/M», fallback на dense-top1, ошибки после retry.
4. Глазами подтвердите поле `relevant`, поставьте `"needs_review": false`.
5. `--mode finalize` → `eval/qrels.jsonl`; `--mode validate` — проверка id в корпусе.
6. Запустите `eval_retrieval.py` (уже офлайн, без внешних API).

### Точные реквизиты: `questions_exact.txt` → `qrels_exact.jsonl`

Для запросов вида «117-ФЗ статья 217» судья LLM-разметки ошибается заметно чаще
(на этом наборе — 6 из 30 запросов: например, на «146-ФЗ статья 11» выбирал *Главу* 11).
Поэтому gold строится **по реквизитам**: у каждого вопроса есть якорь вида
`117-fz_st346_13` / `ukaz-112_app1` / `ukaz-96_secIII`, и в `relevant` попадают ВСЕ чанки
корпуса с этим якорем (реквизит назван в вопросе ⇒ он и есть правильный ответ).
Выбор судьи сохраняется в `meta.judge_relevant` для трассируемости.

```bash
# 1) черновой прогон судьи (кандидаты + pick)
venv/bin/python scripts/build_eval_set.py --mode from-questions \
    --questions eval/questions_exact.txt --auto-judge --candidates 10 \
    --review-out eval/qrels_exact_review.jsonl --out eval/qrels_exact.jsonl
# 2) сборка reference-verified gold (якорь -> все чанки статьи/раздела/приложения) + validate
venv/bin/python scripts/build_eval_set.py --mode validate --out eval/qrels_exact.jsonl
```

match-level для этого набора — `segment`: все чанки одной статьи схлопываются в один ключ,
так что метрика читается как «правильная статья попала в top-k?»

## Тюнинг весов RRF (dense vs BM25), 2026-09-28

Две «популяции» запросов требуют РАЗНЫХ весов. Прогоны: 77 user-style вопросов
(`eval/qrels.jsonl`) и 147 вопросов с реквизитами (`eval/qrels_exact.jsonl`),
корпус `fns_collection` (4723 уникальных чанка), `--match-level segment`.

```bash
venv/bin/python scripts/eval_retrieval.py --qrels eval/qrels.jsonl \
    --configs dense,dense_rerank,hybrid@1.0/0.0,hybrid@0.9/0.1,hybrid@0.65/0.35,hybrid@0.5/0.5,hybrid_rerank@0.65/0.35 \
    --top-k 1,3,5,10 --match-level segment --out eval/weights_gold137/ --per-query
venv/bin/python scripts/eval_retrieval.py --qrels eval/qrels_exact.jsonl \
    --configs dense,dense_rerank,hybrid@1.0/0.0,hybrid@0.9/0.1,hybrid@0.65/0.35,hybrid@0.5/0.5,hybrid_rerank@0.5/0.5 \
    --top-k 1,3,5,10 --match-level segment --out eval/weights_exact/
```

| config | weights | R@1 (user 77) | R@1 (exact 147) | R@3 (user) | R@3 (exact) | Hit@5 (user) | Hit@5 (exact) | MRR@10 (user) | MRR@10 (exact) |
|---|---|---|---|---|---|---|---|---|---|
| dense | — | 0.468 | 0.407 | 0.670 | 0.564 | 0.881 | 0.738 | 0.808 | 0.611 |
| dense_rerank | — | — | — | — | — | — | — | — | — |
| bm25 | — | — | 0.190 | — | 0.353 | — | 0.545 | — | 0.367 |
| hybrid | 0.90/0.10 | 0.444 | — | 0.691 | — | 0.881 | — | 0.785 | — |
| hybrid | 0.65/0.35 (прод) | 0.355 | — | 0.647 | — | 0.855 | — | 0.718 | — |
| hybrid | 0.50/0.50 | — | 0.400 | — | 0.580 | — | 0.759 | — | 0.619 |
| hybrid_rerank | 0.65/0.35 | 0.476 | — | 0.665 | — | 0.868 | — | 0.797 | — |
| hybrid_rerank | 0.50/0.50 (pool=10) | — | 0.456 | — | 0.669 | — | 0.800 | — | 0.672 |
| **adaptive** | 0.9/0.1 / 0.5/0.5 | — | — | — | — | — | — | — | — |

> Реранк-строки пересчитаны **30.10.2026** после фикса бага `--rerank-pool` в `scripts/eval_retrieval.py`
> (дефолт пула замораживался в 30 при импорте; все 3 вызова `rerank_rank()` теперь передают `pool=` явно).
> Прод пулит `combined_nodes[:10]` (engine_rag.py) → строка pool=10 репрезентативна для прода.
> Исторический pool=30: Hit@5=0.862, lat_mean ~20 с — латентность в ~3 раза выше без выигрыша для прода.

**Выводы (метрики пересчитаны 2026-10-28 на пересмотренном датасете; эффективных запросов: 76 из 77 user и 145 из 147 exact — 3 записи без `relevant`):**

- Оптимум весов по-прежнему разный для двух «популяций». На user-наборе **dense** остаётся
  сильным по R@1 (0.468), hybrid@0.9/0.1 — 0.444, прод-веса 0.65/0.35 без реранка — заметно
  хуже (0.355).
- **Реранк на user-наборе теперь помогает**: hybrid_rerank@0.65/0.35 даёт **0.476 R@1**
  (лучший результат: +0.008 к dense, +0.121 к hybrid 0.65/0.35). На прежнем датасете реранк
  НЕ улучшал R@1 — вывод изменился вместе с составом набора.
- Точные реквизиты (exact 145): dense 0.407, BM25 — только 0.190 (на пересобранном наборе
  BM25-лидерство сменилось dense/гибридом), hybrid@0.5/0.5 — 0.400, **hybrid_rerank@0.5/0.5 —
  лучший: 0.456 R@1 и отрыв по R@3 (0.669, pool=10)**.
- Рерanker полезен на обоих наборах: exact +0.056 R@1 к гибриду 0.5/0.5, user +0.121 R@1
  к гибриду 0.65/0.35. Латентность реранка на GPU (pool=10, batch=4) — ~6.9 с/запрос
  (30.10.2026); прежние ~20 с соответствовали бажному pool=30. Выигрыш всё равно достигается
  ценой латентности — см. ниже.
- Прод-конфиг 0.65/0.35 без реранка — компромисс; `adaptive` (ревизия весов в
  `app/rag/engine_rag.py::_get_weights`) после пересмотра датасета требует уточнения
  (пересчёт не выполнялся — в таблице «—»).
- dense_rerank / промежуточные веса hybrid / adaptive на новом датасете не пересчитывались
  (в таблице — «—»).

> **Датасет пересмотрен 2026-10-28:** 117 нормативных вопросов перенесены из USER в EXACT для соответствия логике классификатора. Метрики пересчитаны (эффективных запросов: 76 из 77 user и 145 из 147 exact — запросы без `relevant` исключены из расчёта).

- Артефакт: при bm25_w≈0 hybrid ≠ dense (в прогонах 2026-09-28: 0.468 vs 0.489), потому что в Qdrant есть 62
  дублирующихся payload-id (до 7 копий на id) — RRF складывает вклады дублей и поднимает их
  выше настоящего top-1. Кандидат на фикс: дедуп id в `DenseRetriever.top_ids` и в RRF движка.

Латентность (для контекста, замеры реальных прогонов 2026-10-28): dense ~50 мс, BM25 ~17 мс,
hybrid ~50–90 мс на запрос (Qdrant + CPU, как в проде); cross-encoder (bge-reranker-v2-m3)
на GPU (GTX 1660 Ti, батч 4, pool=10) — mean=6.85 s / p95=7.63 s (exact, 145, 30.10.2026).
Прежние GPU-замеры ~20 s (user 20.7 / exact 20.2) соответствовали pool=30 из-за бага
`--rerank-pool` (исправлен 30.10.2026). Прежний замер ~1.4 s/запрос (прогоны 2026-09-28)
на текущем стенде не воспроизводится. Прод-оценка реранка на CPU
(`eval/weights_gold137_cpu_ref`): mean=11.4 s, p50=10.1 s, p95=13.3 s (без contention),
mean=15.3 s, p95=26.8 s при фоновой загрузке CPU. Вывод: реранк на порядки дороже
retrieval-шага, поэтому его выигрыш по R@1 (см. таблицу) в проде достигается ценой
латентности ~7 с на запрос (GPU, pool=10).

## Как считаются метрики

- **recall@k** — попал ли релевантный чанк в top-k (бинарно, усреднение по запросам).
- **MRR@K** — 1/(ранг первого релевантного), где K = max(top-k).
- **nDCG@K** — бинарный DCG/IDCG (одна релевантная цель ⇒ IDCG=1).
- **match-level**: `segment` (любая часть той же статьи, по умолчанию) | `strict` (точный chunk_id) | `document`.
- **latency**: средний/p50/p95 на весь retrieval-шаг конфигурации (CPU, как в проде).



- **Слабый gold**: вопросы из `llm-gen` сгенерированы из текста чанка и могут частично
  перекрываться с ответом ⇒ метрики завышены. Для честных цифр — `from-questions` + ручная проверка.
- **RAGAS**: в air-gapped контуре внешний judge-API недоступен; в отчёт пишется `status: skipped`
  с причиной, без фейковых чисел. Подключение возможно позже через локальную Ollama.
  (Внешний судья **GigaChat-2-Max** используется только в `build_eval_set.py` на этапе сборки
  gold — `eval_retrieval.py` и RAGAS остаются офлайн.)
- **dense** в режиме `--corpus qdrant` использует реальный поиск Qdrant по продовым векторам —
  это самый близкий к прода вариант. `--corpus local` считает эмбеддинги на лету (дольше).
- **Точные реквизиты**: `qrels_exact.jsonl` — reference-verified (gold = все чанки статьи,
  названной в вопросе), а не дословный вывод судьи LLM; это делает метрику однозначной
  («правильная статья в top-k»), но не проверяет, что выбран *нужный пункт* внутри статьи.
- **Рерanker**: прогоны с реранком на GPU (`--device cuda --rerank-batch 4`) — метрики качества
  переносимы, латентность GPU-специфична; CPU-прогон для прод-оценки латентности —
  `eval/weights_gold137_cpu_ref/`.
- Параметры (0.9/0.1, k=30, top-30, rerank top-10→5) сверены с `app/rag/engine_rag.py`;
  при изменении движка обновлять константы в `eval_retrieval.py`.
## Важные ограничения (честно)

- **Слабый gold**: вопросы из `llm-gen` сгенерированы из текста чанка и могут частично
  перекрываться с ответом ⇒ метрики завышены. Для честных цифр — `from-questions` + ручная проверка.
- **RAGAS**: в air-gapped контуре внешний judge-API недоступен; в отчёт пишется `status: skipped`
  с причиной, без фейковых чисел. Подключение возможно позже через локальную Ollama.
  (Внешний судья **GigaChat-2-Max** используется только в `build_eval_set.py` на этапе сборки
  gold — `eval_retrieval.py` и RAGAS остаются офлайн.)
- **dense** в режиме `--corpus qdrant` использует реальный поиск Qdrant по продовым векторам —
  это самый близкий к прода вариант. `--corpus local` считает эмбеддинги на лету (дольше).
- **Точные реквизиты**: `qrels_exact.jsonl` — reference-verified (gold = все чанки статьи,
  названной в вопросе), а не дословный вывод судьи LLM; это делает метрику однозначной
  («правильная статья в top-k»), но не проверяет, что выбран *нужный пункт* внутри статьи.
- **Рерanker**: прогоны с реранком на GPU (`--device cuda --rerank-batch 4`) — метрики качества
  переносимы, латентность GPU-специфична; CPU-прогон для прод-оценки латентности —
  `eval/weights_gold137_cpu_ref/`.
- Параметры (0.9/0.1, k=30, top-30, rerank top-10→5) сверены с `app/rag/engine_rag.py`;
  при изменении движка обновлять константы в `eval_retrieval.py`.
