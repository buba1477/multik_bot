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

Две «популяции» запросов требуют РАЗНЫХ весов. Прогоны: 137 user-style вопросов
(`eval/qrels.jsonl`) и 30 вопросов с реквизитами (`eval/qrels_exact.jsonl`),
корпус `fns_collection` (4723 уникальных чанка), `--match-level segment`.

```bash
venv/bin/python scripts/eval_retrieval.py --qrels eval/qrels.jsonl \
    --configs dense,dense_rerank,hybrid@1.0/0.0,hybrid@0.9/0.1,hybrid@0.65/0.35,hybrid@0.5/0.5,hybrid_rerank@0.65/0.35 \
    --top-k 1,3,5,10 --match-level segment --out eval/weights_gold137/ --per-query
venv/bin/python scripts/eval_retrieval.py --qrels eval/qrels_exact.jsonl \
    --configs dense,dense_rerank,hybrid@1.0/0.0,hybrid@0.9/0.1,hybrid@0.65/0.35,hybrid@0.5/0.5,hybrid_rerank@0.5/0.5 \
    --top-k 1,3,5,10 --match-level segment --out eval/weights_exact/
```

| config | weights | R@1 (user 137) | R@1 (exact 30) | R@3 (user) | R@3 (exact) | MRR@10 (user) | MRR@10 (exact) |
|---|---|---|---|---|---|---|---|
| dense | — | **0.489** | 0.200 | 0.717 | 0.333 | 0.935 | 0.309 |
| bm25 | — | 0.215 | 0.167 | 0.390 | 0.433 | 0.539 | 0.325 |
| dense_rerank | — | 0.449 | **0.467** | 0.712 | 0.567 | 0.903 | 0.523 |
| hybrid | 1.00/0.00 | 0.468 | 0.200 | 0.699 | 0.333 | 0.894 | 0.309 |
| hybrid | 0.95/0.05 | 0.468 | 0.200 | 0.703 | 0.400 | 0.895 | 0.328 |
| hybrid | 0.90/0.10 | 0.457 | 0.233 | **0.707** | 0.400 | 0.881 | 0.353 |
| hybrid | 0.85/0.15 | 0.446 | 0.267 | 0.700 | 0.467 | 0.874 | 0.381 |
| hybrid | 0.80/0.20 | 0.438 | 0.300 | 0.687 | 0.500 | 0.867 | 0.415 |
| hybrid | 0.75/0.25 | 0.418 | 0.367 | 0.669 | 0.567 | 0.849 | 0.469 |
| hybrid | 0.65/0.35 (прод) | 0.392 | 0.400 | 0.647 | 0.667 | 0.834 | 0.500 |
| hybrid | 0.50/0.50 | 0.369 | 0.400 | 0.610 | 0.667 | 0.803 | 0.521 |
| hybrid_rerank | 0.65/0.35 | 0.446 | 0.467 | 0.691 | 0.600 | 0.892 | 0.524 |
| hybrid_rerank | 0.90/0.10 | 0.447 | — | 0.698 | — | 0.897 | — |
| **adaptive** | 0.9/0.1 / 0.5/0.5 | 0.457 | 0.400 | 0.707 | 0.667 | 0.881 | 0.521 |

**Выводы:**
- Оптимум весов разный. User-style (перефраз): dense / высокий vector-weight — 0.489 (dense) и
  0.457 при 0.9/0.1; BM25 на парафразах даёт шум (14 запросов, где dense попал в top-1,
  hybrid@0.9/0.1 опустил ответ на #2–#4). Точные реквизиты: BM25 решает — R@1 растёт
  0.200 → 0.400 при 0.5/0.5.
- Текущий прод-конфиг 0.65/0.35 — компромисс «ни там, ни там»: 0.392 на user-запросах.
  Отсюда `adaptive`: реквизиты → 0.5/0.5, иначе 0.9/0.1 (в `app/rag/engine_rag.py::_get_weights`,
  зеркало — `eval_retrieval.adaptive_weights`). Классификатор: 30/30 exact распознано, 0/137
  user-запросов ложно отнесено к exact.
- Реранк (bge-reranker-v2-m3) на честном user-gold НЕ улучшает R@1: 0.489 → 0.449 (dense),
  0.392 → 0.446 (hybrid 0.65/0.35, зато R@3 +0.044). На точных реквизитах — наоборот полезен
  (0.400 → 0.467). На «слабом» llm-gen gold (99) реранк давал +0.171 R@1 — прежний вывод был
  построен на завышенном gold.
- Артефакт: при bm25_w≈0 hybrid ≠ dense (0.468 vs 0.489), потому что в Qdrant есть 62
  дублирующихся payload-id (до 7 копий на id) — RRF складывает вклады дублей и поднимает их
  выше настоящего top-1. Кандидат на фикс: дедуп id в `DenseRetriever.top_ids` и в RRF движка.

Латентность (для контекста): dense/BM25/hybrid ~0.01–0.3 с на запрос (CPU),
cross-encoder — ~1.4 с на GPU (GTX 1660 Ti, батч 4) и ~11–12 с на CPU (прод-reжим,
baseline из `eval/weights_gold137_cpu_ref` без contention: mean=11.4 s, p50=10.1 s,
p95=13.3 s; при сильной загрузке CPU (`hybrid_rerank@0.65/0.35`, 137 запросов)
mean=15.3 s, p95=26.8 s — влияние фоновых eval-прогонов).

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
