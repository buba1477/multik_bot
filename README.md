# 🦾 Multik-RAG — Enterprise RAG для НПА РФ

RAG (Retrieval-Augmented Generation) для работы с нормативно-правовыми актами РФ.
Локальный ИИ-ассистент на базе **yagpt5_fns:latest** (Q4_K_M, Ollama).
Спроектирован для air-gapped сред, GPU ≤6 ГБ VRAM.

## Архитектура

```
documents.json
    │
    ▼
download ──► convert ──► pdf-fallback ──► parse ──► chunk ──► index ──► Qdrant
                                                                          │
                                                                          ▼
                                                                   Ollama ◄── RAG API
```

**download** — инкрементальная загрузка PDF/HTML через публикационный API ФНС
(Publication API) + резерв pravo.gov.ru / IPS.
**convert** — HTML→Markdown через BeautifulSoup + pandoc.
**pdf-fallback** — OCR для сканов (pdf_ocr.py).
**parse** — структурный парсинг заголовков (markdown_structure_parser.py).
**chunk** — юридический чанкер (legal_chunker.py): нарезка по article/paragraph,
FRIDA-токенизация (TARGET=450, MAX=512, MIN=40), редакционные блоки не дробятся.
**index** — FRIDA-эмбеддинг (RoBERTa, 1536-dim, CLS-pooling), загрузка в Qdrant
(fns_collection, Cosine).

Подробно: [docs/architecture_for_gnivc.md](docs/architecture_for_gnivc.md)

## Быстрый старт

### Требования

- Python 3.13, GPU (CUDA 12+), ≥6 ГБ VRAM
- Qdrant, Redis, Ollama (docker-compose или локально)
- FRIDA-модель в `hf_cache/FRIDA`

### Установка

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Запуск полного пайплайна

```bash
venv/bin/python scripts/run_pipeline.py --all
```

Флаги: `--force` (перезапустить все шаги), `--no-index` (без Qdrant),
`--chunk --no-index` (только чанкинг).
Подробно флаги и режимы: [docs/DOCUMENT_PIPELINE.md](docs/DOCUMENT_PIPELINE.md) (Приложение)

### Проверка

```bash
# Eval retrieval — dense / hybrid / hybrid_rerank
venv/bin/python scripts/eval_retrieval.py \
    --qrels eval/manual/qrels_user_clean.jsonl \
    --configs dense,hybrid,hybrid_rerank \
    --top-k 10
venv/bin/python scripts/eval_retrieval.py \
    --qrels eval/manual/qrels_exact_clean.jsonl \
    --configs dense,hybrid,hybrid_rerank \
    --top-k 10
```

## Структура проекта

```
app/
├── chunking/               # legal_chunker.py, create_index_qdrant_chunks.py
├── ingestion/              # html_to_markdown.py, html_to_pdf.py,
│                           # markdown_structure_parser.py, pdf_ocr.py
├── rag/                    # engine_rag.py, classifier.py, main_api.py,
│                           # sources.py, chart_engine.py, constants.py
├── publication_api.py      # загрузка через Publication API
├── pravo_resolver.py       # legacy-резерв (pravo.gov.ru / IPS)
├── resolved_documents.json # кэш разрешений документов
└── logger.py

scripts/
├── run_pipeline.py         # единый оркестратор
├── download_documents.py   # загрузчик НПА
├── eval_retrieval.py       # метрики retrieval (FRIDA baseline)
├── eval_e5.py              # метрики retrieval (multilingual-e5-large)
├── create_index_qdrant_e5.py  # индексация E5 в Qdrant
├── build_eval_set.py       # построение gold-датасета
├── pipeline_manifest.py    # работа с pipeline_manifest.json
├── sanitize_qrels.py       # чистка qrels
├── audit_qrels.py          # аудит qrels
├── export_qrels_with_text.py
├── extract_failed_queries.py
└── ...                     # утилиты

docs/
├── METRICS.md              # метрики — краткая выжимка
├── DOCUMENT_PIPELINE.md    # download-часть (Publication API, legacy)
├── FRONTEND_IMPROVEMENTS.md
└── architecture_for_gnivc.md

eval/                       # золотые наборы, qrels, отчёты о прогонах
├── METRICS_SUMMARY.md      # полная таблица по 5 этапам исследований
├── manual/                 # qrels_user_clean (76), qrels_exact_clean (145),
│                           # ROOT_CAUSE_TABLE.md, failed_retrieval_analysis.md
├── user_new/               # eval_report.* (USER FRIDA dense)
├── exact_new/              # eval_report.* (EXACT FRIDA hybrid_rerank)
├── e5_user/                # eval_report.* (E5 USER dense/hybrid)
├── e5_exact/               # eval_report.* (E5 EXACT dense/hybrid)
└── eval_rerank_out/        # отчёты о лёгких реранкерах (8 вариантов)

_archive/                   # временные/устаревшие файлы
├── backups/                # bench_one_query.py, qrels.bak*, pipeline.log
├── scripts/                # rebuild_visual_audit.py, regenerate_audit.py
└── docs/                   # RAG_EVAL_SUMMARY_2026.md (история чистки sig-match)

chunks/                     # результаты чанкинга (.jsonl)
raw/                        # PDF-файлы
raw_html/                   # HTML-файлы
markdown/                   # Markdown после конвертации
structure/                  # структурное представление
pipeline_manifest.json      # состояние пайплайна
```

## Переменные окружения

| Переменная | Описание | Пример |
|---|---|---|
| `OLLAMA_HOST` | Адрес сервера Ollama | `http://host.docker.internal:11434` |
| `CLASSIFIER_BACKEND` | Бэкенд классификатора | `ollama` / `gigachat` |
| `CLASSIFIER_MODEL` | Модель классификатора | `yagpt5_fns:latest` |
| `QDRANT_HOST` | Хост Qdrant | `qdrant` |
| `QDRANT_PORT` | Порт Qdrant | `6333` |
| `REDIS_HOST` | Хост Redis | `redis` |
| `REDIS_PASSWORD` | Пароль Redis | `Redis` |
| `DEBUG_MODE` | Режим отладки | `true` / `false` |

Полный шаблон — [`.env.example`](.env.example).

## Метрики (29.10.2026, eval/manual)
Адаптивный классификатор (`app/rag/classifier.py`) направляет **USER**-запросы в чистый DENSE (FRIDA),
а **EXACT** (точные реквизиты) — в HYBRID 0.5/0.5 + Reranker BAAI/bge-m3.

| Датасет | Конфиг | R@1 | R@3 | Hit@5 | MRR@10 | Latency |
|---|---|---|---|---|---|---|
| **USER** (76 запросов) | DENSE (FRIDA) | 0.468 | 0.670 | **88.1 %** | 0.808 | ~50 мс |
| **EXACT** (145 запросов) | HYBRID 0.5/0.5 + bge-m3 | 0.448 | 0.702 | **86.2 %** | 0.689 | ~20 с |

> Полные таблицы по всем 5 этапам (включая сравнение с multilingual-e5-large и лёгкими реранкерами):
> [`eval/METRICS_SUMMARY.md`](eval/METRICS_SUMMARY.md).  
> Краткая выжимка: [`docs/METRICS.md`](docs/METRICS.md).  
> Архитектура решения: [`docs/architecture_for_gnivc.md`](docs/architecture_for_gnivc.md).  
> Документ для ГНИВЦ (DOCX): [`architecture_for_gnivc.docx`](architecture_for_gnivc.docx).


## Документация

- [`eval/METRICS_SUMMARY.md`](eval/METRICS_SUMMARY.md) — полные результаты retrieval по 5 этапам (FRIDA, лёгкие реранкеры, E5-large)  
- [`docs/METRICS.md`](docs/METRICS.md) — метрики — краткая выжимка  
- [`docs/architecture_for_gnivc.md`](docs/architecture_for_gnivc.md) — полная архитектура (pipeline, Qdrant, chunker, eval, classifier)  
- [`docs/DOCUMENT_PIPELINE.md`](docs/DOCUMENT_PIPELINE.md) — детали download-этапа (Publication API, legacy-резерв)  
- [`docs/FRONTEND_IMPROVEMENTS.md`](docs/FRONTEND_IMPROVEMENTS.md) — фронтенд (тёмная тема, история чатов)  
- [`architecture_for_gnivc.docx`](architecture_for_gnivc.docx) — документ для ГНИВЦ (DOCX, отправлен)

