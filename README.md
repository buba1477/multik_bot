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

Подробно: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

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
Подробно: [docs/OPERATIONS.md](docs/OPERATIONS.md)

### Проверка

```bash
# Eval retrieval — dense baseline
venv/bin/python scripts/eval_retrieval.py \
    --qrels eval/qrels.jsonl \
    --configs dense \
    --top-k 5
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
├── eval_retrieval.py       # метрики retrieval
├── build_eval_set.py       # построение gold-датасета
├── pipeline_manifest.py    # работа с pipeline_manifest.json
└── ...                     # утилиты

docs/
├── ARCHITECTURE.md         # схема, pipeline, Qdrant, chunker, eval, classifier
├── OPERATIONS.md           # идемпотентность, manifest, флаги, troubleshooting
├── DOCUMENT_PIPELINE.md    # download-часть (Publication API, legacy)
└── FRONTEND_IMPROVEMENTS.md

eval/                       # gold-запросы, qrels, отчёты
├── qrels.jsonl             # 138 запросов (segment match)
├── qrels_exact.jsonl       # exact match
└── eval_report.md

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

## Метрики (eval/qrels.jsonl, 138 запросов, match=segment)

| Конфиг | R@5 | MAP@5 | MRR@5 | nDCG@5 | Latency (ms) |
|---|---|---|---|---|---|
| dense (baseline) | 0.818 | 0.719 | 0.917 | 0.796 | 272 |

- **35 документов** в индексе, **5081 чанков** в Qdrant fns_collection
- Размерность эмбеддингов: 1536 (FRIDA / RoBERTa CLS-pooling)

## Документация

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — полная архитектура (pipeline, Qdrant, chunker, eval, classifier)
- [docs/OPERATIONS.md](docs/OPERATIONS.md) — идемпотентность, manifest, режимы, флаги, troubleshooting
- [docs/DOCUMENT_PIPELINE.md](docs/DOCUMENT_PIPELINE.md) — детали download-этапа (Publication API, legacy-резерв)
- [CHANGELOG.md](CHANGELOG.md) — история изменений
- [INTERVIEW_PREP.md](INTERVIEW_PREP.md) — подготовка к интервью (внутреннее)
