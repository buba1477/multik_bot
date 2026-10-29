# Pipeline загрузки и обработки НПА

Документ описывает фактическое состояние pipeline на момент последней проверки.

Дата: 23.10.2026

---

## 1. Общая схема

```
documents.json  (содержит nd для legacy-документов)
    │
    ▼
scripts/download_documents.py
    │   (инкрементальное скачивание; см. секцию 3.3)
    │
    ├── app/publication_api.py           (ОСНОВНОЙ путь)
    │       │
    │       ├── /api/Documents (поиск по number, затем number + name)
    │       ├── /api/Document?eoNumber=… (детали)
    │       ├── /file/pdf?eoNumber=… → raw/<id>.pdf
    │       └── /Document/View/<eoNumber> → raw_html/<id>.html
    │
    └── app/pravo_resolver.py            (LEGACY-резерв, только для актов
            │                              отсутствующих в базе публикации)
            ├── /proxy/ips/ (поиск nd по реквизитам)
            ├── print-представление → raw_html/<id>.html
            └── app/ingestion/html_to_pdf.py
                    └── Playwright + bundled Chromium → raw/<id>.pdf


=== RAG pipeline (на основе HTML) ===

raw_html/<id>.html         raw/<id>.pdf (OCR fallback)
    │                           │
    │              ┌────────────┤
    │              ▼            ▼
    │     pdf_ocr.is_scanned_pdf() → True → pdf_ocr.ocr_pdf()
    │              │            │
    │              ▼            ▼
    │     convert_from_text()   │
    │              │            │
    ▼              ▼            │
app/ingestion/html_to_markdown.py
    │   BeautifulSoup (санитизация)
    │   → pandoc (html→gfm-raw_html)
    │   → clean_markdown
    ▼
markdown/<id>.md
    │
    ▼
app/ingestion/markdown_structure_parser.py
    │   Классификация заголовков и контента
    │   → linear (восстановление порядка)
    │   → tree (структурное представление)
    │   → records (слой для chunker)
    ▼
structure/<id>.json
    │
    ▼
app/chunking/legal_chunker.py
    │   Нарезка по article/paragraph/subparagraph
    │   → FRIDA-токенизация (TARGET=450, MAX=465, MIN=40)
    │   → Редакционные блоки (преамбула) не дробятся
    ▼
chunks/<id>.jsonl
    │
    ▼
app/chunking/create_index_qdrant_chunks.py
    │   FRIDA embedding (Sber RoSBERTa, 1536-dim, CLS-pooling)
    │   → загрузка векторов в Qdrant fns_collection
    │   → спот-чек верификация 10 точек после загрузки
    ▼
Qdrant vector DB (5446 уникальных чанков, коллекция fns_collection)
    │
    ▼
app/rag/engine_rag.py (RerankedEngine)
    │   Классификация запроса: classifier.py → режим EXACT / USER
    │     - USER (1.0/0.0): FAST-TRACK — только DENSE top-30, BM25/RRF/rerank пропущены
    │     - EXACT (0.5/0.5): BM25 top-30 + weighted RRF (k=30)
    │       → SentenceTransformerRerank (bge-m3, /app/reranker, top-10 → top-5)
    │   → Синтезатор: Compact (default) / Tree (сравни, обобщи, анализ...)
    │   → LLM: GigaChat-3-Ultra (Settings.llm, temp=0.0) — генерация
    │     Ollama (yagpt5_fns) — «мультик»-маршрут, chart-engine, классификатор
    │   → ChartEngine (ECharts JSON, опционально)
    │   → Автоподбор фотографий сотрудников
    ▼
app/rag/sources.py — дедупликация и форматирование источников ответа
    │
    ▼
Streaming JSON-ответ → фронтенд
```

Унифицированный запуск пайплайна:

```bash
venv/bin/python scripts/run_pipeline.py --all
# Отдельные шаги: --download, --ocr, --convert, --parse, --chunk
```


---

## 2. Роли файлов и каталогов

| Путь | Назначение |
|------|-----------|
| `documents.json` | Реестр документов (id, number, date, title, type, enabled, legacy nd). Источник истины для списка загружаемых актов. |
| `raw/<id>.pdf` | Скачанные PDF. Для publication-пути — официальный PDF с портала; для legacy-пути — результат Playwright HTML→PDF. |
| `raw_html/<id>.html` | HTML-представление документа. Для publication — официальная HTML-версия с портала; для legacy — print-представление IPS. |
| `markdown/<id>.md` | Конвертированный Markdown (результат html_to_markdown). Единственный источник текста для chunker. |
| `structure/<id>.json` | Результат структурного парсинга (markdown_structure_parser). Содержит linear (точное восстановление), tree, records. |
| `chunks/<id>.jsonl` | Нарезанные чанки (legal_chunker). JSONL-формат: id, title, text, local_img, url. Непосредственный вход для индексации. |
| `app/resolved_documents.json` | Кэш разрешений. Для каждого документа хранит method (publication/legacy), revision, pdf_path, html_path, html_sha256, pdf_size, pdf_pages. |
| `app/rag/sources.py` | Дедупликация и форматирование источников ответа: группировка по URL, объединение структурных частей (пункты 8–10), без hardcode под документы. |
| `app/rag/chart_engine.py` | DynamicChartEngine — детекция запросов на график, генерация ECharts JSON-конфигурации. |
| `app/rag/classifier.py` | Классификатор запросов (EXACT vs USER): strong_exact → LLM → guard_fallback; единый источник RRF-весов WEIGHTS (exact 0.5/0.5, user 1.0/0.0); кэш `_mode_cache`. |
| `app/logger.py` | Единый модуль логирования: ротация 10 MB / 5 бэкапов в `app_audit.log`, дублирование в stdout, ContextVar для request_id. |
| `app/search_fragments.py` | Построение поисковых фрагментов названия (от специфичного к общему) для двухэтапного разрешения документов. |
| `hf_cache/FRIDA/` | Локальный кэш модели Sber RoSBERTa (FRIDA) для embedding-векторов. Offline-режим. |
| `images_cache/` | Кэш изображений сотрудников для автоподбора фотографий в ответах RAG. |
| `employees.txt` | Список сотрудников для автоподбора фотографий (персональные данные, в gitignore). |
| `scripts/download_documents.py` | Инкрементальный загрузчик НПА: publication API + legacy-резерв. |
| `scripts/run_pipeline.py` | Унифицированный пайплайн ingestion: download → ocr → convert → parse → chunk. Флаги: `--all`, `--download`, `--ocr`, `--convert`, `--parse`, `--chunk`. |
| `scripts/eval_retrieval.py` | Офлайн-харнесс оценки retrieval: recall@k, MRR, nDCG + ablation (dense/bm25/hybrid/hybrid_rerank). |
| `scripts/build_eval_set.py` | Утилита сборки golden set (для будущего расширения). |
| `eval/` | Артефакты оценки: qrels, вопросы, отчёты (не коммитятся, кроме шаблонов). |
| `INTERVIEW_PREP.md` | Шпаргалка к собесу ML/LLM Engineer по проекту. |
| `app_audit.log` | Ротируемый лог (10 MB, 5 бэкапов) с request_id для аудита. |


## 3. Разрешение документов (download)

### 3.1. Основной путь: publication.pravo.gov.ru

Модуль: `app/publication_api.py`

1. **Поиск** — `GET /api/Documents?number=<номер>`
   - Ответ: JSON со списком кандидатов (eoNumber, number, documentDate, documentType, title).
   - Параметр number — единственный надёжный фильтр; date и сортировка игнорируются API.

2. **Верификация** — `_pick_unique()`:
   - Сравнение number + date + type со строгими реквизитами из documents.json.
   - Канонизация типов (`_canonical_type`: "Указ" + "Президент Российской Федерации" → "Указ Президента Российской Федерации").
   - **Fail-closed**: при неоднозначности/несовпадении — `DocumentMismatchError` (останов, legacy-резерв НЕ используется).

3. **Детали** — `GET /api/Document?eoNumber=<eoNumber>` 
   - Полная информация о документе (подтверждение реквизитов).

4. **Скачивание**:
   - PDF: `GET /file/pdf?eoNumber=...` → `raw/<id>.pdf` (проверка заголовка `%PDF-`).
   - HTML: `GET /Document/View/<eoNumber>` → `raw_html/<id>.html` (проверка `<!DOCTYPE html>`).

**Двухэтапный поиск** (`resolve_exact`):
   - Этап 1: строго по `number` → `_pick_unique()`.
   - Этап 2: если неоднозначно/не найдено — `number + name` с короткими фрагментами названия
     (`build_search_fragments` из `app/search_fragments.py`, от более специфичного к менее).
   - Итоговое отсутствие → `DocumentNotFoundError` (переход на legacy-резерв).
   - Несовпадение реквизитов → `DocumentMismatchError` (останов, без резерва).

**Ограничение**: портал официального опубликования содержит акты примерно с 2011–2012 гг.
Старые федеральные законы (79-ФЗ от 2004, 58-ФЗ от 2003) — отсутствуют → `DocumentNotFoundError`.

### 3.2. Legacy-резерв: pravo.gov.ru /proxy/ips/

Модуль: `app/pravo_resolver.py`

Используется **только** когда `publication_api.resolve_exact()` вернул `DocumentNotFoundError`.

1. **Поиск nd** — `GET /proxy/ips/?list_itself=&a8=<номер>&page=first`
   - Внутренний идентификатор nd, редкое название, статус.

2. **Верификация** — сравнение date + number + type по карточке документа.

3. **Редакции** — `GET /proxy/ips/?docbody=&nd=<nd>` → парсинг `<select name="doc_editions">`:
   - Определение rdk (последняя доступная редакция, max).

4. **Скачивание HTML** — `GET /proxy/ips/?docview&page=1&print=1&nd=<nd>&rdk=<rdk>&empire=`:
   - Сохраняется как `raw_html/<id>.html`.

5. **Конвертация HTML→PDF** — `app/ingestion/html_to_pdf.py`:
   - Playwright + bundled Chromium → `raw/<id>.pdf`.

### 3.3. Инкрементальное скачивание

`scripts/download_documents.py` — `download_one()`:

```
resolved_documents.json
  └── entry[doc_id] = {
    "method": "publication" | "legacy",
    "revision": { "id": ..., "label": ... },
    "downloaded_at": "ISO-8601",
    "pdf_path": "raw/<id>.pdf",
    "pdf_size": ...,
    "pdf_pages": ...,
    "html_path": "raw_html/<id>.html",
    "html_sha256": "...",
    "detail": { ... }   # специфичные для метода данные
  }
```

- Если revision совпадает и PDF на месте → **skip** ("unchanged → skip download").
- Если revision изменилась / PDF отсутствует → атомарная замена через `os.replace()`.
- При ошибке старый PDF не удаляется, кэш не обновляется.


## 4. Конвертация HTML → Markdown

Модуль: `app/ingestion/html_to_markdown.py`

### 4.1. HTML-путь (основной)

```
raw_html/<id>.html
  │ BeautifulSoup-санитизация
  │   - Удаление script/style/head/iframe/nav/page-navigation/emailDlg
  │   - Таблицы → построчное представление (ячейки через |)
  │   - Сохранение неизвестных элементов (текст не теряется)
  ▼
временный .clean.html
  │ pandoc --from=html --to=gfm-raw_html --wrap=none
  ▼
markdown/<id>.md
  │ clean_markdown() — финальная чистка (пустые строки, спецсимволы)
```

### 4.2. OCR Fallback (сканированные PDF)

Если `pdf_ocr.is_scanned_pdf()` → True:
  - `pdf_ocr.ocr_pdf()` → текст через OCR.
  - `convert_from_text()` → Markdown без структуры.

### 4.3. pdftotext Fallback (текстовые PDF с плохим HTML)

Если HTML-путь дал Markdown ≤ 500 символов:
  - `pdftotext <pdf> -` → извлечение текста.
  - `convert_from_text()` → Markdown.

**Правила**:
- Структура только из DOM; не угадывается по словам (Глава/Статья).
- h1-h6 → pandoc (native).
- ol/ul → markdown-списки.
- `<p>` → абзацы (не превращаются в списки без DOM-тегов).


## 5. Структурный парсинг Markdown

Модуль: `app/ingestion/markdown_structure_parser.py`

### 5.1. Три слоя

| Слой | Описание |
|------|----------|
| **linear** | Линейный порядок лексических блоков (source of truth). Каждый блок: type, text, start_line, end_line, heading_level. Точное восстановление исходного Markdown (`exact_reconstruct()`). |
| **tree** | Иерархическое представление по node_id. parent_id → вложенность. Не источник текста. |
| **records** | Производный плоский слой (type, num, title, text) — вход для chunker. |

### 5.2. Классификация заголовков

| Паттерн | Тип | Уверенность |
|---------|-----|-------------|
| `Глава N. ...` | chapter | 0.95 |
| `Раздел N. ...` | section | 0.95 |
| `Статья N. ...` | article | 0.95 |
| `Приложение №N` | appendix | 0.95 |
| `N.N. ...` (римские) | subsection/section | 0.70–0.75 |
| `N) ...` | item | 0.92 |
| `а) ...` | subparagraph | 0.70–0.80 |

Сомнительное → `unknown` (текст сохраняется). Без правил про конкретный закон.

### 5.3. Валидация

- `exact_reconstruct()` — посимвольное сравнение с исходным Markdown.
- `normalize_reconstruct()` — сравнение без пробелов.
- Статистика: unknown-блоки, table-блоки.


## 6. Чанкинг (legal_chunker)

Модуль: `app/chunking/legal_chunker.py`

### 6.1. Вход/выход

- **Вход**: `markdown/<id>.md` + `structure/<id>.json` (records-слой).
- **Выход**: `chunks/<id>.jsonl` (JSONL, 5 полей: id, title, text, local_img, url).

### 6.2. Логика нарезки

```
records (из structure JSON)
  │
  ├── Статья — логическая граница.
  │   Заголовок статьи присутствует в каждом её чанке.
  │   Разные статьи не смешиваются в одном чанке.
  │
  ├── Большие статьи → деление по paragraph/item/subparagraph.
  │
  ├── Отдельный элемент > MAX_TOKENS → режется по предложениям → по словам.
  │
  ├── Мелкие последовательные части одной статьи → объединение до TARGET_TOKENS.
  │
  └── Редакционные блоки ("(В редакции...)") — не дробятся, единый блок.
```

### 6.3. Лимиты токенов

| Параметр | Значение |
|----------|----------|
| TARGET_TOKENS | 450 |
| MAX_TOKENS | 465 |
| MIN_CHUNK_TOKENS | 40 |

Токенизация — FRIDA (Sber RoSBERTa) tokenizer. Offline. Если модель недоступна → эвристика (~4 символа на токен).

### 6.4. Редакционные блоки

Константа `_EDITORIAL_RE` = `r"\(В редакции[^)]*\)"` — паттерн для выявления редакционных блоков.
- Блоки, соответствующие паттерну, возвращаются как единое целое без дробления.
- При объединении мелких частей редакционный блок прикрепляется к предыдущему чанку.

### 6.5. Валидация JSONL

`validate_jsonl()`:
- Парсинг JSON.
- Проверка ровно 5 ключей (id, title, text, local_img, url).
- Непустой text.
- Подсчёт токенов.


## 7. Индексация в Qdrant

Модуль: `app/chunking/create_index_qdrant_chunks.py`

### 7.1. Процесс

```
1. Чтение и валидация всех chunks/*.jsonl
2. Проверка дубликатов ID
3. Проверка лимита токенов (≤600)
4. Загрузка FRIDA (SentenceTransformer: Transformer + Pooling(CLS), 1536-dim)
   → lru_cache для tokenizer (functools.lru_cache, _tokenize_chunk)
5. Вычисление embeddings (batch=32, префикс "search_document:")
6. Подключение к Qdrant
7. Удаление/пересоздание коллекции fns_collection
8. Загрузка векторов с payload (id, title, text, local_img, source_url)
9. Спот-чек: верификация 10 случайных точек (embedding ≈)
10. Проверка: количество точек == количество чанков, размерность совпадает
```

### 7.2. Параметры

| Параметр | Значение |
|----------|----------|
| Collection | `fns_collection` |
| Qdrant host | `localhost` (переопределяется через `QDRANT_HOST`) |
| Qdrant port | `6333` (переопределяется через `QDRANT_PORT`) |
| Embedding | FRIDA / Sber RoSBERTa (CLS-pooling, 1536-dim) |
| Batch size | 32 |
| Префикс | `search_document:` |
| TARGET_TOKENS | 450 |
| MAX_TOKENS | 600 |

### 7.3. Payload точки

```json
{
  "id": "chunk-<doc_id>-<seq>",
  "title": "Статья N. Название",
  "text": "Текст чанка...",
  "local_img": "images_cache/...jpg" | "",
  "source_url": "http://publication.pravo.gov.ru/file/pdf?eoNumber=..."
}
```


## 8. RAG-движок (engine_rag.py)

Модуль: `app/rag/engine_rag.py`

### 8.1. Архитектура запроса

```
Запрос пользователя (main_api.py: POST /api/v1/predict/stream)
  │
  ├── is_chart_request → пропуск WAF, идёт в RAG-путь
  │
  ├── WAF-проверка (is_malicious_query): длина > 2000, BAD_PATTERNS,
  │     комбо-фильтры, анти-пробел, Unicode-омофоны → блокировка
  │     (ответ "БАЗА_ПУСТА: Я эксперт по вопросам ФНС России.")
  │
  ├── "мультик" (без ФНС-ключевых слов) → Ollama напрямую
  │     (yagpt5_fns:latest, num_ctx=4096, temperature=0.8, num_predict=250)
  │
  └── Redis-кэш (ключ = lower(normalized_query), TTL 86400)
        └── промах → get_ai_streaming_response() → RerankedEngine._sync_query():

            1. DENSE top-30: FRIDA embedding → Qdrant fns_collection
            2. Классификация: classify_mode() (strong_exact → LLM → guard_fallback)
               → get_weights(): USER = (1.0, 0.0), EXACT = (0.5, 0.5)
            3. FAST-TRACK USER (1.0/0.0):
                 чистый DENSE, BM25/RRF/rerank пропущены,
                 chunk_id восстановлены из node_map
               ИЛИ HYBRID EXACT (0.5/0.5):
                 BM25 top-30 (rank_bm25) + weighted RRF (k=30)
                 → rerank только здесь: SentenceTransformerRerank
                   (bge-m3, /app/reranker, top-10 → top-5)
            4. Синтезатор: Compact (default) / Tree (сравни, обобщи,
                 анализ, синтезируй...)
            5. Генерация: Settings.llm = LlamaGigaChat (GigaChat-3-Ultra,
                 temperature=0.0, API_KEY_GIGACHAT)
            6. Источники: app/rag/sources.py (дедупликация, максимум 3),
                 автоподбор фото, стриминг JSON
        │
        ├── 5. Дедупликация и форматирование источников
        │     └── app/rag/sources.py — группировка по URL, объединение структурных частей
        │
        └── 6. Пост-обработка
              ├── Замена спецсимволов (HTML → unicode)
              ├── Автоподбор фотографий (по тексту ответа, employees.txt → images_cache/)
              └── Форматирование: **жирный**, списки, таблицы, пустые строки
```

### 8.2. Промпт

Системный промпт задаёт:
- Роль: "ведущий эксперт ФНС России".
- Язык: строго русский.
- Формат: **жирный** для ключевых терминов, маркированные/нумерованные списки, таблицы Markdown.
- Запрет: вымышленных норм, нецензурной лексики, советов по уклонению.
- При нехватке данных: вежливый отказ ("В моих регламентах про это ни слова").

Параметры LLM (различаются по пути):
| Путь | Модель | temperature | num_ctx | num_predict | Место вызова |
|------|--------|-------------|---------|-------------|--------------|
| Основная RAG-генерация | GigaChat-3-Ultra | 0.0 | 8096 (default) | 512 (default) | `Settings.llm = LlamaGigaChat()` в engine_rag.py |
| «Мультик» (без ФНС-тем) | yagpt5_fns:latest (Ollama) | 0.8 | 4096 | 250 | main_api.py |
| Классификатор (ollama backend) | yagpt5_fns:latest | 0.0 | 512 | 5 | classifier._ollama_generate() |
| Классификатор (gigachat backend) | GigaChat-2-Max | 0.0 | — | 5 | classifier._gigachat_generate() |
| Chart engine (ECharts JSON) | yagpt5_fns:latest (Ollama) | 0.0 (по умолч.) | — | — | chart_engine → httpx POST |

### 8.3. Потоковый ответ

`get_ai_streaming_response()` — асинхронный генератор:
- JSON-объекты по одному на строку: `{"type": "text"|"metadata"|"error"|"end", "content": ...}`.
- Стриминг через Ollama `async chat()` или GigaChat `generate()`.
- Автоподбор фото на основе текста ответа (поиск по employees.txt, сопоставление с images_cache/).

### 8.4. Чарт-режим

`DynamicChartEngine` (модуль `app/rag/chart_engine.py`):
- `is_chart_request()` — определяет запросы на графики (по ключевым словам).
- Генерирует ECharts JSON-конфигурацию → рендерится на фронтенде.

### 8.5. GigaChat — основной LLM для RAG

Модуль: `app/rag/engine_rag.py` (класс `LlamaGigaChat(CustomLLM)`)

- **Основной LLM для генерации RAG-ответов** (Settings.llm = LlamaGigaChat()).
- Модель: `GigaChat-3-Ultra`, temperature=0.0.
- Аутентификация через API-ключ (`API_KEY_GIGACHAT` из .env, Bearer-токен).
- Потоковый и не-потоковый режимы через `gigachat` SDK (GigaChat).
- **Зависимость**: при отсутствии `API_KEY_GIGACHAT` генерация невозможна (ошибка).

### 8.6. Дедупликация источников

Модуль: `app/rag/sources.py`

После генерации ответа:
1. Парсинг текста ответа — выделение ссылок вида `[1], [2-4], [3, 5]`.
2. Группировка source-чанков по `source_url`.
3. Объединение структурных частей (пункты 8–10 одного документа → один источник).
4. Форматирование: "Статья X → пункты Y–Z".
5. Без hardcode под конкретные документы — полностью обобщённая логика.

### 8.7. Автоподбор фотографий

- Файл: `employees.txt` (список сотрудников, не коммитится).
- Кэш изображений: `images_cache/` (названия файлов — фамилии).
- Поиск: после генерации ответа извлекаются фамилии (через employees.txt) → сопоставление с images_cache/ → вставка в `metadata.employees`.


## 9. API

Модуль: `app/rag/main_api.py`

### 9.1. Эндпоинты

| Путь | Метод | Описание |
|------|-------|----------|
| `/` | GET | HTML-страница чата (Jinja2, templates/index.html) |
| `/api/v1/predict/stream` | POST | Streaming-ответ (JSON lines, type: text/metadata/error/end). Параметр JSON: `query`. |

### 9.2. Маршрутизация запросов

```
POST /api/v1/predict/stream  { query: "..." }
  │
  ├── is_chart_request() → True → пропуск WAF
  │
  ├── WAF (is_malicious_query)
  │   └── True → "БАЗА_ПУСТА: Я эксперт по вопросам ФНС России."
  │
  ├── "мультик" + не ФНС-ключевые → Ollama напрямую
  │   (yagpt5_fns:latest, num_ctx=4096, temp=0.8, num_predict=250)
  │
  └── ФНС-запрос / чарт:
        │
        ├── Redis-кэш (ключ: lower(query), TTL: 86400 сек)
        │   └── попали → stream из кэша
        │
        ├── get_ai_streaming_response(query) → RerankedEngine._sync_query()
        │   ├── DENSE top-30 (FRIDA → Qdrant)
        │   ├── classify_mode() → EXACT/USER → get_weights()
        │   ├── FAST-TRACK USER (dense only) / HYBRID EXACT (BM25+RRF+rerank)
        │   ├── Синтезатор (Compact/Tree)
        │   └── GigaChat-3-Ultra (LlamaGigaChat, temp=0.0)
        │
        └── успех → кэшируем полный ответ
          (фильтр: не кэшируются пустые/ошибки/заглушки)
```

### 9.3. Middleware

- Request ID (uuid4, 12 символов) → проброс через ContextVar.
- Замер времени выполнения.
- Логирование с request_id.

### 9.4. Обработка ошибок

- **500** — глобальный exception handler (traceback → лог, пользователь → безопасный JSON).
- **422** — Pydantic validation error (детали → лог, пользователь → общее сообщение).


## 10. Конфигурация

### 10.1. Переменные окружения (.env)

| Переменная | Назначение |
|------------|------------|
| `OLLAMA_HOST` | Адрес Ollama (http://ollama_container:11434) |
| `QDRANT_HOST` | Хост Qdrant (qdrant) |
| `QDRANT_PORT` | Порт Qdrant (6333) |
| `REDIS_HOST` | Хост Redis (redis) |
| `REDIS_PORT` | Порт Redis (6379) |
| `REDIS_PASSWORD` | Пароль Redis |
| `GIGACHAT_CLIENT_ID` | Client ID для GigaChat OAuth 2.0 |
| `GIGACHAT_CLIENT_SECRET` | Client Secret для GigaChat OAuth 2.0 |
| `GIGACHAT_MODEL` | Модель GigaChat (по умолч. `GigaChat:latest`) |
| `GIGACHAT_SCOPE` | Scope авторизации GigaChat |
| `CLASSIFIER_BACKEND` | Бэкенд классификатора (ollama / gigachat, по умолч. ollama) |
| `CLASSIFIER_MODEL` | Модель классификатора (по умолч. yagpt5_fns:latest для ollama) |
| `GIGACHAT_MODEL` | Модель GigaChat для классификатора (по умолч. GigaChat-2-Max) |
| `API_KEY_GIGACHAT` | API-ключ GigaChat (Bearer-токен, для LlamaGigaChat в engine_rag.py и классификатора) |
| `QDRANT_COLLECTION` | Имя коллекции Qdrant (по умолч. fns_collection) |
| `RERANKER_DEVICE` | Устройство для bge-m3 reranker (None → CPU, "cuda" → GPU, только eval) |
| `DEBUG_MODE` | Включение диагностического дампа (True/False, по умолч. False) |

### 10.2. Сервисы Docker Compose

| Сервис | Образ | Порт(ы) | Назначение |
|--------|-------|---------|------------|
| redis | redis:7-alpine | 6381:6379 | Кэш ответов |
| ollama | ollama/ollama:latest | 11434:11434 | LLM (yagpt5_fns) |
| qdrant | qdrant/qdrant:latest | 6333, 6344 | Векторная БД |
| api | multik-core:latest (build .) | 8000:8000 | FastAPI + RAG |

### 10.3. Dockerfile

- База: python:3.13-slim.
- Системные зависимости: Chromium (для Playwright), библиотеки GUI.
- PyTorch CPU (torch==2.4.1, --index-url cpu).
- Ключевые пакеты: llama-index-core==0.10.55, qdrant-client==1.9.0, sentence-transformers==3.1.1, fastapi==0.115.0.
- Предзагрузка NLTK словарей (scripts/setup_nltk.py).
- Установка Chromium для Playwright (`playwright install chromium`).
- CMD: `python main.py` (устарело; актуальный запуск через docker-compose: `uvicorn app.rag.main_api:app`).


## 11. Инструкции по обновлению

### 11.1. Добавление нового документа

1. Добавить запись в `documents.json`:
   ```json
   {
     "id": "123-fz",
     "number": "123-ФЗ",
     "date": "2024-01-15",
     "title": "О внесении изменений...",
     "type": "Федеральный закон",
     "enabled": true
   }
   ```

2. Запустить полный пайплайн ingestion:
   ```bash
   venv/bin/python scripts/run_pipeline.py --all
   ```
   Или пошагово:
   ```bash
   venv/bin/python scripts/download_documents.py         # скачивание
   venv/bin/python scripts/run_pipeline.py --ocr          # OCR fallback (если нужно)
   venv/bin/python scripts/run_pipeline.py --convert      # HTML→Markdown
   venv/bin/python scripts/run_pipeline.py --parse        # структурный парсинг
   venv/bin/python scripts/run_pipeline.py --chunk        # чанкинг
   ```

3. Переиндексация в Qdrant:
   ```bash
   venv/bin/python -m app.chunking.create_index_qdrant_chunks
   ```

4. (Опционально) сброс кэша Redis: `redis-cli -p 6381 FLUSHALL`.

### 11.2. Полный цикл перезапуска (локально)

```bash
# 1. Поднять сервисы (Ollama, Qdrant, Redis)
docker compose up -d redis qdrant ollama

# 2. Убедиться, что модель загружена
curl http://localhost:11434/api/tags

# 3. Запустить API локально
venv/bin/python -m uvicorn app.rag.main_api:app --reload --port 8000
```

### 11.3. Полный цикл (Docker)

```bash
docker compose up -d --build
```

Проверка:
```bash
curl http://localhost:8000
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query":"Что такое НДС?"}'
```

## 12. Оценка retrieval (eval)

### 12.1. Обзор

Модуль: `scripts/eval_retrieval.py`

Офлайн-харнесс для оценки качества поиска (retrieval). Не требует запуска LLM.

### 12.2. Golden set (актуальное состояние)

Основной набор — два раздельных golden set, полученных после чистки sig-match шума:

| Файл | Запросов | Пар (query, chunk) | Описание | Дата фиксации |
|------|---------|-------------------|----------|--------------|
| `eval/manual/qrels_user_clean.jsonl` | 156 | 339 | Смысловые USER-запросы, очищены от мусора sig-match (см. док. `docs/RAG_EVAL_SUMMARY_2026.md`) | 2026-10-14 |
| `eval/manual/qrels_exact_clean.jsonl` | 30 | 88+ | Точные реквизиты (статья, пункт), очищены от over-теггинга больших статей | 2026-10-14 |

**Формат**: JSONL, каждое сообщение — `{"question": "...", "relevant_chunk_ids": [...]}`.
**Корпус**: Qdrant `fns_collection`, 5446 чанков.
**Метрики**: Recall@k, Hit@k, MAP@10, MRR@10, nDCG@10.
**Варианты**: `dense` (векторный FRIDA), `bm25` (ранговая модель), `hybrid` (weighted RRF, α/β), `hybrid_rerank` (reranker bge-m3).

### 12.3. Результаты eval

#### БЛОК A: Изолированный dense-прогон (ROOT_CAUSE_TABLE.md, 2026-10-21)

Запуск `scripts/eval_retrieval.py --mode dense` на исправленных qrels (191 запрос, корпус 5446 чанков, Qdrant read-only).

| Метрика | dense |
|---------|-------|
| R@5 | **0.748** |
| Hit@5 | **0.838** |
| MRR@10 | **0.739** |
| nDCG@10 | **0.711** |
| Провалов (hit@5=0) | 25 / 191 (13.1%) |

(Источник: `eval/manual/ROOT_CAUSE_TABLE.md`, файл закоммичен в репозиторий. Параметры: match-level = segment, top-k = {1,3,5,10}.)

#### БЛОК B: Прогоны полного набора (198 запросов) — сведения из eval-контейнера

> **⚠️ Важно:** первичные отчёты (`eval/eval_report.*`) на хосте **отсутствуют**.  
> Числа ниже восстановлены из логов ранее выполненных eval-прогонов (контейнерная среда) и  
> **не являются независимо верифицируемыми** без перезапуска.  
> Сопоставимость qrels/наборов/параметров между строками таблицы — **не проверена**,  
> поэтому вывод «dense лучше hybrid» (или наоборот) из этих цифр — **неправомерен**.

| Конфиг | R@5 | nDCG@10 | MRR@10 | Hit@5 | Lat mean (ms) | Lat p95 (ms) |
|--------|-----|---------|--------|-------|:-------------:|:------------:|
| **dense** (1.0/0.0) | **0.7309** | **0.7213** | **0.7803** | **0.8535** | 260 | 350 |
| hybrid 0.5/0.5 | 0.6730 | 0.6599 | 0.6891 | 0.8232 | 265 | 354 |
| hybrid\_rerank russian-msmarco | 0.6638 | 0.6450 | 0.6886 | 0.8182 | 4763 | 5400 |
| **exact-30:** hybrid\_rerank russian-msmarco | 0.6389 | 0.5829 | 0.5537 | 0.6667 | 4660 | 5300 |

> **Ключевой вывод (предварительный):** на полном наборе (198 запросов) **dense** (1.0/0.0, без BM25/rerank)  
> показал лучшие метрики по всем показателям при латентности ~260 мс. Reranker (bge-m3) даёт прирост точности  
> только на EXACT-подмножестве (30 запросов), но ценой ~4.5 с. Для USER-запросов reranker не улучшает  
> результат, поэтому отключён в FAST-TRACK USER (см. секцию 8.1).

### 12.4. Запуск

```bash
# Основной eval-харнесс (офлайн, без LLM)
venv/bin/python scripts/eval_retrieval.py --mode dense --top-k 10

# Изолированный анализ провалов (dense coverage top-30)
venv/bin/python scripts/eval_retrieval_dense_detailed.py

# Доступные режимы: dense, bm25, hybrid (weighted RRF), hybrid_rerank (reranker)
# Результат: eval/eval_report.json + eval/eval_report.md
```

### 12.5. Вспомогательные утилиты

| Скрипт | Назначение |
|--------|-----------|
| `scripts/build_eval_set.py` | Сборка golden set (для будущего расширения) |
| `scripts/clean_user_qrels.py` | Чистка USER-блока: удаление шумных срабатываний sig-match |
| `scripts/clean_exact_qrels.py` | Чистка EXACT-блока: устранение over-теггинга больших статей |
