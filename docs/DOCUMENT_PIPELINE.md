# Pipeline загрузки и обработки НПА

Документ описывает фактическое состояние pipeline на момент последней проверки.

Дата: 20.08.2026

---

## 1. Общая схема

```
documents.json  (содержит nd для legacy-документов)
    │
    ▼
scripts/download_documents.py
    │
    ├── app/publication_api.py           (ОСНОВНОЙ путь)
    │       │
    │       ├── /api/Documents (поиск по number)
    │       ├── /api/Document?eoNumber=… (детали)
    │       ├── /file/pdf?eoNumber=… → raw/<id>.pdf
    │       └── /Document/View/<eoNumber> → raw_html/<id>.html
    │
    └── app/pravo_resolver.py            (LEGACY-резерв)
            │
            ├── /proxy/ips/ (поиск nd по реквизитам)
            ├── print-представление → raw_html/<id>.html
            └── app/ingestion/html_to_pdf.py
                    └── Playwright + bundled Chromium → raw/<id>.pdf


=== RAG pipeline (на основе HTML) ===

raw_html/<id>.html
    │
    ▼
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
    │   → FRIDA-токенизация (~350 токенов target, ~400 max)
    ▼
chunks/<id>.jsonl
    │
    ▼
app/chunking/create_index_qdrant_chunks.py
    │   FRIDA embedding (Sber RoSBERTa)
    │   → загрузка векторов в Qdrant fns_collection
    ▼
Qdrant vector DB
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
| `hf_cache/FRIDA/` | Локальный кэш модели Sber RoSBERTa (FRIDA) для embedding-векторов. Offline-режим. |
| `images_cache/` | Кэш изображений сотрудников для автоподбора фотографий в ответах RAG. |
| `app_audit.log` | Ротируемый лог (10 MB, 5 бэкапов) с request_id для аудита. |


## 3. Разрешение документов (download)

### 3.1. Основной путь: publication.pravo.gov.ru

Модуль: `app/publication_api.py`

1. **Поиск** — `GET /api/Documents?number=<номер>`
   - Ответ: JSON со списком кандидатов (eoNumber, number, documentDate, documentType, title).
   - Параметр number — единственный надёжный фильтр; date и сортировка игнорируются API.

2. **Верификация** — `_pick_unique()`:
   - Сравнение number + date + type со строгими реквизитами из documents.json.
   - Канонизация типов ("Указ" + "Президент Российской Федерации" → "Указ Президента Российской Федерации").
   - **Fail-closed**: при неоднозначности/несовпадении — `DocumentMismatchError` (останов, legacy-резерв НЕ используется).

3. **Детали** — `GET /api/Document?eoNumber=<eoNumber>` 
   - Полная информация о документе (подтверждение реквизитов).

4. **Скачивание**:
   - PDF: `GET /file/pdf?eoNumber=...` → `raw/<id>.pdf` (проверка заголовка `%PDF-`).
   - HTML: `GET /Document/View/<eoNumber>` → `raw_html/<id>.html` (проверка `<!DOCTYPE html>`).

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
| TARGET_TOKENS | 350 |
| MAX_TOKENS | 400 |
| MIN_CHUNK_TOKENS | 40 |

Токенизация — FRIDA (Sber RoSBERTa) tokenizer. Offline. Если модель недоступна → эвристика (~4 символа на токен).

### 6.4. Валидация JSONL

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
3. Проверка лимита токенов (≤400)
4. Загрузка FRIDA (SentenceTransformer: Transformer + Pooling(CLS))
5. Вычисление embeddings (batch=32, префикс "search_document:")
6. Подключение к Qdrant
7. Удаление/пересоздание коллекции fns_collection
8. Загрузка векторов с payload (id, title, text, local_img, source_url)
9. Проверка: количество точек == количество чанков, размерность совпадает
```

### 7.2. Параметры

| Параметр | Значение |
|----------|----------|
| Collection | `fns_collection` |
| Qdrant host | localhost (переопределяется через QDRANT_HOST) |
| Qdrant port | 6333 (переопределяется через QDRANT_PORT) |
| Embedding | FRIDA / Sber RoSBERTa (CLS-pooling) |
| Batch size | 32 |
| Префикс | `search_document:` |

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
Запрос пользователя
  │
  ├── is_chart → DynamicChartEngine (ECharts JSON)
  │
  └── Основной RAG-путь:
        │
        ├── 1. Гибридный поиск
        │     ├── Векторный (FRIDA embedding → Qdrant)
        │     ├── BM25 (rank_bm25, по всей коллекции)
        │     └── Fusion: weighted sum (α=0.7 векторный, β=0.3 BM25)
        │
        ├── 2. Re-rank
        │     └── SentenceTransformerRerank (FRIDA cross-encoder, top_k=8)
        │
        ├── 3. Формирование контекста
        │     └── Сбор текстов source-чанков
        │
        ├── 4. Генерация
        │     └── Ollama (модель "yagpt5_fns:latest")
        │
        └── 5. Пост-обработка
              ├── Замена спецсимволов (HTML → unicode)
              ├── Автоподбор фотографий (по тексту ответа)
              └── Форматирование: **жирный**, списки, таблицы, пустые строки
```

### 8.2. Промпт

Системный промпт задаёт:
- Роль: "ведущий эксперт ФНС России".
- Язык: строго русский.
- Формат: **жирный** для ключевых терминов, маркированные/нумерованные списки, таблицы Markdown.
- Запрет: вымышленных норм, нецензурной лексики, советов по уклонению.
- При нехватке данных: вежливый отказ ("В моих регламентах про это ни слова").

### 8.3. Потоковый ответ

`get_ai_streaming_response()` — асинхронный генератор:
- JSON-объекты по одному на строку: `{"type": "text"|"metadata"|"error"|"end", "content": ...}`.
- Стриминг через Ollama `async chat()`.
- Автоподбор фото на основе текста ответа.

### 8.4. Чарт-режим

`DynamicChartEngine.is_chart_request()` — определяет запросы на графики (по ключевым словам).
Генерирует ECharts JSON-конфигурацию → рендерится на фронтенде.


## 9. API

Модуль: `app/rag/main_api.py`

### 9.1. Эндпоинты

| Путь | Метод | Описание |
|------|-------|----------|
| `/` | GET | HTML-страница чата (Jinja2, templates/base.html) |
| `/chat` | GET | HTML-страница чата (та же) |
| `/ask` | POST | Streaming-ответ (SSE, JSON lines). Параметр: `query`. |

### 9.2. Маршрутизация запросов

```
POST /ask  { query: "..." }
  │
  ├── "мультик" + не ФНС-ключевые → Ollama (yagpt5_fns, num_ctx=4096, temp=0.8)
  │
  └── ФНС-запрос / чарт:
        │
        ├── Redis-кэш (ключ: lower(query), TTL: 86400 сек)
        │   └── попали → stream из кэша
        │
        └── не попали → engine_rag (get_ai_streaming_response)
              └── успех → кэшируем полный ответ
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
| `REDIS_PASSWORD` | Пароль Redis |

### 10.2. Сервисы Docker Compose

| Сервис | Образ | Порт(ы) | Назначение |
|--------|-------|---------|------------|
| redis | redis:7-alpine | 6381:6379 | Кэш ответов |
| ollama | ollama/ollama:latest | 11434:11434 | LLM (yagpt5_fns) |
| qdrant | qdrant/qdrant:latest | 6333, 6344 | Векторная БД |
| api | multik-core:latest (build .) | 8000:8000 | FastAPI + RAG |

### 10.3. Dockerfile

- База: python:3.11-slim.
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

2. Запустить скачивание:
   ```bash
   venv/bin/python scripts/download_documents.py
   ```

3. Конвертация HTML → Markdown:
   ```bash
   venv/bin/python -m app.ingestion.html_to_markdown
   ```

4. Структурный парсинг:
   ```bash
   venv/bin/python -m app.ingestion.markdown_structure_parser
   ```
   (или `batch_parse(markdown_dir='markdown', structure_dir='structure')`)

5. Чанкинг:
   ```bash
   venv/bin/python -m app.chunking.legal_chunker --all
   ```

6. Переиндексация в Qdrant:
   ```bash
   venv/bin/python -m app.chunking.create_index_qdrant_chunks
   ```

7. (Опционально) сброс кэша Redis: `redis-cli -p 6381 FLUSHALL`.

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

