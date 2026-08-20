# Pipeline загрузки НПА

Документ описывает фактическое состояние pipeline на момент последней проверки.

Дата: 19.08.2026

---

## 1. Общая схема

```
documents.json
    │
    ▼
scripts/download_documents.py   ←── app/pdf_ocr.py (классификация PDF)
    │                                     │
    ├── app/publication_api.py            ├── Сканированные PDF
    │       │                             │      OCR → Markdown
    │       ▼                             │
    │   publication.pravo.gov.ru          └── Текстовые PDF
    │       │                                    │
    │       ├── Документ найден (eoNumber)        ▼
    │       │       ├── get_document(eo)   scripts/run_pipeline.py
    │       │       └── download_pdf(eo, dest)     │
    │       │         → raw/<id>.pdf              ├── Шаг 2: HTML → Markdown
    │       │                                      │     (html_to_markdown.py)
    │       └── НЕ найден (DocumentNotFoundError)  │
    │               │                             ├── Шаг 3 (опционально):
    │               ▼                             │     OCR fallback для
    │           app/pravo_resolver.py              │     сканированных PDF
    │               │                             │   (pdf_ocr.py)
    │               ▼                             │
    │           pravo.gov.ru/proxy/ips/            ├── Шаг 4: Markdown → структура
    │               │                             │   (markdown_structure_parser.py)
    │               ├── resolve_document → nd      │
    │               ├── find_latest_rdk → rdk      └── Шаг 5: Структура → чанки
    │               └── print_url → HTML →              (legal_chunker.py)
    │                   Playwright + bundled Chromium → raw/<id>.pdf
    │
    └── app/resolved_documents.json (кэш)
```

---

## 2. Роль каждого файла

### `documents.json`

- **Зачем**: реестр документов, которые нужно скачать. Это точка входа — единственное место, где пользователь объявляет новый НПА.
- **Кто вызывает**: `scripts/download_documents.py` (функция `main()`).
- **Вход**: не принимает — файл читается.
- **Выход**: список словарей с полями `id`, `type`, `number`, `date`, `title`, `source`, `enabled`.
- **Основной/резервный**: основной реестр.

### `scripts/download_documents.py`

- **Зачем**: оркестратор — читает `documents.json`, для каждого документа определяет способ разрешения (publication API или legacy-резерв), скачивает PDF инкрементально, обновляет кэш.
- **Кто вызывает**: пользователь напрямую (`python scripts/download_documents.py`).
- **Вход**: `documents.json` (registry), `app/resolved_documents.json` (кэш).
- **Выход**: PDF в `raw/<id>.pdf`, обновлённый `app/resolved_documents.json`.
- **Основной/резервный**: основной запускаемый скрипт.

### `app/publication_api.py`

- **Зачем**: работа с официальным JSON API портала «Официальное опубликование правовых актов» (publication.pravo.gov.ru).
- **Кто вызывает**: `scripts/download_documents.py` (импортируется как `import app.publication_api as pub`).
- **Вход**: реквизиты документа (словарь с `number`, `date`, `type`).
- **Выход**: JSON-объект документа API (`eoNumber`, `number`, `documentDate`, `pagesCount`, `pdfFileLength` и т.д.) или исключение (`DocumentNotFoundError`, `DocumentMismatchError`, `PublicAPIError`).
- **Основной/резервный**: основной путь для современных документов. Не является отдельным запускаемым загрузчиком (см. п. 3).

### `app/pravo_resolver.py`

- **Зачем**: поиск документов в устаревшей HTML-системе `/proxy/ips/` pravo.gov.ru — резерв для актов, отсутствующих в официальном API публикации.
- **Кто вызывает**: `scripts/download_documents.py` (импортируется как `import app.pravo_resolver as legacy`).
- **Вход**: номер, название, дата документа.
- **Выход**: словарь с `nd`, `title`, `name`, `status`, `date` или исключение `DocumentNotFoundError`.
- **Основной/резервный**: **только резервный** (fallback). В модуле есть явный маркер `DEPRECATED / legacy RESERVE`.

### `app/resolved_documents.json`

- **Зачем**: кэш разрешения (какой метод использован для каждого документа), состояния ревизий и метаданных скачанных PDF.
- **Кто вызывает**: `scripts/download_documents.py` (функции `load_cache`, `save_cache`, `persist_entry`), также `app/pravo_resolver.py` (только для чтения legacy-кэша).
- **Вход**: словарь `{doc_id: {method, resolved_at, detail, revision, downloaded_at, pdf_path, pdf_size, pdf_pages}}`.
- **Выход**: тот же словарь, записанный атомарно через `.tmp` → `os.replace`.
- **Основной/резервный**: служебный кэш, не предназначен для ручного редактирования.

### `raw/`

- **Зачем**: директория для хранения скачанных PDF-файлов.
- **Кто вызывает**: `scripts/download_documents.py` (функция `_download_to_tmp`, `download_one`).
- **Вход**: PDF-байты.
- **Выход**: `raw/<id>.pdf`.
- **Основной/резервный**: хранилище артефактов.
### `app/ingestion/pdf_ocr.py`

- **Зачем**: определяет, является ли PDF сканированным (без текстового слоя), и запускает OCR через ocrmypdf + Tesseract для извлечения текста.
- **Кто вызывает**: `scripts/run_pipeline.py` (step `--ocr`), `app/ingestion/html_to_markdown.py` (batch_convert), а также напрямую.
- **Вход**: путь к PDF-файлу (`raw/<id>.pdf`).
- **Выход**: извлечённый текст (str) или классификация (сканированный / текстовый).
- **Зависимости**: `tesseract-ocr` (системный), `ocrmypdf` (pip), `pypdf`.
- **Переменные окружения**: `TESSDATA_PREFIX` — если нестандартный путь к tessdata.
- **Основной/резервный**: OCR fallback — основной для сканированных PDF.
- **Ключевые функции**:
  - `is_scanned_pdf(pdf_path)` → `bool` — проверка, есть ли текстовый слой.
  - `ocr_pdf(pdf_path)` → `str` — выполнить OCR и вернуть текст.
  - `classify_pdf_directory(pdf_dir)` → `dict` — статистика по всем PDF в директории.

### `app/ingestion/html_to_markdown.py`

- **Зачем**: конвертирует очищенный HTML в Markdown через Pandoc, а также сохраняет OCR-текст как Markdown.
- **Кто вызывает**: `scripts/run_pipeline.py` (step `--convert`), напрямую через `convert()` или `batch_convert()`.
- **Вход**: HTML-файлы из `raw_html/` или текст из OCR (через `convert_from_text()`).
- **Выход**: `.md` файлы в `markdown/`.
- **Основной/резервный**: основной конвертер для HTML; для OCR — единственный путь.
- **Ключевые функции**:
  - `convert(fname)` — конвертировать один HTML в Markdown.
  - `convert_from_text(text, doc_id)` — сохранить текст как Markdown (OCR fallback).
  - `batch_convert(in_dir, out_dir, raw_dir)` — пакетная конвертация + OCR fallback.

### `app/ingestion/markdown_structure_parser.py`

- **Зачем**: парсит Markdown-документы в структурные JSON (с заголовками, статьями, главами, параграфами).
- **Кто вызывает**: `scripts/run_pipeline.py` (step `--parse`).
- **Вход**: `.md` файлы из `markdown/`.
- **Выход**: `.json` файлы в `structure/` (линейные ноды, дерево, records).
- **Основной/резервный**: основной.

### `app/chunking/legal_chunker.py`

- **Зачем**: нарезает structure JSON в поисковые чанки (chunks) для RAG-системы.
- **Кто вызывает**: напрямую через `batch_convert()` или `process_markdown_file()`.
- **Вход**: `.json` файлы из `structure/`.
- **Выход**: `.jsonl` файлы в `chunks/` (по одному JSON-объекту на строку, поля: id, title, text, local_img, url).
- **Основной/резервный**: основной чанкер.

### `scripts/run_pipeline.py`

- **Зачем**: оркестратор пайплайна — запускает загрузку, конвертацию, OCR fallback, парсинг и валидацию.
- **Кто вызывает**: пользователь (`python scripts/run_pipeline.py --all` или `--ocr --convert --parse`).
- **Вход**: реестр `documents.json`, PDF в `raw/`, HTML в `raw_html/`.
- **Выход**: Markdown в `markdown/`, structure в `structure/`, чанки в `chunks/`.
- **Флаги**:
  - `--all` — полный пайплайн.
  - `--download` — только скачивание.
  - `--ocr` — классификация PDF + OCR fallback.
  - `--convert` — HTML → Markdown.
  - `--parse` — Markdown → структура + чанки.
---

## 3. Почему `publication_api.py` не является отдельным загрузчиком

`app/publication_api.py` — это библиотека, а не запускаемый скрипт. В нём нет `if __name__ == "__main__"`. Он предоставляет:

- `pub.search_documents(number)` — поиск кандидатов по номеру;
- `pub.resolve_exact(record)` — fail-closed поиск с валидацией даты и типа;
- `pub.get_document(eoNumber)` — детали документа по `eoNumber`;
- `pub.download_pdf(eoNumber, dest)` — скачивание PDF.

Все эти функции вызываются из `download_documents.py`:

```python
import app.publication_api as pub   # scripts/download_documents.py, строка 41

# resolve_doc → основная ветка:
matched = pub.resolve_exact(doc)                    # поиск по реквизитам

# _revision_current → publication:
m = pub.get_document(eo)                            # детали для фингерпринта

# _download_to_tmp → publication:
size = pub.download_pdf(eo, tmp_pdf)               # скачивание PDF
```

Аналогично для `pravo_resolver.py`:

```python
import app.pravo_resolver as legacy  # scripts/download_documents.py, строка 40

# resolve_doc → резервная ветка (при DocumentNotFoundError):
legacy_rec = legacy.resolve_document(doc["number"], doc.get("title"), doc.get("date"))

# _revision_current → legacy:
rev = legacy.find_latest_revision(entry["detail"]["nd"])

# _download_to_tmp → legacy:
latest = legacy.find_latest_rdk(nd)
data = get_bytes(legacy.print_url(nd, rdk))
```

Ни один из этих модулей не предназначен для прямого запуска — они импортируются в `download_documents.py`.

---

## 4. Алгоритм выбора источника

```
documents.json → doc {number, date, type, title}
    │
    ▼
pub.resolve_exact(doc)
    │
    ├── DocumentNotFoundError
    │       │
    │       ▼
    │   legacy.resolve_document(number, title, date)
    │       │
    │       ├── найден (nd) → method = "legacy"
    │       │
    │       └── не найден (DocumentNotFoundError)
    │               → ResolutionError ("не найден ни в API, ни в legacy")
    │
    ├── DocumentMismatchError (неоднозначность / несовпадение реквизитов)
    │       → PublicAPIError / DocumentMismatchError
    │         НЕ переключается на legacy — fail-closed останов
    │
    └── PublicAPIError (сеть недоступна, HTTP-ошибка)
            → PublicAPIError
              НЕ переключается на legacy — fail-closed останов
```

**Ключевое правило**: переключение на legacy происходит **только** при `DocumentNotFoundError` — когда API подтверждённо не содержит документа. Ошибки сети, неоднозначности, несовпадения реквизитов **не** приводят к молчаливому fallback.
---

## 5. Инкрементальное скачивание

```
documents.json
    │
    ▼
resolve_doc(doc) → entry {method, detail, revision?}
    │
    ▼
_revision_current(doc, entry)
    │   publication: pub.get_document(eo) → {id: eo, fingerprint, label}
    │   legacy:      legacy.find_latest_revision(nd) → {id: rdk, label, date}
    │
    ▼
_revision_unchanged(entry, current_rev, out_pdf)
    │
    │   entry["revision"] == current_rev  И  out_pdf.exists()
    │   ├── True  → "unchanged → skip download" (выход)
    │   └── False → "ревизия изменилась / PDF отсутствует → скачивание"
    │
    ▼
_download_to_tmp(doc, entry)
    │   publication: pub.download_pdf(eo, tmp.pdf)
    │   legacy:      get_bytes → Playwright HTML→PDF (tmp.html → tmp.pdf)
    │   validate_pdf(tmp.pdf) → pages
    │
    ▼
os.replace(tmp.pdf, out.pdf)   # атомарная замена
    │
    ▼
entry["revision"] = current_rev
entry["downloaded_at"] = iso_now()
entry["pdf_path"] = str(out.pdf)
entry["pdf_size"] = size
entry["pdf_pages"] = pages
persist_entry(doc["id"], entry, cache_path)
```

**При ошибке** на любом этапе `_download_to_tmp`:
- временные файлы `.new.pdf` и `.new.html` удаляются (`_silent_unlink`);
- целевой `raw/<id>.pdf` **не трогается**;
- кэш **не обновляется** — остаётся предыдущая ревизия.

---

## 6. Как определяется новая редакция

### Через publication API

Вызов `pub.get_document(eo)` возвращает полный объект документа. Формируется фингерпринт:

```python
fingerprint = "|".join([
    eoNumber,
    documentDate,
    publishDateShort,
    pdfFileLength,
    pagesCount,
])
```

Сравнение — по всему словарю `{id, fingerprint, label, publishDateShort, pagesCount}`. Изменение любого поля считается новой редакцией.

### Через legacy (pravo.gov.ru)

Вызов `legacy.find_latest_revision(nd)` возвращает словарь:

```python
{
    "rdk": 98,                              # номер редакции (int)
    "label": "98 - от 08.03.2026 № 52-ФЗ (изм.)",
    "date": "08.03.2026",
}
```

Сравнение — по всему словарю `{id: rdk, label, date}`. Изменение `rdk` (или метки) считается новой редакцией.

### Что хранится в `resolved_documents.json`

```json
{
  "79-FZ": {
    "method": "legacy",
    "resolved_at": "2026-08-17T08:42:50+00:00",
    "detail": {
      "nd": "102088054"
    },
    "revision": {
      "id": 98,
      "label": "98 - от 08.03.2026 № 52-ФЗ (изм.)",
      "date": "08.03.2026"
    },
    "downloaded_at": "2026-08-17T10:13:07+00:00",
    "pdf_path": "/home/amlin04/multik_bot/raw/79-FZ.pdf",
    "pdf_size": 1247979,
    "pdf_pages": 121
  }
}
```

Поле `revision` — это текущая успешно скачанная редакция. Сравнение с `_revision_current()` даёт ответ на вопрос «нужно ли перекачивать PDF».
---

## 7. Пример для трёх текущих документов

| id | number | date | метод | detail |
|---|---|---|---|---|
| `79-FZ` | 79-ФЗ | 27.07.2004 | **legacy** | nd=102088054 |
| `58-FZ` | 58-ФЗ | 27.05.2003 | **legacy** | nd=102081744 |
| `ukaz-112-2005` | 112 | 01.02.2005 | **legacy** | nd=102090878 |

Все три документа — старые (2003–2005), их нет в publication.pravo.gov.ru. Поэтому `resolve_exact` вернул `DocumentNotFoundError`, и они разрешены через legacy-резерв.

Если бы добавить современный документ (например, Федеральный закон от 09.04.2026 № 79-ФЗ, который есть в publication API), метод был бы `publication`, а `detail` содержал бы `eoNumber`.

---

## 8. Как добавить новый НПА

Добавить запись в `documents.json`. Пример:

```json
{
  "id": "fz-150-2026",
  "type": "Федеральный закон",
  "number": "150-ФЗ",
  "date": "01.06.2026",
  "title": "О внесении изменений в отдельные законодательные акты",
  "source": "pravo.gov.ru",
  "enabled": true
}
```

**Правила**:
- `id` — наш внутренний уникальный идентификатор, используется как имя файла (`raw/fz-150-2026.pdf`). Не путать с `eoNumber` или `nd` с pravo.gov.ru.
- `type` — полное название вида документа (например, «Федеральный закон», «Указ Президента Российской Федерации»).
- `number` — номер документа как в официальном тексте (с дефисом, римскими цифрами и т.д.).
- `date` — дата подписания/принятия в формате `ДД.ММ.ГГГГ`.
- `title` — официальное название.
- `source` — всегда `"pravo.gov.ru"`.
- `enabled` — `true` для включения, `false` для пропуска без удаления из реестра.

После добавления — запустить `python scripts/download_documents.py`. Система сама определит, есть ли документ в publication API или нужен legacy-резерв, скачает PDF и обновит кэш.
---

## 9. Что НЕ нужно делать

- ❌ **Не указывать `nd` в `documents.json`**. `nd` — внутренний идентификатор pravo.gov.ru, он определяется автоматически при разрешении документа. Указание вручную нарушает pipeline.
- ❌ **Не искать вручную `eoNumber`**. `eoNumber` — стабильный идентификатор publication API, он возвращается `pub.resolve_exact()`.
- ❌ **Не скачивать PDF вручную** и не класть его в `raw/`. Только `download_documents.py` должен записывать PDF, чтобы кэш ревизий оставался консистентным.
- ❌ **Не редактировать `resolved_documents.json` вручную**. Файл перезаписывается скриптом; ручные правки потеряются или приведут к рассинхронизации.
- ❌ **Не использовать `pravo_resolver.py` как основной путь**. Это устаревший резерв; для современных документов доступных через publication API, legacy не должен вызываться.
- ❌ **Не запускать `publication_api.py` напрямую** — в нём нет точки входа.

---

## 10. Что будет при повторном запуске

| Ситуация | Поведение |
|---|---|
| Редакция не изменилась, PDF на месте | `unchanged → skip download` — ни одного сетевого запроса. |
| Редакция изменилась (новый `rdk` / `fingerprint`) | Скачивается новый PDF во временный файл, проверяется, атомарно заменяет старый. Кэш обновляется. |
| PDF удалён из `raw/`, ревизия в кэше есть | Определяется как «PDF отсутствует» → скачивание заново (той же редакции). |
| Кэш повреждён (невалидный JSON) | `load_cache` возвращает `{}` → первый запуск как для нового документа. |
| Ошибка при скачивании нового PDF (сеть, невалидный PDF) | Старый PDF **не удаляется**, временные файлы подчищаются, кэш не обновляется. |
| `DocumentNotFoundError` и в API, и в legacy | `ResolutionError` — документ не может быть найден никаким способом. |
| `DocumentMismatchError` / `PublicAPIError` | Fail-closed останов — legacy не используется. |

---

## 11. Карта проекта

```
multik_bot/
├── docs/
│   └── DOCUMENT_PIPELINE.md          ← этот файл
├── documents.json                    ← реестр НПА (точка входа)
├── scripts/
│   └── download_documents.py         ← оркестратор скачивания
├── app/
│   ├── publication_api.py            ← основной путь (официальный API)
│   ├── pravo_resolver.py             ← резервный путь (legacy /proxy/ips/)
│   ├── ingestion/
│   │   └── html_to_pdf.py            ← Playwright HTML→PDF конвертация
│   └── resolved_documents.json       ← кэш метода, ревизии, состояния
├── raw/                              ← скачанные PDF (28 шт.)
├── raw_html/                         ← конвертированные HTML (17 шт.)
├── markdown/                         ← Markdown после конвертации (29 .md)
├── structure/                        ← структура JSON после парсинга (29 .json)
├── chunks/                           ← чанки JSONL после нарезки (29 .jsonl)
├── .tmp_convert/                     ← временные файлы html_to_markdown
├── tests/
│   ├── test_downloader.py
│   ├── test_publication_api.py
│   ├── test_pravo_resolver.py
│   ├── test_legal_chunker.py
│   ├── test_markdown_structure_parser.py
│   └── test_pdf_ocr.py              ← тесты OCR fallback
```

---

Документ описывает фактическое состояние pipeline на момент последней проверки.
