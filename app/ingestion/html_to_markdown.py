"""Универсальный lossless HTML -> Markdown конвертер нормативных документов.

Архитектура: raw HTML -> universal parser -> lossless Markdown/text.
Структурная разметка (document/chapter/article/appendix...) - ОТДЕЛЬНЫЙ слой,
делается позже, по Markdown, а не зашита в заголовки конвертера.

Правила:
  - Структура только из DOM. НЕ угадываем по словам (Глава/Статья/Раздел/Приказ/...).
  - h1-h6 -> pandoc даёт #..###### сам (их не трогаем).
    Вендорский класс "H" (generic heading marker) -> фиксированный уровень ##
    (плоский DOM; настоящую иерархию восстанавливает структурный парсер позже).
  - ol/ul -> настоящие Markdown-списки; plain <p> остаются абзацами (1), 2., а)
    НЕ превращаются в списки, если DOM не <ol>/<ul>.
  - Таблицы -> безопасное построчное текстовое представление (ячейки через |):
    сохраняет строки/ячейки/порядок, не теряется ни при каких обёртках.
  - Удаляется только однозначный технический мусор (script/style/head/span-обёртки/
    k6clnthook/<hr>/атрибуты). Текст не удаляется "за необычность".
"""
import re
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from bs4 import BeautifulSoup

from app.ingestion import pdf_ocr

PANDOC = "/usr/bin/pandoc"
H_HEADING_LEVEL = 2

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
TMP_CONVERT = PROJECT_DIR / ".tmp_convert"


def _silent_unlink(p: Path) -> None:
    """Безопасно удалить файл, игнорируя отсутствие."""
    try:
        p.unlink(missing_ok=True)
    except OSError:
        pass


def _make_temp_file(suffix: str = ".html") -> Path:
    """Создать пустой временный файл в .tmp_convert/ и вернуть путь.

    Файл существует (создан), но пуст — вызывающий должен записать содержимое.
    """
    TMP_CONVERT.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=suffix, dir=str(TMP_CONVERT))
    os.close(fd)
    return Path(path)

NO_CONTENT_TAGS = ["script", "style", "noscript", "meta", "link", "head", "base",
                   "template", "iframe", "object", "embed"]
WRAP_TAGS = ["span", "div", "font"]


def _tables_to_rows(soup: BeautifulSoup) -> None:
    for tbl in soup.find_all("table"):
        container = soup.new_tag("div")
        for tr in tbl.find_all("tr"):
            cells = [re.sub(r"[ \t\n\r]+", " ", td.get_text(" ", strip=True)).strip()
                     for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            p = soup.new_tag("p")
            p.string = " | ".join(cells) if cells else ""
            container.append(p)
        tbl.replace_with(container)


def sanitize_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for t in soup.find_all(NO_CONTENT_TAGS):
        t.decompose()
    for hr in soup.find_all("hr"):
        hr.decompose()

    # Удаление UI-элементов pravo.gov.ru (интерфейс, не контент)
    # Навигация по страницам
    for nav in soup.select("div.page-navigation"):
        nav.decompose()
    # Элементы выбора размера страницы (содержат "ВСЕ")
    for ps in soup.select("div.document-page-size"):
        ps.decompose()
    # Модальное окно "Отправить документ"
    for modal in soup.select("div#emailDlg"):
        modal.decompose()
    # Строки с номером страницы в таблицах изображений
    for tr in soup.select("table.document-images tr.notforprint"):
        tr.decompose()
    # Ссылки на скачивание PDF
    for a in soup.select('a[href*="/file/pdf?"]'):
        a.decompose()
    # Футер
    for ft in soup.select("footer.notforprint"):
        ft.decompose()

    _tables_to_rows(soup)

    for p in soup.find_all("p"):
        if "H" in (p.get("class") or []):
            txt = p.get_text(" ", strip=True)
            if txt:
                h = soup.new_tag(f"h{H_HEADING_LEVEL}")
                h.string = txt
                p.replace_with(h)

    for t in soup.find_all(WRAP_TAGS):
        t.unwrap()

    for el in soup.find_all(True):
        if not el.attrs:
            continue
        keep = el.attrs.get("href") if el.name == "a" and "href" in el.attrs else None
        el.attrs.clear()
        if keep is not None:
            el["href"] = keep

    return soup.prettify()


def clean_markdown(md: str) -> str:
    md = re.sub(r"\[([^\]]*)\]\(k6clnthook:[^)]*\)", r"\1", md)
    md = re.sub(r"k6clnthook://&?(?:amp;)?nd=\d+", "", md)
    md = re.sub(r"(?m)^[ \t]*Complex([ \t][^\n]*)?[ \t]*$", "", md)
    md = re.sub(r"<[^>]+>", "", md)
    md = re.sub(r"\[(?:TABLE|PH|CAPTION|FIGURE)\]\n?", "", md)
    md = re.sub(r"(?m)^[ \t]*(?:(?:-{3,})|(?:_+))[ \t]*$", "", md)
    md = re.sub(r"[ \t]{2,}", " ", md)
    md = re.sub(r" +([.,;:)\)])", r"\1", md)
    md = re.sub(r"\( +", "(", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def convert(fname: str, in_dir: Path = None, out_dir: Path = None) -> None:
    in_dir = in_dir or PROJECT_DIR / "raw_html"
    out_dir = out_dir or PROJECT_DIR / "markdown"
    src = in_dir / fname
    if not src.exists():
        raise FileNotFoundError(src)
    html_bytes = src.read_bytes()
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html = html_bytes.decode("windows-1251", "replace")
    clean_html = sanitize_html(html)

    tmp = _make_temp_file(".clean.html")
    try:
        tmp.write_bytes(clean_html.encode("utf-8"))

        out = out_dir / (src.stem + ".md")
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [PANDOC, "--from=html", "--to=gfm-raw_html", "--wrap=none", "-o", str(out), str(tmp)],
            check=True,
        )

        md = out.read_text(encoding="utf-8")
        out.write_text(clean_markdown(md), encoding="utf-8")
        print(f"done: {src.name} -> {out.relative_to(out_dir)}")
    finally:
        _silent_unlink(tmp)


def convert_from_text(text: str, doc_id: str, out_dir: Path | None = None) -> Path:
    """Сохранить извлечённый текст (OCR) как Markdown.

    Args:
        text: Текст для сохранения (обычно из OCR сканированного PDF).
        doc_id: Идентификатор документа (без расширения).
        out_dir: Директория для .md файла (по умолчанию markdown/).

    Returns:
        Путь к созданному .md файлу.
    """
    out_dir = out_dir or PROJECT_DIR / "markdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{doc_id}.md"
    md = clean_markdown(text)
    # Если clean_markdown уже удалил лишнее, но не добавил переносы:
    # гарантируем завершающий перевод строки
    if not md.endswith("\n"):
        md += "\n"
    out.write_text(md, encoding="utf-8")
    print(f"done (OCR): {doc_id} -> {out.relative_to(out_dir)}")
    return out



def batch_convert(
    in_dir: Path | None = None,
    out_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> list[Path]:
    """Конвертировать все .html файлы из in_dir в Markdown в out_dir.

    Если для PDF нет соответствующего HTML-файла, проверяет PDF
    на наличие текстового слоя. Если PDF сканированный — запускает OCR.

    Args:
        in_dir: Директория с HTML-файлами (по умолчанию raw_html/).
        out_dir: Директория для Markdown (по умолчанию markdown/).
        raw_dir: Директория с PDF-файлами (по умолчанию raw/).

    Returns:
        Список созданных .md файлов.
    """
    in_dir = in_dir or PROJECT_DIR / "raw_html"
    out_dir = out_dir or PROJECT_DIR / "markdown"
    raw_dir = raw_dir or PROJECT_DIR / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    # Шаг 1: HTML -> Markdown
    html_files = sorted(in_dir.glob("*.html"))
    html_ids = set()
    for src in html_files:
        if src.name.startswith("__"):
            continue
        html_ids.add(src.stem)
        tmp = None
        try:
            html_bytes = src.read_bytes()
            try:
                html = html_bytes.decode("utf-8")
            except UnicodeDecodeError:
                html = html_bytes.decode("windows-1251", "replace")
            clean_html = sanitize_html(html)

            tmp = _make_temp_file(".clean.html")
            tmp.write_bytes(clean_html.encode("utf-8"))

            out = out_dir / (src.stem + ".md")
            subprocess.run(
                [PANDOC, "--from=html", "--to=gfm-raw_html", "--wrap=none", "-o", str(out), str(tmp)],
                check=True,
            )

            md = out.read_text(encoding="utf-8")
            out.write_text(clean_markdown(md), encoding="utf-8")
            results.append(out)
            print(f"done: {src.name} -> {out.relative_to(out_dir)}")
        except Exception as exc:
            errors.append(f"HTML {src.name}: {exc}")
            print(f"FAIL: {src.name}: {exc}")
        finally:
            if tmp is not None:
                _silent_unlink(tmp)

    if html_files:
        summary = f"HTML -> Markdown: {len(results)} uspeshno, {len(errors)} oshibok"
        print(summary)

    # Шаг 2: OCR fallback для сканированных PDF + fallback для текстовых PDF с плохим HTML
    # Для сканированных PDF используется OCR-текст, даже если HTML присутствует,
    # т.к. HTML на pravo.gov.ru для таких документов — только интерфейс просмотра
    # изображений, а не текст документа.
    # Для текстовых PDF: если HTML дал непригодный Markdown (<= 500 символов),
    # извлекаем текст напрямую из PDF через pdftotext.
    pdf_files = sorted(raw_dir.glob("*.pdf"))
    ocr_count = 0
    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        try:
            if not pdf_ocr.is_scanned_pdf(pdf_path):
                if doc_id not in html_ids:
                    print(f"SKIP: {pdf_path.name} (текстовый PDF, нет HTML)")
                    continue

                # Текстовый PDF с HTML — проверяем, что Markdown не пустой
                md_path = out_dir / f"{doc_id}.md"
                if md_path.exists():
                    md_text = md_path.read_text(encoding="utf-8")
                    if len(md_text.strip()) > 500:
                        continue  # Markdown нормальный, оставляем HTML-версию

                # HTML дал непригодный Markdown — используем текст из PDF
                print(f"FALLBACK: {pdf_path.name} (текстовый PDF, HTML непригоден)...")
                result = subprocess.run(
                    ["pdftotext", str(pdf_path), "-"],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0 and len(result.stdout.strip()) > 200:
                    text = result.stdout
                    out = convert_from_text(text, doc_id, out_dir=out_dir)
                    results.append(out)
                else:
                    print(f"SKIP: {pdf_path.name} (pdftotext не дал текста)")
                continue

            print(f"OCR: {pdf_path.name} (сканированный PDF)...")
            text = pdf_ocr.ocr_pdf(pdf_path)
            out = convert_from_text(text, doc_id, out_dir=out_dir)
            results.append(out)
            ocr_count += 1
        except Exception as exc:
            errors.append(f"OCR {pdf_path.name}: {exc}")
            print(f"FAIL OCR: {pdf_path.name}: {exc}")

    if ocr_count:
        print(f"OCR fallback: {ocr_count} dokumentov obrabotano")

    if errors:
        print("Ошибки:")
        for e in errors:
            print(f"  - {e}")
    return results

if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "79-FZ.html")
