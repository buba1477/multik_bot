"""Конвертация IPS print HTML → PDF через Playwright + bundled Chromium.

Заменяет отсутствующий PDF для документов, полученных через legacy-резерв,
а также служит fallback для publication-документов, если оригинальный PDF
недоступен, но IPS HTML найден и является textual.

Установка bundled Chromium:
    python -m playwright install chromium

Зависимость:
    playwright==1.62.0
"""

from pathlib import Path

from playwright.sync_api import sync_playwright


def convert_html_to_pdf(html_path: Path, pdf_path: Path) -> dict:
    """Конвертировать локальный HTML в PDF через bundled Chromium.

    Args:
        html_path: Путь к локальному HTML-файлу (file:// URI).
        pdf_path: Путь для сохранения сгенерированного PDF.

    Returns:
        Словарь с метриками сгенерированного PDF:
            {
                "title": str | None,   # <title> HTML
                "pages": int,          # количество страниц
                "pdf_size": int,       # размер в байтах
                "html_len": int,       # длина HTML в байтах
            }

    Raises:
        RuntimeError: Если Playwright/Chromium недоступен или конвертация не удалась.
    """
    if not html_path.exists():
        raise FileNotFoundError(f"HTML не найден: {html_path}")

    import os

    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        try:
            context = browser.new_context(
                viewport={"width": 1024, "height": 768},
                locale="ru-RU",
                timezone_id="Europe/Moscow",
                device_scale_factor=1,
            )
            try:
                page = context.new_page()
                try:
                    page.goto(html_path.as_uri(), wait_until="networkidle")
                    page.wait_for_timeout(2000)

                    title = page.title()
                    html_len = html_path.stat().st_size

                    page.pdf(
                        path=str(pdf_path),
                        format="A4",
                        print_background=True,
                        margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
                    )

                    pdf_size = pdf_path.stat().st_size
                    pages = _count_pdf_pages(pdf_path)

                    return {
                        "title": title,
                        "pages": pages,
                        "pdf_size": pdf_size,
                        "html_len": html_len,
                    }
                finally:
                    page.close()
            finally:
                context.close()
        finally:
            browser.close()


def _count_pdf_pages(pdf_path: Path) -> int:
    """Определить количество страниц в PDF через PyPDF."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        raise RuntimeError(f"Не удалось определить количество страниц в PDF: {pdf_path}")