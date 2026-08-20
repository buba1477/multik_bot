"""OCR fallback для сканированных PDF без текстового слоя.

Определяет, является ли PDF сканированным (нет текстового слоя),
и запускает OCR через ocrmypdf + Tesseract для извлечения текста.

Зависимости (системные):
  - tesseract-ocr (>= 5.0)
  - tesseract-ocr-rus (русский языковой пакет)
  - Python: ocrmypdf (>= 17.0)

Переменные окружения:
  TESSDATA_PREFIX — если нестандартный путь к tessdata Tesseract.
"""
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)

# Порог: если после очистки от form feed / пробелов / переводов строк
# в PDF остаётся меньше min_text_chars символов — считаем его сканированным.
DEFAULT_MIN_TEXT_CHARS = 50

# Язык OCR по умолчанию
DEFAULT_OCR_LANG = "rus"


# ============================================================================
# Вспомогательные функции
# ============================================================================

def _clean_text(text: str) -> str:
    """Удалить form feed, управляющие символы, лишние пробелы."""
    # Удаляем form feed и прочие непечатные символы, кроме \n и \t
    cleaned = "".join(c for c in text if c.isprintable() or c in "\n\t")
    return cleaned.strip()


def _count_meaningful_chars(text: str) -> int:
    """Количество «полезных» символов: без form feed, пробелов и переводов строк."""
    cleaned = "".join(c for c in text if c not in "\f\n\r\t ")
    return len(cleaned)


def _get_pdf_page_count(pdf_path: str | Path) -> int:
    """Количество страниц в PDF через pypdf (без запуска внешних утилит)."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as exc:
        logger.warning("Не удалось определить число страниц для %s: %s", pdf_path, exc)
        return 0


# ============================================================================
# Определение типа PDF
# ============================================================================

def is_scanned_pdf(pdf_path: str | Path, min_text_chars: int = DEFAULT_MIN_TEXT_CHARS) -> bool:
    """Проверить, является ли PDF сканированным (не содержит текстового слоя).

    Args:
        pdf_path: Путь к PDF-файлу.
        min_text_chars: Минимальное количество полезных символов для
                        признания PDF текстовым.

    Returns:
        True, если PDF сканированный (требуется OCR).
        False, если PDF содержит текстовый слой.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")

    page_count = _get_pdf_page_count(pdf_path)
    logger.info("Проверка %s: %d стр.", pdf_path.name, page_count)

    # Пробуем извлечь текст через pdftotext (быстрее и надёжнее pypdf для этой задачи)
    text = _extract_text_via_pdftotext(pdf_path)

    if text is None:
        # pdftotext не сработал — пробуем pypdf как fallback
        text = _extract_text_via_pypdf(pdf_path)

    meaningful = _count_meaningful_chars(text)
    logger.info("  %s: полезных символов: %d (порог: %d)", pdf_path.name, meaningful, min_text_chars)

    if meaningful < min_text_chars:
        logger.info("  -> PDF СКАНИРОВАННЫЙ (текстовый слой отсутствует или недостаточен)")
        return True
    else:
        logger.info("  -> PDF ТЕКСТОВЫЙ (содержит текстовый слой)")
        return False


def _extract_text_via_pdftotext(pdf_path: Path) -> str | None:
    """Извлечь текст из PDF через pdftotext (poppler-utils)."""
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning("pdftotext вернул код %d: %s", result.returncode, result.stderr.strip())
            return None
    except FileNotFoundError:
        logger.warning("pdftotext не найден в PATH")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pdftotext превысил таймаут для %s", pdf_path.name)
        return None
    except Exception as exc:
        logger.warning("pdftotext ошибка для %s: %s", pdf_path.name, exc)
        return None


def _extract_text_via_pypdf(pdf_path: Path) -> str:
    """Извлечь текст из PDF через pypdf."""
    text_parts = []
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text_parts.append(extracted)
    except Exception as exc:
        logger.warning("pypdf ошибка для %s: %s", pdf_path.name, exc)
    return "\n".join(text_parts)
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as exc:
        logger.warning("Не удалось определить число страниц для %s: %s", pdf_path, exc)
# ============================================================================
# OCR
# ============================================================================

def ocr_pdf(pdf_path: str | Path, lang: str = DEFAULT_OCR_LANG) -> str:
    """Выполнить OCR сканированного PDF и вернуть извлечённый текст.

    Работа:
      1. Создаёт временный PDF с OCR-слоем через ocrmypdf.
      2. Извлекает текст из результата через pdftotext.
      3. Удаляет временные файлы.

    Args:
        pdf_path: Путь к сканированному PDF.
        lang: Язык(и) OCR (через '+', например 'rus+eng').

    Returns:
        Извлечённый текст (str).

    Raises:
        FileNotFoundError: PDF не найден.
        RuntimeError: OCR не удался.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF не найден: {pdf_path}")

    page_count = _get_pdf_page_count(pdf_path)
    logger.info("OCR для %s: %d стр., язык=%s", pdf_path.name, page_count, lang)

    # Ищем ocrmypdf: сначала через shutil (полный путь), потом просто 'ocrmypdf'
    ocrmypdf_exe = shutil.which("ocrmypdf")
    if not ocrmypdf_exe:
        # fallback: maybe it's in the venv bin
        venv_bin = Path(sys.executable).parent
        candidate = venv_bin / "ocrmypdf"
        if candidate.is_file():
            ocrmypdf_exe = str(candidate)
        else:
            ocrmypdf_exe = "ocrmypdf"  # hope it's in PATH

    # ocrmypdf требует TESSDATA_PREFIX в окружении если он нестандартный
    env = os.environ.copy()
    tess_prefix = os.environ.get("TESSDATA_PREFIX")
    if tess_prefix:
        env["TESSDATA_PREFIX"] = tess_prefix

    # Создаём временный PDF с OCR-слоем
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        ocr_pdf_path = Path(tmp_file.name)

    try:
        logger.info("  Запуск ocrmypdf (lang=%s)...", lang)
        subprocess.run(
            [
                ocrmypdf_exe,
                "--language", lang,
                "--output-type", "pdf",
                "--skip-text",
                "-d",
                str(pdf_path),
                str(ocr_pdf_path),
            ],
            check=True,
            timeout=600,
            capture_output=True,
            text=True,
            env=env,
        )
        logger.info("  ocrmypdf завершён успешно")
    except subprocess.TimeoutExpired:
        _safe_unlink(ocr_pdf_path)
        raise RuntimeError(f"OCR превысил таймаут для {pdf_path.name}")
    except subprocess.CalledProcessError as exc:
        _safe_unlink(ocr_pdf_path)
        error_msg = exc.stderr or exc.stdout or ""
        raise RuntimeError(
            f"OCR не удался для {pdf_path.name}: {error_msg[:500]}"
        )

    # Извлекаем текст из OCR-результата
    try:
        text = _extract_text_via_pdftotext(ocr_pdf_path)
        if text is None:
            # fallback: pypdf
            text = _extract_text_via_pypdf(ocr_pdf_path)

        text = _clean_text(text)
        text_len = len(text)
        logger.info("  OCR текст: %d символов", text_len)

        if text_len < 20:
            logger.warning("  OCR вернул очень мало текста (%d символов)", text_len)

        return text
    finally:
        _safe_unlink(ocr_pdf_path)


def _safe_unlink(path: Path) -> None:
    """Безопасно удалить файл."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Не удалось удалить временный файл %s: %s", path, exc)


# ============================================================================
# Пакетная проверка
# ============================================================================

def classify_pdf_directory(pdf_dir: str | Path, min_text_chars: int = DEFAULT_MIN_TEXT_CHARS) -> dict:
    """Проверить все PDF в директории и вернуть статистику.

    Returns:
        Словарь с ключами:
          - scanned: список имён PDF, определённых как сканированные
          - text_pdf: список имён PDF с текстовым слоем
          - total: общее количество PDF
    """
    pdf_dir = Path(pdf_dir)
    result = {"scanned": [], "text_pdf": [], "total": 0, "errors": []}

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        result["total"] += 1
        try:
            if is_scanned_pdf(pdf_path, min_text_chars=min_text_chars):
                result["scanned"].append(pdf_path.name)
            else:
                result["text_pdf"].append(pdf_path.name)
        except Exception as exc:
            result["errors"].append(f"{pdf_path.name}: {exc}")
            logger.error("Ошибка проверки %s: %s", pdf_path.name, exc)

    return result
