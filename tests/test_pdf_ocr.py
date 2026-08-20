"""Тесты для OCR fallback (pdf_ocr).

Проверяют:
  - Определение сканированного PDF (is_scanned_pdf)
  - Извлечение текста через OCR (ocr_pdf)
  - Классификацию директории (classify_pdf_directory)

Зависимости (system): tesseract-ocr, tesseract-ocr-rus
Зависимости (pip): ocrmypdf, pypdf
"""
from pathlib import Path

import pytest

from app.ingestion.pdf_ocr import (
    _clean_text,
    _count_meaningful_chars,
    _get_pdf_page_count,
    classify_pdf_directory,
    is_scanned_pdf,
    ocr_pdf,
)

# ============================================================================
# Fixtures
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "raw"


# ============================================================================
# Вспомогательные функции
# ============================================================================

def test_clean_text() -> None:
    """Форм-фид и управляющие символы удаляются."""
    assert _clean_text("Привет\nмир") == "Привет\nмир"
    assert _clean_text("Привет\fмир") == "Приветмир"
    assert _clean_text(" \t\n ") == ""


def test_count_meaningful_chars() -> None:
    """Считаются только печатные символы, не пробелы."""
    assert _count_meaningful_chars("abc def") == 6
    assert _count_meaningful_chars("   \n\t") == 0
    assert _count_meaningful_chars("УКАЗ\nПРЕЗИДЕНТА") == 14


def test_get_pdf_page_count() -> None:
    """Количество страниц для реального PDF."""
    pdf_path = RAW_DIR / "ukaz-613.pdf"
    if not pdf_path.exists():
        pytest.skip("raw/ukaz-613.pdf не найден")
    count = _get_pdf_page_count(pdf_path)
    assert count > 0, "Должны получить >0 страниц"
    assert isinstance(count, int)
# ============================================================================
# Определение типа PDF
# ============================================================================

def test_is_scanned_scanned_pdf() -> None:
    """ukaz-613.pdf должен быть определён как сканированный."""
    pdf_path = RAW_DIR / "ukaz-613.pdf"
    if not pdf_path.exists():
        pytest.skip("raw/ukaz-613.pdf не найден")
    assert is_scanned_pdf(pdf_path) is True


def test_is_scanned_text_pdf() -> None:
    """79-FZ.pdf (текстовый PDF) не должен быть определён как сканированный."""
    pdf_path = RAW_DIR / "79-FZ.pdf"
    if not pdf_path.exists():
        pytest.skip("raw/79-FZ.pdf не найден")
    assert is_scanned_pdf(pdf_path) is False


def test_is_scanned_file_not_found() -> None:
    """Несуществующий файл — FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        is_scanned_pdf("/nonexistent/pdf.pdf")


def test_classify_pdf_directory() -> None:
    """Классификация всей директории raw/ должна дать минимум 1 сканированный PDF."""
    if not RAW_DIR.exists():
        pytest.skip("raw/ не найдена")
    result = classify_pdf_directory(RAW_DIR)
    assert result["total"] > 0
    assert len(result["scanned"]) > 0, "Должен быть хотя бы 1 сканированный PDF"
    assert len(result["text_pdf"]) > 0, "Должен быть хотя бы 1 текстовый PDF"
    assert len(result["errors"]) == 0, f"Не должно быть ошибок: {result['errors']}"
# ============================================================================
# OCR (интеграционные тесты)
# ============================================================================

class TestOCR:
    """Тесты OCR — требуют установленного tesseract-ocr-rus и TESSDATA_PREFIX."""

    @pytest.fixture(autouse=True)
    def check_env(self) -> None:
        if not RAW_DIR.joinpath("ukaz-613.pdf").exists():
            pytest.skip("raw/ukaz-613.pdf не найден")

    def test_ocr_pdf_text_length(self) -> None:
        """OCR для ukaz-613.pdf должен вернуть > 1000 символов."""
        text = ocr_pdf(RAW_DIR / "ukaz-613.pdf")
        assert len(text) > 1000, f"OCR вернул слишком мало текста: {len(text)}"

    def test_ocr_pdf_contains_keywords(self) -> None:
        """OCR-текст должен содержать ключевые слова указа."""
        text = ocr_pdf(RAW_DIR / "ukaz-613.pdf")
        assert "УКАЗ" in text
        assert "ПРЕЗИДЕНТА" in text
        assert "коррупции" in text.lower()
        assert "273-ФЗ" in text

    def test_ocr_pdf_page_count(self) -> None:
        """Страницы ukaz-613.pdf."""
        page_count = _get_pdf_page_count(RAW_DIR / "ukaz-613.pdf")
        assert page_count == 10, f"ukaz-613.pdf должен иметь 10 страниц, получено {page_count}"

    def test_ocr_pdf_not_scanned(self) -> None:
        """OCR для текстового PDF должен работать."""
        text = ocr_pdf(RAW_DIR / "79-FZ.pdf")
        assert len(text) > 100, "Текстовый PDF должен давать текст через OCR"


# ============================================================================
# Пограничные случаи
# ============================================================================

def test_ocr_default_lang() -> None:
    """Язык по умолчанию — rus+eng."""
    from app.ingestion.pdf_ocr import DEFAULT_OCR_LANG
    assert DEFAULT_OCR_LANG == "rus+eng"