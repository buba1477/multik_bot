#!/usr/bin/env python3
"""Запуск только step_pdf_fallback (OCR для scanned PDF без HTML). 
Без download, без resolve, без конвертации HTML.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import step_pdf_fallback

stats = step_pdf_fallback(force=False)
print(f"\n=== Результат PDF fallback ===")
print(f"  Всего PDF: {stats['total']}")
print(f"  Scanned (OCR): {stats['scanned']}")
print(f"  OCR OK: {stats['ocr_ok']}")
print(f"  OCR FAIL: {stats['ocr_fail']}")
print(f"  PDF with text layer: {stats['text_pdf']}")
print(f"  pdftotext OK: {stats['pdftotext_ok']}")
print(f"  pdftotext FAIL: {stats['pdftotext_fail']}")
print(f"  Skipped (have HTML): {stats['skipped']}")