"""PDF -> Markdown конвертер для publication-документов.

Использует Docling для извлечения текста из PDF.
Архитектура:
  raw/<document>.pdf -> Docling -> Markdown -> markdown/<document>.md

Правила:
  - Полный offline режим Docling.
  - Только очистка технического мусора (номера страниц, колонтитулы).
  - НИКАКОЙ логики распознавания структуры НПА (это делает markdown_structure_parser.py).
"""

import os
import re
import sys
from pathlib import Path


# Offline-режим ДО инициализации Docling/Transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent



def _clean_markdown(content: str) -> str:
    """Удалить технический мусор из Docling-маркдауна.

    Копирует логику очистки из chunker_docling_chat_146_v7.py.
    """
    # Номера страниц: "Стр. 68 из 104" или "Страница 12"
    content = re.sub(r"(?i)Стр\.\s+\d+\s+из\s+\d+", "", content)
    content = re.sub(r"(?i)Страница\s+\d+", "", content)

    # Зачистка мусора разметки списков и отступов
    content = re.sub(r"(?m)^\s*\d+\.\s+([а-яА-Яa-zA-Z]\))", r"\1", content)
    content = re.sub(r"(?m)^\s*\d+\.\s+(\d+\))", r"\1", content)

    # Нормализация пробелов
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\r\n?", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip() + "\n"



def convert(
    fname: str,
    in_dir: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """Конвертировать один PDF в Markdown.

    Args:
        fname: Имя PDF-файла (например, "79-FZ.pdf").
        in_dir: Директория с PDF (по умолчанию raw/).
        out_dir: Директория для Markdown (по умолчанию markdown/).
    """
    in_dir = in_dir or PROJECT_DIR / "raw"
    out_dir = out_dir or PROJECT_DIR / "markdown"
    src = in_dir / fname
    if not src.exists():
        raise FileNotFoundError(f"PDF ne najden: {src}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + ".md")

    # Настройка Docling pipeline (полный offline, без таблиц)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.enable_remote_services = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)},
    )

    result = converter.convert(str(src))
    content = result.document.export_to_markdown()

    # Очистка технического мусора
    content = _clean_markdown(content)

    out.write_text(content, encoding="utf-8")
    print(f"done: {src.name} -> {out.relative_to(out_dir)}")



def batch_convert(
    in_dir: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Конвертировать все PDF из in_dir в Markdown в out_dir.

    Returns:
        Список созданных .md файлов.
    """
    in_dir = in_dir or PROJECT_DIR / "raw"
    out_dir = out_dir or PROJECT_DIR / "markdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(in_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Net PDF-fajlov v {in_dir}")
        return []

    results = []
    errors = []
    for src in pdf_files:
        try:
            convert(src.name, in_dir=in_dir, out_dir=out_dir)
            results.append(out_dir / (src.stem + ".md"))
        except Exception as exc:
            errors.append(f"{src.name}: {exc}")
            print(f"FAIL: {src.name}: {exc}")

    summary = f"PDF -> Markdown: {len(results)} uspeshno, {len(errors)} oshibok"
    print(summary)
    if errors:
        print("Oshibki:")
        for e in errors:
            print(f"  - {e}")
    return results



if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Ispolzovanie: venv/bin/python -m app.ingestion.pdf_to_markdown <file.pdf> [--all]")
        sys.exit(1)

    if "--all" in args:
        batch_convert()
    else:
        for fname in args:
            if fname == "--all":
                continue
            convert(fname)
