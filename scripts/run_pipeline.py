import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import html_to_markdown
from app.ingestion import markdown_structure_parser
from app.ingestion import pdf_ocr

PROJECT = Path(__file__).resolve().parent.parent
REGISTRY = PROJECT / "documents.json"
MARKDOWN_DIR = PROJECT / "markdown"
STRUCTURE_DIR = PROJECT / "structure"
RAW_DIR = PROJECT / "raw"
RAW_HTML_DIR = PROJECT / "raw_html"


def step_download() -> None:
    print("=" * 60)
    print("SHAG 1: Skachivanie dokumentov (PDF + HTML)")
    print("=" * 60)
    from scripts.download_documents import main as download_main
    download_main()


def step_convert() -> list[Path]:
    print("\n" + "=" * 60)
    print("SHAG 2: Konvertatsiya HTML -> Markdown")
    print("=" * 60)
    return html_to_markdown.batch_convert()


def step_parse() -> list[dict]:
    print("\n" + "=" * 60)
    print("SHAG 3: Parsing Markdown -> struktura")
    print("=" * 60)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return markdown_structure_parser.batch_parse(
        MARKDOWN_DIR, STRUCTURE_DIR,
        registry=registry.get("documents", []),
    )
def step_ocr() -> dict:
    """Определить сканированные PDF и выполнить OCR.

    Returns:
        Словарь со статистикой: scanned, text, ocr_ok, ocr_fail, ocr_ids.
    """
    print("\n" + "=" * 60)
    print("SHAG 1b: Proverka PDF na nalichie tekstovogo sloya (OCR fallback)")
    print("=" * 60)

    # Классифицируем PDF
    classification = pdf_ocr.classify_pdf_directory(RAW_DIR)
    scanned = classification["scanned"]
    text_pdf = classification["text_pdf"]

    print(f"  Vsego PDF: {classification['total']}")
    print(f"  S tekstovym sloem: {len(text_pdf)}")
    print(f"  Skanirovannyh (nuzhen OCR): {len(scanned)}")

    if scanned:
        print(f"  Skanirovannye PDF: {scanned}")

    if classification["errors"]:
        print(f"  Oshibki proverki: {len(classification['errors'])}")
        for e in classification["errors"]:
            print(f"    ! {e}")

    # Выполняем OCR для сканированных PDF, у которых нет HTML
    html_ids = {h.stem for h in sorted(RAW_HTML_DIR.glob("*.html"))}
    ocr_ok = []
    ocr_fail = []

    for pdf_name in scanned:
        doc_id = Path(pdf_name).stem
        if doc_id in html_ids:
            print(f"  Propusk OCR: {pdf_name} (est' HTML)")
            continue

        pdf_path = RAW_DIR / pdf_name
        try:
            print(f"  OCR: {pdf_name}...")
            text = pdf_ocr.ocr_pdf(pdf_path)
            html_to_markdown.convert_from_text(text, doc_id)
            ocr_ok.append(pdf_name)
            print(f"    -> OK ({len(text)} simvolov)")
        except Exception as exc:
            ocr_fail.append(pdf_name)
            print(f"    -> FAIL: {exc}")

    print(f"\n  OCR vypolnen: {len(ocr_ok)}, oshibok: {len(ocr_fail)}")
    if ocr_ok:
        print(f"  Obrabotano: {ocr_ok}")

    return {
        "scanned": scanned,
        "text_pdf": text_pdf,
        "total": classification["total"],
        "ocr_ok": ocr_ok,
        "ocr_fail": ocr_fail,
    }



def validate_pairs() -> dict:
    print("\n" + "=" * 60)
    print("VALIDATsIYa: parnost' raw/ i raw_html/")
    print("=" * 60)
    raw_dir = PROJECT / "raw"
    raw_html_dir = PROJECT / "raw_html"
    pdfs = sorted(raw_dir.glob("*.pdf"))
    htmls = sorted(raw_html_dir.glob("*.html"))
    pdf_ids = {p.stem for p in pdfs}
    html_ids = {h.stem for h in htmls}
    missing_html = pdf_ids - html_ids
    missing_pdf = html_ids - pdf_ids
    paired = pdf_ids & html_ids
    print(f"  PDF files : {len(pdfs)}")
    print(f"  HTML files: {len(htmls)}")
    print(f"  Paired    : {len(paired)}")
    if missing_html:
        print(f"  ! PDF bez HTML: {sorted(missing_html)}")
    if missing_pdf:
        print(f"  ! HTML bez PDF: {sorted(missing_pdf)}")
    return {"pdf_count": len(pdfs), "html_count": len(htmls),
            "paired": len(paired), "missing_html": sorted(missing_html),
            "missing_pdf": sorted(missing_pdf)}


def report_parser_results(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("OTChYoT PO PARSINGU")
    print("=" * 60)
    for r in results:
        doc = r["doc"]
        doc_id = doc.get("id", "?")
        doc_type = doc.get("type", "document")
        linear = r.get("linear", [])
        records = r.get("records", [])
        unknown = sum(1 for n in linear if n["type"] == "unknown")
        tables = sum(1 for n in linear if n["type"] == "table")
        empty = len(linear) == 0
        all_unknown = unknown == len(linear) if linear else False
        print(f"\n  [{doc_id}] ({doc_type})")
        print(f"    Nodes (linear): {len(linear)}")
        print(f"    Records: {len(records)}")
        dist = {}
        for n in linear:
            dist[n["type"]] = dist.get(n["type"], 0) + 1
        print(f"    Type distribution: {dict(sorted(dist.items()))}")
        print(f"    Unknown: {unknown}")
        print(f"    Tables: {tables}")
        if empty:
            print("    !!! PUSTOJ REZUL'TAT")
        if all_unknown:
            print("    !!! VSE NODY UNKNOWN")


def main() -> None:
    args = set(sys.argv[1:]) if len(sys.argv) > 1 else {"--all"}
    do_all = "--all" in args or not any(a.startswith("--") for a in args)
    do_download = "--download" in args or do_all
    do_ocr = "--ocr" in args or do_all
    do_convert = "--convert" in args or do_all
    do_parse = "--parse" in args or do_all
    t_start = time.time()

    if do_download:
        step_download()

    ocr_stats = {}
    if do_ocr:
        ocr_stats = step_ocr()

    if do_convert:
        step_convert()

    if do_parse:
        results = step_parse()
        report_parser_results(results)

    pairs = validate_pairs()
    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"PAJPLAJN ZAVERShYON za {elapsed:.1f} sec")
    print(f"  PDF: {pairs['pdf_count']}, HTML: {pairs['html_count']}, Par: {pairs['paired']}")
    if ocr_stats:
        print(f"  OCR: {len(ocr_stats.get('ocr_ok', []))} ok, {len(ocr_stats.get('ocr_fail', []))} fail")
        if ocr_stats.get("scanned"):
            print(f"  Skanirovannyh PDF: {len(ocr_stats['scanned'])}")


if __name__ == "__main__":
    main()
