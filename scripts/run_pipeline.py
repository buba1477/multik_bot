import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import html_to_markdown
from app.ingestion import markdown_structure_parser
from app.ingestion import pdf_ocr
from app.chunking import legal_chunker
from scripts import pipeline_manifest

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


def step_convert(force: bool = False) -> tuple[list[Path], int, int]:
    print("\n" + "=" * 60)
    print("SHAG 2: Konvertatsiya HTML -> Markdown")
    print("=" * 60)
    return html_to_markdown.batch_convert(force=force)


def step_parse(force: bool = False) -> tuple[list[dict], int, int]:
    print("\n" + "=" * 60)
    print("SHAG 3: Parsing Markdown -> struktura")
    print("=" * 60)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return markdown_structure_parser.batch_parse(
        MARKDOWN_DIR, STRUCTURE_DIR,
        registry=registry.get("documents", []),
        force=force,
    )
def step_pdf_fallback(force: bool = False) -> dict:
    """PDF fallback: OCR + pdftotext dlya PDF bez HTML ili s plohim Markdown.

    Zapuskaetsya POSLE step_convert(). Analiziruet rezultat convert
    i dorabatyvaet: skanirovannye PDF (OCR), tekstovye PDF (pdftotext),
    esli HTML otsutstvuet ili Markdown poluchilsya hilym (<=500 simvolov).
    """
    print("\n" + "=" * 60)
    print("SHAG 2b: PDF fallback (OCR / pdftotext)")
    print("=" * 60)

    classification = pdf_ocr.classify_pdf_directory(RAW_DIR)
    scanned = classification["scanned"]
    text_pdfs = classification["text_pdf"]
    html_ids = {h.stem for h in sorted(RAW_HTML_DIR.glob("*.html"))}

    total = classification["total"]
    print(f"  Vsego PDF: {total}")
    print(f"  S tekstovym sloem: {len(text_pdfs)}")
    print(f"  Skanirovannyh (nuzhen OCR): {len(scanned)}")

    stats = {
        "total": total, "scanned": scanned, "text_pdf": text_pdfs,
        "ocr_ok": [], "ocr_fail": [],
        "pdftotext_ok": [], "pdftotext_fail": [], "skipped": [],
    }

    for pdf_name in sorted(scanned + text_pdfs):
        doc_id = Path(pdf_name).stem
        pdf_path = RAW_DIR / pdf_name
        is_scanned = pdf_name in scanned

        if doc_id in html_ids:
            md_path = MARKDOWN_DIR / f"{doc_id}.md"
            if md_path.exists():
                md_text = md_path.read_text(encoding="utf-8")
                if len(md_text.strip()) > 500:
                    stats["skipped"].append(pdf_name)
                    print(f"  Propusk: {pdf_name} (est HTML, md={len(md_text.strip())}s, >500)")
                    continue
                print(f"  Plohoj Markdown ({len(md_text.strip())}s), fallback...")
            else:
                print(f"  .md ne najden, fallback...")

        try:
            if is_scanned:
                print(f"  OCR: {pdf_name}...")
                text = pdf_ocr.ocr_pdf(pdf_path)
                stats["ocr_ok"].append(pdf_name)
                print(f"    -> OK ({len(text)} simvolov)")
            else:
                print(f"  pdftotext: {pdf_name}...")
                import subprocess
                result_proc = subprocess.run(
                    ["pdftotext", str(pdf_path), "-"],
                    capture_output=True, text=True, timeout=60,
                )
                if result_proc.returncode == 0 and len(result_proc.stdout.strip()) > 200:
                    text = result_proc.stdout
                    stats["pdftotext_ok"].append(pdf_name)
                else:
                    print(f"    -> SKIP (net teksta)")
                    stats["pdftotext_fail"].append(pdf_name)
                    continue
            html_to_markdown.convert_from_text(text, doc_id)
        except Exception as exc:
            print(f"    -> FAIL: {exc}")
            (stats["ocr_fail"] if is_scanned else stats["pdftotext_fail"]).append(pdf_name)

    print(f"\n  OCR: {len(stats['ocr_ok'])} ok, {len(stats['ocr_fail'])} fail")
    print(f"  pdftotext: {len(stats['pdftotext_ok'])} ok, {len(stats['pdftotext_fail'])} fail")
    print(f"  Propusheno: {len(stats['skipped'])}")

    return stats



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


def step_chunk(force: bool = False) -> tuple[list[Path], int, int]:
    """Structure JSON -> chunks JSONL (legal_chunker)."""
    print("\n" + "=" * 60)
    print("SHAG 4: Chanking structure -> chunks")
    print("=" * 60)
    results, processed, skipped = legal_chunker.batch_convert(force=force)
    print(f"\n  Obrabotano: {processed}, propusheno: {skipped}")
    print(f"  Sozdano .jsonl faylov: {len(results)}")
    for p in results:
        print(f"  - {p}")
    return results, processed, skipped


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


def update_manifest(manifest: dict, step: str, processed: int | None = None,
                    skipped: int | None = None) -> None:
    """Obnovlyaet manifest i sohranyaet."""
    if step == "chunk":
        manifest["chunker_rev"] = pipeline_manifest.chunker_rev()
    if step == "convert" or step == "chunk":
        manifest["embedder_rev"] = pipeline_manifest.embedder_rev()
    pipeline_manifest.save(manifest)


def main() -> None:
    args = set(sys.argv[1:]) if len(sys.argv) > 1 else {"--all"}
    force = "--force" in args
    do_all = "--all" in args or not any(a.startswith("--") for a in args)
    do_download = "--download" in args or do_all
    do_pdf_fallback = "--pdf-fallback" in args or do_all
    do_convert = "--convert" in args or do_all
    do_parse = "--parse" in args or do_all
    do_chunk = "--chunk" in args or do_all
    t_start = time.time()

    manifest = pipeline_manifest.load()

    if do_download:
        step_download()

    convert_processed = convert_skipped = 0
    if do_convert:
        _conv_results, convert_processed, convert_skipped = step_convert(force=force)

    pdf_stats = {}
    if do_pdf_fallback:
        pdf_stats = step_pdf_fallback(force=force)

    parse_processed = parse_skipped = 0
    parse_results: list[dict] = []
    if do_parse:
        parse_results, parse_processed, parse_skipped = step_parse(force=force)
        report_parser_results(parse_results)

    chunk_processed = chunk_skipped = 0
    chunk_results: list[Path] = []
    if do_chunk:
        chunk_results, chunk_processed, chunk_skipped = step_chunk(force=force)
        update_manifest(manifest, "chunk", chunk_processed, chunk_skipped)

    pairs = validate_pairs()
    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    if do_convert:
        print(f"  Konvert: {convert_processed} obrabotano, {convert_skipped} propusheno")
    if do_parse:
        print(f"  Parse: {parse_processed} obrabotano, {parse_skipped} propusheno")
    if do_chunk:
        print(f"  Chunk: {chunk_processed} obrabotano, {chunk_skipped} propusheno")
        print(f"  Chunks: {len(chunk_results)} faylov")
    print(f"PAJPLAJN ZAVERShYON za {elapsed:.1f} sec")
    print(f"  PDF: {pairs['pdf_count']}, HTML: {pairs['html_count']}, Par: {pairs['paired']}")
    if pdf_stats:
        print(f"  PDF fallback: OCR {len(pdf_stats.get('ocr_ok', []))}/{len(pdf_stats.get('ocr_fail', []))} "
              f"pdftotext {len(pdf_stats.get('pdftotext_ok', []))}/{len(pdf_stats.get('pdftotext_fail', []))} "
              f"skip {len(pdf_stats.get('skipped', []))}")


if __name__ == "__main__":
    main()
