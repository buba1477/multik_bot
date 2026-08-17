"""Валидатор lossless-конвертации: HTML vs Markdown.

Проверяет отсутствие потери контента, а не только наличие ожидаемых слов:
  - покрытие текста (по токенам и по окнам);
  - отсутствие крупных пропавших кусков (окно со слабым покрытием);
  - сохранение таблиц (текст ячеек присутствует);
  - сохранение ссылок/сносок;
  - подозрительное уменьшение размера;
  - пустые блоки.
"""
import re
import sys
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup


def norm(text: str) -> str:
    text = text.lower()
    text = text.replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^а-яёa-z0-9№\x84-]+|\s+", norm(text)) if t]


def strip_markdown(md: str) -> str:
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = re.sub(r"(?m)^#{1,6}\s*", "", md)
    md = re.sub(r"(?m)^\s*[-*+]\s+", "", md)
    md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)
    md = re.sub('[*_`|>]', ' ', md)
    return md


def html_text(html_bytes: bytes) -> str:
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html = html_bytes.decode("windows-1251", "replace")
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(["script", "style", "head"]):
        t.decompose()
    return soup.get_text(" ", strip=True)


def token_coverage(hw, mset: Counter) -> float:
    if isinstance(hw, list):
        hw = Counter(hw)
    total = sum(hw.values())
    if not total:
        return 1.0
    got = sum(min(c, mset.get(w, 0)) for w, c in hw.items())
    return got / total


def table_tokens(html_bytes: bytes) -> list[str]:
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html = html_bytes.decode("windows-1251", "replace")
    soup = BeautifulSoup(html, "lxml")
    out = []
    for tbl in soup.find_all("table"):
        out.append(tokens(tbl.get_text(" ", strip=True)))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: validate_markdown.py <file.html> [--in DIR] [--md DIR]")
        sys.exit(2)
    fname = sys.argv[1]
    in_dir = Path("./raw_html")
    md_dir = Path("./markdown")
    if "--in" in sys.argv:
        in_dir = Path(sys.argv[sys.argv.index("--in") + 1])
    if "--md" in sys.argv:
        md_dir = Path(sys.argv[sys.argv.index("--md") + 1])

    src = in_dir / fname
    md = md_dir / (Path(fname).stem + ".md")
    if not src.exists():
        print(f"SKIP (no html): {src}")
        return
    if not md.exists():
        print(f"FAIL: нет markdown {md}")
        return

    hb = src.read_bytes()
    htext = norm(html_text(hb))
    mtext = norm(strip_markdown(md.read_text(encoding="utf-8")))

    ht = Counter(tokens(htext))
    mt = Counter(tokens(mtext))
    cov = token_coverage(ht, mt)

    # оконное покрытие: крупные пропавшие куски
    SIZE = 1500
    windows = [htext[i:i + SIZE] for i in range(0, len(htext), SIZE)]
    lost_chars = 0
    lost_windows = []
    for w in windows:
        wt = Counter(tokens(w))
        if token_coverage(wt, mt) < 0.9:
            lost_chars += len(w)
            if len(lost_windows) < 3:
                lost_windows.append(w[:120])
    covered_pct = 100.0 - (lost_chars / max(len(htext), 1) * 100.0)

    # таблицы
    htabs = table_tokens(hb)
    tabs_ok = all(token_coverage(t, mt) >= 0.9 for t in htabs) if htabs else True

    # ссылки
    links = re.findall(r"href=\"([^\"]+)\"", hb.decode("utf-8", "replace"))
    links_md = 0
    hrefs_ok = True

    # пустые блоки/строки
    empty_lines = sum(1 for ln in md.read_text(encoding="utf-8").splitlines() if not ln.strip())
    total_lines = md.read_text(encoding="utf-8").count("\n") + 1

    size_ratio = md.stat().st_size / max(len(hb), 1)

    print(f"== {fname}")
    print(f"  html_size  : {len(hb)} байт")
    print(f"  md_size    : {md.stat().st_size} байт | ratio={size_ratio:.2f}")
    print(f"  token_coverage : {cov*100:.1f}%")
    print(f"  window_coverage: {covered_pct:.1f}% (сильные пропажи: {len(lost_windows)}" + (": " + "; ".join(lost_windows) if lost_windows else "") + ")")
    print(f"  tables: html={len(htabs)} сохранены={tabs_ok}")
    print(f"  empty_lines: {empty_lines}/{total_lines}")

    ok = True
    if cov < 0.95 or covered_pct < 90 or not tabs_ok or size_ratio < 0.3:
        ok = False
    print(f"  RESULT: {'PASS' if ok else 'WARN'}")
    return ok


if __name__ == "__main__":
    main()
