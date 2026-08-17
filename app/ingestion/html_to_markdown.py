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
from pathlib import Path
from bs4 import BeautifulSoup

PANDOC = "/usr/bin/pandoc"
H_HEADING_LEVEL = 2

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
    in_dir = in_dir or (Path(__file__).resolve().parent.parent.parent / "raw_html")
    out_dir = out_dir or (Path(__file__).resolve().parent.parent.parent / "markdown")
    src = in_dir / fname
    if not src.exists():
        raise FileNotFoundError(src)
    html_bytes = src.read_bytes()
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html = html_bytes.decode("windows-1251", "replace")
    clean_html = sanitize_html(html)

    tmp = in_dir / ("__" + src.stem + ".clean.html")
    tmp.write_bytes(clean_html.encode("utf-8"))

    out = out_dir / (src.stem + ".md")
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [PANDOC, "--from=html", "--to=gfm-raw_html", "--wrap=none", "-o", str(out), str(tmp)],
        check=True,
    )
    tmp.unlink(missing_ok=True)

    md = out.read_text(encoding="utf-8")
    out.write_text(clean_markdown(md), encoding="utf-8")
    print(f"done: {src.name} -> {out.relative_to(out_dir)}")


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "79-FZ.html")
