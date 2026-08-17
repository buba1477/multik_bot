"""Универсальный структурный парсер Markdown для нормативных документов.

Принципы:
  - Markdown - единственный источник текста. Парсер НИЧЕГО не переписывает.
  - linear - линейный порядок лексических блоков (source of truth, exact reconstruction).
  - tree - только структурное представление по node_id, не источник текста.
  - records - производный слой для chunker.
  - Классификация по MD-признакам + универсальным шаблонам юридической нумерации.
    Сомнительное -> unknown (текст сохраняется). Без правил про конкретный закон.
"""
import json
import re
import sys
from pathlib import Path

SPACE = "[ \\t\\xa0]"

STRUCTURAL_HEADING = {"chapter", "section", "subsection", "article", "appendix", "note", "unknown"}


def _norm_number(s: str) -> str:
    return re.sub(SPACE + "+", ".", s.strip()).strip(".")


def classify_heading(text: str):
    t = text.strip()
    m = re.match(r"^Глава" + SPACE + r"+([0-9IVX]+)[.]?" + SPACE + r"*(.*)", t)
    if m:
        return "chapter", _norm_number(m.group(1)), m.group(2).strip(" ."), 0.95
    m = re.match(r"^Раздел" + SPACE + r"+([0-9IVX]+)[.]?" + SPACE + r"*(.*)", t)
    if m:
        return "section", _norm_number(m.group(1)), m.group(2).strip(" ."), 0.95
    m = re.match(r"^Статья" + SPACE + r"+(\d+(?:" + SPACE + r"+\d+)*?)[.]?" + SPACE + r"*(.*)", t)
    if m:
        return "article", _norm_number(m.group(1)), m.group(2).strip(" ."), 0.95
    m = re.match(r"^Приложение" + SPACE + r"+№?" + SPACE + r"*(\d+)[.]?" + SPACE + r"*(.*)", t)
    if m:
        return "appendix", m.group(1), m.group(2).strip(" ."), 0.95
    if re.match(r"^\(Наименование", t):
        return "note", None, t, 0.6
    m = re.match(r"^(\d+(?:\.\d+)*)[.]?" + SPACE + r"*(.*)", t)
    if m:
        num = _norm_number(m.group(1))
        return ("subsection" if "." in num else "section"), num, m.group(2).strip(" ."), 0.7
    m = re.match(r"^([IVX]+)[.)]" + SPACE + r"*(.*)", t)
    if m:
        return "section", m.group(1), m.group(2).strip(" ."), 0.75
    return "unknown", None, t, 0.3


def classify_content(lines):
    stripped = lines[0].lstrip()
    if stripped.startswith(">"):
        return "blockquote", None, 0.9
    m = re.match(r"^(\d+)\)" + SPACE + r"+", stripped)
    if m:
        return "item", m.group(1), 0.92
    m = re.match(r"^([а-яА-Яa-zA-Z])\)" + SPACE + r"*", stripped)
    if m:
        return "subparagraph", m.group(1), 0.8
    m = re.match(r"^([а-яА-Яa-zA-Z])\." + SPACE + r"*", stripped)
    if m and len(stripped) < 60:
        return "subparagraph", m.group(1), 0.7
    m = re.match(r"^(\d+)\." + SPACE + r"+", stripped)
    if m and len(lines) == 1:
        return "paragraph", m.group(1), 0.6
    return "text", None, 0.5


def is_table_line(line: str) -> bool:
    return line.strip() != "" and ("|" in line or "\\|" in line)


def split_segments(lines):
    segs = []
    cur = []
    for i, ln in enumerate(lines):
        if ln.strip() == "":
            if cur:
                segs.append(cur)
                cur = []
        else:
            cur.append(i)
    if cur:
        segs.append(cur)
    return segs
# ============================================================================
# НОВЫЙ PIPELINE (достраивается поверх существующих функций)
# ============================================================================

RECORD_TYPES = {
    "article", "paragraph", "subparagraph", "item", "appendix",
    "table", "text", "unknown", "note", "blockquote",
}

CONTEXT_FLAT_FIELDS = [
    "chapter", "section", "subsection", "article",
    "paragraph", "item", "appendix",
]

STRUCTURAL_NODE_TYPES = {
    "chapter", "section", "subsection", "article", "appendix", "note",
}

TYPE_TREE_DEPTH = {
    "document": 0, "chapter": 1, "section": 2, "subsection": 3,
    "article": 2, "appendix": 1, "paragraph": 3, "subparagraph": 4,
    "item": 4, "table": 3, "blockquote": 2, "note": 1,
    "text": 2, "unknown": 2,
}


def _make_node_id(index: int) -> str:
    return f"n{index:04d}"


def _md_heading_level(line: str) -> int:
    level = 0
    for ch in line:
        if ch == "#":
            level += 1
        else:
            break
    return level if level > 0 else 0


def _strip_md_heading(line: str) -> str:
    return re.sub(r"^#+\s*", "", line).strip()


def read_markdown(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines(keepends=False)


# ============================================================================
# 1. PARSING
# ============================================================================

def parse_blocks(lines: list[str]) -> list[dict]:
    segs = split_segments(lines)
    blocks = []
    for idx, seg in enumerate(segs):
        block_lines = [lines[i] for i in seg]
        blocks.append({
            "lines": block_lines,
            "source_index": idx,
            "source_lines": {"start": seg[0], "end": seg[-1]},
        })
    return blocks
# ============================================================================
# 2. CLASSIFICATION
# ============================================================================

def _classify_block(block: dict) -> tuple:
    blines = block["lines"]
    first = blines[0]

    hlevel = _md_heading_level(first)
    if hlevel > 0:
        heading_text = _strip_md_heading(first)
        btype, num, title, conf = classify_heading(heading_text)
        return btype, num, title, conf

    if all(is_table_line(ln) for ln in blines):
        return "table", None, None, 0.9

    if first.lstrip().startswith(">"):
        return "blockquote", None, None, 0.9

    stripped_first = first.lstrip()
    hyphen_item = re.match(r"^[-*]\s+(.*)", stripped_first)
    if hyphen_item:
        return "item", None, hyphen_item.group(1), 0.7

    btype, num, conf = classify_content(blines)
    if btype == "text":
        # Fallback: экранированные markdown-маркеры (1\. → 1.)
        unescaped = re.sub(r"\\\.", ".", stripped_first)
        if unescaped != stripped_first:
            m = re.match(r"^(\d+)\." + SPACE + r"+", unescaped)
            if m:
                return "paragraph", m.group(1), None, 0.55
            m = re.match(r"^([а-яА-Яa-zA-Z])\." + SPACE + r"*", unescaped)
            if m and len(stripped_first) < 60:
                return "subparagraph", m.group(1), None, 0.55
    return btype, num, None, conf


def _parse_table_rows(blines: list[str]) -> list[list[str]]:
    rows = []
    for ln in blines:
        cells = [c.strip() for c in ln.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        rows.append(cells)
    return rows


# ============================================================================
# 3. BUILD LINEAR
# ============================================================================

def build_linear(blocks: list[dict]) -> list[dict]:
    nodes = []
    for idx, block in enumerate(blocks):
        blines = block["lines"]
        btype, num, title, conf = _classify_block(block)
        content = "\n".join(blines)
        node = {
            "id": _make_node_id(idx),
            "type": btype,
            "number": num,
            "title": title or "",
            "content": content,
            "children": [],
            "level": 0,
            "source_index": idx,
            "source_lines": block["source_lines"],
            "context": {"ancestors": []},
            "context_flat": {f: None for f in CONTEXT_FLAT_FIELDS},
            "confidence": conf,
        }
        if btype == "table":
            node["rows"] = _parse_table_rows(blines)
        nodes.append(node)
    return nodes


# ============================================================================
# 4. BUILD TREE
# ============================================================================

def _find_tree_parent(stack: list[dict], child_type: str) -> dict | None:
    if child_type in STRUCTURAL_NODE_TYPES | {"paragraph", "subparagraph", "item"}:
        for i in range(len(stack) - 1, -1, -1):
            parent = stack[i]
            pd = TYPE_TREE_DEPTH.get(parent["type"], 99)
            cd = TYPE_TREE_DEPTH.get(child_type, 99)
            if pd < cd:
                return parent
        return stack[0] if stack else None
    for i in range(len(stack) - 1, -1, -1):
        if stack[i]["type"] in STRUCTURAL_NODE_TYPES:
            return stack[i]
    return stack[-1] if stack else None


def build_tree(linear: list[dict]) -> dict:
    root = {
        "id": "n0000",
        "type": "document",
        "number": None,
        "title": "",
        "content": "",
        "children": [],
        "level": 0,
        "source_index": -1,
        "source_lines": {"start": 0, "end": 0},
        "context": {"ancestors": []},
        "context_flat": {f: None for f in CONTEXT_FLAT_FIELDS},
        "confidence": 0.0,
    }
    stack = [root]
    for node in linear:
        ntype = node["type"]
        if ntype in STRUCTURAL_NODE_TYPES | {"paragraph", "subparagraph", "item"}:
            parent = _find_tree_parent(stack, ntype)
            if parent is None:
                parent = root
            node["level"] = parent["level"] + 1
            parent["children"].append(node)
            stack.append(node)
        else:
            parent = _find_tree_parent(stack, ntype)
            if parent is None:
                parent = root
            node["level"] = parent["level"] + 1
            parent["children"].append(node)
    return root
# ============================================================================
# 5. CONTEXT
# ============================================================================

def _build_context_node(node: dict, parent_ctx: dict) -> None:
    ancestors = list(parent_ctx.get("ancestors", []))
    if parent_ctx.get("type") in STRUCTURAL_NODE_TYPES:
        ancestors.append({
            "type": parent_ctx["type"],
            "number": parent_ctx["number"],
            "title": parent_ctx.get("title", ""),
        })
    node["context"] = {"ancestors": ancestors}
    flat = dict(parent_ctx.get("context_flat", {}))
    ntype = node["type"]
    if ntype in CONTEXT_FLAT_FIELDS:
        flat[ntype] = node["number"]
    node["context_flat"] = flat


def build_context_recursive(node: dict, parent_ctx: dict | None = None) -> None:
    if node["type"] == "document":
        node["context"] = {"ancestors": []}
        node["context_flat"] = {f: None for f in CONTEXT_FLAT_FIELDS}
        parent_meta = {
            "type": "document", "number": None,
            "title": node.get("title", ""),
            "context_flat": node["context_flat"],
            "ancestors": [],
        }
    else:
        _build_context_node(node, parent_ctx)
        parent_meta = {
            "type": node["type"], "number": node["number"],
            "title": node.get("title", ""),
            "context_flat": node["context_flat"],
            "ancestors": node["context"]["ancestors"],
        }
    for child in node["children"]:
        build_context_recursive(child, parent_meta)
# ============================================================================
# 6. RECORDS
# ============================================================================

def build_records(tree_root: dict) -> list[dict]:
    records = []

    def _walk(node: dict):
        ntype = node["type"]
        if ntype in {"document"} or (ntype in {"chapter", "section", "subsection"} and node["children"]):
            for c in node["children"]:
                _walk(c)
            return
        records.append({
            "node_id": node["id"],
            "text": node["content"],
            "structure": {
                "type": ntype,
                "number": node["number"],
                "context_flat": node.get("context_flat", {}),
            },
        })
        for c in node["children"]:
            _walk(c)

    _walk(tree_root)
    return records


# ============================================================================
# 7. DOC META
# ============================================================================

def extract_doc_info(lines: list[str]) -> dict:
    title = ""
    number = ""
    for ln in lines[:30]:
        s = ln.strip()
        if not s:
            continue
        m = re.search(r"№\s*([\dA-Za-zА-Яа-я\-]+)", s)
        if m and not number:
            number = m.group(1)
        if not title:
            # Снимаем markdown-маркер заголовка и проверяем, похоже ли на заголовок документа
            candidate = _strip_md_heading(s)
            if candidate.isupper() and len(candidate) > 5:
                title = candidate
    if not title:
        for ln in lines[:10]:
            if "Федеральный закон" in ln or "закон" in ln.lower():
                title = ln.strip()
                break
    return {"title": title, "number": number}


# ============================================================================
# 8. RECONSTRUCTION
# ============================================================================

def exact_reconstruct(linear: list[dict], total_source_lines: int | None = None) -> str:
    """Восстанавливает исходный markdown из linear, сохраняя пустые строки."""
    if not linear:
        return ""
    if total_source_lines is None:
        total_source_lines = max(n["source_lines"]["end"] for n in linear) + 1
    buf = [""] * total_source_lines
    for node in linear:
        start = node["source_lines"]["start"]
        node_lines = node["content"].split("\n")
        for i, ln in enumerate(node_lines):
            buf[start + i] = ln
    return "\n".join(buf)


def normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()
# ============================================================================
# 9. CLI
# ============================================================================

def parse_and_save(md_path: str | Path, output_path: str | Path | None = None) -> dict:
    md_path = Path(md_path)
    lines = read_markdown(md_path)
    doc_info = extract_doc_info(lines)

    blocks = parse_blocks(lines)
    linear = build_linear(blocks)
    tree_root = build_tree(linear)
    build_context_recursive(tree_root)
    records = build_records(tree_root)

    result = {
        "doc": {
            "title": doc_info["title"],
            "number": doc_info["number"],
            "type": "document",
            "source_md": str(md_path.resolve()),
        },
        "linear": linear,
        "tree": tree_root,
        "records": records,
    }

    out = Path(output_path) if output_path else Path("structure") / f"{md_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved: {out.resolve()}")
    print(f"   Nodes (linear): {len(linear)}")
    print(f"   Records: {len(records)}")
    dist = {}
    for n in linear:
        dist[n["type"]] = dist.get(n["type"], 0) + 1
    print(f"   Type distribution: {dict(sorted(dist.items()))}")
    print(f"   Unknown: {sum(1 for n in linear if n['type'] == 'unknown')}")
    print(f"   Tables: {sum(1 for n in linear if n['type'] == 'table')}")

    original = md_path.read_text(encoding="utf-8")
    reconstructed = exact_reconstruct(linear, total_source_lines=len(lines))
    # Сравниваем без учёта хвостового \n (файл может оканчиваться \n, реконструкция — нет)
    original_stripped = original.rstrip("\n")
    if original_stripped == reconstructed:
        print("✅ EXACT RECONSTRUCTION: PASS (по символам)")
    else:
        print("⚠️ EXACT RECONSTRUCTION: FAIL")
        ol = original_stripped.splitlines(keepends=False)
        rl = reconstructed.splitlines(keepends=False)
        for i, (o, r) in enumerate(zip(ol, rl)):
            if o != r:
                print(f"   Первое расхождение на строке {i}:")
                print(f"   orig: {repr(o[:120])}")
                print(f"   recon: {repr(r[:120])}")
                break
        if len(ol) != len(rl):
            print(f"   Длина строк: orig={len(ol)}, recon={len(rl)}")

    if normalize_text(original_stripped) == normalize_text(reconstructed):
        print("✅ NORMALIZED RECONSTRUCTION: PASS")
    else:
        print("⚠️ NORMALIZED RECONSTRUCTION: FAIL")

    return result


def main():
    if len(sys.argv) < 2:
        print("Использование: python markdown_structure_parser.py <markdown.md> [--output result.json]")
        sys.exit(1)
    md_path = Path(sys.argv[1])
    if not md_path.exists():
        print(f"❌ Файл не найден: {md_path}")
        sys.exit(1)
    output_path = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]
    parse_and_save(md_path, output_path)


if __name__ == "__main__":
    main()
