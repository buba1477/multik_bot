"""Разделение qrels_exact_clean.jsonl на strong_exact и complex_exact.

Правила разделения (повторяют strong_exact из classifier.py):
- strong_exact: query содержит цифры ИЛИ слова из _STRONG_EXACT_REQUISITES
- complex_exact: все остальные
"""

import json
import sys
from pathlib import Path

_STRONG_EXACT_REQUISITES = (
    "статья", "ст.", "фз", "указ", "постановление", "%", "процент",
)


def classify_query(query: str) -> str:
    """Вернуть 'strong' или 'complex' по правилам strong_exact."""
    q_low = query.lower()
    has_numbers = any(char.isdigit() for char in query)
    has_strong_requisites = any(x in q_low for x in _STRONG_EXACT_REQUISITES)
    return "strong" if (has_numbers or has_strong_requisites) else "complex"


def main():
    src = Path("eval/manual/qrels_exact_clean.jsonl")
    out_strong = Path("eval/manual/qrels_strong_exact.jsonl")
    out_complex = Path("eval/manual/qrels_complex_exact.jsonl")

    if not src.exists():
        print(f"❌ Файл не найден: {src}", file=sys.stderr)
        sys.exit(1)

    strong = []
    complex_ = []

    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            category = classify_query(obj["query"])
            if category == "strong":
                strong.append(obj)
            else:
                complex_.append(obj)

    with open(out_strong, "w", encoding="utf-8") as f:
        for obj in strong:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(out_complex, "w", encoding="utf-8") as f:
        for obj in complex_:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"📊 Разделение qrels_exact_clean.jsonl ({len(strong) + len(complex_)} запросов):")
    print(f"   • strong_exact: {len(strong)} запросов → {out_strong}")
    print(f"   • complex_exact: {len(complex_)} запросов → {out_complex}")

    # Показать примеры
    print(f"\n📝 Примеры strong_exact (первые 5):")
    for obj in strong[:5]:
        print(f"   • {obj['query']}")
    print(f"\n📝 Примеры complex_exact (первые 5):")
    for obj in complex_[:5]:
        print(f"   • {obj['query']}")


if __name__ == "__main__":
    main()
