#!/usr/bin/env python3
"""Финальная санитизация qrels_user_clean.jsonl: удаление 14 кросс-доменных мусорных чанков.

Программный in-place фикс: загружает eval/manual/qrels_user_clean.jsonl,
находит по частичному совпадению field `query` (case-insensitive) 14 записей
и хирургически очищает их списки `relevant` согласно экспертному контент-анализу.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
QRELS_PATH = PROJECT_DIR / "eval" / "manual" / "qrels_user_clean.jsonl"

# (query_substring, ids_to_remove, ids_to_add)
# ids_to_remove: list[str] — мусорные чанки, которые нужно удалить из relevant
# ids_to_add:    list[str] — корректные чанки, которые нужно добавить (если есть)
FIXES: list[tuple[str, list[str], list[str]]] = [
    # 1. "Могут ли уволить за коррупцию?" → удалить ukaz-870_app1_s2_p3
    ("уволить госслужащего за коррупцию", ["ukaz-870_app1_s2_p3"], []),
    # 2. "Что такое утрата доверия?" → удалить 33-fz_st21_p1
    ("утрата доверия", ["33-fz_st21_p1"], []),
    # 3. "Из чего складывается зарплата?" / "Зарплата" → удалить ukaz-609_app1_s2_p8
    ("зарплата", ["ukaz-609_app1_s2_p8"], []),
    # 4. "Какие надбавки и доплаты?" / "Надбавки" → удалить 117-fz_st255_p1
    ("надбавки и доплаты", ["117-fz_st255_p1"], []),
    # 5. "Приходит налог на имущество" → удалить 117-fz_st346_35_p2
    ("налог на имущество", ["117-fz_st346_35_p2"], []),
    # 6. "Что я обязан делать как налогоплательщик?" → удалить 117-fz_st324_1_p1, 146-fz_st32_p1
    ("обязан делать как налогоплательщик", ["117-fz_st324_1_p1", "146-fz_st32_p1"], []),
    # 7. "Что можно делать в личном кабинете?" → удалить 117-fz_st219_p8
    ("личном кабинете", ["117-fz_st219_p8"], []),
    # 8. "Каким ФЗ определены основы ГГС?" → удалить 273-fz_pre_p3
    ("основы государственной гражданской службы", ["273-fz_pre_p3"], []),
    # 9. "Какая максимальная надбавка за особые условия труда?" → удалить 117-fz_st428_p1, добавить 79-fz_st50_p1
    ("максимальная надбавка за особые условия", ["117-fz_st428_p1"], ["79-fz_st50_p1"]),
    # 10. "Госслужащий ФНС самозанятый для сдачи квартиры?" → удалить 117-fz_st217_p1
    ("сдачи квартиры в аренду", ["117-fz_st217_p1"], []),
    # 11. "Чем взятка отличается от подарка?" → удалить 117-fz_st217_p29
    ("взятка отличается от подарка", ["117-fz_st217_p29"], []),
    # 12. "Что нужно принести на конкурс?" → удалить ukaz-870_app1_s4_p18
    ("принести на конкурс", ["ukaz-870_app1_s4_p18"], []),
    # 13. "Кто присваивает классный чин?" → удалить ukaz-113_app1_s3_p1
    ("присваивает классный чин", ["ukaz-113_app1_s3_p1"], []),
    # 14. "Как инспектору ФНС поступить, если родственник" → удалить ukaz-821_app1_s2_p11
    ("родственник", ["ukaz-821_app1_s2_p11"], []),
]


def sanitize(content: str = "") -> int:
    lines = content.splitlines()
    lines = [line for line in lines if line.strip()]
    total = len(lines)
    print(f"📥 Загружено {total} строк из {QRELS_PATH}")

    fixed_count = 0
    out_lines: list[str] = []

    for line in lines:
        record = json.loads(line)
        query = record.get("query", "")
        query_lower = query.lower()
        relevant: list[str] = record.get("relevant", [])

        # Ищем подходящий фикс
        matched = False
        for q_sub, to_remove, to_add in FIXES:
            if q_sub.lower() in query_lower:
                original_len = len(relevant)
                new_relevant = [v for v in relevant if v not in to_remove]
                for add_id in to_add:
                    if add_id not in new_relevant:
                        new_relevant.append(add_id)

                if len(new_relevant) != original_len or to_add:
                    record["relevant"] = new_relevant
                    print(f"  🔧 Фикс #{FIXES.index((q_sub, to_remove, to_add)) + 1:>2}: "
                          f"«{query[:60].strip()}...» → "
                          f"{'удалены ' + ', '.join(to_remove) if to_remove else ''}"
                          f"{' + ' if to_remove and to_add else ''}"
                          f"{'добавлен ' + ', '.join(to_add) if to_add else ''}"
                          f" (relevant: {original_len} → {len(new_relevant)})")
                    fixed_count += 1
                matched = True
                break

        out_lines.append(json.dumps(record, ensure_ascii=False))

    # Проверка: ровно 200 строк
    assert len(out_lines) == 200, (
        f"❌ Ожидалось 200 строк, получено {len(out_lines)}"
    )

    # Перезапись in-place
    QRELS_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\n✅ Файл перезаписан: {QRELS_PATH}")
    print(f"🔧 Исправлено записей: {fixed_count}")
    print(f"📊 Всего строк: {len(out_lines)}")

    # Проверочное чтение
    verify_lines = [l for l in QRELS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(verify_lines) == 200, (
        f"❌ Проверка: ожидалось 200 строк, прочитано {len(verify_lines)}"
    )
    print("✅ Assert: ровно 200 строк подтверждён.")

    return 0


def main() -> int:
    if not QRELS_PATH.exists():
        print(f"❌ Файл не найден: {QRELS_PATH}")
        return 1

    # Читаем содержимое ДО создания бэкапа
    content = QRELS_PATH.read_text(encoding="utf-8")

    # Бэкап
    backup = QRELS_PATH.with_suffix(".jsonl.bak2")
    if not backup.exists():
        backup.write_text(content, encoding="utf-8")
        print(f"📦 Создан бэкап: {backup}")
    else:
        print(f"ℹ️  Бэкап уже существует: {backup}")

    return sanitize(content)


if __name__ == "__main__":
    sys.exit(main())