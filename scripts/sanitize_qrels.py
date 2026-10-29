#!/usr/bin/env python3
"""Программная санитизация eval/manual/qrels_user_clean.jsonl.

Задача: устранить кросс-доменный мусор, дубликаты и искусственно раздутые
списки `relevant` на основе экспертного контент-анализа. Все правки вносятся
строго программно, файл перезаписывается in-place (с резервной копией в /tmp).

Поиск целевых записей по тексту поля `query` (case-insensitive):
  1. Нормализация: lower, отбрасывание «?», схлопывание пробелов.
  2. Сначала ищем ТОЧНОЕ совпадение нормализованного текста с формулировкой
     правила из ТЗ.
  3. Если точного совпадения нет — ищем записи, чей нормализованный текст
     содержит ВСЕ «якорные» подстроки правила. При ровно ОДНОМ кандидате
     применяем правку; при 0 или >1 кандидатов — аварийная остановка БЕЗ
     записи (гарантия, что правка не заденет чужую запись).

Операции:
  remove(chunk_ids, dedupe=True) — удалить перечисленные чанки (если есть);
    dedupe=True дополнительно схлопывает дубликаты id, сохраняя первое вхождение.
  set(chunk_ids) — переписать `relevant` строго заданным списком (прямой ответ).

В конце выполняется мягкая проверка корпуса chunks/: каждый результирующий и
каждый удаляемый чанк логируется (WARN, если id отсутствует в корпусе —
не блокирует запись, но привлекает внимание эксперта).

Идемпотентность: повторный запуск не меняет файл (remove/set уже применены).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
QRELS = PROJECT_DIR / "eval" / "manual" / "qrels_user_clean.jsonl"
CHUNKS_DIR = PROJECT_DIR / "chunks"
BACKUP = Path("/tmp/qrels_user_clean.jsonl.bak")


def normalize(text: str) -> str:
    """Нормализация текста запроса для case-insensitive поиска."""
    return " ".join(text.lower().replace("?", "").split())


@dataclass(frozen=True)
class Rule:
    """Одна правка санитизации."""

    tag: str            # короткая метка для лога
    target_query: str   # формулировка цели из ТЗ (для лога и точного матча)
    anchors: tuple[str, ...]  # якоря-подстроки нормализованного текста записи
    op: str             # "remove" | "set"
    chunk_ids: tuple[str, ...]  # удаляемые чанки / итоговый список (set)
    dedupe: bool = True  # для remove: схлопывать дубликаты id (оставить первое вхождение)

# ---------------------------------------------------------------------------
# Правила из ТЗ. target_query — формулировка пользователя; anchors подобраны
# так, чтобы уникально матчить фактическую запись в датасете (проверено: каждая
# цель даёт ровно одного кандидата на актуальном файле).
# ---------------------------------------------------------------------------
RULES: list[Rule] = [
    # --- 1. ХИРУРГИЧЕСКАЯ ОЧИСТКА МУСОРА И ДУБЛЕЙ ---
    Rule(
        tag="налог/вычеты -> убрать НДС",
        target_query="Хочу вернуть налог, какие вычеты бывают?",
        anchors=("хочу вернуть налог, какие вычеты бывают",),
        op="remove",
        chunk_ids=("117-fz_st171_p7", "117-fz_st171_p5", "117-fz_st171_p1", "117-fz_st171_p11"),
    ),
    Rule(
        tag="запреты госслужащему -> мусор+дубль",
        target_query="Что запрещено госслужащему?",
        anchors=("запрещено", "делать"),
        op="remove",
        chunk_ids=("ukaz-111_app1_s2_p2",),  # 79-fz_st17_p1-дубль убирает dedupe
    ),
    Rule(
        tag="выплаты при призыве -> убрать анкету",
        target_query="Сохраняются ли выплаты при призыве?",
        anchors=("выплаты при призыве",),
        op="remove",
        chunk_ids=("ukaz-870_app1_s2_p6",),
    ),
    Rule(
        tag="бизнес -> убрать госдолжности",
        target_query="Может ли госслужащий заниматься бизнесом?",
        anchors=("заниматься бизнесом",),
        op="remove",
        chunk_ids=("273-fz_st12_1_p1",),
    ),
    Rule(
        tag="служебное поведение -> убрать комиссии",
        target_query="Какие требования к служебному поведению?",
        anchors=("требования к служебному поведению",),
        op="remove",
        chunk_ids=("ukaz-821_app1_s2_p13",),
    ),
    Rule(
        tag="служебная проверка -> убрать 273-фз",
        target_query="В каких случаях проводится служебная проверка?",
        anchors=("в каких случаях проводится служебная проверка",),
        op="remove",
        chunk_ids=("273-fz_st13_5_p1",),
    ),
    # --- 2. УРЕЗАНИЕ ИСКУССТВЕННО РАЗДУТЫХ СПИСКОВ (только прямой ответ) ---
    Rule(
        tag="повышение в должности -> резерв+продвижение",
        target_query="Как повышают в должности?",
        anchors=("повышают в должности",),
        op="set",
        chunk_ids=("79-fz_st64_p3", "ukaz-96_secII_s4_p1"),
    ),
    Rule(
        tag="представитель нанимателя -> один чанк",
        target_query="Кто такой представитель нанимателя?",
        anchors=("представитель нанимателя и какие у него полномочия",),
        op="set",
        chunk_ids=("ukaz-159_secIII_s5_p1",),
    ),
    Rule(
        tag="увольнение -> только базовые статьи",
        target_query="По каким основаниям можно уволить?",
        anchors=("по каким основаниям можно уволить",),
        op="set",
        chunk_ids=("79-fz_st33_p2", "79-fz_st37_p1", "79-fz_st34_p1", "79-fz_st39_p1"),
    ),
]
def load_records(path: Path) -> list[dict]:
    """Читает JSONL, отбрасывая пустые строки."""
    out = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"❌ Некорректный JSON в {path} строка {line_no}: {exc}") from None
    return out


def resolve_record(rows: list[dict], rule: Rule) -> int:
    """Находит единственный индекс записи для правила (exception при неудаче)."""
    normalized = [normalize(r["query"]) for r in rows]

    # 1) точное case-insensitive совпадение с формулировкой ТЗ
    exact = normalize(rule.target_query)
    exact_hits = [i for i, q in enumerate(normalized) if q == exact]
    if len(exact_hits) == 1:
        return exact_hits[0]
    if len(exact_hits) > 1:
        raise ValueError(
            f"Правило [{rule.tag}]: неожиданно несколько точных совпадений: {exact_hits}"
        )

    # 2) все якоря входят в нормализованный текст записи -> ровно один кандидат
    anchor_hits = [
        i for i, q in enumerate(normalized)
        if all(a in q for a in rule.anchors)
    ]
    if len(anchor_hits) == 1:
        return anchor_hits[0]
    if len(anchor_hits) == 0:
        raise ValueError(
            f"Правило [{rule.tag}]: запись не найдена (якоря {rule.anchors}). "
            f"Формулировка в файле могла измениться — требуется ручная сверка."
        )
    raise ValueError(
        f"Правило [{rule.tag}]: неоднозначное совпадение — {len(anchor_hits)} кандидатов "
        f"(индексы {anchor_hits}). Требуется уточнить якоря."
    )


def apply_rule(
    relevant: list[str], rule: Rule
) -> tuple[list[str], bool, list[str], list[str]]:
    """Применяет правило к списку relevant. Возвращает (новый, changed, removed, added)."""
    if rule.op == "remove":
        drop = set(rule.chunk_ids)
        kept = [c for c in relevant if c not in drop]
        removed = sorted(set(c for c in relevant if c in drop))
        if rule.dedupe:
            seen: set[str] = set()
            deduped: list[str] = []
            for c in kept:
                if c not in seen:
                    deduped.append(c)
                    seen.add(c)
            kept = deduped
        added: list[str] = []
        return kept, kept != relevant, removed, added

    if rule.op == "set":
        new = list(rule.chunk_ids)
        removed = [c for c in relevant if c not in rule.chunk_ids]
        added = [c for c in rule.chunk_ids if c not in relevant]
        return new, new != relevant, removed, added

    raise ValueError(f"Неизвестная операция: {rule.op!r}")


def corpus_chunk_ids() -> set[str]:
    """Все id чанков из chunks/*.jsonl (мягкая проверка)."""
    ids: set[str] = set()
    for f in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line).get("id", ""))
            except json.JSONDecodeError:
                continue
    return ids

def main() -> int:
    p = argparse.ArgumentParser(description="Санитизация qrels_user_clean.jsonl")
    p.add_argument("--dry-run", action="store_true",
                   help="Показать лог изменений без записи в файл")
    p.add_argument("--qrels", type=Path, default=QRELS,
                   help="Путь к qrels-файлу (по умолчанию eval/manual/qrels_user_clean.jsonl)")
    args = p.parse_args()

    qrels_path = args.qrels
    if not qrels_path.exists():
        raise SystemExit(f"❌ Файл не найден: {qrels_path}")

    rows = load_records(qrels_path)
    print(f"📄 Загружено записей: {len(rows)}  ({qrels_path})")
    print("=" * 100)

    # ---------- Резолвинг (безопасно: все правила резолвятся ДО каких-либо правок) ----------
    resolved: list[tuple[Rule, int]] = []
    for rule in RULES:
        idx = resolve_record(rows, rule)
        resolved.append((rule, idx))
        print(f"🎯 {rule.tag:44s} → [{idx}] «{rows[idx]['query']}»")

    used_idx: set[int] = set()
    for rule, idx in resolved:
        if idx in used_idx:
            raise SystemExit(f"❌ Правила [{rule.tag}] и другое правило ссылаются на запись [{idx}] — конфликт")
        used_idx.add(idx)

    # ---------- Применение ----------
    changes = 0
    print("=" * 100)
    for rule, idx in resolved:
        rec = rows[idx]
        before = list(rec["relevant"])
        after, changed, removed, added = apply_rule(before, rule)
        status = "✏️  ИЗМЕНЕНО" if changed else "✔️  без изменений"
        print(f"\n{status} | [{idx}] «{rec['query']}»")
        print(f"   цель ТЗ : «{rule.target_query}»")
        print(f"   было    ({len(before)}): {before}")
        print(f"   стало   ({len(after)}): {after}")
        if removed:
            print(f"   удалено : {removed}")
        if added:
            print(f"   добавлено: {added}")
        if rule.op == "remove" and rule.dedupe:
            dup_before = len(before) - len(set(before))
            dup_after = len(after) - len(set(after))
            print(f"   дубликаты до/после: {dup_before} / {dup_after}")
        if changed:
            rec["relevant"] = after
            changes += 1

    # ---------- Мягкая проверка корпуса ----------
    print(f"\n{'=' * 100}")
    if CHUNKS_DIR.exists():
        corpus = corpus_chunk_ids()
        print(f"🗂  Корпус: {len(corpus)} чанков (chunks/*.jsonl)")
        for rule, idx in resolved:
            for cid in set(rule.chunk_ids) | set(rows[idx]["relevant"]):
                if cid not in corpus:
                    print(f"   ⚠️  WARN: чанк {cid!r} не найден в корпусе "
                          f"(запись [{idx}] «{rows[idx]['query']}»)")
    else:
        print("🗂  Каталог chunks/ не найден — проверка корпуса пропущена")

    # ---------- Запись ----------
    if args.dry_run:
        print(f"\n⏭️  DRY-RUN: запись не выполнялась (изменено записей: {changes})")
        return 0
    if changes == 0:
        print("\nℹ️  Изменений нет — файл уже санитизирован, перезапись не требуется.")
        return 0

    if args.qrels == QRELS:  # бэкап только для боевого файла
        shutil.copy2(QRELS, BACKUP)
        print(f"💾 Резервная копия: {BACKUP}")

    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    qrels_path.write_text(payload, encoding="utf-8")

    print(f"✅ Файл перезаписан: {qrels_path}  (изменено записей: {changes})")

    # ---------- Верификация после записи ----------
    verify = load_records(qrels_path)
    assert len(verify) == len(rows), "❌ Количество записей изменилось после перезаписи"
    for rule, idx in resolved:
        if rule.op == "set":
            got = verify[idx]["relevant"]
            assert got == list(rule.chunk_ids), (
                f"❌ Верификация не прошла для [{idx}]: {got} != {rule.chunk_ids}"
            )
    print(f"✅ Верификация после записи пройдена ({len(verify)} записей, всё читается корректно).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
