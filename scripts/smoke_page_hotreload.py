#!/usr/bin/env python3
"""Smoke-проверка hot-reload mapping/chunk_to_page.json.

Запуск:  venv/bin/python scripts/smoke_page_hotreload.py

ПОЛНОСТЬЮ локальная проверка, без рестарта API и без правки прод-карты:
  1) копирует реальный mapping/chunk_to_page.json во временный файл;
  2) подменяет app.rag.sources._MAPPING_FILE на эту копию;
  3) дёргает collect_sources до и после "пересборки" карты
     (смена page + сдвиг mtime вперед, как делает save_mapping);
  4) показывает смену #page в URL;
  5) восстанавливает _MAPPING_FILE/_MAPPING_MTIME/_CHUNK_TO_PAGE в finally.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import app.rag.sources as src  # noqa: E402

PROD_MAPPING = PROJECT / "mapping" / "chunk_to_page.json"
FAKE_PAGE = 99999


def _make_node(node_id: str, url: str, score: float):
    meta = {"title": "Smoke — Hot-reload", "source_url": url, "point": ""}
    return type(
        "FakeNodeWithScore",
        (),
        {"node": type("FakeNode", (), {"node_id": node_id, "metadata": meta, "text_template": "{content}"})(), "score": score},
    )


def _set_mtime(p: Path, ns: int) -> None:
    os.utime(p, ns=(ns, ns))


def main() -> int:
    if not PROD_MAPPING.exists():
        print(f"[!] Прод-карта не найдена: {PROD_MAPPING}", file=sys.stderr)
        return 1

    state = (src._MAPPING_FILE, src._MAPPING_MTIME, src._CHUNK_TO_PAGE)
    try:
        data = json.loads(PROD_MAPPING.read_text(encoding="utf-8"))
        mapped = [(k, v) for k, v in data.items() if isinstance(v, int)]
        if not mapped:
            print("[!] В карте нет записей со страницей", file=sys.stderr)
            return 1
        chunk_id, old_page = mapped[0]

        with tempfile.TemporaryDirectory(prefix="dl_smoke_hotreload_") as tmpd:
            tmp_map = Path(tmpd) / "chunk_to_page.json"
            json.dump(data, tmp_map.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

            src._MAPPING_FILE = tmp_map
            src._MAPPING_MTIME = None  # форсируем первое чтение temp-копии
            _set_mtime(tmp_map, 1_000_000_000)

            nodes = [_make_node(chunk_id, "raw/fake-doc.pdf", 0.9)]
            url_before = src.collect_sources(nodes, max_sources=1)[0]["url"]

            # "пересборка карты": новое значение page + сдвиг mtime вперёд
            data[chunk_id] = FAKE_PAGE
            json.dump(data, tmp_map.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
            _set_mtime(tmp_map, 2_000_000_000)

            url_after = src.collect_sources(nodes, max_sources=1)[0]["url"]

        print(f"Чанк                         : {chunk_id}")
        print(f"page в прод-mapping          : {old_page}")
        print(f"ДО   пересборки (URL)        : {url_before}")
        print(f"ПОСЛЕ пересборки (hot-reload): {url_after}")
        ok = url_after.endswith(f"#page={FAKE_PAGE}") and url_before != url_after
        print("\nOK: hot-reload работает — новая page подхвачена без рестарта" if ok
              else "\nFAIL: URL после пересборки не изменился")
        print("Прод-карта не изменена; состояние модуля восстановлено.")
        return 0 if ok else 2
    finally:
        src._MAPPING_FILE, src._MAPPING_MTIME, src._CHUNK_TO_PAGE = state


if __name__ == "__main__":
    raise SystemExit(main())
