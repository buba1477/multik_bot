"""Manifest dlya idempotentnosti pajplajna.

Hranit versii komponentov (chunker, embedder) i per-doc sostoyanie dlya Qdrant.
Sohranyaetsya posle kazhdogo shaga pajplajna.
"""

import json
import subprocess
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "pipeline_manifest.json"
from app.rag.constants import FRIDA_REV
EMBEDDER_REV = FRIDA_REV


def load() -> dict:
    if not MANIFEST_PATH.exists():
        return {"chunker_rev": None, "embedder_rev": None, "indexed": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def chunker_rev() -> str:
    """Vozvrashaet git-reviziyu legal_chunker.py ili 'dirty'."""
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "app/chunking/legal_chunker.py"],
        capture_output=True,
        cwd=MANIFEST_PATH.parent,
    )
    if diff.returncode != 0:
        return "dirty"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD:app/chunking/legal_chunker.py"],
        capture_output=True, text=True,
        cwd=MANIFEST_PATH.parent,
    )
    return result.stdout.strip() or "unknown"


def embedder_rev() -> str:
    return EMBEDDER_REV