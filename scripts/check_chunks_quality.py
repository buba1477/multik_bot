#!/usr/bin/env python3
"""Анализ качества chunks перед загрузкой в Qdrant.

Использует тот же FRIDA tokenizer, что и legal_chunker.py.
Ничего не меняет — только диагностика.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

# === Настройка ===
PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNKS_DIR = PROJECT_DIR / "chunks"
STRUCTURE_DIR = PROJECT_DIR / "structure"
MARKDOWN_DIR = PROJECT_DIR / "markdown"
OCR_DOCS = [
    "ukaz-613", "postanovlenie-1000", "postanovlenie-1387",
    "postanovlenie-397", "postanovlenie-9", "rasporyazhenie-2867-r",
    "ukaz-16", "ukaz-460", "ukaz-615", "ukaz-68",
]

# === Токенизатор FRIDA (из legal_chunker.py) ===
_TOKENIZER = None


def _load_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER or None
    model_dir = PROJECT_DIR / "hf_cache" / "FRIDA"
    if not model_dir.exists():
        _TOKENIZER = False
        return None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        _TOKENIZER = tok
        return tok
    except Exception as e:
        print(f"  [WARN] FRIDA tokenizer failed to load: {e}", file=sys.stderr)
        _TOKENIZER = False
        return None

def count_tokens(text: str) -> int:
    if not text:
        return 0
    tok = _load_tokenizer()
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, (len(text) + 3) // 4)

