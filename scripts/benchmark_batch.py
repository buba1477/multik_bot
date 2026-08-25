#!/usr/bin/env python3
"""Benchmark batch sizes for FRIDA embedding on GPU."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, models

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "chunks" / "79-FZ.jsonl"
MODEL_PATH = BASE_DIR / "hf_cache" / "FRIDA"
N_SAMPLES = 64

print("=" * 76)
print("BENCHMARK: FRIDA batch sizes")
print("=" * 76)

# Загрузить модель
print(f"\nModel: {MODEL_PATH}")
word_embedding_model = models.Transformer(str(MODEL_PATH))
dim = word_embedding_model.get_word_embedding_dimension()
print(f"Dimension: {dim}")
pooling_model = models.Pooling(dim, pooling_mode="cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda")
model.eval()

# Загрузить 64 chunks
print(f"\nLoading {N_SAMPLES} chunks from {CHUNKS_FILE.name}")
texts = []
with open(CHUNKS_FILE) as f:
    for i, line in enumerate(f):
        if i >= N_SAMPLES:
            break
        texts.append(json.loads(line)["text"])
print(f"Loaded {len(texts)} texts")

# Prefix
prefixed = [f"search_document: {t}" for t in texts]
print(f"Avg text length: {sum(len(t) for t in prefixed) / len(prefixed):.0f} chars")
print(f"Total chars: {sum(len(t) for t in prefixed)}")

gpu_name = torch.cuda.get_device_name(0)
total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"\nGPU: {gpu_name}, Total VRAM: {total_mem:.1f} GiB")

# Warm-up
print("\nWarm-up (batch=2, 4 texts)...")
with torch.inference_mode():
    _ = model.encode(prefixed[:4], batch_size=2, show_progress_bar=False, normalize_embeddings=False)
torch.cuda.empty_cache()

# Benchmark different batch sizes
batch_sizes = [2, 4, 8, 16, 32, 64]
results = []
reserved_before = torch.cuda.memory_reserved()

for bs in batch_sizes:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    if bs > N_SAMPLES:
        n = N_SAMPLES
    else:
        n = N_SAMPLES // bs * bs  # round to multiple of bs
    
    batch = prefixed[:n]
    
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            emb = model.encode(batch, batch_size=bs, show_progress_bar=False, normalize_embeddings=False)
        t1 = time.perf_counter()
        
        mem_alloc = torch.cuda.memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        
        elapsed = t1 - t0
        rate = len(batch) / elapsed
        
        results.append((bs, n, elapsed, rate, mem_alloc, peak_alloc, "OK"))
        print(f"  batch={bs:<4}  {n:>3} texts  {elapsed:.2f}s  {rate:.1f} ch/s  "
              f"alloc={mem_alloc:.2f}GiB  peak={peak_alloc:.2f}GiB  OK")
        
    except torch.cuda.OutOfMemoryError as e:
        results.append((bs, 0, 0, 0, 0, 0, "OOM"))
        print(f"  batch={bs:<4}  OOM — SKIPPED")
        torch.cuda.empty_cache()
        break

print(f"\n{'=' * 76}")
print("SUMMARY")
print(f"{'=' * 76}")
print(f"{'batch':>6} {'texts':>6} {'time(s)':>8} {'ch/s':>8} {'alloc(GiB)':>10} {'peak(GiB)':>10} {'status':>8}")
for bs, n, elapsed, rate, mem_alloc, peak_alloc, status in results:
    if status == "OK":
        print(f"{bs:>6} {n:>6} {elapsed:>8.2f} {rate:>8.1f} {mem_alloc:>10.2f} {peak_alloc:>10.2f} {status:>8}")
    else:
        print(f"{bs:>6} {'-':>6} {'-':>8} {'-':>8} {'-':>10} {'-':>10} {status:>8}")

# Recommendation
ok_results = [(bs, n, elapsed, rate) for bs, n, elapsed, rate, _, _, status in results if status == "OK"]
if ok_results:
    best = max(ok_results, key=lambda x: x[3])
    print(f"\nRecommended batch_size: {best[0]} ({best[3]:.1f} ch/s)")
    print(f"Embedding estimate for 5317 chunks: {5317 / best[3]:.0f} seconds ≈ {5317 / best[3] / 60:.1f} min")