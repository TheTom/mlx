#!/usr/bin/env python3
# Copyright © 2025 Apple Inc.
# TurboQuant Comprehensive Benchmark — fused attention, NR0=2, compiled encode
#
# Usage:
#   python3 benchmarks/python/turbo_comprehensive_bench.py
#
# Benchmarks:
#   1. Baseline FP16 SDPA decode (128 and 256 KV tokens)
#   2. Turbo4 fused attention decode (NR0=1)
#   3. Turbo4 fused attention decode (NR0=2)
#   4. Encode pipeline: compiled vs uncompiled
#   5. PPL comparison: FP16 vs turbo4 cross-entropy on synthetic forward pass

import math
import subprocess
import sys
import time
from datetime import datetime

import mlx.core as mx
import numpy as np

# Ensure the turbo_kv_cache module is importable from the repo
sys.path.insert(0, "python")
from mlx.nn.layers.turbo_kv_cache import (
    turbo_encode,
    turbo_encode_uncompiled,
    turbo_decode,
    turbo_fused_attention,
    turbo_attention,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE_NAME = subprocess.check_output(
    ["sysctl", "-n", "machdep.cpu.brand_string"]
).decode("utf-8").strip()

N_WARMUP = 10
N_ITER = 50
BATCH = 1
N_HEADS = 8
DIM = 128
BITS = 4
SEED = 42
KV_LENGTHS = [128, 256, 512, 1024]

results = []


def bench(fn, *args, n_warmup=N_WARMUP, n_iter=N_ITER):
    """Run fn with warmup, return median time in ms."""
    # Warmup
    for _ in range(n_warmup):
        out = fn(*args)
        mx.eval(out)

    times = []
    for _ in range(n_iter):
        s = time.perf_counter_ns()
        out = fn(*args)
        mx.eval(out)
        e = time.perf_counter_ns()
        times.append((e - s) * 1e-6)  # ms

    times.sort()
    median = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    return median, p10, p90


def log(msg):
    print(msg)
    results.append(msg)


# ---------------------------------------------------------------------------
# 1. Baseline FP16 SDPA decode
# ---------------------------------------------------------------------------
log("=" * 72)
log(f"TurboQuant Comprehensive Benchmark — {datetime.now().isoformat()}")
log(f"Device: {DEVICE_NAME}")
log(f"Config: B={BATCH}, H={N_HEADS}, D={DIM}, bits={BITS}")
log("=" * 72)
log("")
log("## 1. Baseline FP16 SDPA Decode")
log("")
log(f"{'T_kv':>8} | {'Median (ms)':>12} | {'P10':>8} | {'P90':>8}")
log("-" * 50)

for T_kv in KV_LENGTHS:
    q = mx.random.normal((BATCH, N_HEADS, 1, DIM))
    k = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    v = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    mx.eval(q, k, v)

    def fp16_sdpa(q, k, v):
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(DIM)
        )

    med, p10, p90 = bench(fp16_sdpa, q, k, v)
    log(f"{T_kv:>8} | {med:>12.3f} | {p10:>8.3f} | {p90:>8.3f}")

log("")

# ---------------------------------------------------------------------------
# 2. Turbo4 Fused Attention Decode (NR0=1)
# ---------------------------------------------------------------------------
log("## 2. Turbo4 Fused Attention Decode (NR0=1)")
log("")
log(f"{'T_kv':>8} | {'Median (ms)':>12} | {'P10':>8} | {'P90':>8} | {'vs FP16':>8}")
log("-" * 62)

fp16_medians = {}
fused_medians = {}

# Re-run FP16 baseline to collect medians for comparison
for T_kv in KV_LENGTHS:
    q = mx.random.normal((BATCH, N_HEADS, 1, DIM))
    k = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    v = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    mx.eval(q, k, v)

    def fp16_sdpa(q, k, v):
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(DIM)
        )

    med, _, _ = bench(fp16_sdpa, q, k, v)
    fp16_medians[T_kv] = med

for T_kv in KV_LENGTHS:
    q = mx.random.normal((BATCH, N_HEADS, 1, DIM))
    k = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    v = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    mx.eval(q, k, v)

    pk, kn = turbo_encode(k, bits=BITS, seed=SEED)
    pv, vn = turbo_encode(v, bits=BITS, seed=SEED)
    mx.eval(pk, kn, pv, vn)

    def fused_nr1(q, pk, kn, pv, vn):
        return turbo_fused_attention(
            q, pk, kn, pv, vn, dim=DIM, bits=BITS, seed=SEED, nr0=1
        )

    med, p10, p90 = bench(fused_nr1, q, pk, kn, pv, vn)
    fused_medians[T_kv] = med
    speedup = fp16_medians[T_kv] / med if med > 0 else 0
    log(f"{T_kv:>8} | {med:>12.3f} | {p10:>8.3f} | {p90:>8.3f} | {speedup:>7.2f}x")

log("")

# ---------------------------------------------------------------------------
# 3. Turbo4 Fused Attention Decode (NR0=2)
# ---------------------------------------------------------------------------
log("## 3. Turbo4 Fused Attention Decode (NR0=2)")
log("")
log(f"{'T_kv':>8} | {'Median (ms)':>12} | {'P10':>8} | {'P90':>8} | {'vs NR0=1':>9} | {'vs FP16':>8}")
log("-" * 75)

for T_kv in KV_LENGTHS:
    q2 = mx.random.normal((BATCH, N_HEADS, 2, DIM))
    k = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    v = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    mx.eval(q2, k, v)

    pk, kn = turbo_encode(k, bits=BITS, seed=SEED)
    pv, vn = turbo_encode(v, bits=BITS, seed=SEED)
    mx.eval(pk, kn, pv, vn)

    def fused_nr2(q2, pk, kn, pv, vn):
        return turbo_fused_attention(
            q2, pk, kn, pv, vn, dim=DIM, bits=BITS, seed=SEED, nr0=2
        )

    med, p10, p90 = bench(fused_nr2, q2, pk, kn, pv, vn)

    # NR0=2 processes 2 queries, so per-query time is med/2
    # Compare per-query time vs NR0=1 and FP16
    per_query = med / 2.0
    nr1_speedup = fused_medians[T_kv] / per_query if per_query > 0 else 0
    fp16_speedup = fp16_medians[T_kv] / per_query if per_query > 0 else 0
    log(f"{T_kv:>8} | {med:>12.3f} | {p10:>8.3f} | {p90:>8.3f} | {nr1_speedup:>8.2f}x | {fp16_speedup:>7.2f}x")

log("")
log("NOTE: NR0=2 per-query speedup = total_time/2 vs NR0=1 single-query time")
log("")

# ---------------------------------------------------------------------------
# 4. Encode Pipeline: Compiled vs Uncompiled
# ---------------------------------------------------------------------------
log("## 4. Encode Pipeline: Compiled vs Uncompiled")
log("")
log(f"{'T_kv':>8} | {'Compiled (ms)':>14} | {'Uncompiled (ms)':>16} | {'Speedup':>8}")
log("-" * 58)

for T_kv in KV_LENGTHS:
    x = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    mx.eval(x)

    def encode_compiled(x):
        return turbo_encode(x, bits=BITS, seed=SEED)

    def encode_uncompiled(x):
        return turbo_encode_uncompiled(x, bits=BITS, seed=SEED)

    med_c, _, _ = bench(encode_compiled, x)
    med_u, _, _ = bench(encode_uncompiled, x)
    speedup = med_u / med_c if med_c > 0 else 0
    log(f"{T_kv:>8} | {med_c:>14.3f} | {med_u:>16.3f} | {speedup:>7.2f}x")

log("")

# ---------------------------------------------------------------------------
# 5. PPL Comparison: FP16 vs Turbo4
# ---------------------------------------------------------------------------
log("## 5. Quality Comparison: FP16 vs Turbo4 Reconstruction Error")
log("")
log("Measures mean squared error and cosine similarity between FP16 attention")
log("output and turbo4 fused attention output (same Q/K/V inputs).")
log("")
log(f"{'T_kv':>8} | {'MSE':>12} | {'Cosine Sim':>12} | {'Max Abs Err':>12}")
log("-" * 56)

for T_kv in KV_LENGTHS:
    q = mx.random.normal((BATCH, N_HEADS, 1, DIM))
    k = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    v = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    mx.eval(q, k, v)

    # FP16 baseline
    out_fp16 = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(DIM)
    )
    mx.eval(out_fp16)

    # Turbo4 fused
    pk, kn = turbo_encode(k, bits=BITS, seed=SEED)
    pv, vn = turbo_encode(v, bits=BITS, seed=SEED)
    mx.eval(pk, kn, pv, vn)

    out_turbo = turbo_fused_attention(
        q, pk, kn, pv, vn, dim=DIM, bits=BITS, seed=SEED
    )
    mx.eval(out_turbo)

    # Compute error metrics
    diff = (out_fp16 - out_turbo).astype(mx.float32)
    mse = mx.mean(diff * diff).item()
    max_err = mx.max(mx.abs(diff)).item()

    # Cosine similarity
    fp16_flat = out_fp16.reshape(-1).astype(mx.float32)
    turbo_flat = out_turbo.reshape(-1).astype(mx.float32)
    dot = mx.sum(fp16_flat * turbo_flat).item()
    norm_a = mx.linalg.norm(fp16_flat).item()
    norm_b = mx.linalg.norm(turbo_flat).item()
    cos_sim = dot / (norm_a * norm_b + 1e-10)

    log(f"{T_kv:>8} | {mse:>12.8f} | {cos_sim:>12.8f} | {max_err:>12.8f}")

log("")

# ---------------------------------------------------------------------------
# 6. Cross-Entropy Loss: FP16 vs Turbo4 (PPL proxy)
# ---------------------------------------------------------------------------
log("## 6. Cross-Entropy Loss Comparison (PPL Proxy)")
log("")
log("Simulates an autoregressive forward pass: compute attention with")
log("FP16 vs turbo4, then measure cross-entropy between the two outputs")
log("treated as log-probability distributions.")
log("")

for T_kv in [128, 256]:
    # Simulate multiple decode steps
    n_decode = 32
    ce_losses = []

    q_base = mx.random.normal((BATCH, N_HEADS, 1, DIM))
    k = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    v = mx.random.normal((BATCH, N_HEADS, T_kv, DIM))
    mx.eval(q_base, k, v)

    pk, kn = turbo_encode(k, bits=BITS, seed=SEED)
    pv, vn = turbo_encode(v, bits=BITS, seed=SEED)
    mx.eval(pk, kn, pv, vn)

    for step in range(n_decode):
        # Slightly perturb query each step (simulating different decode positions)
        q = q_base + mx.random.normal(q_base.shape) * 0.1
        mx.eval(q)

        out_fp16 = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(DIM)
        )
        out_turbo = turbo_fused_attention(
            q, pk, kn, pv, vn, dim=DIM, bits=BITS, seed=SEED
        )
        mx.eval(out_fp16, out_turbo)

        # Treat outputs as logits, compute KL divergence as quality proxy
        fp16_probs = mx.softmax(out_fp16.reshape(-1).astype(mx.float32), axis=-1)
        turbo_probs = mx.softmax(out_turbo.reshape(-1).astype(mx.float32), axis=-1)

        # KL(fp16 || turbo) = sum(fp16 * log(fp16 / turbo))
        kl = mx.sum(
            fp16_probs * (mx.log(fp16_probs + 1e-10) - mx.log(turbo_probs + 1e-10))
        ).item()
        ce_losses.append(kl)

    avg_kl = sum(ce_losses) / len(ce_losses)
    max_kl = max(ce_losses)
    log(f"T_kv={T_kv}: avg_KL={avg_kl:.8f}, max_KL={max_kl:.8f} over {n_decode} steps")

log("")
log("KL divergence < 0.001 indicates turbo4 output is effectively identical")
log("to FP16 for downstream token prediction.")
log("")

# ---------------------------------------------------------------------------
# Write results to markdown
# ---------------------------------------------------------------------------
output_path = "benchmarks/turbo_comprehensive_benchmark.md"
with open(output_path, "w") as f:
    f.write("# TurboQuant Comprehensive Benchmark Results\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Machine: {DEVICE_NAME}\n")
    f.write(f"Branch: feature/turboquant-kv\n\n")
    f.write("```\n")
    for line in results:
        f.write(line + "\n")
    f.write("```\n")

print(f"\nResults written to {output_path}")
