# TurboQuant Comprehensive Benchmark Results

Date: 2026-04-05
Machine: Apple M5 Max
Branch: feature/turboquant-kv
Config: B=1, H=8, D=128, bits=4 (turbo4)

## Summary

| Optimization | Result | Notes |
|---|---|---|
| NR0=2 multi-row amortization | **~2x per-query throughput** | Near-perfect amortization of K/V dequant |
| Compiled encode (`mx.compile`) | **1.03–1.18x** encode speedup | Modest win from graph fusion |
| Quality (turbo4 vs FP16) | **>0.989 cosine similarity** | KL < 0.001, negligible PPL impact |
| Fused NR0=1 vs FP16 SDPA | **0.33–0.56x** (slower) | Python dispatch + custom kernel overhead dominates at small T_kv |

## Key Findings

### NR0=2 is a near-perfect 2x
The NR0=2 kernel processes 2 queries for essentially the same wall-clock time as
1 query (~0.337ms vs 0.334ms at T_kv=128). This means the K/V dequant work was
completely amortized — the bottleneck was memory bandwidth for reading packed data,
and NR0=2 reads it once for both queries.

Per-query improvement: **1.97–1.98x** across all context lengths. This is
bandwidth-bound nirvana — almost zero overhead from the second query.

### Fused kernel is slower than FP16 SDPA at these sizes
The fused kernel (NR0=1) is 0.33–0.56x of FP16 SDPA speed. This is expected at
small context (128–1024 tokens) because:

1. **MLX's SDPA is heavily optimized** — Apple's built-in kernel uses tiled matmul
   with shared memory, vs our per-element centroid lookup
2. **Python dispatch overhead** — pre-rotate Q, struct.pack scale, mx.fast.metal_kernel
   launch all add constant overhead that dominates sub-ms kernels
3. **The win comes at large context** — when memory bandwidth for FP16 K/V exceeds
   the GPU's capacity, our 1/8th bandwidth packed format wins. The crossover point
   is likely T_kv > 4096 where FP16 K/V exceeds L2 cache

**TODO: Benchmark at T_kv=4096, 8192, 16384** to find the crossover point.

### Compiled encode gives modest speedup
`mx.compile()` fusion gives 1.03–1.18x encode speedup. The graph was already
fairly efficient (MLX lazy eval batches ops naturally). The win is from eliminating
kernel launch gaps between norm → sign_flip → WHT → quantize → pack (5 Metal
dispatches → 1 fused dispatch).

### Quality is excellent
- **Cosine similarity**: 0.989–0.992 across all context lengths
- **MSE**: decreases with context (0.00035 at 128 → 0.00005 at 1024) — errors average out
- **KL divergence**: < 0.00024 — turbo4 output is statistically indistinguishable from FP16
  for next-token prediction

## Raw Results

### 1. Baseline FP16 SDPA Decode

| T_kv | Median (ms) | P10 | P90 |
|------|------------|-----|-----|
| 128 | 0.190 | 0.162 | 0.253 |
| 256 | 0.159 | 0.142 | 0.203 |
| 512 | 0.167 | 0.142 | 0.206 |
| 1024 | 0.242 | 0.213 | 0.293 |

### 2. Turbo4 Fused Attention (NR0=1)

| T_kv | Median (ms) | P10 | P90 | vs FP16 |
|------|------------|-----|-----|---------|
| 128 | 0.334 | 0.286 | 0.602 | 0.51x |
| 256 | 0.293 | 0.281 | 0.331 | 0.56x |
| 512 | 0.353 | 0.341 | 0.397 | 0.42x |
| 1024 | 0.478 | 0.461 | 0.516 | 0.33x |

### 3. Turbo4 Fused Attention (NR0=2)

| T_kv | Median (ms) | P10 | P90 | Per-query vs NR0=1 | Per-query vs FP16 |
|------|------------|-----|-----|-------------------|------------------|
| 128 | 0.337 | 0.276 | 0.395 | 1.98x | 1.01x |
| 256 | 0.296 | 0.283 | 0.341 | 1.98x | 1.10x |
| 512 | 0.359 | 0.341 | 0.385 | 1.97x | 0.83x |
| 1024 | 0.482 | 0.460 | 0.518 | 1.99x | 0.65x |

NOTE: NR0=2 per-query time = total_time / 2. The 2 queries share K/V dequant work.

### 4. Encode Pipeline: Compiled vs Uncompiled

| T_kv | Compiled (ms) | Uncompiled (ms) | Speedup |
|------|--------------|----------------|---------|
| 128 | 0.223 | 0.258 | 1.16x |
| 256 | 0.249 | 0.257 | 1.03x |
| 512 | 0.248 | 0.273 | 1.10x |
| 1024 | 0.281 | 0.331 | 1.18x |

### 5. Quality: FP16 vs Turbo4 Reconstruction Error

| T_kv | MSE | Cosine Sim | Max Abs Err |
|------|-----|-----------|-------------|
| 128 | 0.00035424 | 0.99081338 | 0.06787653 |
| 256 | 0.00017624 | 0.99057180 | 0.04338371 |
| 512 | 0.00009334 | 0.99177332 | 0.04101210 |
| 1024 | 0.00005293 | 0.98963200 | 0.03073353 |

### 6. KL Divergence (PPL Proxy)

| T_kv | Avg KL | Max KL | Steps |
|------|--------|--------|-------|
| 128 | 0.00022528 | 0.00023872 | 32 |
| 256 | 0.00012652 | 0.00013745 | 32 |

KL < 0.001 = turbo4 and FP16 produce statistically identical token distributions.

## Next Steps

1. **Benchmark at T_kv=4096+** — find the crossover where fused beats FP16 SDPA
2. **Reduce Python dispatch overhead** — pre-compute rotated Q in C++, or batch the pre/post transforms
3. **NR0=4 or NR0=8** — if speculative decoding generates 4-8 candidates, we could amortize further
4. **Threadgroup-local centroid table** — currently in device memory, moving to threadgroup constant memory
   would eliminate per-word latency for centroid[idx] lookups
