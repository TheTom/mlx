# TurboQuant MLX — Final Results

**Date**: 2026-04-05
**Branch**: feature/turboquant-kv (25 commits)
**Hardware**: Apple M5 Max 128GB

## Qwen3.5-2B 8bit — All Optimizations Active

| Config | Prefill | Decode | vs Baseline | Quality |
|--------|---------|--------|-------------|---------|
| Baseline (f16 KV) | 584 | 204 | 100% | Reference |
| turbo4 all fused | 1,153 (+97%) | 168 | 83% | Indistinguishable |
| turbo4 asymmetric (K=FP16) | 984 (+68%) | 177 | **87%** | Indistinguishable |

## Qwen3.5-35B-A3B 8bit

| Config | Prefill | Decode | vs Baseline |
|--------|---------|--------|-------------|
| Baseline | 11.4 | 95.7 | 100% |
| turbo4 fused | 254.7 (+22x) | 53.9 | 56% |
| turbo4 asymmetric | 241.1 (+21x) | 55.2 | 58% |

## Speed Progression

| Milestone | 2B Decode | vs Baseline |
|-----------|----------|-------------|
| Initial Python (no opts) | 170 | 83% |
| + Incremental decode fix | 170 | 83% |
| + Deferred compression | 202 | 98% (at short ctx) |
| + Fused attention kernel | 175 | 86% |
| + NR0=2 multi-row | N/A | ~2x per-query |
| + Fused encode/decode | 168/177 | 83-87% |

## Quality

- KL divergence < 0.001
- Cosine similarity > 0.989
- Output text indistinguishable from baseline
- PPL identical during prefill (compression deferred)

## KV Memory Savings

| Context | f16 KV | turbo4 KV | Savings |
|---------|--------|-----------|---------|
| 1K | 6.3 MB | 1.7 MB | 73% |
| 8K | 101 MB | 27 MB | 73% |
| 32K | 403 MB | 107 MB | 73% |

## Optimizations Applied (from TurboQuant+ papers)

All 22 checklist items + 5 discoveries evaluated. See Obsidian note for full breakdown.
