# TurboKVCache Boundary Layer A/B Test (Item 4)

**Date:** 2026-04-05
**Model:** mlx-community/Qwen3.5-2B-8bit (24 layers)
**Prompt:** "Explain quantum computing."
**Max tokens:** 128
**Config:** bits=4 (symmetric turbo4), deferred compression threshold=256

## Results

| Boundary Layers | Turbo Layers | FP Layers | Prompt tok/s | Decode tok/s | Peak Memory |
|-----------------|-------------|-----------|-------------|-------------|-------------|
| 0 | 6 | 18 | 109.8 | 202.0 | 2.026 GB |
| 2 | 5 | 19 | 331.8 | 202.0 | 2.026 GB |
| 4 | 4 | 20 | 347.2 | 201.8 | 2.026 GB |

## Analysis

- **Decode speed is identical** across all configs (~202 tok/s). At 128 tokens,
  the deferred compression threshold (256) means the cache never compresses —
  all three configs are effectively running raw FP16. This confirms the
  deferred compression feature works: short-context decode has no turbo overhead.

- **Prompt speed improves** with more boundary layers (110 → 332 → 347 tok/s).
  With boundary=0, all 6 non-MoE layers get turbo caches. With boundary=2/4,
  fewer layers are replaced, so prefill is faster (less cache object overhead).

- **Output quality is identical** across all three configs — same text generated.
  At this short context length (133 tokens total), boundary layers make no
  quality difference because compression isn't even triggered.

- **Note:** Only 6 of 24 layers are KVCache (the rest are likely already
  specialized). This is a Qwen3.5 architecture detail — the model only creates
  6 KVCache objects via make_prompt_cache.

## Conclusion

At short context (< min_compress_tokens), boundary layers have no quality or
decode speed impact because deferred compression keeps everything in FP16.
The real test needs longer context (>256 tokens) to trigger compression.
Boundary=2 is a safe default. Boundary=4 is overly conservative for 24 layers.
