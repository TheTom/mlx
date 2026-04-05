# Long Context Memory Test (Item 16)

**Date:** 2026-04-05
**Model:** mlx-community/Qwen3.5-2B-8bit (24 layers, 6 KV cache layers)
**Prompt:** "word " * 4000 (~4001 tokens)
**Max tokens:** 32

## Results

| Config | Prompt tok/s | Decode tok/s | Cache Memory | Peak GPU Memory |
|--------|-------------|-------------|-------------|----------------|
| Baseline (KVCache) | 8,898 | 202.8 | 56.9 MB | 3.267 GB |
| TurboKVCache bits=4 min_compress=256 | 11,084 | 159.0 | 106.7 MB | 3.262 GB |

## Analysis

### Memory savings are negligible on this model
- Peak GPU memory difference: **4.9 MB (0.1%)**
- The cache is only 57 MB baseline vs 107 MB turbo — turbo is actually LARGER
- Why? TurboKVCache stores both packed indices AND decoded FP16 cache (for O(1)
  incremental decode). The decoded FP16 cache is the same size as baseline, and
  the packed indices are additional overhead.

### The double-storage problem
TurboKVCache keeps:
1. `_packed_keys` / `_packed_values` (compressed, ~25% of FP16 for 4-bit)
2. `_decoded_keys` / `_decoded_values` (full FP16, same size as baseline)

This means memory usage is ~125% of baseline, not less. The compressed storage
is useful for:
- Saving to disk / serialization
- Future fused kernel that reads from packed directly (no decoded cache needed)
- Models with much larger KV (bigger models, more layers)

### Decode speed penalty at long context
At 4K tokens with compression active: 159 tok/s vs 203 tok/s baseline (21% slower).
The encode step on each new token is the bottleneck — turbo_encode involves
normalize → sign_flip → hadamard_transform → boundary quantize → pack.

### This model is too small to show KV compression benefits
Only 6 of 24 layers use KVCache. The KV cache is <2% of total GPU memory.
To see meaningful savings, need:
- Larger models (7B+, 70B) where KV is a bigger fraction
- Models with more KV layers
- Much longer context (32K+) where KV dominates memory

## TODO
- [ ] Drop the decoded FP16 cache — read from packed storage during attention
      (requires fused Metal kernel or decode-on-read approach)
- [ ] Test on larger model (e.g., Qwen3.5-7B or Llama-3-8B)
- [ ] Test at 32K context where KV memory becomes dominant
