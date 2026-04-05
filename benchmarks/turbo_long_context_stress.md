# Item 5: Long Context Stress Test

## Test Setup
- Model: Qwen3.5-2B-8bit (6 attention layers, dim=256, 2 KV heads)
- Context lengths: 512, 1024, 2048, 4000 tokens
- Decode: 32 tokens after prefill

## Speed Results

| Context | FP16 Decode | turbo4sym Decode | turbo0v4 Decode | turbo4 % |
|---------|------------|-----------------|-----------------|----------|
| 512     | 174.1      | 139.2           | 145.8           | 80%      |
| 1024    | 173.7      | 139.2           | 144.9           | 80%      |
| 2048    | 171.9      | 137.0           | 142.5           | 80%      |
| 4000    | 168.8      | 130.7           | -               | 77%      |

## Memory Results
- FP16 KV cache at 4K tokens: 46.9 MB
- Turbo KV cache at 4K tokens: 46.9 MB (same — double-storage problem)

## Analysis

1. **Turbo decode speed is 77-84% of FP16** across all context lengths — consistent.
   The overhead is the encode/decode step, not context-length dependent.
2. **Memory savings: ZERO** due to double-storage problem. TurboKVCache keeps both
   packed storage AND decoded FP16 cache for standard SDPA compatibility.
3. **Asymmetric (turbo0v4) is ~3% faster than symmetric (turbo4sym)** because it
   skips K encoding/decoding entirely.
4. **To get memory savings**, must use fused_attention=True path (eliminates decoded
   FP16 buffers). Or use the packed-only storage with turbo_fused_attention.
5. **At this model size (2B), KV cache is only 47MB at 4K** — too small for memory
   savings to matter. The real benefit is at 7B+ models with 32K+ context.

## Key Limitation
The current update_and_fetch approach maintains decoded FP16 caches for SDPA
compatibility, negating memory savings. The fused attention path (already implemented)
is the correct solution but requires model-level integration.
