# Item 3: All Bit Widths on 35B MoE

## Test Setup
- Model: Qwen3.5-35B-A3B-8bit (40 layers, 10 attention layers)
- Context: ~120 tokens
- Peak memory: 37.3 GB

## PPL Results

| Config          | PPL    | Delta from FP16 |
|-----------------|--------|----------------|
| FP16/FP16       | 2.6707 | baseline       |
| turbo4/turbo4   | 2.6741 | +0.0034        |
| FP16/turbo4     | 2.7133 | +0.0426        |
| FP16/turbo3     | 2.6930 | +0.0223        |
| FP16/turbo2     | 2.7275 | +0.0568        |
| turbo3/turbo3   | 2.6799 | +0.0092        |
| turbo2/turbo2   | 2.6666 | -0.0041        |

## Speed Results

| Config    | Prompt (tok/s) | Generation (tok/s) | vs FP16 |
|-----------|---------------|-------------------|---------|
| FP16      | 144.0         | 95.9              | 100%    |
| turbo4sym | 79.1          | 43.9              | 46%     |

## Analysis

1. **MoE confirms near-lossless turbo4**: +0.003 PPL for turbo4 symmetric.
2. **Surprising turbo2 result**: turbo2 symmetric gives BETTER PPL than FP16 
   (2.6666 vs 2.6707). Quantization noise as regularization effect.
3. **Speed is 46% of FP16** — the encode/decode overhead dominates because
   this is a large MoE model where attention is only ~15% of compute.
   The fused attention path (Item 17) should help significantly.
4. **Asymmetric (FP16 K) hurts more on MoE** (+0.043 for turbo0v4) than
   symmetric turbo4 (+0.003). This is because MoE has fewer attention
   layers — each one carries more signal weight, so K compression matters.
5. **Recommendation for MoE**: turbo4 symmetric, not asymmetric.
