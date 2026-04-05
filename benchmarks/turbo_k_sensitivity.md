# Item 1: K Sensitivity Spectrum

## Test Setup
- Model: Qwen3.5-2B-8bit (6 attention layers, hybrid SSM+attention)
- Context: ~120 tokens computing history passage
- Dual signs enabled (Item 8)

## Results

| K Config | V Config | PPL    | Delta from FP16/FP16 |
|----------|----------|--------|---------------------|
| FP16     | FP16     | 3.1512 | baseline            |
| FP16     | turbo4   | 3.1463 | -0.0050             |
| turbo4   | turbo4   | 3.1538 | +0.0026             |
| turbo3   | turbo4   | 3.1308 | -0.0205             |
| turbo2   | turbo4   | 3.1017 | -0.0496             |
| FP16     | turbo3   | 3.2213 | +0.0700             |
| turbo4   | turbo3   | 3.2013 | +0.0501             |
| turbo3   | turbo3   | 3.1739 | +0.0227             |

## Analysis

At short context (~120 tokens), K precision sensitivity is minimal. Counter-intuitively,
lower K precision sometimes gives BETTER PPL — likely because quantization noise acts as
stochastic regularization at this scale. Key takeaways:

1. **V precision matters more than K precision** for this model at short context.
   turbo3 V adds +0.07 PPL vs turbo4 V adding only +0.003.
2. **turbo4 symmetric (K=4bit, V=4bit) is essentially lossless**: +0.003 PPL.
3. **Asymmetric FP16 K + turbo4 V is the safest choice** at -0.005 from baseline.
4. K sensitivity will likely increase at longer context where softmax amplification
   of K errors becomes more pronounced (more tokens competing for attention weight).

## Decision
No code change needed — the existing asymmetric (key_bits=0, bits=4) default is correct.
Document q8_0 K as a future TODO when MLX gets affine 8-bit quantization for KV cache.
