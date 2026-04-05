# Item 2: turbo2/turbo3 V with Boundary Protection

## Test Setup
- Model: Qwen3.5-2B-8bit (6 attention layers at idx 3,7,11,15,19,23)
- Context: ~120 tokens
- K=FP16 for all tests, only V compressed
- Boundary = N first + N last attention layers kept at FP16

## Results

| Config          | Boundary | PPL    | Delta from FP16 |
|-----------------|----------|--------|----------------|
| FP16/FP16       | -        | 3.1512 | baseline       |
| turbo0v2        | 0        | 3.2339 | +0.0826        |
| turbo0v2        | 1        | 3.2031 | +0.0518        |
| turbo0v2        | 2        | 3.1905 | +0.0393        |
| turbo0v2        | 3 (=FP)  | 3.1512 | +0.0000        |
| turbo0v3        | 0        | 3.2213 | +0.0700        |
| turbo0v3        | 1        | 3.1689 | +0.0177        |
| turbo0v3        | 2        | 3.1542 | +0.0029        |

## Analysis

1. **Boundary protection is very effective** — especially for turbo3:
   - turbo3 + boundary=2 is essentially lossless (+0.003 PPL)
   - turbo2 + boundary=2 reduces penalty by 52% (0.083 → 0.039)
2. **First/last layers carry disproportionate signal** — confirmed by
   dhawalc's TurboQuantDC independent validation.
3. **With only 6 attention layers, boundary=2 protects 4/6 (67%) of layers**.
   For models with more layers (32+), boundary=2 protects only ~12%.
4. **Recommendation for aggressive compression**: turbo3 + boundary=2.
   turbo2 is not worth the quality hit even with boundary protection.
