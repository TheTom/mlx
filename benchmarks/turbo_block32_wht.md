# Item 6: block_size=32 WHT E2E Results

## Test Setup
- Model: Qwen3.5-2B-8bit (dim=256, 6 attention layers)
- Context: ~120 tokens
- Compared full-dim WHT (dim=256) vs block=32 WHT

## PPL Results

| Config         | Full WHT | Block=32 | Delta    |
|----------------|----------|----------|----------|
| FP16 baseline  | 3.1512   | -        | -        |
| turbo4 sym     | 3.1633   | 3.1425   | -0.0208  |
| turbo0v4       | 3.1471   | 3.1423   | -0.0048  |

## Speed Results (dim=256, 100 tokens, Python path)

| Operation | Full WHT | Block=32 | Speedup |
|-----------|----------|----------|---------|
| Encode    | 0.34ms   | 0.27ms   | 21%     |
| Decode    | 0.25ms   | 0.24ms   | 5%      |

## Analysis

**Block=32 is better on ALL metrics:**
1. **Better PPL** (-0.02 for symmetric, -0.005 for asymmetric)
2. **Faster encode** (21% speedup on Python path)
3. **Slightly faster decode** (5%)

The PPL improvement is counter-intuitive. Possible explanation: block=32 WHT
only decorrelates within 32-element blocks, creating a "gentler" transform that
preserves more local structure. The Beta-distribution centroids were fitted for
the full-dim distribution, and block WHT may produce coordinates that are
slightly easier to quantize.

## Implementation Path
This is a positive result worth implementing:
1. Add `block_size` parameter to turbo_encode/turbo_decode (default=32)
2. Update fused Metal kernels — the WHT butterfly only runs log2(32)=5 stages
   instead of log2(256)=8, reducing shared memory pressure
3. Update fused attention query pre-rotation to use block WHT
4. Need new centroids: block=32 WHT produces different distribution than full-dim
   (current centroids are for dim=256, may not be optimal for block=32)

## TODO
- [ ] Implement block_size parameter in Python encode/decode
- [ ] Update Metal encode/decode kernels for block WHT
- [ ] Compute Beta-distribution centroids for (4, 32) block size
- [ ] Benchmark on more models (especially dim=128 models)
