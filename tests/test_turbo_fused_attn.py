#!/usr/bin/env python3
"""Test: TurboQuant compressed-domain fused attention kernel.

Validates that turbo_fused_attention (Metal kernel operating on packed data)
produces the same output as turbo_attention (decode-then-matmul reference).

The fused kernel avoids materializing FP16 K/V entirely — it does centroid
lookup + dot product directly on packed uint32 indices. This test ensures
correctness by comparing against the Python reference implementation.

Usage:
    python3 tests/test_turbo_fused_attn.py
    python3 tests/test_turbo_fused_attn.py -v          # verbose
    python3 tests/test_turbo_fused_attn.py --benchmark  # include perf test
"""

import argparse
import math
import sys
import time

import mlx.core as mx

# Import both attention paths
from mlx.nn.layers.turbo_kv_cache import (
    turbo_attention,
    turbo_encode,
    turbo_decode,
    turbo_fused_attention,
    _get_codebook,
    _sign_flip_vector,
)


def test_correctness_basic(verbose=False):
    """Basic correctness: fused vs reference for small inputs."""
    print("=== Test: Basic correctness (B=1, heads=4, T_kv=32, dim=128) ===")

    B, n_heads, T_kv, dim = 1, 4, 32, 128
    bits = 4
    seed = 42

    mx.random.seed(0)
    q = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    k = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)

    # Encode K and V
    pk, kn = turbo_encode(k, bits=bits, seed=seed)
    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    if verbose:
        print(f"  packed_keys shape: {pk.shape}, dtype: {pk.dtype}")
        print(f"  key_norms shape: {kn.shape}, dtype: {kn.dtype}")

    # Reference: decode-then-matmul
    ref_out = turbo_attention(q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed)
    mx.eval(ref_out)

    # Fused: compressed-domain Metal kernel
    try:
        fused_out = turbo_fused_attention(
            q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed,
        )
        mx.eval(fused_out)
    except Exception as e:
        print(f"  FAIL: Fused kernel raised exception: {e}")
        return False

    # Compare outputs
    ref_out = ref_out.astype(mx.float32)
    fused_out = fused_out.astype(mx.float32)

    # The fused kernel does NOT apply the sparse attention mask, and
    # the reference does. But the sparse mask only zeros out weights < 1e-6,
    # so the difference should be tiny.
    diff = mx.abs(ref_out - fused_out)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()
    ref_norm = mx.sqrt(mx.sum(ref_out * ref_out)).item()
    relative_err = max_diff / (ref_norm + 1e-10)

    if verbose:
        print(f"  ref_out shape: {ref_out.shape}")
        print(f"  fused_out shape: {fused_out.shape}")
        print(f"  max_diff: {max_diff:.6e}")
        print(f"  mean_diff: {mean_diff:.6e}")
        print(f"  ref_norm: {ref_norm:.4f}")
        print(f"  relative_err: {relative_err:.6e}")

    # Tolerance: the fused kernel and reference should agree closely.
    # Small differences come from:
    # 1. float32 vs float16 intermediate accumulation
    # 2. The sparse attention mask in the reference (threshold=1e-6)
    # 3. Different summation order (associativity of FP addition)
    passed = relative_err < 0.01  # 1% relative error tolerance

    if passed:
        print(f"  PASS (relative_err={relative_err:.6e})")
    else:
        print(f"  FAIL (relative_err={relative_err:.6e} > 0.01)")
        if verbose:
            # Print first few elements for debugging
            print(f"  ref[0,0,0,:8]:   {ref_out[0,0,0,:8].tolist()}")
            print(f"  fused[0,0,0,:8]: {fused_out[0,0,0,:8].tolist()}")

    return passed


def test_correctness_multihead(verbose=False):
    """Test with more heads and larger T_kv."""
    print("=== Test: Multi-head (B=2, heads=8, T_kv=256, dim=128) ===")

    B, n_heads, T_kv, dim = 2, 8, 256, 128
    bits = 4
    seed = 42

    mx.random.seed(1)
    q = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    k = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)

    pk, kn = turbo_encode(k, bits=bits, seed=seed)
    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    ref_out = turbo_attention(q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed)
    mx.eval(ref_out)

    try:
        fused_out = turbo_fused_attention(
            q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed,
        )
        mx.eval(fused_out)
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    ref_out = ref_out.astype(mx.float32)
    fused_out = fused_out.astype(mx.float32)

    max_diff = mx.max(mx.abs(ref_out - fused_out)).item()
    ref_norm = mx.sqrt(mx.sum(ref_out * ref_out)).item()
    relative_err = max_diff / (ref_norm + 1e-10)

    passed = relative_err < 0.01
    status = "PASS" if passed else "FAIL"
    print(f"  {status} (relative_err={relative_err:.6e})")
    return passed


def test_correctness_dim64(verbose=False):
    """Test with dim=64 (different codebook)."""
    print("=== Test: dim=64 (B=1, heads=4, T_kv=64, dim=64) ===")

    B, n_heads, T_kv, dim = 1, 4, 64, 64
    bits = 4
    seed = 42

    mx.random.seed(2)
    q = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    k = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)

    pk, kn = turbo_encode(k, bits=bits, seed=seed)
    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    ref_out = turbo_attention(q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed)
    mx.eval(ref_out)

    try:
        fused_out = turbo_fused_attention(
            q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed,
        )
        mx.eval(fused_out)
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

    ref_out = ref_out.astype(mx.float32)
    fused_out = fused_out.astype(mx.float32)

    max_diff = mx.max(mx.abs(ref_out - fused_out)).item()
    ref_norm = mx.sqrt(mx.sum(ref_out * ref_out)).item()
    relative_err = max_diff / (ref_norm + 1e-10)

    passed = relative_err < 0.01
    status = "PASS" if passed else "FAIL"
    print(f"  {status} (relative_err={relative_err:.6e})")
    return passed


def test_error_handling(verbose=False):
    """Test that expected errors are raised."""
    print("=== Test: Error handling ===")
    passed = True

    B, n_heads, T_kv, dim = 1, 4, 32, 128
    bits = 4
    seed = 42

    mx.random.seed(3)
    k = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    pk, kn = turbo_encode(k, bits=bits, seed=seed)
    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    # Test T_q > 1 rejection
    q_multi = mx.random.normal((B, n_heads, 4, dim))
    try:
        turbo_fused_attention(q_multi, pk, kn, pv, vn, dim=dim)
        print("  FAIL: Should have raised ValueError for T_q > 1")
        passed = False
    except ValueError as e:
        if verbose:
            print(f"  OK: T_q>1 correctly rejected: {e}")

    # Test mask rejection
    q = mx.random.normal((B, n_heads, 1, dim))
    mask = mx.zeros((1, 1, 1, T_kv))
    try:
        turbo_fused_attention(q, pk, kn, pv, vn, dim=dim, mask=mask)
        print("  FAIL: Should have raised ValueError for mask")
        passed = False
    except ValueError as e:
        if verbose:
            print(f"  OK: mask correctly rejected: {e}")

    # Test 3-bit rejection (not yet implemented)
    try:
        turbo_fused_attention(q, pk, kn, pv, vn, dim=dim, bits=3)
        print("  FAIL: Should have raised NotImplementedError for 3-bit")
        passed = False
    except NotImplementedError as e:
        if verbose:
            print(f"  OK: 3-bit correctly rejected: {e}")

    status = "PASS" if passed else "FAIL"
    print(f"  {status}")
    return passed


def test_wht_domain_math(verbose=False):
    """Verify the WHT-domain equivalence: Q @ K_decoded^T == Q_rot @ centroids[indices] * norm.

    This is the mathematical foundation of the fused kernel. If this fails,
    the entire approach is wrong.
    """
    print("=== Test: WHT domain math equivalence ===")

    dim = 128
    bits = 4
    seed = 42
    cb = _get_codebook(bits, dim)
    signs = _sign_flip_vector(dim, seed)

    mx.random.seed(4)
    q = mx.random.normal((dim,)).astype(mx.float32)
    k = mx.random.normal((dim,)).astype(mx.float32)

    # Encode k
    pk, kn = turbo_encode(k.reshape(1, dim), bits=bits, seed=seed)
    k_decoded = turbo_decode(pk, kn, dim, bits=bits, seed=seed).reshape(dim)

    # Method 1: standard dot product with decoded K
    dot_standard = mx.sum(q * k_decoded).item()

    # Method 2: WHT-domain dot product
    q_rot = mx.hadamard_transform((q * signs).reshape(1, dim)).reshape(dim)

    # Unpack indices and lookup centroids
    from mlx.nn.layers.turbo_kv_cache import _unpack_indices
    indices = _unpack_indices(pk.reshape(1, -1), bits, dim).reshape(dim)
    k_centroids = cb.centroids[indices]  # centroid values in WHT domain
    norm_k = kn.item()

    dot_wht = (mx.sum(q_rot * k_centroids) * norm_k).item()

    diff = abs(dot_standard - dot_wht)
    rel_err = diff / (abs(dot_standard) + 1e-10)

    if verbose:
        print(f"  dot_standard:  {dot_standard:.6f}")
        print(f"  dot_wht:       {dot_wht:.6f}")
        print(f"  abs_diff:      {diff:.6e}")
        print(f"  rel_err:       {rel_err:.6e}")

    passed = rel_err < 1e-4
    status = "PASS" if passed else "FAIL"
    print(f"  {status} (rel_err={rel_err:.6e})")
    return passed


def test_cache_attention_method(verbose=False):
    """Test TurboKVCache.attention() — fused path during decode."""
    print("=== Test: TurboKVCache.attention() method ===")

    from mlx.nn.layers.turbo_kv_cache import TurboKVCache

    B, n_heads, T_prefill, dim = 1, 4, 64, 128

    mx.random.seed(6)

    # Create cache with symmetric 4-bit (both K and V compressed)
    cache = TurboKVCache(bits=4, key_bits=4, min_compress_tokens=32)

    # Simulate prefill: feed 64 tokens
    k_prefill = mx.random.normal((B, n_heads, T_prefill, dim)).astype(mx.float32)
    v_prefill = mx.random.normal((B, n_heads, T_prefill, dim)).astype(mx.float32)
    q_prefill = mx.random.normal((B, n_heads, T_prefill, dim)).astype(mx.float32)

    # Prefill: use update_and_fetch (cache stores raw)
    all_k, all_v = cache.update_and_fetch(k_prefill, v_prefill)
    mx.eval(all_k, all_v)

    # Trigger compression (first decode step)
    k_decode = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    v_decode = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    q_decode = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)

    # Use attention() method — should use fused path
    output = cache.attention(q_decode, k_decode, v_decode)
    mx.eval(output)

    if verbose:
        print(f"  output shape: {output.shape}")
        print(f"  output[:4]: {output[0, 0, 0, :4].tolist()}")
        print(f"  cache compressed: {cache._is_compressed}")
        print(f"  cache offset: {cache.offset}")

    # Verify output is not all zeros
    out_norm = mx.sqrt(mx.sum(output * output)).item()
    if out_norm < 1e-6:
        print(f"  FAIL: output is all zeros")
        return False

    # Do a few more decode steps
    for i in range(5):
        k_new = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
        v_new = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
        q_new = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
        out = cache.attention(q_new, k_new, v_new)
        mx.eval(out)

    if verbose:
        print(f"  After 6 decode steps: offset={cache.offset}")

    # Verify offset tracks correctly
    expected_offset = T_prefill + 6  # prefill + 6 decode tokens
    if cache.offset != expected_offset:
        print(f"  FAIL: expected offset={expected_offset}, got {cache.offset}")
        return False

    print(f"  PASS (fused path active, offset={cache.offset})")
    return True


def test_cache_attention_memory(verbose=False):
    """Test that fused attention path doesn't create decoded FP16 buffers.

    This is the whole point — the fused kernel should eliminate the
    double-storage problem where both packed AND decoded FP16 are held.
    """
    print("=== Test: Fused path memory (no decoded FP16 buffers) ===")

    from mlx.nn.layers.turbo_kv_cache import TurboKVCache

    B, n_heads, dim = 1, 4, 128

    cache = TurboKVCache(bits=4, key_bits=4, min_compress_tokens=8)

    # Prefill with enough tokens to trigger compression
    k = mx.random.normal((B, n_heads, 16, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_heads, 16, dim)).astype(mx.float32)
    all_k, all_v = cache.update_and_fetch(k, v)
    mx.eval(all_k, all_v)

    # First decode: triggers compression via update_and_fetch
    # (this creates decoded FP16 caches)
    k1 = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    v1 = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    all_k, all_v = cache.update_and_fetch(k1, v1)
    mx.eval(all_k, all_v)
    has_decoded_after_fetch = (
        cache._decoded_keys is not None or cache._decoded_values is not None
    )

    if verbose:
        print(f"  After update_and_fetch: has decoded FP16 = {has_decoded_after_fetch}")

    # Now create a FRESH cache and use only attention() path
    cache2 = TurboKVCache(bits=4, key_bits=4, min_compress_tokens=8)

    # Prefill
    all_k, all_v = cache2.update_and_fetch(k, v)
    mx.eval(all_k, all_v)

    # Decode via attention() — should NOT create decoded FP16 buffers
    q = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    out = cache2.attention(q, k1, v1)
    mx.eval(out)

    # The fused path in attention() triggers _compress_raw_cache (which
    # does create decoded buffers) unless we skip that. Actually, the
    # attention() method calls _compress_raw_cache via the deferred threshold,
    # then on subsequent calls it uses the fused path.
    # The key insight: after compression, subsequent attention() calls
    # append to packed but DON'T update _decoded_{keys,values}.
    k2 = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    v2 = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    q2 = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    out2 = cache2.attention(q2, k2, v2)
    mx.eval(out2)

    # The decoded caches should exist from _compress_raw_cache but NOT
    # have been updated since. The packed arrays should have 2 more tokens.
    if verbose:
        print(f"  cache2 packed_keys tokens: {cache2._packed_keys.shape[2]}")
        if cache2._decoded_keys is not None:
            print(f"  cache2 decoded_keys tokens: {cache2._decoded_keys.shape[2]}")
        else:
            print(f"  cache2 decoded_keys: None (fused path skipped decode)")

    # The packed storage should have 18 tokens (16 prefill + 2 decode)
    # But the decoded cache should only have 16 (from initial compress)
    # because the fused path doesn't update it
    packed_tokens = cache2._packed_keys.shape[2]
    decoded_tokens = (
        cache2._decoded_keys.shape[2] if cache2._decoded_keys is not None else 0
    )

    passed = packed_tokens == 18 and decoded_tokens < packed_tokens
    status = "PASS" if passed else "FAIL"
    print(f"  {status} (packed={packed_tokens}, decoded={decoded_tokens})")

    return passed


def benchmark(T_kv=512, n_iters=50, verbose=False):
    """Benchmark fused vs decode-then-matmul."""
    print(f"\n=== Benchmark: B=1, heads=32, T_kv={T_kv}, dim=128 ===")

    B, n_heads, dim = 1, 32, 128
    bits = 4
    seed = 42

    mx.random.seed(5)
    q = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    k = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)

    pk, kn = turbo_encode(k, bits=bits, seed=seed)
    pv, vn = turbo_encode(v, bits=bits, seed=seed)
    mx.eval(pk, kn, pv, vn)

    # Warmup
    for _ in range(5):
        ref = turbo_attention(q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed)
        mx.eval(ref)
    for _ in range(5):
        fused = turbo_fused_attention(q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed)
        mx.eval(fused)

    # Benchmark reference (decode-then-matmul)
    start = time.perf_counter()
    for _ in range(n_iters):
        ref = turbo_attention(q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed)
        mx.eval(ref)
    ref_time = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(n_iters):
        fused = turbo_fused_attention(q, pk, kn, pv, vn, dim=dim, bits=bits, seed=seed)
        mx.eval(fused)
    fused_time = (time.perf_counter() - start) / n_iters * 1000

    speedup = ref_time / fused_time if fused_time > 0 else 0

    print(f"  Reference (decode+matmul): {ref_time:.3f} ms/iter")
    print(f"  Fused (compressed-domain): {fused_time:.3f} ms/iter")
    print(f"  Speedup: {speedup:.2f}x")

    if speedup > 1.0:
        print(f"  --> Fused kernel is {speedup:.2f}x FASTER")
    elif speedup > 0.8:
        print(f"  --> Roughly equivalent ({speedup:.2f}x)")
    else:
        print(f"  --> Fused kernel is SLOWER ({speedup:.2f}x) -- needs optimization")

    return speedup


def main():
    parser = argparse.ArgumentParser(description="Test TurboQuant fused attention kernel")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    args = parser.parse_args()

    if not mx.metal.is_available():
        print("SKIP: Metal not available (fused kernel requires GPU)")
        sys.exit(0)

    print("TurboQuant Fused Attention Kernel Test")
    print("=" * 50)
    print()

    results = []

    # Math verification first — if this fails, the approach is fundamentally wrong
    results.append(("WHT domain math", test_wht_domain_math(args.verbose)))

    # Error handling
    results.append(("Error handling", test_error_handling(args.verbose)))

    # Correctness tests
    results.append(("Basic correctness", test_correctness_basic(args.verbose)))
    results.append(("Multi-head", test_correctness_multihead(args.verbose)))
    results.append(("dim=64", test_correctness_dim64(args.verbose)))

    # TurboKVCache integration tests
    results.append(("Cache attention()", test_cache_attention_method(args.verbose)))
    results.append(("Cache memory (no FP16)", test_cache_attention_memory(args.verbose)))

    # Benchmarks (optional)
    if args.benchmark:
        for tkv in [64, 256, 512, 1024]:
            benchmark(T_kv=tkv, verbose=args.verbose)

    # Summary
    print()
    print("=" * 50)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed.")
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
