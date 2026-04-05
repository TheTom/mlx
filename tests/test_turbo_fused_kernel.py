#!/usr/bin/env python3
"""Test and benchmark fused turbo_encode/turbo_decode Metal kernels.

Validates:
1. Correctness: fused output matches Python graph output
2. Performance: timing comparison (fused vs compiled vs uncompiled)

Usage:
    python3 tests/test_turbo_fused_kernel.py
"""

import time
import mlx.core as mx
import numpy as np


def test_encode_correctness():
    """Verify fused encode matches Python (uncompiled) encode output."""
    from mlx.nn.layers.turbo_kv_cache import (
        turbo_encode_uncompiled as turbo_encode,
        turbo_encode_fused,
    )

    print("=" * 60)
    print("TEST: Fused encode correctness")
    print("=" * 60)

    for dim in [64, 128, 256]:
        for shape_prefix in [(1, 8, 1), (2, 4, 16), (1, 32, 1)]:
            shape = (*shape_prefix, dim)
            x = mx.random.normal(shape).astype(mx.float32)
            mx.eval(x)

            # Python path
            packed_py, norms_py = turbo_encode(x, bits=4, seed=42)
            mx.eval(packed_py, norms_py)

            # Fused kernel path
            packed_fused, norms_fused = turbo_encode_fused(x, bits=4, seed=42)
            mx.eval(packed_fused, norms_fused)

            # Compare norms
            norms_py_np = np.array(norms_py.reshape(-1))
            norms_fused_np = np.array(norms_fused.reshape(-1))
            norm_diff = np.max(np.abs(norms_py_np - norms_fused_np))

            # Compare packed indices
            packed_py_np = np.array(packed_py.reshape(-1))
            packed_fused_np = np.array(packed_fused.reshape(-1))
            packed_match = np.array_equal(packed_py_np, packed_fused_np)

            status = "PASS" if (norm_diff < 1e-4 and packed_match) else "FAIL"
            print(f"  {status} shape={shape} | norm_diff={norm_diff:.6f} | packed_match={packed_match}")

            if status == "FAIL":
                # Print first mismatches for debugging
                if not packed_match:
                    mismatches = np.where(packed_py_np != packed_fused_np)[0]
                    print(f"    First 5 packed mismatches at indices: {mismatches[:5]}")
                    for idx in mismatches[:5]:
                        print(f"      [{idx}] py=0x{packed_py_np[idx]:08x} fused=0x{packed_fused_np[idx]:08x}")
                if norm_diff >= 1e-4:
                    print(f"    Norm diff too large: {norm_diff}")

    print()


def test_decode_correctness():
    """Verify fused decode matches Python (graph) decode output."""
    import os
    os.environ["TURBO_DISABLE_FUSED_KERNEL"] = "1"
    from mlx.nn.layers.turbo_kv_cache import (
        turbo_encode_uncompiled as turbo_encode,
        turbo_decode as _turbo_decode_dispatch,
        turbo_decode_fused,
    )
    # Force Python graph path for reference decode
    def turbo_decode(packed, norms, dim, bits=4, seed=42):
        from mlx.nn.layers.turbo_kv_cache import (
            _get_codebook, _unpack_indices, _sign_flip_vector,
        )
        cb = _get_codebook(bits, dim)
        indices = _unpack_indices(packed, bits, dim)
        x_rotated = cb.centroids[indices]
        x_flipped = mx.hadamard_transform(x_rotated)
        signs = _sign_flip_vector(dim, seed)
        x_unit = x_flipped * signs
        return x_unit * norms
    os.environ.pop("TURBO_DISABLE_FUSED_KERNEL", None)

    print("=" * 60)
    print("TEST: Fused decode correctness")
    print("=" * 60)

    for dim in [64, 128, 256]:
        for shape_prefix in [(1, 8, 1), (2, 4, 16), (1, 32, 1)]:
            shape = (*shape_prefix, dim)
            x = mx.random.normal(shape).astype(mx.float32)
            mx.eval(x)

            # Encode with Python path (known correct)
            packed, norms = turbo_encode(x, bits=4, seed=42)
            mx.eval(packed, norms)

            # Decode: Python path
            decoded_py = turbo_decode(packed, norms, dim=dim, bits=4, seed=42)
            mx.eval(decoded_py)

            # Decode: Fused kernel path
            decoded_fused = turbo_decode_fused(packed, norms, dim=dim, bits=4, seed=42)
            mx.eval(decoded_fused)

            # Compare
            py_np = np.array(decoded_py.reshape(-1))
            fused_np = np.array(decoded_fused.reshape(-1))
            max_diff = np.max(np.abs(py_np - fused_np))
            mean_diff = np.mean(np.abs(py_np - fused_np))

            status = "PASS" if max_diff < 1e-3 else "FAIL"
            print(f"  {status} shape={shape} | max_diff={max_diff:.6f} | mean_diff={mean_diff:.8f}")

            if status == "FAIL":
                # Show worst mismatches
                worst_indices = np.argsort(np.abs(py_np - fused_np))[-5:]
                for idx in worst_indices:
                    print(f"    [{idx}] py={py_np[idx]:.6f} fused={fused_np[idx]:.6f} diff={abs(py_np[idx]-fused_np[idx]):.6f}")

    print()


def test_roundtrip():
    """Verify fused encode→fused decode preserves information (same as Python roundtrip)."""
    from mlx.nn.layers.turbo_kv_cache import (
        turbo_encode,
        turbo_decode,
        turbo_encode_fused,
        turbo_decode_fused,
    )

    print("=" * 60)
    print("TEST: Fused roundtrip quality")
    print("=" * 60)

    dim = 128
    x = mx.random.normal((1, 8, 64, dim)).astype(mx.float32)
    mx.eval(x)

    # Python roundtrip
    packed_py, norms_py = turbo_encode(x, bits=4, seed=42)
    decoded_py = turbo_decode(packed_py, norms_py, dim=dim, bits=4, seed=42)
    mx.eval(decoded_py)

    # Fused roundtrip
    packed_f, norms_f = turbo_encode_fused(x, bits=4, seed=42)
    decoded_f = turbo_decode_fused(packed_f, norms_f, dim=dim, bits=4, seed=42)
    mx.eval(decoded_f)

    # Compare reconstruction error
    x_np = np.array(x.reshape(-1))
    py_np = np.array(decoded_py.reshape(-1))
    fused_np = np.array(decoded_f.reshape(-1))

    py_mse = np.mean((x_np - py_np) ** 2)
    fused_mse = np.mean((x_np - fused_np) ** 2)

    py_cos = np.dot(x_np, py_np) / (np.linalg.norm(x_np) * np.linalg.norm(py_np))
    fused_cos = np.dot(x_np, fused_np) / (np.linalg.norm(x_np) * np.linalg.norm(fused_np))

    print(f"  Python roundtrip:  MSE={py_mse:.8f}  cos_sim={py_cos:.8f}")
    print(f"  Fused roundtrip:   MSE={fused_mse:.8f}  cos_sim={fused_cos:.8f}")
    print(f"  MSE ratio (fused/python): {fused_mse / py_mse:.4f}")
    print()


def benchmark_encode():
    """Time fused vs compiled vs uncompiled encode."""
    from mlx.nn.layers.turbo_kv_cache import (
        turbo_encode,
        turbo_encode_uncompiled,
        turbo_encode_fused,
    )

    print("=" * 60)
    print("BENCHMARK: Encode speed (single token, 8 heads, dim=128)")
    print("=" * 60)

    dim = 128
    n_iters = 200
    warmup = 50

    # Typical decode shape: (1, n_heads, 1, dim)
    for n_heads in [8, 32]:
        x = mx.random.normal((1, n_heads, 1, dim)).astype(mx.float32)
        mx.eval(x)

        # Warm up all paths
        for _ in range(warmup):
            p, n = turbo_encode(x, bits=4, seed=42)
            mx.eval(p, n)
        for _ in range(warmup):
            p, n = turbo_encode_uncompiled(x, bits=4, seed=42)
            mx.eval(p, n)
        for _ in range(warmup):
            p, n = turbo_encode_fused(x, bits=4, seed=42)
            mx.eval(p, n)

        # Benchmark uncompiled
        t0 = time.perf_counter()
        for _ in range(n_iters):
            p, n = turbo_encode_uncompiled(x, bits=4, seed=42)
            mx.eval(p, n)
        t_uncompiled = (time.perf_counter() - t0) / n_iters * 1000

        # Benchmark compiled (mx.compile)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            p, n = turbo_encode(x, bits=4, seed=42)
            mx.eval(p, n)
        t_compiled = (time.perf_counter() - t0) / n_iters * 1000

        # Benchmark fused Metal kernel
        t0 = time.perf_counter()
        for _ in range(n_iters):
            p, n = turbo_encode_fused(x, bits=4, seed=42)
            mx.eval(p, n)
        t_fused = (time.perf_counter() - t0) / n_iters * 1000

        print(f"\n  n_heads={n_heads}, shape=(1, {n_heads}, 1, {dim})")
        print(f"    Uncompiled:  {t_uncompiled:.3f} ms/call")
        print(f"    Compiled:    {t_compiled:.3f} ms/call  ({t_uncompiled/t_compiled:.2f}x vs uncompiled)")
        print(f"    Fused Metal: {t_fused:.3f} ms/call  ({t_uncompiled/t_fused:.2f}x vs uncompiled, {t_compiled/t_fused:.2f}x vs compiled)")

    print()


def benchmark_decode():
    """Time fused vs Python decode."""
    from mlx.nn.layers.turbo_kv_cache import (
        turbo_encode,
        turbo_decode,
        turbo_decode_fused,
    )

    print("=" * 60)
    print("BENCHMARK: Decode speed (64 tokens, 8 heads, dim=128)")
    print("=" * 60)

    dim = 128
    n_iters = 200
    warmup = 50

    for n_tokens in [1, 64, 256]:
        for n_heads in [8, 32]:
            x = mx.random.normal((1, n_heads, n_tokens, dim)).astype(mx.float32)
            mx.eval(x)
            packed, norms = turbo_encode(x, bits=4, seed=42)
            mx.eval(packed, norms)

            # Warm up
            for _ in range(warmup):
                d = turbo_decode(packed, norms, dim=dim, bits=4, seed=42)
                mx.eval(d)
            for _ in range(warmup):
                d = turbo_decode_fused(packed, norms, dim=dim, bits=4, seed=42)
                mx.eval(d)

            # Benchmark Python decode
            t0 = time.perf_counter()
            for _ in range(n_iters):
                d = turbo_decode(packed, norms, dim=dim, bits=4, seed=42)
                mx.eval(d)
            t_python = (time.perf_counter() - t0) / n_iters * 1000

            # Benchmark fused decode
            t0 = time.perf_counter()
            for _ in range(n_iters):
                d = turbo_decode_fused(packed, norms, dim=dim, bits=4, seed=42)
                mx.eval(d)
            t_fused = (time.perf_counter() - t0) / n_iters * 1000

            print(f"\n  tokens={n_tokens}, heads={n_heads}")
            print(f"    Python:      {t_python:.3f} ms/call")
            print(f"    Fused Metal: {t_fused:.3f} ms/call  ({t_python/t_fused:.2f}x speedup)")

    print()


if __name__ == "__main__":
    print("Fused turbo_encode / turbo_decode Metal kernel tests")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"Device: {mx.default_device()}")
    print()

    test_encode_correctness()
    test_decode_correctness()
    test_roundtrip()
    benchmark_encode()
    benchmark_decode()
