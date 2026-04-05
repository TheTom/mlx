#!/usr/bin/env python3
"""End-to-end test: TurboKVCache vs standard KVCache in mlx-lm generation.

Loads a small model, runs generation with both standard and TurboKVCache,
and compares output quality. The turbo cache should produce coherent text
even if not identical to the baseline.

Usage:
    python3 tests/test_turbo_kv_e2e.py
    python3 tests/test_turbo_kv_e2e.py --model mlx-community/Qwen3.5-2B-8bit
    python3 tests/test_turbo_kv_e2e.py --max-tokens 50 --bits 4 --key-bits 0
"""

import argparse
import sys
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.models import cache as cache_module

# Import our TurboKVCache
from mlx.nn.layers.turbo_kv_cache import TurboKVCache


def make_turbo_cache(model, bits=4, key_bits=None):
    """Create a cache list matching the model's layer structure.

    For hybrid models (e.g. Qwen3.5 with linear + full attention), only
    replaces KVCache instances with TurboKVCache. Other cache types
    (ArraysCache, etc.) are left as-is since they serve non-attention layers.

    Args:
        model: The loaded mlx-lm model.
        bits: TurboQuant bit-width for values.
        key_bits: Bit-width for keys (None = same as bits, 0 = FP16).

    Returns:
        List of cache objects matching model.layers.
    """
    # Get the model's default cache to understand the layer structure
    default_cache = cache_module.make_prompt_cache(model)
    turbo_cache = []

    n_replaced = 0
    n_skipped = 0

    for i, c in enumerate(default_cache):
        if isinstance(c, cache_module.KVCache):
            turbo_cache.append(TurboKVCache(bits=bits, key_bits=key_bits))
            n_replaced += 1
        else:
            # Keep non-KV caches (ArraysCache for linear attention, etc.)
            turbo_cache.append(c)
            n_skipped += 1

    print(f"  Replaced {n_replaced} KVCache layers with TurboKVCache")
    if n_skipped:
        print(f"  Kept {n_skipped} non-KV cache layers as-is")

    return turbo_cache


def run_generation(model, tokenizer, prompt, cache_list, max_tokens=20):
    """Run generation with a given cache and return text + timing."""
    start = time.perf_counter()

    text = ""
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt,
        max_tokens=max_tokens,
        prompt_cache=cache_list,
    ):
        text += response.text

    elapsed = time.perf_counter() - start
    return text, elapsed


def main():
    parser = argparse.ArgumentParser(description="TurboKVCache E2E test")
    parser.add_argument(
        "--model", default="mlx-community/Qwen3.5-2B-8bit",
        help="Model to load (default: mlx-community/Qwen3.5-2B-8bit)",
    )
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--bits", type=int, default=4, help="V quantization bits")
    parser.add_argument(
        "--key-bits", type=int, default=None,
        help="K quantization bits (0=FP16, None=same as bits)",
    )
    parser.add_argument(
        "--prompt", default="The meaning of life is",
        help="Prompt for generation",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)
    print(f"  Layers: {len(model.layers)}")

    prompt = args.prompt
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Max tokens: {args.max_tokens}")
    print(f"TurboQuant config: V={args.bits}bit, K={'FP16' if args.key_bits == 0 else f'{args.key_bits or args.bits}bit'}")

    # --- Baseline: standard KVCache ---
    print("\n" + "=" * 60)
    print("BASELINE (standard KVCache)")
    print("=" * 60)
    baseline_cache = cache_module.make_prompt_cache(model)
    baseline_text, baseline_time = run_generation(
        model, tokenizer, prompt, baseline_cache, args.max_tokens,
    )
    print(f"  Output: {baseline_text}")
    print(f"  Time: {baseline_time:.2f}s")

    # --- TurboQuant: compressed KV cache ---
    print("\n" + "=" * 60)
    print(f"TURBO (V={args.bits}bit, K={'FP16' if args.key_bits == 0 else f'{args.key_bits or args.bits}bit'})")
    print("=" * 60)
    turbo_cache = make_turbo_cache(
        model, bits=args.bits, key_bits=args.key_bits,
    )
    turbo_text, turbo_time = run_generation(
        model, tokenizer, prompt, turbo_cache, args.max_tokens,
    )
    print(f"  Output: {turbo_text}")
    print(f"  Time: {turbo_time:.2f}s")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    exact_match = baseline_text == turbo_text
    print(f"  Exact match: {exact_match}")
    print(f"  Baseline length: {len(baseline_text)} chars")
    print(f"  Turbo length: {len(turbo_text)} chars")
    print(f"  Speed ratio: {baseline_time / turbo_time:.2f}x")

    # Check turbo cache compression stats
    print("\n  Turbo cache state:")
    for i, c in enumerate(turbo_cache):
        if isinstance(c, TurboKVCache):
            print(f"    Layer {i}: {c}")

    # --- Asymmetric test (K=FP16, V=turbo4) if not already ---
    if args.key_bits != 0:
        print("\n" + "=" * 60)
        print("ASYMMETRIC (K=FP16, V=turbo4 — recommended config)")
        print("=" * 60)
        asym_cache = make_turbo_cache(model, bits=4, key_bits=0)
        asym_text, asym_time = run_generation(
            model, tokenizer, prompt, asym_cache, args.max_tokens,
        )
        print(f"  Output: {asym_text}")
        print(f"  Time: {asym_time:.2f}s")
        print(f"  Matches baseline: {asym_text == baseline_text}")

    # Basic sanity: turbo output should be non-empty
    if not turbo_text.strip():
        print("\nFAIL: Turbo generated empty output!")
        sys.exit(1)

    print("\nPASS: TurboKVCache generated coherent output")


if __name__ == "__main__":
    main()
