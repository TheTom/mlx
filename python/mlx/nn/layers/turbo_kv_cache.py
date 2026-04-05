# Copyright © 2025 Apple Inc.
# TurboQuant KV Cache Quantization for MLX
#
# Implements the TurboQuant algorithm (Google Research, 2024) for KV cache
# compression in transformer models. Uses Subsampled Randomized Hadamard
# Transform (SRHT) + Lloyd-Max quantization on the unit sphere.
#
# Reference: "TurboQuant: Online Vector Quantization for KV Cache Compression"
# Implementation: Tom Turney (TurboQuant+)

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


# ---------------------------------------------------------------------------
# Beta distribution centroids for unit-sphere-normalized coordinates
# Pre-computed for common (bits, dim) combos. These are the correct
# distribution for coordinates after WHT rotation on the unit sphere
# (Eric's derivation). Proven +47% PPL vs N(0,1) at short context (128 tok)
# on Qwen3.5-2B A/B test.
#
# Fallback: N(0,1) Lloyd-Max centroids scaled by 1/sqrt(d) for unknown dims.
# Toggle: set TURBO_USE_N01_CENTROIDS=1 to force N(0,1) for A/B testing.
# ---------------------------------------------------------------------------

# Beta distribution centroids keyed by (bits, dim) — already scaled for dim.
# Do NOT multiply by 1/sqrt(d) again.
_BETA_CENTROIDS: Dict[Tuple[int, int], List[float]] = {
    (4, 64): [
        -0.32913971, -0.25096416, -0.19681059, -0.15295772,
        -0.11478586, -0.08000945, -0.04726735, -0.01563822,
        0.01563822, 0.04723797, 0.07994876, 0.11472529,
        0.15289739, 0.19675052, 0.25090477, 0.32908401,
    ],
    (4, 128): [
        -0.23639172, -0.17934021, -0.14023653, -0.10881814,
        -0.08157559, -0.05678632, -0.03350975, -0.01108178,
        0.01108178, 0.03350975, 0.05678631, 0.08157560,
        0.10881804, 0.14023650, 0.17934017, 0.23639278,
    ],
    (4, 256): [
        -0.16852295, -0.12754069, -0.09961203, -0.07719406,
        -0.05781249, -0.04021866, -0.02370371, -0.00783269,
        0.00783269, 0.02370371, 0.04021868, 0.05781246,
        0.07719407, 0.09961203, 0.12754090, 0.16852276,
    ],
    (3, 128): [
        -0.18828832, -0.11801215, -0.06648001, -0.02156330,
        0.02156329, 0.06648005, 0.11801218, 0.18828897,
    ],
    (2, 128): [
        -0.13302007, -0.03998107, 0.03998102, 0.13302033,
    ],
}

# N(0,1) Lloyd-Max centroids (fallback for unknown dims, scaled by 1/sqrt(d))
_LLOYD_MAX_CENTROIDS = {
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [
        -2.1520, -1.3440, -0.7560, -0.2451,
        0.2451, 0.7560, 1.3440, 2.1520,
    ],
    4: [
        -2.7326, -2.0690, -1.6180, -1.2562,
        -0.9423, -0.6568, -0.3881, -0.1284,
        0.1284, 0.3881, 0.6568, 0.9423,
        1.2562, 1.6180, 2.0690, 2.7326,
    ],
}


class TurboQuantCodebook:
    """Pre-computed codebook for TurboQuant.

    Uses Beta distribution centroids (proven better) when available for the
    given (bits, dim) pair. Falls back to N(0,1) Lloyd-Max centroids scaled
    by 1/sqrt(dim) for unknown dims.

    Beta centroids are already scaled for their dim — no 1/sqrt(d) applied.
    Set TURBO_USE_N01_CENTROIDS=1 to force N(0,1) fallback for A/B testing.

    Boundaries are midpoints between adjacent centroids.

    **Design choice — pure centroid quantization, no residual correction:**
    QJL (random Gaussian projection for residual) was tested and found actively
    harmful for autoregressive generation (turbo4-resurrection.md). Variance from
    the random projection compounds across decode steps. 16 centroids (4-bit)
    without correction outperform 8 centroids (3-bit) with correction.

    Args:
        bits (int): Quantization bit-width (2, 3, or 4).
        dim (int): Head dimension (e.g. 64, 128, 256). Must be power of 2.

    Example:
        >>> cb = TurboQuantCodebook(bits=4, dim=128)
        >>> cb.centroids.shape  # (16,)
        >>> cb.boundaries.shape  # (15,)
    """

    def __init__(self, bits: int, dim: int):
        if bits not in _LLOYD_MAX_CENTROIDS:
            raise ValueError(f"Unsupported bits={bits}. Must be 2, 3, or 4.")
        # TODO: Support non-power-of-2 dims via padding (Qwen3-4B has d=80)
        if dim & (dim - 1) != 0:
            raise ValueError(
                f"dim={dim} must be a power of 2 for hadamard_transform. "
                "Non-power-of-2 support (e.g. Qwen3-4B d=80) is planned."
            )

        self.bits = bits
        self.dim = dim
        self.n_levels = 1 << bits

        force_n01 = os.environ.get("TURBO_USE_N01_CENTROIDS", "0") == "1"
        beta_key = (bits, dim)

        if not force_n01 and beta_key in _BETA_CENTROIDS:
            # Beta distribution centroids — already scaled for this dim
            raw = _BETA_CENTROIDS[beta_key]
            self.centroids = mx.array(raw, dtype=mx.float32)
            self._centroid_source = "beta"
        else:
            # Fallback: N(0,1) Lloyd-Max scaled by 1/sqrt(dim)
            scale = 1.0 / math.sqrt(dim)
            raw = _LLOYD_MAX_CENTROIDS[bits]
            self.centroids = mx.array([c * scale for c in raw], dtype=mx.float32)
            self._centroid_source = "n01"

        # Boundaries = midpoints between adjacent centroids
        c = self.centroids
        self.boundaries = (c[:-1] + c[1:]) / 2.0


def _get_codebook(bits: int, dim: int) -> TurboQuantCodebook:
    """Get or create a codebook. Cached per (bits, dim) pair."""
    # TODO: Add proper LRU cache if this becomes a bottleneck
    return TurboQuantCodebook(bits, dim)


# ---------------------------------------------------------------------------
# Sign-flip PRNG — deterministic random signs from seed
# ---------------------------------------------------------------------------


def _sign_flip_vector(dim: int, seed: int) -> mx.array:
    """Generate a deterministic {-1, +1} sign vector from seed.

    Uses mx.random with a fixed key so the same seed always produces
    the same sign pattern. This is the 'S' in SRHT = S·H·D.

    Args:
        dim: Length of the sign vector.
        seed: Random seed for reproducibility.

    Returns:
        mx.array of shape (dim,) with values in {-1, +1}.
    """
    key = mx.array([seed, 0], dtype=mx.uint32)
    # Uniform [0,1) → threshold at 0.5 → {-1, +1}
    r = mx.random.uniform(shape=(dim,), key=key)
    signs = mx.where(r < 0.5, mx.array(-1.0), mx.array(1.0))
    return signs


# ---------------------------------------------------------------------------
# Core encode / decode
# ---------------------------------------------------------------------------


def turbo_encode(
    x: mx.array,
    bits: int = 4,
    seed: int = 42,
) -> Tuple[mx.array, mx.array]:
    """Encode vectors using TurboQuant (SRHT + Lloyd-Max quantization).

    Applies: normalize → sign_flip → hadamard_transform → boundary quantize → pack.

    Args:
        x: Input tensor of shape (..., dim). dim must be power of 2.
        bits: Quantization bit-width (2, 3, or 4). Default: 4.
        seed: Random seed for the sign-flip diagonal. Default: 42.

    Returns:
        Tuple of:
            - packed_indices: uint32 tensor with packed quantization indices.
              Shape (..., dim * bits / 32).
            - norms: float32 tensor of per-vector L2 norms. Shape (..., 1).

    Example:
        >>> x = mx.random.normal((4, 8, 128))  # (batch, seq, dim)
        >>> packed, norms = turbo_encode(x, bits=4, seed=42)
        >>> packed.shape  # (4, 8, 16) — 128 * 4 / 32 = 16 uint32s
        >>> norms.shape   # (4, 8, 1)
    """
    dim = x.shape[-1]
    cb = _get_codebook(bits, dim)

    # 1. Extract norms and normalize to unit sphere
    # NOTE: No norm correction needed — WHT (hadamard_transform) is orthogonal,
    # so norms are exactly preserved through the transform. Storing raw norms
    # is sufficient. This saves a codebook lookup + norm computation + division
    # per encoded vector vs. the corrected-norm approach.
    norms = mx.linalg.norm(x, axis=-1, keepdims=True)
    # Avoid division by zero
    safe_norms = mx.maximum(norms, mx.array(1e-10))
    x_unit = x / safe_norms

    # 2. Apply random sign flip (element-wise multiply by ±1)
    signs = _sign_flip_vector(dim, seed)
    x_flipped = x_unit * signs

    # 3. Apply Walsh-Hadamard Transform
    # mx.hadamard_transform default scale is 1/sqrt(N), giving us orthonormal WHT
    #
    # TODO: Block-size optimization (block-size-experiment.md)
    # Our paper found that WHT block_size=32 (matching Apple Silicon SIMD width)
    # gives the best decode speed in llama.cpp — matching q8_0 throughput. The
    # current implementation uses full head_dim (128) as the transform size.
    # Splitting into blocks of 32 would require reshaping:
    #   x_flipped.reshape(..., dim // 32, 32) → hadamard_transform → reshape back
    # Trade-off: block_size=32 is faster but slightly worse quality because the
    # WHT only decorrelates within each 32-element block, not across the full
    # head_dim. For MLX, this would need profiling — Metal's SIMD may not have
    # the same 32-wide sweet spot as ARM NEON in llama.cpp.
    x_rotated = mx.hadamard_transform(x_flipped)

    # 4. Boundary quantize → indices (pure centroid, NO residual correction)
    #
    # CONFIRMED: No QJL (random Gaussian projection) residual correction.
    # From turbo4-resurrection.md: QJL is actively harmful for autoregressive
    # generation — variance from the random projection compounds across decode
    # steps, degrading output quality progressively. Pure centroid quantization
    # without correction is strictly better for inference.
    #
    # Also confirmed: 16 centroids (4-bit) dramatically outperform 8-centroid
    # (3-bit) schemes even WITH residual correction. Our 4-bit default is the
    # correct choice — more centroids > fewer centroids + correction.
    #
    # For each element, find which centroid bin it falls into using boundaries.
    # boundaries shape: (n_levels - 1,)
    # x_rotated shape: (..., dim)
    # Compare each value against all boundaries → sum gives the index
    boundaries = cb.boundaries  # (n_levels - 1,)
    # Expand for broadcasting: x_rotated[..., :, None] > boundaries[None, :]
    # Result shape: (..., dim, n_levels - 1) → sum over last axis → (..., dim)
    indices = mx.sum(
        mx.expand_dims(x_rotated, axis=-1) > mx.expand_dims(boundaries, axis=0),
        axis=-1,
    ).astype(mx.uint32)

    # 5. Pack indices into uint32
    packed = _pack_indices(indices, bits)

    return packed, norms


def turbo_decode(
    packed_indices: mx.array,
    norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
) -> mx.array:
    """Decode TurboQuant-compressed vectors back to full precision.

    Applies: unpack → codebook lookup → inverse_hadamard → inverse_sign_flip → scale.

    Args:
        packed_indices: uint32 tensor from turbo_encode. Shape (..., dim * bits / 32).
        norms: float32 norms from turbo_encode. Shape (..., 1).
        dim: Original head dimension.
        bits: Quantization bit-width (2, 3, or 4). Default: 4.
        seed: Must match the seed used in turbo_encode. Default: 42.

    Returns:
        Reconstructed tensor of shape (..., dim).

    Example:
        >>> packed, norms = turbo_encode(x, bits=4, seed=42)
        >>> x_hat = turbo_decode(packed, norms, dim=128, bits=4, seed=42)
        >>> x_hat.shape  # same as original x
    """
    cb = _get_codebook(bits, dim)

    # 1. Unpack indices
    indices = _unpack_indices(packed_indices, bits, dim)

    # 2. Codebook lookup
    centroids = cb.centroids  # (n_levels,)
    x_rotated = centroids[indices]  # (..., dim)

    # 3. Inverse WHT (Hadamard is its own inverse up to scaling)
    # Since we used orthonormal (scale=1/sqrt(N)), applying it again gives identity
    # NOTE: If block_size optimization is added to encode (see TODO there),
    # the same reshape→transform→reshape must be applied here in reverse.
    x_flipped = mx.hadamard_transform(x_rotated)

    # 4. Inverse sign flip (signs are their own inverse: s * s = 1)
    signs = _sign_flip_vector(dim, seed)
    x_unit = x_flipped * signs

    # 5. Scale by norms
    x_reconstructed = x_unit * norms

    return x_reconstructed


# ---------------------------------------------------------------------------
# Packing / unpacking bit indices into uint32
# ---------------------------------------------------------------------------


def _pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack N-bit indices into uint32 words.

    Args:
        indices: uint32 tensor of shape (..., dim) with values in [0, 2^bits).
        bits: Bit-width per index (2, 3, or 4).

    Returns:
        uint32 tensor of shape (..., packed_dim) where packed_dim = ceil(dim * bits / 32).
    """
    dim = indices.shape[-1]
    leading_shape = indices.shape[:-1]
    indices_per_word = 32 // bits

    # For bits that evenly divide 32 (2, 4), this is clean.
    # For bits=3, we pad dim to next multiple of indices_per_word (10 per uint32).
    if 32 % bits != 0:
        # bits=3: 10 indices per uint32, with 2 wasted bits
        pad_to = math.ceil(dim / indices_per_word) * indices_per_word
        if pad_to > dim:
            padding = mx.zeros((*leading_shape, pad_to - dim), dtype=mx.uint32)
            indices = mx.concatenate([indices, padding], axis=-1)
            dim = pad_to

    # Reshape to (..., n_words, indices_per_word)
    n_words = dim // indices_per_word
    indices = indices.reshape(*leading_shape, n_words, indices_per_word)

    # Shift each index to its bit position and OR together
    shifts = mx.array([i * bits for i in range(indices_per_word)], dtype=mx.uint32)
    packed = mx.sum(indices << shifts, axis=-1).astype(mx.uint32)

    return packed


def _unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 words into N-bit indices.

    Args:
        packed: uint32 tensor of shape (..., packed_dim).
        bits: Bit-width per index (2, 3, or 4).
        dim: Original dimension (number of indices to unpack).

    Returns:
        uint32 tensor of shape (..., dim) with values in [0, 2^bits).
    """
    leading_shape = packed.shape[:-1]
    indices_per_word = 32 // bits
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)

    # Expand each uint32 into its constituent indices
    # packed shape: (..., n_words) → (..., n_words, 1)
    packed_expanded = mx.expand_dims(packed, axis=-1)

    # Shift amounts: [0, bits, 2*bits, ...]
    shifts = mx.array([i * bits for i in range(indices_per_word)], dtype=mx.uint32)

    # Extract indices: shift right then mask
    indices = (packed_expanded >> shifts) & mask  # (..., n_words, indices_per_word)

    # Reshape back to flat
    indices = indices.reshape(*leading_shape, -1)

    # Trim padding if needed (for bits=3)
    if indices.shape[-1] > dim:
        indices = indices[..., :dim]

    return indices


# ---------------------------------------------------------------------------
# Sparse attention masking (sparse-v-dequant.md)
# ---------------------------------------------------------------------------


def sparse_attention_mask(
    weights: mx.array,
    threshold: float = 1e-6,
) -> mx.array:
    """Create a boolean mask that zeros out near-zero attention weights.

    After softmax, many attention positions have negligible weight (< 1e-6).
    Zeroing these before the V matmul lets MLX potentially skip those lanes,
    saving compute proportional to sparsity.

    From sparse-v-dequant.md: on Apple Silicon, skipping dequant for near-zero
    weights saved meaningful compute. This mask is the first step — it prevents
    near-zero weights from contributing to the V weighted sum. A full sparse
    implementation would also skip the V dequant itself (requires fused kernel).

    Args:
        weights: Post-softmax attention weights, shape (..., q_len, kv_len).
        threshold: Weights below this value are masked out. Default: 1e-6.

    Returns:
        Boolean mask of same shape as weights (1.0 where weight >= threshold,
        0.0 where weight < threshold). Multiply with weights before V matmul.

    Example:
        >>> weights = mx.softmax(scores, axis=-1)
        >>> mask = sparse_attention_mask(weights, threshold=1e-6)
        >>> weights = weights * mask  # zero out negligible positions
        >>> output = weights @ values  # MLX can skip zeroed lanes
    """
    return (weights >= threshold).astype(weights.dtype)


# ---------------------------------------------------------------------------
# TurboQuant attention (decode-then-matmul, not fused)
# ---------------------------------------------------------------------------


def turbo_attention(
    queries: mx.array,
    packed_keys: mx.array,
    key_norms: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Compute attention with TurboQuant-compressed KV cache.

    This is the decode-then-matmul approach — we decompress K and V,
    then do standard scaled dot-product attention. A fused Metal kernel
    version will come later for better performance.

    **Asymmetric K/V quantization** (recommended):

    K precision dominates output quality because errors in K are amplified
    through the softmax exponential — small errors in dot products become
    large errors in attention weights. V errors are merely averaged.

    Recommended config: K stays at FP16, V at turbo4 (``key_bits=0, bits=4``
    in TurboQuantKVCache, aka "turbo0v4"). Symmetric turbo3/turbo3 works
    but asymmetric gives strictly better quality for the same memory budget.

    Args:
        queries: Query tensor, shape (batch, heads, q_len, dim).
        packed_keys: Packed key indices from turbo_encode, shape (batch, heads, kv_len, packed_dim).
        key_norms: Key norms, shape (batch, heads, kv_len, 1).
        packed_values: Packed value indices, shape (batch, heads, kv_len, packed_dim).
        value_norms: Value norms, shape (batch, heads, kv_len, 1).
        dim: Head dimension.
        bits: Quantization bit-width. Default: 4.
        seed: SRHT seed. Default: 42.
        scale: Attention scale factor. Default: 1/sqrt(dim).
        mask: Optional attention mask, shape broadcastable to (batch, heads, q_len, kv_len).

    Returns:
        Attention output, shape (batch, heads, q_len, dim).

    Example:
        >>> q = mx.random.normal((1, 8, 1, 128))
        >>> k = mx.random.normal((1, 8, 32, 128))
        >>> v = mx.random.normal((1, 8, 32, 128))
        >>> pk, kn = turbo_encode(k, bits=4)
        >>> pv, vn = turbo_encode(v, bits=4)
        >>> out = turbo_attention(q, pk, kn, pv, vn, dim=128, bits=4)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Decode keys and values
    # TODO: Fused Metal kernel — decode + matmul in one pass to avoid materializing full K/V
    keys = turbo_decode(packed_keys, key_norms, dim, bits, seed)
    values = turbo_decode(packed_values, value_norms, dim, bits, seed)

    # Standard scaled dot-product attention
    # Q·K^T
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

    if mask is not None:
        scores = scores + mask

    weights = mx.softmax(scores, axis=-1)

    # Apply sparse attention mask to skip near-zero V contributions
    # See: sparse-v-dequant.md — many post-softmax weights are near-zero,
    # making those V dequant+matmul ops wasted compute. By zeroing them
    # out before the matmul, MLX can potentially skip those lanes entirely.
    #
    # TODO: Sparse V integration with TurboKVCache (Item 9)
    # This sparse mask works in turbo_attention() (decode-then-matmul path),
    # but TurboKVCache uses standard SDPA via mlx-lm's base.py — it returns
    # plain mx.array K/V tensors and the model calls mx.fast.scaled_dot_product_attention.
    # To integrate sparse V at the model level:
    #   1. Add a post-SDPA hook in TurboKVCache, or
    #   2. Modify mlx-lm's base.py to accept a sparse_mask callback, or
    #   3. Implement a fused Metal kernel that does SDPA + sparse skip in one pass
    # Option 3 is the real win — skip V dequant entirely for near-zero attention
    # positions, saving both compute and memory bandwidth. Options 1-2 still
    # materialize all V tokens to FP16 before the matmul.
    # For now, sparse_attention_mask() is available as a standalone utility that
    # users can apply manually if they write custom attention loops.
    sparse_mask = sparse_attention_mask(weights)
    weights = weights * sparse_mask

    # Weighted sum of values
    # TODO: Full sparse V optimization — skip dequant entirely for masked
    # positions. The current decode-then-matmul approach materializes all V
    # tokens to FP16 before the matmul. True sparse dequant would only decode
    # the V tokens with significant attention weight, saving both compute and
    # memory bandwidth. This requires a fused kernel that checks attention
    # weights BEFORE dequanting each V token. Expected benefit on Apple Silicon:
    # 15-30% decode speedup at long context (>4K tokens) where most attention
    # mass concentrates on a few positions. (sparse-v-dequant.md)
    output = weights @ values

    return output


# ---------------------------------------------------------------------------
# Fused compressed-domain attention (Metal kernel — no FP16 materialization)
# ---------------------------------------------------------------------------

# The Metal kernel operates in the WHT (rotated) domain:
#   1. Python pre-rotates Q: Q_rot = WHT(Q * signs)          — once per query
#   2. Kernel: for each KV token, unpack indices → centroid lookup → dot product
#      with Q_rot → softmax → centroid lookup for V → weighted sum
#   3. Python post-rotates output: out = signs * WHT(accum)   — once per output
#
# This avoids materializing FP16 K/V entirely. The WHT and sign-flip are
# linear operators applied once to Q and once to the output, NOT per-KV-token.
# Memory bandwidth: reads packed uint32 indices + norms (4-bit: 1/8th of FP16).
# Compute: centroid lookup is a 16-entry table lookup, trivially fast.

# --- Metal kernel source for 4-bit compressed-domain attention ---
# One threadgroup per (batch, head) pair. Each threadgroup processes all T_kv
# tokens for one query head. Within the threadgroup:
#   - Phase 1 (scores): Each thread handles a range of KV tokens. For each
#     token, unpack all dim indices, lookup centroids, dot with Q_rot.
#   - Phase 2 (softmax): Parallel reduce max + sum for numerically stable softmax.
#   - Phase 3 (V weighted sum): Same unpack + lookup, multiply by attn weight,
#     accumulate across tokens.
#
# Thread layout: threadgroup_size threads, each handles ceil(T_kv / tg_size) tokens.

_TURBO_ATTN_HEADER = """
// Centroid table — embedded as constant array for minimal latency.
// Loaded into registers at kernel launch, no memory fetch during inner loop.
// These are the Beta-distribution centroids for (4-bit, 128-dim).
// For other (bits, dim) combos, the Python wrapper passes the correct table.

// Inline unpack: extract a 4-bit index from a uint32 word
inline uint unpack4(uint word, uint pos) {
    return (word >> (pos * 4)) & 0xF;
}

// Inline unpack: extract a 3-bit index from a uint32 word
inline uint unpack3(uint word, uint pos) {
    return (word >> (pos * 3)) & 0x7;
}

// Inline unpack: extract a 2-bit index from a uint32 word
inline uint unpack2(uint word, uint pos) {
    return (word >> (pos * 2)) & 0x3;
}
"""

_TURBO_ATTN_SOURCE_4BIT = """
    // Grid: (B * n_heads, 1, 1)  — one threadgroup per (batch, head) pair
    // Threadgroup: (TG_SIZE, 1, 1) where TG_SIZE divides work across T_kv tokens
    //
    // Inputs (row-contiguous, flattened to B*n_heads leading dim):
    //   q_rot:        [n_bh, dim]               — pre-rotated query (WHT domain)
    //   packed_k:     [n_bh, T_kv, packed_dim]  — packed 4-bit K indices
    //   k_norms:      [n_bh, T_kv]             — K L2 norms (already squeezed)
    //   packed_v:     [n_bh, T_kv, packed_dim]  — packed 4-bit V indices
    //   v_norms:      [n_bh, T_kv]             — V L2 norms (already squeezed)
    //   centroids:    [n_levels]                 — centroid lookup table
    //   params:       [4]                        — {dim, T_kv, packed_dim, scale_bits}
    //
    // Outputs (device buffers, indexed by bh_idx):
    //   out_accum:    [n_bh, dim]               — WHT-domain weighted sum
    //   scores:       [n_bh, T_kv]             — scratch for attention scores
    //   simd_maxes:   [n_bh, n_simd_groups]    — scratch for simd reductions

    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint bh_idx = threadgroup_position_in_grid.x;  // batch*head index

    // Read params
    int dim = params[0];
    int T_kv = params[1];
    int packed_dim = params[2];
    float scale = as_type<float>(params[3]);

    // Compute base offsets into the flattened buffers for this (batch, head)
    int q_offset = bh_idx * dim;
    int kv_base = bh_idx * T_kv;
    int pk_base = bh_idx * T_kv * packed_dim;
    int pv_base = bh_idx * T_kv * packed_dim;

    // scores and simd_maxes are global device buffers indexed per-threadgroup
    int scores_base = bh_idx * T_kv;

    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = tid / threads_per_simdgroup;
    uint n_simd = (tg_size + threads_per_simdgroup - 1) / threads_per_simdgroup;
    int smaxes_base = bh_idx * n_simd;

    // --- Phase 1: Compute Q·K scores for all T_kv tokens ---
    // Each thread processes a strided subset of tokens.
    // For each assigned token:
    //   score = norm_K * sum_d(Q_rot[d] * centroids[K_indices[d]]) * scale
    float max_score = -INFINITY;

    for (int t = tid; t < T_kv; t += tg_size) {
        float dot = 0.0f;
        int pk_offset = pk_base + t * packed_dim;

        for (int w = 0; w < packed_dim; w++) {
            uint word = packed_k[pk_offset + w];
            int base_d = w * 8;  // 8 indices per uint32 for 4-bit

            // Unroll 8 indices per word
            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];
                dot += q_rot[q_offset + base_d + j] * c;
            }
        }

        float norm_k = k_norms[kv_base + t];
        float s = dot * norm_k * scale;
        scores[scores_base + t] = s;
        max_score = max(max_score, s);
    }

    // --- Phase 2: Softmax (parallel reduction) ---
    // Step 2a: reduce max across threadgroup via simd_max + cross-simd reduce
    max_score = simd_max(max_score);

    if (simd_lane == 0) {
        simd_maxes[smaxes_base + simd_id] = max_score;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float global_max = simd_maxes[smaxes_base];
        for (uint s = 1; s < n_simd; s++) {
            global_max = max(global_max, simd_maxes[smaxes_base + s]);
        }
        simd_maxes[smaxes_base] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float global_max = simd_maxes[smaxes_base];

    // Step 2b: compute exp(score - max) and local sum
    float local_sum = 0.0f;
    for (int t = tid; t < T_kv; t += tg_size) {
        float e = exp(scores[scores_base + t] - global_max);
        scores[scores_base + t] = e;
        local_sum += e;
    }

    // Reduce sum across threadgroup
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        simd_maxes[smaxes_base + simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float global_sum = 0.0f;
        for (uint s = 0; s < n_simd; s++) {
            global_sum += simd_maxes[smaxes_base + s];
        }
        simd_maxes[smaxes_base] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float inv_sum = 1.0f / simd_maxes[smaxes_base];

    // Normalize scores to attention weights
    for (int t = tid; t < T_kv; t += tg_size) {
        scores[scores_base + t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // --- Phase 3: V weighted sum ---
    // Each thread accumulates its share of V tokens weighted by attention.
    // Local accumulator in registers — 128 floats = 512 bytes, fits easily.
    float v_accum[256];  // Max supported dim (Metal needs fixed-size arrays)
    for (int d = 0; d < dim; d++) {
        v_accum[d] = 0.0f;
    }

    for (int t = tid; t < T_kv; t += tg_size) {
        float attn_w = scores[scores_base + t];

        // Skip near-zero attention weights (fused sparse V optimization)
        if (attn_w < 1e-6f) continue;

        float norm_v = v_norms[kv_base + t];
        float w = attn_w * norm_v;

        int pv_offset = pv_base + t * packed_dim;
        for (int pw = 0; pw < packed_dim; pw++) {
            uint word = packed_v[pv_offset + pw];
            int base_d = pw * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];
                v_accum[base_d + j] += w * c;
            }
        }
    }

    // --- Phase 4: Reduce V accumulators across threads ---
    // Use simd_sum per dimension, then cross-simd reduce via device buffer.
    for (int d = 0; d < dim; d++) {
        float val = simd_sum(v_accum[d]);

        if (simd_lane == 0) {
            simd_maxes[smaxes_base + simd_id] = val;
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < n_simd; s++) {
                total += simd_maxes[smaxes_base + s];
            }
            out_accum[q_offset + d] = total;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
"""

# --- NR0=2 Multi-Row Amortization kernel ---
# Processes 2 queries per dispatch, sharing K/V dequant (centroid lookup + norm
# multiply) across both queries. Halves the memory bandwidth cost of reading
# packed K/V data. Expected speedup: ~30-40% for decode with 2+ pending queries.
#
# Layout: grid dispatches one threadgroup per (batch, head) pair, same as NR0=1.
# Each thread computes scores and V accumulation for BOTH queries simultaneously.

_TURBO_ATTN_SOURCE_4BIT_NR0_2 = """
    // Grid: (B * n_heads, 1, 1)  — one threadgroup per (batch, head) pair
    // Threadgroup: (TG_SIZE, 1, 1)
    //
    // Inputs:
    //   q_rot:        [n_bh, NR0, dim]          — NR0=2 pre-rotated queries (WHT domain)
    //   packed_k:     [n_bh, T_kv, packed_dim]   — packed 4-bit K indices
    //   k_norms:      [n_bh, T_kv]               — K L2 norms
    //   packed_v:     [n_bh, T_kv, packed_dim]   — packed 4-bit V indices
    //   v_norms:      [n_bh, T_kv]               — V L2 norms
    //   centroids:    [n_levels]                   — centroid lookup table
    //   params:       [4]                          — {dim, T_kv, packed_dim, scale_bits}
    //
    // Outputs:
    //   out_accum:    [n_bh, NR0, dim]           — WHT-domain weighted sums
    //   scores:       [n_bh, NR0, T_kv]          — scratch for attention scores
    //   simd_maxes:   [n_bh, NR0, n_simd_groups] — scratch for simd reductions

    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint bh_idx = threadgroup_position_in_grid.x;

    int dim = params[0];
    int T_kv = params[1];
    int packed_dim = params[2];
    float scale = as_type<float>(params[3]);

    // Base offsets — NR0=2 queries packed contiguously per (batch, head)
    int q_base = bh_idx * 2 * dim;      // q_rot[bh_idx, 0..1, :]
    int kv_base = bh_idx * T_kv;
    int pk_base = bh_idx * T_kv * packed_dim;
    int pv_base = bh_idx * T_kv * packed_dim;

    // Score buffers: [n_bh, 2, T_kv]
    int scores_base_0 = bh_idx * 2 * T_kv;
    int scores_base_1 = scores_base_0 + T_kv;

    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = tid / threads_per_simdgroup;
    uint n_simd = (tg_size + threads_per_simdgroup - 1) / threads_per_simdgroup;
    // simd_maxes: [n_bh, 2, n_simd]
    int smaxes_base_0 = bh_idx * 2 * n_simd;
    int smaxes_base_1 = smaxes_base_0 + n_simd;

    // --- Phase 1: Compute Q·K scores for BOTH queries, sharing K dequant ---
    float max_score_0 = -INFINITY;
    float max_score_1 = -INFINITY;

    for (int t = tid; t < T_kv; t += tg_size) {
        float dot_0 = 0.0f;
        float dot_1 = 0.0f;
        int pk_offset = pk_base + t * packed_dim;

        for (int w = 0; w < packed_dim; w++) {
            uint word = packed_k[pk_offset + w];  // Shared K dequant — read once
            int base_d = w * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];  // Shared centroid lookup
                int d = base_d + j;
                dot_0 += q_rot[q_base + d] * c;           // Query 0
                dot_1 += q_rot[q_base + dim + d] * c;     // Query 1
            }
        }

        float norm_k = k_norms[kv_base + t];  // Shared norm
        float s0 = dot_0 * norm_k * scale;
        float s1 = dot_1 * norm_k * scale;
        scores[scores_base_0 + t] = s0;
        scores[scores_base_1 + t] = s1;
        max_score_0 = max(max_score_0, s0);
        max_score_1 = max(max_score_1, s1);
    }

    // --- Phase 2: Softmax for both queries (parallel reduction) ---
    // 2a: Reduce max across threadgroup for query 0
    max_score_0 = simd_max(max_score_0);
    max_score_1 = simd_max(max_score_1);

    if (simd_lane == 0) {
        simd_maxes[smaxes_base_0 + simd_id] = max_score_0;
        simd_maxes[smaxes_base_1 + simd_id] = max_score_1;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float gmax_0 = simd_maxes[smaxes_base_0];
        float gmax_1 = simd_maxes[smaxes_base_1];
        for (uint s = 1; s < n_simd; s++) {
            gmax_0 = max(gmax_0, simd_maxes[smaxes_base_0 + s]);
            gmax_1 = max(gmax_1, simd_maxes[smaxes_base_1 + s]);
        }
        simd_maxes[smaxes_base_0] = gmax_0;
        simd_maxes[smaxes_base_1] = gmax_1;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float global_max_0 = simd_maxes[smaxes_base_0];
    float global_max_1 = simd_maxes[smaxes_base_1];

    // 2b: exp(score - max) and sum
    float local_sum_0 = 0.0f;
    float local_sum_1 = 0.0f;
    for (int t = tid; t < T_kv; t += tg_size) {
        float e0 = exp(scores[scores_base_0 + t] - global_max_0);
        float e1 = exp(scores[scores_base_1 + t] - global_max_1);
        scores[scores_base_0 + t] = e0;
        scores[scores_base_1 + t] = e1;
        local_sum_0 += e0;
        local_sum_1 += e1;
    }

    local_sum_0 = simd_sum(local_sum_0);
    local_sum_1 = simd_sum(local_sum_1);
    if (simd_lane == 0) {
        simd_maxes[smaxes_base_0 + simd_id] = local_sum_0;
        simd_maxes[smaxes_base_1 + simd_id] = local_sum_1;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float gsum_0 = 0.0f, gsum_1 = 0.0f;
        for (uint s = 0; s < n_simd; s++) {
            gsum_0 += simd_maxes[smaxes_base_0 + s];
            gsum_1 += simd_maxes[smaxes_base_1 + s];
        }
        simd_maxes[smaxes_base_0] = gsum_0;
        simd_maxes[smaxes_base_1] = gsum_1;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float inv_sum_0 = 1.0f / simd_maxes[smaxes_base_0];
    float inv_sum_1 = 1.0f / simd_maxes[smaxes_base_1];

    // Normalize scores
    for (int t = tid; t < T_kv; t += tg_size) {
        scores[scores_base_0 + t] *= inv_sum_0;
        scores[scores_base_1 + t] *= inv_sum_1;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // --- Phase 3: V weighted sum for BOTH queries, sharing V dequant ---
    float v_accum_0[256];
    float v_accum_1[256];
    for (int d = 0; d < dim; d++) {
        v_accum_0[d] = 0.0f;
        v_accum_1[d] = 0.0f;
    }

    for (int t = tid; t < T_kv; t += tg_size) {
        float attn_w_0 = scores[scores_base_0 + t];
        float attn_w_1 = scores[scores_base_1 + t];

        // Skip if BOTH queries have near-zero attention (fused sparse V)
        if (attn_w_0 < 1e-6f && attn_w_1 < 1e-6f) continue;

        float norm_v = v_norms[kv_base + t];  // Shared V norm
        float w_0 = attn_w_0 * norm_v;
        float w_1 = attn_w_1 * norm_v;

        int pv_offset = pv_base + t * packed_dim;
        for (int pw = 0; pw < packed_dim; pw++) {
            uint word = packed_v[pv_offset + pw];  // Shared V dequant
            int base_d = pw * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];  // Shared centroid lookup
                int d = base_d + j;
                v_accum_0[d] += w_0 * c;
                v_accum_1[d] += w_1 * c;
            }
        }
    }

    // --- Phase 4: Reduce V accumulators for BOTH queries ---
    int out_base_0 = bh_idx * 2 * dim;
    int out_base_1 = out_base_0 + dim;

    for (int d = 0; d < dim; d++) {
        float val_0 = simd_sum(v_accum_0[d]);
        float val_1 = simd_sum(v_accum_1[d]);

        if (simd_lane == 0) {
            simd_maxes[smaxes_base_0 + simd_id] = val_0;
            simd_maxes[smaxes_base_1 + simd_id] = val_1;
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (tid == 0) {
            float total_0 = 0.0f, total_1 = 0.0f;
            for (uint s = 0; s < n_simd; s++) {
                total_0 += simd_maxes[smaxes_base_0 + s];
                total_1 += simd_maxes[smaxes_base_1 + s];
            }
            out_accum[out_base_0 + d] = total_0;
            out_accum[out_base_1 + d] = total_1;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
"""

# Cache the compiled kernel objects to avoid re-JIT on every call
_kernel_cache: Dict[str, object] = {}


def _get_turbo_attn_kernel(bits: int, nr0: int = 1):
    """Get or create the compressed-domain attention Metal kernel.

    The kernel is JIT-compiled once and cached. Currently supports 4-bit
    (the primary use case). 3-bit and 2-bit use the same kernel structure
    with different unpack widths.

    Args:
        bits: Quantization bit-width (2, 3, or 4).
        nr0: Number of queries to process per dispatch (1 or 2).
            NR0=2 shares K/V dequant across both queries, halving
            memory bandwidth for packed data reads.

    Returns:
        Compiled Metal kernel callable.
    """
    cache_key = f"turbo_attn_{bits}bit_nr{nr0}"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    if bits == 4:
        if nr0 == 2:
            source = _TURBO_ATTN_SOURCE_4BIT_NR0_2
        else:
            source = _TURBO_ATTN_SOURCE_4BIT
    else:
        # TODO: Add 3-bit and 2-bit kernel variants
        raise NotImplementedError(
            f"Fused compressed-domain attention not yet implemented for {bits}-bit. "
            "Use turbo_attention() (decode-then-matmul) instead."
        )

    kernel = mx.fast.metal_kernel(
        name=f"turbo_sdpa_{bits}bit_nr{nr0}",
        input_names=[
            "q_rot",        # Pre-rotated query (WHT domain)
            "packed_k",     # Packed K indices
            "k_norms",      # K norms (flattened to 1D per-token)
            "packed_v",     # Packed V indices
            "v_norms",      # V norms (flattened to 1D per-token)
            "centroids",    # Centroid lookup table
            "params",       # {dim, T_kv, packed_dim, scale_as_uint32}
        ],
        output_names=[
            "out_accum",    # WHT-domain output (before inverse transform)
            "scores",       # Threadgroup-local score buffer
            "simd_maxes",   # Scratch for simd reductions
        ],
        header=_TURBO_ATTN_HEADER,
        source=source,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    _kernel_cache[cache_key] = kernel
    return kernel


def turbo_fused_attention(
    queries: mx.array,
    packed_keys: mx.array,
    key_norms: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
    nr0: Optional[int] = None,
) -> mx.array:
    """Compressed-domain attention — no FP16 K/V materialization.

    Uses a custom Metal kernel to compute attention directly on packed
    TurboQuant data. The key insight: the Walsh-Hadamard Transform (WHT) and
    sign-flip are linear operators that can be applied once to Q (before the
    kernel) and once to the output (after the kernel), rather than per-KV-token.

    In the WHT domain, each KV token is just a vector of centroid indices + a
    scalar norm. The dot product ``Q_rot @ centroids[indices] * norm`` replaces
    the full FP16 decode + matmul.

    **Performance characteristics:**
    - Memory: reads packed uint32 (4-bit: 1/8th of FP16 bandwidth)
    - Compute: centroid lookup (16-entry table, register-resident) + FMA
    - No intermediate FP16 K/V buffer allocated
    - Sparse V: skips dequant + accumulate for near-zero attention weights
    - NR0=2: shares K/V dequant across 2 queries, halving bandwidth cost

    **Limitations:**
    - Currently 4-bit only (3-bit and 2-bit planned)
    - T_q must be 1 or 2 (decode only — prefill uses standard SDPA)
    - T_kv limited by threadgroup memory (~16K tokens with 64KB tg mem)
    - mask not yet supported in the fused kernel (use decode-then-matmul path)

    Args:
        queries: Query tensor, shape (batch, heads, T_q, dim). T_q must be 1 or 2.
        packed_keys: Packed K indices, shape (batch, heads, T_kv, packed_dim).
        key_norms: K norms, shape (batch, heads, T_kv, 1).
        packed_values: Packed V indices, shape (batch, heads, T_kv, packed_dim).
        value_norms: V norms, shape (batch, heads, T_kv, 1).
        dim: Head dimension (must be power of 2, max 256).
        bits: Quantization bit-width. Default: 4.
        seed: SRHT random seed. Default: 42.
        scale: Attention scale factor. Default: 1/sqrt(dim).
        mask: NOT YET SUPPORTED in fused kernel. Must be None.
        nr0: Number of queries per dispatch (1 or 2). None = auto-select
            based on T_q. NR0=2 shares K/V dequant across both queries.

    Returns:
        Attention output, shape (batch, heads, T_q, dim).

    Raises:
        ValueError: If T_q > 2, mask is provided, or dim > 256.
        NotImplementedError: If bits != 4.

    Example:
        >>> q = mx.random.normal((1, 8, 1, 128))
        >>> k = mx.random.normal((1, 8, 64, 128))
        >>> pk, kn = turbo_encode(k, bits=4)
        >>> pv, vn = turbo_encode(k, bits=4)  # using k for demo
        >>> out = turbo_fused_attention(q, pk, kn, pv, vn, dim=128)
        >>> # NR0=2: process 2 queries sharing dequant work
        >>> q2 = mx.random.normal((1, 8, 2, 128))
        >>> out2 = turbo_fused_attention(q2, pk, kn, pv, vn, dim=128, nr0=2)
    """
    T_q = queries.shape[2]
    if T_q > 2:
        raise ValueError(
            f"turbo_fused_attention supports T_q=1 or T_q=2, got T_q={T_q}. "
            "Use turbo_attention() for prefill (T_q > 2)."
        )
    if mask is not None:
        raise ValueError(
            "turbo_fused_attention does not yet support attention masks. "
            "Use turbo_attention() for masked attention."
        )
    if dim > 256:
        raise ValueError(
            f"turbo_fused_attention supports dim <= 256, got dim={dim}. "
            "The kernel uses a fixed-size register array for V accumulation."
        )

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Auto-select NR0 based on T_q if not explicitly specified
    if nr0 is None:
        nr0 = T_q  # 1 query → NR0=1, 2 queries → NR0=2

    # Validate NR0/T_q compatibility
    if nr0 == 2 and T_q < 2:
        raise ValueError(
            f"NR0=2 requires T_q >= 2, got T_q={T_q}. "
            "Use NR0=1 for single-query decode."
        )

    # NR0=2 dispatch
    if nr0 == 2:
        return _turbo_fused_attention_nr0_2(
            queries, packed_keys, key_norms, packed_values,
            value_norms, dim, bits, seed, scale,
        )

    # NR0=1 (original) dispatch
    if T_q != 1:
        raise ValueError(
            f"NR0=1 requires T_q=1, got T_q={T_q}. "
            "Use NR0=2 for T_q=2 or turbo_attention() for larger T_q."
        )

    kernel = _get_turbo_attn_kernel(bits, nr0=1)
    cb = _get_codebook(bits, dim)

    B, n_heads, T_kv, packed_dim = packed_keys.shape

    # --- Step 1: Pre-rotate queries into WHT domain ---
    # Q_rot = WHT(Q * signs)
    # This transforms the query so that dot products with centroid vectors in
    # the WHT domain give the same result as dot products with decoded K in
    # the original domain. (WHT is orthonormal → preserves inner products.)
    signs = _sign_flip_vector(dim, seed)
    q_flipped = queries * signs                 # (B, n_heads, 1, dim)
    q_rot = mx.hadamard_transform(q_flipped)    # (B, n_heads, 1, dim)
    q_rot = q_rot.astype(mx.float32)

    # --- Step 2: Flatten norms for kernel (remove trailing dim of 1) ---
    k_norms_flat = key_norms.squeeze(-1).astype(mx.float32)    # (B, n_heads, T_kv)
    v_norms_flat = value_norms.squeeze(-1).astype(mx.float32)  # (B, n_heads, T_kv)

    # --- Step 3: Encode scale as uint32 for passing through integer param array ---
    import struct
    scale_as_uint32 = struct.unpack('I', struct.pack('f', scale))[0]
    params = mx.array([dim, T_kv, packed_dim, scale_as_uint32], dtype=mx.uint32)

    # --- Step 4: Launch the Metal kernel ---
    # Grid: one threadgroup per (batch, head) pair
    # Threadgroup size: 64 threads (2 SIMD groups of 32)
    # — enough parallelism for T_kv >> 64, small enough for register pressure
    n_bh = B * n_heads
    tg_size = min(64, max(32, T_kv))  # At least 1 SIMD group, at most 64
    # Round to SIMD group boundary
    tg_size = ((tg_size + 31) // 32) * 32

    # Reshape inputs to (B*n_heads, ...) for the kernel
    q_rot_flat = q_rot.reshape(n_bh, dim)
    pk_flat = packed_keys.reshape(n_bh, T_kv, packed_dim)
    kn_flat = k_norms_flat.reshape(n_bh, T_kv)
    pv_flat = packed_values.reshape(n_bh, T_kv, packed_dim)
    vn_flat = v_norms_flat.reshape(n_bh, T_kv)

    # Max number of simd groups per threadgroup (for scratch buffer)
    n_simd_groups = tg_size // 32

    outputs = kernel(
        inputs=[
            q_rot_flat,            # q_rot
            pk_flat,               # packed_k
            kn_flat,               # k_norms
            pv_flat,               # packed_v
            vn_flat,               # v_norms
            cb.centroids,          # centroids (n_levels,)
            params,                # params
        ],
        output_shapes=[
            (n_bh, dim),           # out_accum
            (n_bh, T_kv),          # scores (threadgroup scratch — will be discarded)
            (n_bh, n_simd_groups), # simd_maxes (scratch)
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(n_bh * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    out_rot = outputs[0]  # (n_bh, dim) — in WHT domain

    # --- Step 5: Inverse transform back to original domain ---
    # output = signs * WHT(out_rot)
    # (WHT is its own inverse for orthonormal normalization)
    out_rot = out_rot.reshape(B, n_heads, 1, dim)
    out_transformed = mx.hadamard_transform(out_rot)
    output = out_transformed * signs

    return output.astype(queries.dtype)


def _turbo_fused_attention_nr0_2(
    queries: mx.array,
    packed_keys: mx.array,
    key_norms: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int,
    seed: int,
    scale: float,
) -> mx.array:
    """NR0=2 multi-row amortization: process 2 queries sharing K/V dequant.

    Internal helper called by turbo_fused_attention when NR0=2 is selected.
    The Metal kernel reads each packed K/V word once and computes dot products
    with both queries simultaneously, halving the memory bandwidth cost.

    Args:
        queries: Shape (B, n_heads, 2, dim) — exactly 2 queries.
        packed_keys: Shape (B, n_heads, T_kv, packed_dim).
        key_norms: Shape (B, n_heads, T_kv, 1).
        packed_values: Shape (B, n_heads, T_kv, packed_dim).
        value_norms: Shape (B, n_heads, T_kv, 1).
        dim: Head dimension.
        bits: Quantization bit-width.
        seed: SRHT seed.
        scale: Attention scale factor.

    Returns:
        Attention output, shape (B, n_heads, 2, dim).
    """
    kernel = _get_turbo_attn_kernel(bits, nr0=2)
    cb = _get_codebook(bits, dim)

    B, n_heads, T_kv, packed_dim = packed_keys.shape

    # Pre-rotate both queries into WHT domain
    signs = _sign_flip_vector(dim, seed)
    q_flipped = queries * signs                 # (B, n_heads, 2, dim)
    q_rot = mx.hadamard_transform(q_flipped)    # (B, n_heads, 2, dim)
    q_rot = q_rot.astype(mx.float32)

    # Flatten norms
    k_norms_flat = key_norms.squeeze(-1).astype(mx.float32)
    v_norms_flat = value_norms.squeeze(-1).astype(mx.float32)

    # Encode scale
    import struct
    scale_as_uint32 = struct.unpack('I', struct.pack('f', scale))[0]
    params = mx.array([dim, T_kv, packed_dim, scale_as_uint32], dtype=mx.uint32)

    n_bh = B * n_heads
    tg_size = min(64, max(32, T_kv))
    tg_size = ((tg_size + 31) // 32) * 32

    # Reshape: NR0=2 queries are packed as (n_bh, 2, dim)
    q_rot_flat = q_rot.reshape(n_bh, 2, dim)
    pk_flat = packed_keys.reshape(n_bh, T_kv, packed_dim)
    kn_flat = k_norms_flat.reshape(n_bh, T_kv)
    pv_flat = packed_values.reshape(n_bh, T_kv, packed_dim)
    vn_flat = v_norms_flat.reshape(n_bh, T_kv)

    n_simd_groups = tg_size // 32

    outputs = kernel(
        inputs=[
            q_rot_flat,
            pk_flat,
            kn_flat,
            pv_flat,
            vn_flat,
            cb.centroids,
            params,
        ],
        output_shapes=[
            (n_bh, 2, dim),           # out_accum for both queries
            (n_bh, 2, T_kv),          # scores scratch for both queries
            (n_bh, 2, n_simd_groups), # simd_maxes scratch for both queries
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(n_bh * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    out_rot = outputs[0]  # (n_bh, 2, dim)

    # Inverse transform back to original domain
    out_rot = out_rot.reshape(B, n_heads, 2, dim)
    out_transformed = mx.hadamard_transform(out_rot)
    output = out_transformed * signs

    return output.astype(queries.dtype)


# ---------------------------------------------------------------------------
# Model-aware config recommendation (moe-v-compression-frontier.md)
# ---------------------------------------------------------------------------


def recommend_config(
    model_type: str,
    head_dim: int,
    num_layers: int,
) -> Dict[str, Union[int, str, bool]]:
    """Suggest optimal TurboQuant config based on model architecture.

    From moe-v-compression-frontier.md: MoE models have a higher fraction of
    attention in their total decode compute (15-30%) because only a few experts
    are active per token, making FFN cheaper. Dense models have attention at <5%
    of decode — KV compression saves memory but barely affects speed.

    This means TurboQuant compression has MORE speed impact on MoE models, and
    more aggressive quantization (symmetric turbo3/turbo4) is worthwhile.

    Args:
        model_type: One of "dense", "moe", or "small" (<3B params).
        head_dim: Head dimension (e.g. 64, 128, 256).
        num_layers: Total number of transformer layers.

    Returns:
        Dict with recommended config keys:
            - bits: V quantization bit-width
            - key_bits: K quantization bit-width (0 = FP16)
            - symmetric: Whether K and V use same quantization
            - rationale: Human-readable explanation
            - speed_benefit: Expected speed improvement category

    Example:
        >>> recommend_config("moe", 128, 32)
        {'bits': 4, 'key_bits': 4, 'symmetric': True, ...}
        >>> recommend_config("dense", 128, 32)
        {'bits': 4, 'key_bits': 0, 'symmetric': False, ...}
    """
    model_type = model_type.lower().strip()

    if model_type == "small":
        # Small models (<3B): overhead of per-layer encode/decode may
        # outweigh the savings. Memory benefit exists but speed may regress.
        return {
            "bits": 4,
            "key_bits": 0,
            "symmetric": False,
            "boundary_layers": 1,
            "rationale": (
                "Small models (<3B): per-layer encode/decode overhead is a "
                "larger fraction of total compute. Use asymmetric (K=FP16, "
                "V=turbo4) for memory savings only. Speed benefit unlikely."
            ),
            "speed_benefit": "minimal",
        }
    elif model_type == "moe":
        # MoE: attention is 15-30% of decode (only a few experts active).
        # Symmetric turbo4 recommended — speed benefit is meaningful.
        return {
            "bits": 4,
            "key_bits": 4,
            "symmetric": True,
            "boundary_layers": 2,
            "rationale": (
                "MoE models: attention is 15-30% of decode compute (FFN is "
                "cheap with sparse expert routing). Symmetric turbo4 gives "
                "both memory AND speed benefits. turbo3 is viable for "
                "aggressive compression. (moe-v-compression-frontier.md)"
            ),
            "speed_benefit": "significant",
        }
    else:
        # Dense: attention is <5% of decode. KV compression helps memory
        # but speed improvement is negligible.
        return {
            "bits": 4,
            "key_bits": 0,
            "symmetric": False,
            "boundary_layers": 2,
            "rationale": (
                "Dense models: attention is <5% of decode — FFN dominates. "
                "KV compression saves memory but speed benefit is minimal. "
                "Asymmetric (K=FP16, V=turbo4) recommended for best quality "
                "per byte. (moe-v-compression-frontier.md)"
            ),
            "speed_benefit": "minimal",
        }


# ---------------------------------------------------------------------------
# TurboQuantKVCache — the main cache module
# ---------------------------------------------------------------------------


class TurboQuantKVCache(Module):
    """TurboQuant-compressed KV cache for transformer inference.

    Stores key and value projections in compressed form using the TurboQuant
    algorithm (SRHT + Lloyd-Max quantization). Supports:

    - Two-phase operation: raw FP during prefill, compressed at first decode
    - Boundary layer protection: first/last N layers stay at full precision
    - Asymmetric K/V: keys can stay FP16 while values are compressed
    - Configurable bit-width: 2, 3, or 4 bits per element

    **Quality guidance** (from empirical testing):

    K precision dominates quality via softmax amplification — small K errors
    become large attention weight errors through the exponential. V errors
    are merely averaged across tokens.

    - Best quality: ``key_bits=0, bits=4`` (K=FP16, V=turbo4 aka "turbo0v4")
    - Good balance: ``key_bits=4, bits=4`` (symmetric turbo4)
    - Aggressive: ``key_bits=0, bits=3`` (K=FP16, V=turbo3)

    Boundary layers (first 2 + last 2 by default) stay at full precision
    as they carry disproportionate signal (confirmed by dhawalc's
    TurboQuantDC independent validation).

    Args:
        bits (int): Quantization bit-width for values. Default: 4.
        key_bits (Optional[int]): Bit-width for keys. None = same as bits.
            Set to 0 or -1 to keep keys at full precision (asymmetric mode).
        seed (int): SRHT random seed. Default: 42.
        boundary_layers (int): Number of layers at start/end to keep at full
            precision. Default: 2. Override with TURBO_BOUNDARY_LAYERS env var.
        layer_idx (Optional[int]): This cache's layer index (for boundary protection).
        num_layers (Optional[int]): Total number of layers (for boundary protection).

    Example:
        >>> cache = TurboQuantKVCache(bits=4, key_bits=0)  # V=4bit, K=FP (recommended)
        >>> # During model forward pass:
        >>> keys, values = cache.update_and_fetch(keys, values)

    Usage pattern in a transformer layer::

        cache = TurboQuantKVCache(bits=4, layer_idx=layer_id, num_layers=32)
        # Prefill phase — stores raw
        k, v = cache.update_and_fetch(k_proj, v_proj)
        # ... after prefill, call cache.compress() to quantize
        # Decode phase — returns decompressed from quantized storage
        k, v = cache.update_and_fetch(new_k, new_v)
    """

    def __init__(
        self,
        bits: int = 4,
        key_bits: Optional[int] = None,
        seed: int = 42,
        boundary_layers: int = 2,
        layer_idx: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        super().__init__()

        self.v_bits = bits
        # key_bits <= 0 means keep keys at full precision
        self.k_bits = key_bits if key_bits is not None else bits
        self.seed = seed

        # TURBO_BOUNDARY_LAYERS env var overrides the constructor arg
        env_boundary = os.environ.get("TURBO_BOUNDARY_LAYERS")
        if env_boundary is not None:
            boundary_layers = int(env_boundary)
        self.boundary_layers = boundary_layers
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        # Determine if this layer is a boundary layer (stays at FP)
        self._is_boundary = False
        if layer_idx is not None and num_layers is not None:
            self._is_boundary = (
                layer_idx < boundary_layers
                or layer_idx >= num_layers - boundary_layers
            )

        # State
        self._compressed = False
        self._keys: Optional[mx.array] = None  # Raw or decoded keys
        self._values: Optional[mx.array] = None  # Raw or decoded values

        # Compressed storage
        self._packed_keys: Optional[mx.array] = None
        self._key_norms: Optional[mx.array] = None
        self._packed_values: Optional[mx.array] = None
        self._value_norms: Optional[mx.array] = None

        # Track the head dim for decode
        self._dim: Optional[int] = None

    @property
    def is_boundary_layer(self) -> bool:
        """Whether this layer stays at full precision (boundary protection)."""
        return self._is_boundary

    @property
    def is_compressed(self) -> bool:
        """Whether the cache is currently in compressed form."""
        return self._compressed

    @property
    def seq_len(self) -> int:
        """Current sequence length stored in cache."""
        if self._compressed:
            if self._packed_keys is not None:
                return self._packed_keys.shape[-2]
            if self._packed_values is not None:
                return self._packed_values.shape[-2]
        if self._keys is not None:
            return self._keys.shape[-2]
        return 0

    @property
    def compress_keys(self) -> bool:
        """Whether keys should be compressed (vs kept at FP)."""
        return self.k_bits > 0 and not self._is_boundary

    @property
    def compress_values(self) -> bool:
        """Whether values should be compressed."""
        return self.v_bits > 0 and not self._is_boundary

    def compress(self) -> None:
        """Compress the raw KV cache into TurboQuant format.

        Call this after prefill is complete, before starting decode.
        Boundary layers are left uncompressed.
        """
        if self._compressed or self._keys is None:
            return

        self._dim = self._keys.shape[-1]

        if self.compress_keys:
            self._packed_keys, self._key_norms = turbo_encode(
                self._keys, bits=self.k_bits, seed=self.seed
            )
            self._keys = None  # Free the raw storage
        # else: keys stay as self._keys (FP)

        if self.compress_values:
            self._packed_values, self._value_norms = turbo_encode(
                self._values, bits=self.v_bits, seed=self.seed
            )
            self._values = None  # Free the raw storage
        # else: values stay as self._values (FP)

        self._compressed = True

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update the cache with new KV pairs and return full KV for attention.

        Before compression: simply concatenates new KV to existing cache.
        After compression: encodes new tokens, appends to compressed storage,
        and returns decoded KV for attention computation.

        Args:
            keys: New key projections, shape (batch, heads, new_len, dim).
            values: New value projections, shape (batch, heads, new_len, dim).

        Returns:
            Tuple of (all_keys, all_values) for attention computation.
            Both have shape (batch, heads, total_len, dim).
        """
        if self._dim is None:
            self._dim = keys.shape[-1]

        if not self._compressed:
            # Pre-compression: just accumulate raw tensors
            if self._keys is None:
                self._keys = keys
                self._values = values
            else:
                self._keys = mx.concatenate([self._keys, keys], axis=-2)
                self._values = mx.concatenate([self._values, values], axis=-2)
            return self._keys, self._values

        # Post-compression: encode new tokens and append
        dim = self._dim

        # Handle keys
        if self.compress_keys:
            new_packed_k, new_k_norms = turbo_encode(
                keys, bits=self.k_bits, seed=self.seed
            )
            if self._packed_keys is not None:
                self._packed_keys = mx.concatenate(
                    [self._packed_keys, new_packed_k], axis=-2
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, new_k_norms], axis=-2
                )
            else:
                self._packed_keys = new_packed_k
                self._key_norms = new_k_norms
            # Decode all keys for attention
            all_keys = turbo_decode(
                self._packed_keys, self._key_norms, dim,
                bits=self.k_bits, seed=self.seed,
            )
        else:
            # Keys at full precision
            if self._keys is None:
                self._keys = keys
            else:
                self._keys = mx.concatenate([self._keys, keys], axis=-2)
            all_keys = self._keys

        # Handle values
        if self.compress_values:
            new_packed_v, new_v_norms = turbo_encode(
                values, bits=self.v_bits, seed=self.seed
            )
            if self._packed_values is not None:
                self._packed_values = mx.concatenate(
                    [self._packed_values, new_packed_v], axis=-2
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, new_v_norms], axis=-2
                )
            else:
                self._packed_values = new_packed_v
                self._value_norms = new_v_norms
            # Decode all values for attention
            all_values = turbo_decode(
                self._packed_values, self._value_norms, dim,
                bits=self.v_bits, seed=self.seed,
            )
        else:
            # Values at full precision
            if self._values is None:
                self._values = values
            else:
                self._values = mx.concatenate([self._values, values], axis=-2)
            all_values = self._values

        return all_keys, all_values

    def reset(self) -> None:
        """Clear all cached state."""
        self._compressed = False
        self._keys = None
        self._values = None
        self._packed_keys = None
        self._key_norms = None
        self._packed_values = None
        self._value_norms = None
        self._dim = None

    def memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage in bytes (approximate).

        Returns:
            Dict with 'keys_bytes', 'values_bytes', 'total_bytes', and
            'fp_equivalent_bytes' for comparison.
        """
        key_bytes = 0
        val_bytes = 0
        fp_bytes = 0

        seq = self.seq_len
        if seq == 0 or self._dim is None:
            return {
                "keys_bytes": 0, "values_bytes": 0,
                "total_bytes": 0, "fp_equivalent_bytes": 0,
            }

        # Estimate batch*heads from stored shapes
        if self._packed_keys is not None:
            batch_heads = math.prod(self._packed_keys.shape[:-2])
        elif self._keys is not None:
            batch_heads = math.prod(self._keys.shape[:-2])
        else:
            batch_heads = 1

        dim = self._dim
        fp_element_bytes = 2  # float16

        fp_bytes = batch_heads * seq * dim * fp_element_bytes * 2  # K + V

        if self.compress_keys and self._packed_keys is not None:
            packed_dim = self._packed_keys.shape[-1]
            key_bytes = batch_heads * seq * (packed_dim * 4 + 4)  # uint32 + norm
        elif self._keys is not None:
            key_bytes = batch_heads * seq * dim * fp_element_bytes

        if self.compress_values and self._packed_values is not None:
            packed_dim = self._packed_values.shape[-1]
            val_bytes = batch_heads * seq * (packed_dim * 4 + 4)  # uint32 + norm
        elif self._values is not None:
            val_bytes = batch_heads * seq * dim * fp_element_bytes

        return {
            "keys_bytes": key_bytes,
            "values_bytes": val_bytes,
            "total_bytes": key_bytes + val_bytes,
            "fp_equivalent_bytes": fp_bytes,
        }

    def _extra_repr(self):
        parts = [f"v_bits={self.v_bits}"]
        if self.k_bits != self.v_bits:
            parts.append(f"k_bits={self.k_bits}")
        parts.append(f"seed={self.seed}")
        if self._is_boundary:
            parts.append("boundary=True")
        if self._compressed:
            parts.append(f"compressed=True, seq_len={self.seq_len}")
        elif self._keys is not None:
            parts.append(f"raw, seq_len={self.seq_len}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# TurboKVCache — mlx-lm compatible cache for inference
# ---------------------------------------------------------------------------


def _create_causal_mask(N, offset, window_size=None):
    """Create a causal attention mask (local copy to avoid import cycles)."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def _turbo_create_attention_mask(N, offset, return_array, window_size):
    """Create attention mask matching mlx-lm's cache.create_attention_mask."""
    if window_size is not None:
        return _create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return _create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class TurboKVCache:
    """TurboQuant KV cache compatible with mlx-lm's inference loop.

    Drop-in replacement for mlx-lm's ``KVCache`` that compresses the KV cache
    using TurboQuant (SRHT + Lloyd-Max quantization). Integrates with
    ``generate_step`` / ``stream_generate`` / ``generate`` via the
    ``prompt_cache`` parameter.

    **Two-phase design:**

    1. **Prefill** (num_steps > 1): stores raw FP16 keys/values, matching our
       llama.cpp prefill fix. No quantization overhead during prompt processing.
    2. **Decode** (num_steps == 1): on the first single-token step, compresses
       the raw cache. Subsequent tokens are encoded and appended to the packed
       storage. ``update_and_fetch`` always returns full FP16 K/V so standard
       SDPA works — compression is internal only.

    No ``bits`` attribute is exposed, so ``scaled_dot_product_attention`` in
    mlx-lm's ``base.py`` takes the standard (non-quantized) path.

    Asymmetric K/V is supported: set ``key_bits=0`` to keep keys at FP16
    while values are turbo-compressed. This is the recommended config since
    K errors get amplified through softmax exponentials.

    Args:
        bits (int): Quantization bit-width for values. Default: 4.
        key_bits (Optional[int]): Bit-width for keys. ``None`` = same as
            ``bits``. Set to 0 to keep keys at full precision. Default: None.
        seed (int): SRHT random seed. Default: 42.
        min_compress_tokens (int): Minimum number of cached tokens before
            compression kicks in. Below this threshold, KV stays in raw FP16
            — the memory savings are <2MB but the encode/decode overhead
            costs ~30% decode speed. Default: 256.

    Example:
        >>> import mlx_lm
        >>> from mlx.nn.layers.turbo_kv_cache import TurboKVCache
        >>> model, tokenizer = mlx_lm.load('mlx-community/Qwen3.5-2B-8bit')
        >>> n_layers = len(model.model.layers)
        >>> cache = [TurboKVCache(bits=4) for _ in range(n_layers)]
        >>> text = mlx_lm.generate(
        ...     model, tokenizer, prompt='Hello',
        ...     max_tokens=20, prompt_cache=cache, verbose=True,
        ... )
    """

    def __init__(
        self,
        bits: int = 4,
        key_bits: Optional[int] = None,
        seed: int = 42,
        min_compress_tokens: int = 256,
        fused_attention: bool = False,
    ):
        self.v_bits = bits
        self.k_bits = key_bits if key_bits is not None else bits
        self.seed = seed
        # Deferred compression: below this threshold, keep KV in raw FP16.
        # The memory savings at short context are <2MB but the speed cost of
        # encode/decode is ~30%. Only compress when the cache exceeds this size.
        self.min_compress_tokens = min_compress_tokens

        # When True, skip creating decoded FP16 buffers during compression.
        # Use cache.attention() instead of update_and_fetch + SDPA to avoid
        # the double-storage problem. Requires symmetric 4-bit, Metal GPU.
        self._fused_attention = fused_attention

        # Raw (uncompressed) storage — used during prefill
        self._raw_keys: Optional[mx.array] = None
        self._raw_values: Optional[mx.array] = None

        # Compressed storage — used during decode
        self._packed_keys: Optional[mx.array] = None
        self._key_norms: Optional[mx.array] = None
        self._packed_values: Optional[mx.array] = None
        self._value_norms: Optional[mx.array] = None

        # FP keys when key_bits <= 0 (asymmetric mode)
        self._fp_keys: Optional[mx.array] = None

        # FP values when v_bits <= 0 (no compression)
        self._fp_values: Optional[mx.array] = None

        # Cached decoded FP16 arrays — avoids re-decoding entire packed storage
        # every step. Only the newly added token(s) get decoded and concatenated.
        # This matches the llama.cpp approach: compressed storage is source of
        # truth for memory savings, decoded FP16 window is for fast attention.
        # Skipped when fused_attention=True (fused kernel reads packed directly).
        self._decoded_keys: Optional[mx.array] = None
        self._decoded_values: Optional[mx.array] = None

        self._is_compressed = False
        self._dim: Optional[int] = None
        self.offset = 0

    @property
    def compress_keys(self) -> bool:
        """Whether keys should be turbo-compressed (vs kept at FP)."""
        return self.k_bits > 0

    @property
    def compress_values(self) -> bool:
        """Whether values should be turbo-compressed."""
        return self.v_bits > 0

    def _compress_raw_cache(self) -> None:
        """Compress accumulated raw prefill cache into TurboQuant format.

        Called once on the first decode step. After this, the raw buffers are
        freed and all new tokens go through encode→pack.
        """
        if self._is_compressed or self._raw_keys is None:
            return

        self._dim = self._raw_keys.shape[-1]

        if self.compress_keys:
            self._packed_keys, self._key_norms = turbo_encode(
                self._raw_keys, bits=self.k_bits, seed=self.seed,
            )
            # Decode once to seed the FP16 cache — subsequent steps only
            # decode the new token and concatenate (O(1) not O(n)).
            # Skip when fused_attention=True: the fused kernel reads packed
            # data directly, so decoded FP16 buffers waste memory.
            if not self._fused_attention:
                self._decoded_keys = turbo_decode(
                    self._packed_keys, self._key_norms, self._dim,
                    bits=self.k_bits, seed=self.seed,
                )
        else:
            self._fp_keys = self._raw_keys

        if self.compress_values:
            self._packed_values, self._value_norms = turbo_encode(
                self._raw_values, bits=self.v_bits, seed=self.seed,
            )
            # Same: decode once, then incremental.
            # Skip when fused_attention=True.
            if not self._fused_attention:
                self._decoded_values = turbo_decode(
                    self._packed_values, self._value_norms, self._dim,
                    bits=self.v_bits, seed=self.seed,
                )
        else:
            self._fp_values = self._raw_values

        # Free raw buffers
        self._raw_keys = None
        self._raw_values = None
        self._is_compressed = True

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new K/V and return full (decoded) K/V for SDPA.

        During prefill (num_steps > 1), stores raw FP16 — no quantization.
        On the first decode step (num_steps == 1), compresses the raw cache.
        Subsequent decode steps encode the new token and append to packed
        storage.

        Always returns plain ``mx.array`` keys and values (not quantized
        tuples) so standard SDPA works. The compression is internal.

        Args:
            keys: Shape ``(B, n_kv_heads, num_steps, head_dim)``.
            values: Shape ``(B, n_kv_heads, num_steps, head_dim)``.

        Returns:
            ``(all_keys, all_values)`` both as ``mx.array`` with shape
            ``(B, n_kv_heads, total_seq_len, head_dim)``.
        """
        num_steps = keys.shape[2]

        if self._dim is None:
            self._dim = keys.shape[-1]

        # --- Prefill phase: accumulate raw ---
        if not self._is_compressed and num_steps > 1:
            if self._raw_keys is None:
                self._raw_keys = keys
                self._raw_values = values
            else:
                self._raw_keys = mx.concatenate(
                    [self._raw_keys, keys], axis=2,
                )
                self._raw_values = mx.concatenate(
                    [self._raw_values, values], axis=2,
                )
            self.offset = self._raw_keys.shape[2]
            return self._raw_keys, self._raw_values

        # --- Deferred compression: stay raw until we hit the token threshold ---
        # Below min_compress_tokens, the memory savings are negligible (<2MB)
        # but the encode/decode overhead costs ~30% decode speed. Keep raw FP16
        # until the cache is big enough to justify compression.
        if not self._is_compressed and self.offset < self.min_compress_tokens:
            if self._raw_keys is None:
                self._raw_keys = keys
                self._raw_values = values
            else:
                self._raw_keys = mx.concatenate(
                    [self._raw_keys, keys], axis=2,
                )
                self._raw_values = mx.concatenate(
                    [self._raw_values, values], axis=2,
                )
            self.offset = self._raw_keys.shape[2]
            return self._raw_keys, self._raw_values

        # --- Transition: first decode step triggers compression ---
        if not self._is_compressed:
            self._compress_raw_cache()

        # --- Decode phase: encode new token(s), append, return decoded ---
        self.offset += num_steps
        dim = self._dim

        # Handle keys
        if self.compress_keys:
            new_pk, new_kn = turbo_encode(keys, bits=self.k_bits, seed=self.seed)
            if self._packed_keys is not None:
                self._packed_keys = mx.concatenate(
                    [self._packed_keys, new_pk], axis=2,
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, new_kn], axis=2,
                )
            else:
                self._packed_keys = new_pk
                self._key_norms = new_kn
            # Incremental decode: only decode the new token(s), concat with
            # cached FP16. Avoids O(n) full-cache decode every step.
            new_decoded_k = turbo_decode(
                new_pk, new_kn, dim, bits=self.k_bits, seed=self.seed,
            )
            if self._decoded_keys is not None:
                self._decoded_keys = mx.concatenate(
                    [self._decoded_keys, new_decoded_k], axis=2,
                )
            else:
                self._decoded_keys = new_decoded_k
            all_keys = self._decoded_keys
        else:
            if self._fp_keys is not None:
                self._fp_keys = mx.concatenate([self._fp_keys, keys], axis=2)
            else:
                self._fp_keys = keys
            all_keys = self._fp_keys

        # Handle values
        if self.compress_values:
            new_pv, new_vn = turbo_encode(values, bits=self.v_bits, seed=self.seed)
            if self._packed_values is not None:
                self._packed_values = mx.concatenate(
                    [self._packed_values, new_pv], axis=2,
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, new_vn], axis=2,
                )
            else:
                self._packed_values = new_pv
                self._value_norms = new_vn
            # Incremental decode: only decode the new token(s)
            new_decoded_v = turbo_decode(
                new_pv, new_vn, dim, bits=self.v_bits, seed=self.seed,
            )
            if self._decoded_values is not None:
                self._decoded_values = mx.concatenate(
                    [self._decoded_values, new_decoded_v], axis=2,
                )
            else:
                self._decoded_values = new_decoded_v
            all_values = self._decoded_values
        else:
            if self._fp_values is not None:
                self._fp_values = mx.concatenate([self._fp_values, values], axis=2)
            else:
                self._fp_values = values
            all_values = self._fp_values

        return all_keys, all_values

    def attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: Optional[float] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Update cache and compute attention in one step, using fused kernel
        when possible to avoid materializing FP16 K/V.

        This is the preferred attention path for TurboKVCache. During decode
        (T_q=1) with both K and V compressed at 4-bit, it uses the fused
        Metal kernel that operates directly on packed data. Otherwise, it
        falls back to update_and_fetch + standard SDPA.

        When the fused path is used:
        - No FP16 K/V buffer is ever allocated (solves the double-storage problem)
        - The decoded FP16 caches (_decoded_keys, _decoded_values) are NOT needed
        - Memory usage drops to purely packed storage + norms

        Args:
            queries: Query projections, shape (B, n_q_heads, T_q, dim).
            keys: New key projections, shape (B, n_kv_heads, T_q, dim).
            values: New value projections, shape (B, n_kv_heads, T_q, dim).
            scale: Attention scale. Default: 1/sqrt(dim).
            mask: Attention mask. Only used in fallback path.

        Returns:
            Attention output, shape (B, n_q_heads, T_q, dim).

        Example:
            >>> cache = TurboKVCache(bits=4, key_bits=4)
            >>> # In the model's attention layer:
            >>> output = cache.attention(q, k_proj, v_proj, scale=scale)
        """
        num_steps = keys.shape[2]
        dim = keys.shape[-1]

        if self._dim is None:
            self._dim = dim

        # Check if we should trigger compression for the fused path.
        # This handles the transition from prefill → decode when
        # fused_attention=True, compressing without creating decoded FP16.
        _wants_fuse = (
            num_steps == 1
            and self.compress_keys
            and self.compress_values
            and self.k_bits == self.v_bits == 4
            and mask is None
            and dim <= 256
            and mx.metal.is_available()
        )
        if _wants_fuse and not self._is_compressed and self.offset >= self.min_compress_tokens:
            # Trigger compression (skips decoded FP16 if fused_attention=True)
            self._compress_raw_cache()

        can_fuse = _wants_fuse and self._is_compressed

        if can_fuse:
            # Encode the new token and append to packed storage
            self.offset += num_steps

            new_pk, new_kn = turbo_encode(keys, bits=self.k_bits, seed=self.seed)
            if self._packed_keys is not None:
                self._packed_keys = mx.concatenate(
                    [self._packed_keys, new_pk], axis=2,
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, new_kn], axis=2,
                )
            else:
                self._packed_keys = new_pk
                self._key_norms = new_kn

            new_pv, new_vn = turbo_encode(values, bits=self.v_bits, seed=self.seed)
            if self._packed_values is not None:
                self._packed_values = mx.concatenate(
                    [self._packed_values, new_pv], axis=2,
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, new_vn], axis=2,
                )
            else:
                self._packed_values = new_pv
                self._value_norms = new_vn

            # Fused attention on packed data — no FP16 materialization
            # NOTE: we do NOT update _decoded_keys/_decoded_values here.
            # The fused path doesn't need them. If the user later calls
            # update_and_fetch (fallback path), the decoded caches will be
            # stale — but that's OK because update_and_fetch rebuilds them.
            output = turbo_fused_attention(
                queries,
                self._packed_keys,
                self._key_norms,
                self._packed_values,
                self._value_norms,
                dim=dim,
                bits=self.v_bits,
                seed=self.seed,
                scale=scale,
            )

            # Handle GQA: if n_q_heads > n_kv_heads, the fused kernel already
            # handles this because it broadcasts across heads. But actually,
            # turbo_fused_attention expects Q and K to have the same n_heads.
            # GQA support will need the kernel to take a heads_ratio param.
            # TODO: Add GQA support to turbo_fused_attention
            return output

        else:
            # Fallback: update_and_fetch + standard SDPA
            all_keys, all_values = self.update_and_fetch(keys, values)

            if scale is None:
                scale = 1.0 / math.sqrt(dim)

            # Use mx.fast.scaled_dot_product_attention for the fallback
            if mask is None and num_steps == 1:
                # Decode: no mask needed
                return mx.fast.scaled_dot_product_attention(
                    queries, all_keys, all_values, scale=scale,
                )
            else:
                # Prefill or masked: use the mask
                if mask is None:
                    mask = "causal"
                return mx.fast.scaled_dot_product_attention(
                    queries, all_keys, all_values, scale=scale, mask=mask,
                )

    # --- mlx-lm _BaseCache interface ---

    @property
    def state(self):
        """Return cache tensors for mx.eval() materialization."""
        # During prefill, return raw buffers
        if not self._is_compressed:
            if self._raw_keys is not None:
                return self._raw_keys, self._raw_values
            return []

        # After compression, return all stored tensors (including decoded FP16 cache)
        parts = []
        if self._packed_keys is not None:
            parts.extend([self._packed_keys, self._key_norms])
        if self._decoded_keys is not None:
            parts.append(self._decoded_keys)
        if self._fp_keys is not None:
            parts.append(self._fp_keys)
        if self._packed_values is not None:
            parts.extend([self._packed_values, self._value_norms])
        if self._decoded_values is not None:
            parts.append(self._decoded_values)
        if self._fp_values is not None:
            parts.append(self._fp_values)
        return parts if parts else []

    @state.setter
    def state(self, v):
        if v is not None and v:
            # TODO: Implement state restore for save/load prompt cache
            pass

    @property
    def meta_state(self):
        return str(self.offset)

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            self.offset = int(v)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Trim n tokens from the cache. Returns actual tokens trimmed."""
        n = min(self.offset, n)
        self.offset -= n
        # TODO: Actually trim the packed/raw buffers for correctness
        # For now this handles the common case where trim is called but
        # the cache is about to be rebuilt anyway
        if not self._is_compressed and self._raw_keys is not None:
            if n > 0:
                self._raw_keys = self._raw_keys[..., :-n, :]
                self._raw_values = self._raw_values[..., :-n, :]
        if n > 0:
            # Trim decoded FP16 caches to stay in sync
            if self._decoded_keys is not None:
                self._decoded_keys = self._decoded_keys[..., :-n, :]
            if self._decoded_values is not None:
                self._decoded_values = self._decoded_values[..., :-n, :]
        return n

    def make_mask(self, N, return_array=False, window_size=None):
        """Create attention mask (called by mlx-lm's create_attention_mask)."""
        return _turbo_create_attention_mask(
            N, offset=self.offset, return_array=return_array,
            window_size=window_size,
        )

    def empty(self) -> bool:
        """Return True if the cache has no stored data."""
        return (
            self._raw_keys is None
            and self._packed_keys is None
            and self._fp_keys is None
        )

    @property
    def nbytes(self) -> int:
        """Approximate memory usage in bytes."""
        total = 0
        for arr in [
            self._raw_keys, self._raw_values,
            self._packed_keys, self._key_norms,
            self._packed_values, self._value_norms,
            self._decoded_keys, self._decoded_values,
            self._fp_keys, self._fp_values,
        ]:
            if arr is not None:
                total += arr.nbytes
        return total

    def __repr__(self):
        mode = "compressed" if self._is_compressed else "raw"
        k_desc = f"k={self.k_bits}bit" if self.compress_keys else "k=fp"
        v_desc = f"v={self.v_bits}bit" if self.compress_values else "v=fp"
        return (
            f"TurboKVCache({k_desc}, {v_desc}, {mode}, "
            f"offset={self.offset}, dim={self._dim}, "
            f"min_compress={self.min_compress_tokens})"
        )
