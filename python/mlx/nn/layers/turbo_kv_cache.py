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
    x_rotated = mx.hadamard_transform(x_flipped)

    # 4. Boundary quantize → indices
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

    # Weighted sum of values
    output = weights @ values

    return output


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
    ):
        self.v_bits = bits
        self.k_bits = key_bits if key_bits is not None else bits
        self.seed = seed

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
            # decode the new token and concatenate (O(1) not O(n))
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
            # Same: decode once, then incremental
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
            f"offset={self.offset}, dim={self._dim})"
        )
