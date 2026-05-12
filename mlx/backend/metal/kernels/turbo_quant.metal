// Copyright © 2026 Eric Kryski. TurboQuant Metal kernels for compressed-domain
// attention.
//
// Framework-level compiled kernels. Runtime-varying parameters (token_count,
// num_blocks, repeat_count, etc.) are buffer arguments instead of template
// constants, eliminating per-token pipeline recompilation from the JIT version.
//
// Compile-time template params: Bits, Dim, PackedWidth (determine data layout).

#include <metal_atomic>
#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// ============================================================================
// Scoring kernel: Q×K dot product from packed codebook indices
// ============================================================================
template <int Bits, int Dim, int PackedWidth>
[[kernel]] void turbo_score(
    const device float* q_rot [[buffer(0)]],
    const device uint32_t* packed [[buffer(1)]],
    const device float* norms [[buffer(2)]],
    const device float* codebook [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant int& token_count [[buffer(5)]],
    constant int& repeat_count [[buffer(6)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint MASK = (1u << Bits) - 1u;
  constexpr uint LEVELS = 1u << Bits;

  uint lane = pos.x;
  uint q_idx = pos.y;
  uint k_idx = pos.z;
  uint kv_idx = q_idx / uint(repeat_count);

  const device float* q_ptr = q_rot + q_idx * Dim;
  const device uint32_t* packed_ptr =
      packed + kv_idx * uint(token_count) * PackedWidth + k_idx * PackedWidth;
  float norm_val = norms[kv_idx * uint(token_count) + k_idx];

  float cb[LEVELS];
  for (uint i = 0; i < LEVELS; i++)
    cb[i] = codebook[i];

  float acc = 0.0f;
  for (uint d = lane; d < uint(Dim); d += 32) {
    uint bit_offset = d * Bits;
    uint word_idx = bit_offset / 32;
    uint shift = bit_offset % 32;
    uint value = (packed_ptr[word_idx] >> shift);
    int spill = (int)shift + (int)Bits - 32;
    if (spill > 0) {
      value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
    }
    value &= MASK;
    acc += q_ptr[d] * cb[value];
  }

  acc = simd_sum(acc);
  if (lane == 0) {
    scores[q_idx * uint(token_count) + k_idx] = acc * norm_val;
  }
}

// ============================================================================
// Fused encode: norm + rotate + quantize + pack + norm correction
// ============================================================================
template <int Bits, int Dim, int PackedWidth>
[[kernel]] void turbo_fused_encode(
    const device float* input [[buffer(0)]],
    const device float* rotation [[buffer(1)]],
    const device float* boundaries [[buffer(2)]],
    const device float* codebook [[buffer(3)]],
    device uint32_t* packed_out [[buffer(4)]],
    device float* norms_out [[buffer(5)]],
    uint d [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  constexpr uint LEVELS = 1u << Bits;

  float val = input[row * Dim + d];

  // L2 norm via SIMD + threadgroup reduction
  float sq = val * val;
  float norm_sq = simd_sum(sq);
  threadgroup float shared_norm[16];
  uint sg_id = d / 32;
  if (d % 32 == 0)
    shared_norm[sg_id] = norm_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_norm_sq = 0;
  uint num_groups = (Dim + 31) / 32;
  for (uint i = 0; i < num_groups; i++)
    total_norm_sq += shared_norm[i];
  float norm_val = sqrt(total_norm_sq);
  float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

  float unit_val = val * inv_norm;

  // Rotate via shared memory matmul
  threadgroup float shared_unit[1024];
  shared_unit[d] = unit_val;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rotated = 0.0f;
  for (uint j = 0; j < uint(Dim); j++) {
    rotated += rotation[d * Dim + j] * shared_unit[j];
  }

  // Quantize via branchless boundary comparison
  uint idx = 0;
  for (uint b = 0; b < LEVELS - 1; b++)
    idx += (uint)(rotated > boundaries[b]);

  // Pack bits via atomic OR on threadgroup memory
  uint bit_offset = d * Bits;
  uint word_idx = bit_offset / 32;
  uint shift = bit_offset % 32;
  uint masked = idx & ((1u << Bits) - 1u);

  // Sized for max PackedWidth across all instantiated (D, bits) combos:
  // D=512 B=8 → PW=128. Smaller (D, bits) use the prefix only.
  threadgroup uint shared_packed[128];
  if (d < uint(PackedWidth))
    shared_packed[d] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  atomic_fetch_or_explicit(
      (threadgroup atomic_uint*)&shared_packed[word_idx],
      masked << shift,
      memory_order_relaxed);
  int spill_bits = (int)shift + (int)Bits - 32;
  if (spill_bits > 0) {
    atomic_fetch_or_explicit(
        (threadgroup atomic_uint*)&shared_packed[word_idx + 1],
        masked >> ((uint)Bits - (uint)spill_bits),
        memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (d < uint(PackedWidth))
    packed_out[row * PackedWidth + d] = shared_packed[d];

  // Norm correction
  float centroid_val = codebook[idx];
  float recon_sq = centroid_val * centroid_val;
  float recon_norm_sq = simd_sum(recon_sq);
  if (d % 32 == 0)
    shared_norm[sg_id] = recon_norm_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_recon_sq = 0;
  for (uint i = 0; i < num_groups; i++)
    total_recon_sq += shared_norm[i];
  float recon_norm = sqrt(total_recon_sq);
  float corrected_norm =
      (recon_norm > 1e-8f) ? (norm_val / recon_norm) : norm_val;

  if (d == 0)
    norms_out[row] = corrected_norm;
}

// ============================================================================
// Fused WHT encode: norm + Walsh-Hadamard butterfly + quantize + pack
// ============================================================================
template <int Bits, int Dim, int PackedWidth, int LogDim>
[[kernel]] void turbo_fused_encode_wht(
    const device float* input [[buffer(0)]],
    const device float* wht_signs [[buffer(1)]],
    const device float* boundaries [[buffer(2)]],
    device uint32_t* packed_out [[buffer(3)]],
    device float* norms_out [[buffer(4)]],
    uint d [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  constexpr uint LEVELS = 1u << Bits;

  float val = input[row * Dim + d];

  // L2 norm
  float sq = val * val;
  float norm_sq = simd_sum(sq);
  threadgroup float shared_norm[16];
  uint sg_id = d / 32;
  if (d % 32 == 0)
    shared_norm[sg_id] = norm_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total_norm_sq = 0;
  uint num_groups = (Dim + 31) / 32;
  for (uint i = 0; i < num_groups; i++)
    total_norm_sq += shared_norm[i];
  float norm_val = sqrt(total_norm_sq);
  float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

  // Normalize + sign flip (fused)
  float wht_val = val * (inv_norm * wht_signs[d]);

  // WHT butterfly: intra-SIMD via shuffle, cross-SIMD via shared memory
  uint simd_stages = min(uint(LogDim), 5u);
  uint lane_in_simd = d % 32;
  for (uint s = 0; s < simd_stages; s++) {
    uint step = 1u << s;
    float other = simd_shuffle_xor(wht_val, step);
    wht_val = (lane_in_simd & step) ? (other - wht_val) : (other + wht_val);
  }

  threadgroup float shared_buf[1024];
  if (uint(LogDim) > 5u) {
    shared_buf[d] = wht_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = simd_stages; s < uint(LogDim); s++) {
      uint half_block = 1u << s;
      uint block_size = half_block << 1;
      uint block_id = d / block_size;
      uint pos_in_block = d % block_size;
      float a, b;
      if (pos_in_block < half_block) {
        a = shared_buf[block_id * block_size + pos_in_block];
        b = shared_buf[block_id * block_size + pos_in_block + half_block];
        shared_buf[d] = a + b;
      } else {
        a = shared_buf[block_id * block_size + pos_in_block - half_block];
        b = shared_buf[block_id * block_size + pos_in_block];
        shared_buf[d] = a - b;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    wht_val = shared_buf[d];
  }

  float rotated = wht_val * (1.0f / sqrt((float)Dim));

  // Quantize + pack (same as dense encode)
  uint idx = 0;
  for (uint b = 0; b < LEVELS - 1; b++)
    idx += (uint)(rotated > boundaries[b]);

  uint bit_offset = d * Bits;
  uint word_idx = bit_offset / 32;
  uint shift = bit_offset % 32;
  uint masked = idx & ((1u << Bits) - 1u);

  // Sized for max PackedWidth across all instantiated (D, bits) combos:
  // D=512 B=8 → PW=128. Smaller (D, bits) use the prefix only.
  threadgroup uint shared_packed[128];
  if (d < uint(PackedWidth))
    shared_packed[d] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  atomic_fetch_or_explicit(
      (threadgroup atomic_uint*)&shared_packed[word_idx],
      masked << shift,
      memory_order_relaxed);
  int spill_bits = (int)shift + (int)Bits - 32;
  if (spill_bits > 0) {
    atomic_fetch_or_explicit(
        (threadgroup atomic_uint*)&shared_packed[word_idx + 1],
        masked >> ((uint)Bits - (uint)spill_bits),
        memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (d < uint(PackedWidth))
    packed_out[row * PackedWidth + d] = shared_packed[d];
  if (d == 0)
    norms_out[row] = norm_val; // WHT is orthogonal — no norm correction
}

// ============================================================================
// TurboFlash Pass 2: Cross-block online softmax reduction
//
// Output dtype: `bfloat`. All current MLX LMs run bf16 weights; writing bf16
// directly skips one fp32→bf16 conversion step in the consumer's graph.
// Accumulators stay fp32 (m, l, o[]) and only the final write narrows — same
// pattern MLX SDPA uses.
//
// History: previously fp32 output forced an `output.asType(queries.dtype)`
// cast in the Swift caller (see git history of TurboQuantKVCache.swift,
// PR #104) to defeat an upstream MLX graph-compile interaction (issues
// #87/#92) that produced `!!!!!` decoding on certain shapes (Qwen3.5-9B
// nKVH=4 was the reliable reproducer). Switching this kernel to bf16
// output eliminated the bug at the source on every model tested. If
// the bug ever returns on a new model/shape, restore the cast as a
// short-term workaround and investigate fusion behavior at the upstream
// MLX scheduler / graph-compile level.
// ============================================================================
template <int Dim>
[[kernel]] void turbo_flash_pass2(
    const device float* o_partials [[buffer(0)]],
    const device float* m_partials [[buffer(1)]],
    const device float* l_partials [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant int& num_blocks [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;
  uint lane = pos.x;
  uint q_idx = pos.y;

  float m = -INFINITY;
  float l = 0.0f;
  float o[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++)
    o[i] = 0.0f;

  for (uint b = 0; b < uint(num_blocks); b++) {
    uint ml_idx = q_idx * uint(num_blocks) + b;
    float block_m = m_partials[ml_idx];
    float block_l = l_partials[ml_idx];
    if (block_l == 0.0f)
      continue;

    float new_m = max(m, block_m);
    float exp_old = exp(m - new_m);
    float exp_block = exp(block_m - new_m);

    uint partial_base = (q_idx * uint(num_blocks) + b) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d < uint(Dim)) {
        o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
      }
    }
    l = l * exp_old + block_l * exp_block;
    m = new_m;
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    if (d < uint(Dim)) {
      output[q_idx * Dim + d] = (bfloat)(o[i] * inv_l);
    }
  }
}

// ============================================================================
// TurboFlash Pass 2 with fused output rotation
// ============================================================================
template <int Dim>
[[kernel]] void turbo_flash_pass2_fused_rot(
    const device float* o_partials [[buffer(0)]],
    const device float* m_partials [[buffer(1)]],
    const device float* l_partials [[buffer(2)]],
    const device float* val_rotation [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    constant int& num_blocks [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;
  uint lane = pos.x;
  uint q_idx = pos.y;

  float m = -INFINITY;
  float l = 0.0f;
  float o[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++)
    o[i] = 0.0f;

  for (uint b = 0; b < uint(num_blocks); b++) {
    uint ml_idx = q_idx * uint(num_blocks) + b;
    float block_m = m_partials[ml_idx];
    float block_l = l_partials[ml_idx];
    if (block_l == 0.0f)
      continue;

    float new_m = max(m, block_m);
    float exp_old = exp(m - new_m);
    float exp_block = exp(block_m - new_m);

    uint partial_base = (q_idx * uint(num_blocks) + b) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d < uint(Dim)) {
        o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
      }
    }
    l = l * exp_old + block_l * exp_block;
    m = new_m;
  }

  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

  // Gather into threadgroup memory for rotation matmul
  threadgroup float shared_out[Dim];
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    if (d < uint(Dim))
      shared_out[d] = o[i] * inv_l;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Inverse rotation: output[d] = Σ_j shared_out[j] * Π_val[j][d]
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    if (d < uint(Dim)) {
      float acc = 0.0f;
      for (uint j = 0; j < uint(Dim); j++) {
        acc += shared_out[j] * val_rotation[j * Dim + d];
      }
      output[q_idx * Dim + d] = (bfloat)acc;
    }
  }
}

// ============================================================================
// Value aggregation: weighted sum of codebook-quantized values
// ============================================================================
template <int Bits, int Dim, int PackedWidth>
[[kernel]] void turbo_value(
    const device float* weights [[buffer(0)]],
    const device uint32_t* packed [[buffer(1)]],
    const device float* norms [[buffer(2)]],
    const device float* codebook [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& token_count [[buffer(5)]],
    constant int& repeat_count [[buffer(6)]],
    constant float& sparse_threshold [[buffer(7)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint MASK = (1u << Bits) - 1u;
  constexpr uint LEVELS = 1u << Bits;

  uint lane = pos.x;
  uint head_idx = pos.y;
  uint dim_block = pos.z;
  uint d = dim_block * 32 + lane;
  if (d >= uint(Dim))
    return;

  uint kv_head = head_idx / uint(repeat_count);

  float cb[LEVELS];
  for (uint i = 0; i < LEVELS; i++)
    cb[i] = codebook[i];

  float acc = 0.0f;
  for (uint t = 0; t < uint(token_count); t++) {
    float w = weights[head_idx * uint(token_count) + t];
    if (w < sparse_threshold)
      continue;

    float norm_val = norms[kv_head * uint(token_count) + t];
    const device uint32_t* packed_ptr =
        packed + kv_head * uint(token_count) * PackedWidth + t * PackedWidth;

    uint bit_offset = d * Bits;
    uint word_idx = bit_offset / 32;
    uint shift = bit_offset % 32;
    uint value = (packed_ptr[word_idx] >> shift);
    int spill_bits = (int)shift + (int)Bits - 32;
    if (spill_bits > 0) {
      value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill_bits));
    }
    value &= MASK;
    acc += w * norm_val * cb[value];
  }

  output[head_idx * Dim + d] = acc;
}

// ============================================================================
// Bulk dequant in rotated codec space
// ============================================================================
// Decompresses [B, H, T, PackedWidth] uint32 + norms[B, H, T] +
// codebook[2^Bits] into [B, H, T, Dim] FP16/BF16, in rotated Π space (caller
// applies the inverse rotation downstream — typically by passing the output to
// MLXFast.scaledDotProductAttention with the matching rotated query, then
// matmul-by-V_rotation on the SDPA output).
//
// One thread per packed `uint32` word, emitting `dims_per_word = 32 / Bits`
// dim outputs each. For Bits ∈ {2, 4, 8}, dims_per_word ∈ {16, 8, 4}, all
// clean (no cross-word spill). For Bits=3, words straddle dims and the
// per-thread loop checks `spill > 0` like the score/value kernels do.
//
// Grid:  (PackedWidth, T, B*H)  — one thread per (packed-word, token, head)
// Group: (PackedWidth, 1, 1)    — clamped to ≤32 in the host dispatcher
template <int Bits, int Dim, int PackedWidth, typename T>
[[kernel]] void turbo_dequant_rotated(
    const device uint32_t* packed [[buffer(0)]],
    const device float* norms [[buffer(1)]],
    const device float* codebook [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant int& tokens [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]]) {
  uint w = pos.x;
  uint t = pos.y;
  uint bh = pos.z;
  if (w >= uint(PackedWidth) || t >= uint(tokens))
    return;

  constexpr uint MASK = (1u << Bits) - 1u;

  uint base = bh * uint(tokens) * PackedWidth + t * PackedWidth;
  uint word = packed[base + w];
  float norm_val = norms[bh * uint(tokens) + t];

  // Bits ∈ {2, 4, 8}: clean nibble/byte packing — fast inner loop.
  // Bits == 3: per-dim spill check matches turbo_score / turbo_value.
  if (Bits == 2 || Bits == 4 || Bits == 8) {
    constexpr uint DIMS_PER_WORD = 32u / Bits;
    uint d_base = w * DIMS_PER_WORD;
    uint out_base = bh * uint(tokens) * Dim + t * Dim + d_base;
    for (uint k = 0; k < DIMS_PER_WORD; k++) {
      uint d = d_base + k;
      if (d >= uint(Dim))
        break;
      uint val = (word >> (k * Bits)) & MASK;
      float result = codebook[val] * norm_val;
      out[out_base + k] = static_cast<T>(result);
    }
  } else {
    // 3-bit (or any other non-divisor): per-dim with spill handling.
    // Each thread emits ceil(32 / Bits) dims worth of output, but the
    // bit-extract is keyed on absolute dim index `d` so cross-word spills
    // pull from word+1 correctly.
    constexpr uint DIMS_PER_WORD = (32u + Bits - 1u) / Bits;
    uint d_base = w * DIMS_PER_WORD;
    for (uint k = 0; k < DIMS_PER_WORD; k++) {
      uint d = d_base + k;
      if (d >= uint(Dim))
        break;
      uint bit_offset = d * Bits;
      uint word_idx = bit_offset / 32;
      uint shift = bit_offset % 32;
      // Re-fetch in case d straddles into the next word for this thread.
      uint local_word = packed[base + word_idx];
      uint val = (local_word >> shift);
      int spill = (int)shift + (int)Bits - 32;
      if (spill > 0) {
        val |= (packed[base + word_idx + 1] << ((uint)Bits - (uint)spill));
      }
      val &= MASK;
      float result = codebook[val] * norm_val;
      out[bh * uint(tokens) * Dim + t * Dim + d] = static_cast<T>(result);
    }
  }
}

// ============================================================================
// Instantiation macros
// ============================================================================

// Common (Bits, Dim) combinations for real models
// PackedWidth = (Dim * Bits + 31) / 32
#define TQ_PW(dim, bits) (((dim) * (bits) + 31) / 32)

#define instantiate_turbo_score(bits, dim)                              \
  template [[host_name("turbo_score_" #bits "_" #dim)]] [[kernel]] void \
  turbo_score<bits, dim, TQ_PW(dim, bits)>(                             \
      const device float*,                                              \
      const device uint32_t*,                                           \
      const device float*,                                              \
      const device float*,                                              \
      device float*,                                                    \
      constant int&,                                                    \
      constant int&,                                                    \
      uint3);

#define instantiate_turbo_encode(bits, dim)                                    \
  template [[host_name("turbo_fused_encode_" #bits "_" #dim)]] [[kernel]] void \
  turbo_fused_encode<bits, dim, TQ_PW(dim, bits)>(                             \
      const device float*,                                                     \
      const device float*,                                                     \
      const device float*,                                                     \
      const device float*,                                                     \
      device uint32_t*,                                                        \
      device float*,                                                           \
      uint,                                                                    \
      uint);

#define instantiate_turbo_encode_wht(bits, dim, logdim)                       \
  template                                                                    \
      [[host_name("turbo_fused_encode_wht_" #bits "_" #dim)]] [[kernel]] void \
      turbo_fused_encode_wht<bits, dim, TQ_PW(dim, bits), logdim>(            \
          const device float*,                                                \
          const device float*,                                                \
          const device float*,                                                \
          device uint32_t*,                                                   \
          device float*,                                                      \
          uint,                                                               \
          uint);

#define instantiate_turbo_pass2(dim)                                   \
  template [[host_name("turbo_flash_p2_" #dim)]] [[kernel]] void       \
  turbo_flash_pass2<dim>(                                              \
      const device float*,                                             \
      const device float*,                                             \
      const device float*,                                             \
      device bfloat*,                                                  \
      constant int&,                                                   \
      uint3);                                                          \
  template [[host_name("turbo_flash_p2_fused_" #dim)]] [[kernel]] void \
  turbo_flash_pass2_fused_rot<dim>(                                    \
      const device float*,                                             \
      const device float*,                                             \
      const device float*,                                             \
      const device float*,                                             \
      device bfloat*,                                                  \
      constant int&,                                                   \
      uint3);

#define instantiate_turbo_value(bits, dim)                              \
  template [[host_name("turbo_value_" #bits "_" #dim)]] [[kernel]] void \
  turbo_value<bits, dim, TQ_PW(dim, bits)>(                             \
      const device float*,                                              \
      const device uint32_t*,                                           \
      const device float*,                                              \
      const device float*,                                              \
      device float*,                                                    \
      constant int&,                                                    \
      constant int&,                                                    \
      constant float&,                                                  \
      uint3);

// Bulk dequant kernel — one host_name per (bits, dim, output dtype) tuple.
// Output dtype suffix matches MLX's standard conventions: `bf16` and `f16`.
#define instantiate_turbo_dequant_rotated(bits, dim, dtype, dtype_suffix) \
  template [[host_name(                                                   \
      "turbo_dequant_rotated_" #bits "_" #dim                             \
      "_" #dtype_suffix)]] [[kernel]] void                                \
  turbo_dequant_rotated<bits, dim, TQ_PW(dim, bits), dtype>(              \
      const device uint32_t*,                                             \
      const device float*,                                                \
      const device float*,                                                \
      device dtype*,                                                      \
      constant int&,                                                      \
      uint3);

// Bits × Dim combinations for real models. Dim=512 added for Gemma 4
// family (E2B, 26B-A4B, 31B), which uses headDim=512.
#define instantiate_all_for_bits(bits)                                                           \
  instantiate_turbo_score(bits, 64) instantiate_turbo_score(                                     \
      bits,                                                                                      \
      80) instantiate_turbo_score(bits, 96) instantiate_turbo_score(bits, 128)                   \
      instantiate_turbo_score(bits, 256) instantiate_turbo_score(                                \
          bits,                                                                                  \
          512) instantiate_turbo_encode(bits, 64) instantiate_turbo_encode(bits, 80)             \
          instantiate_turbo_encode(bits, 96) instantiate_turbo_encode(                           \
              bits, 128) instantiate_turbo_encode(bits, 256)                                     \
              instantiate_turbo_encode(bits, 512) instantiate_turbo_value(                       \
                  bits, 64) instantiate_turbo_value(bits, 80)                                    \
                  instantiate_turbo_value(bits, 96) instantiate_turbo_value(                     \
                      bits, 128) instantiate_turbo_value(bits, 256)                              \
                      instantiate_turbo_value(bits, 512) instantiate_turbo_dequant_rotated(      \
                          bits,                                                                  \
                          64,                                                                    \
                          bfloat,                                                                \
                          bf16) instantiate_turbo_dequant_rotated(bits, 80, bfloat, bf16)        \
                          instantiate_turbo_dequant_rotated(                                     \
                              bits, 96, bfloat, bf16)                                            \
                              instantiate_turbo_dequant_rotated(                                 \
                                  bits, 128, bfloat, bf16)                                       \
                                  instantiate_turbo_dequant_rotated(                             \
                                      bits, 256, bfloat, bf16)                                   \
                                      instantiate_turbo_dequant_rotated(                         \
                                          bits, 512, bfloat, bf16)                               \
                                          instantiate_turbo_dequant_rotated(                     \
                                              bits, 64, half, f16)                               \
                                              instantiate_turbo_dequant_rotated(                 \
                                                  bits, 80, half, f16)                           \
                                                  instantiate_turbo_dequant_rotated(             \
                                                      bits, 96, half, f16)                       \
                                                      instantiate_turbo_dequant_rotated(         \
                                                          bits,                                  \
                                                          128,                                   \
                                                          half,                                  \
                                                          f16)                                   \
                                                          instantiate_turbo_dequant_rotated(     \
                                                              bits,                              \
                                                              256,                               \
                                                              half,                              \
                                                              f16)                               \
                                                              instantiate_turbo_dequant_rotated( \
                                                                  bits,                          \
                                                                  512,                           \
                                                                  half,                          \
                                                                  f16)

// WHT encode only for power-of-2 dims
#define instantiate_wht_for_bits(bits)               \
  instantiate_turbo_encode_wht(bits, 64, 6)          \
      instantiate_turbo_encode_wht(bits, 128, 7)     \
          instantiate_turbo_encode_wht(bits, 256, 8) \
              instantiate_turbo_encode_wht(bits, 512, 9)

instantiate_all_for_bits(2) instantiate_all_for_bits(3)
    instantiate_all_for_bits(4) instantiate_all_for_bits(8)

        instantiate_wht_for_bits(2) instantiate_wht_for_bits(3)
            instantiate_wht_for_bits(4) instantiate_wht_for_bits(8)

                instantiate_turbo_pass2(64) instantiate_turbo_pass2(80)
                    instantiate_turbo_pass2(96) instantiate_turbo_pass2(128)
                        instantiate_turbo_pass2(256)
                            instantiate_turbo_pass2(512)
