// Copyright © 2024-2026 Apple Inc.

// Fused RMSNorm + RoPE kernel
// Combines RMS normalization with rotary position embedding in a single
// dispatch. Saves one Metal kernel launch per Q/K vs separate rms_norm + rope
// calls.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

/// Fused RMSNorm + RoPE for non-traditional (paired) layout.
///
/// Input/output: one row per (batch, seq_pos, head), each of length
/// `axis_size`. Each threadgroup processes one row. Thread count = axis_size /
/// 2 (one per rotation pair).
///
/// Phase 1: Compute inv_rms = rsqrt(mean(x^2) + eps) via SIMD + threadgroup
/// reduction.
///          Each thread loads x[tid] and x[tid + half], accumulating two
///          squared values.
/// Phase 2: Apply weight scaling and RoPE rotation:
///          normed_a = w[tid] * x[tid] * inv_rms
///          normed_b = w[tid+half] * x[tid+half] * inv_rms
///          out[tid]      = normed_a * cos(theta) - normed_b * sin(theta)
///          out[tid+half] = normed_a * sin(theta) + normed_b * cos(theta)
///          where theta = position * inv_freqs[tid]
template <typename T>
[[kernel]] void rms_norm_rope(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    const device float* inv_freqs [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& axis_size [[buffer(5)]],
    constant int& offset [[buffer(6)]],
    constant int& n_heads [[buffer(7)]],
    constant int& seq_len [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  uint half_dim = axis_size / 2;

  const device T* x_row = x + gid * size_t(axis_size);

  // Phase 1: Load paired elements and compute sum(x^2)
  float v1 = (lid < half_dim) ? float(x_row[lid]) : 0.0f;
  float v2 = (lid < half_dim) ? float(x_row[lid + half_dim]) : 0.0f;
  float sum_sq = v1 * v1 + v2 * v2;

  // SIMD reduction
  sum_sq = simd_sum(sum_sq);

  // Cross-SIMD-group reduction via threadgroup memory
  threadgroup float local_sums[SIMD_SIZE];
  threadgroup float local_inv_rms[1];

  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = sum_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id == 0) {
    float total = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_rms[0] = metal::precise::rsqrt(total / float(axis_size) + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid >= half_dim)
    return;

  float inv_rms = local_inv_rms[0];

  // Phase 2: Apply weight * inv_rms * RoPE rotation
  // Row layout is [B, L, nHeads, headDim], position = offset + (gid / nHeads) %
  // seqLen
  uint l = (gid / uint(n_heads)) % uint(seq_len);
  float pos = float(offset + int(l));

  float normed_a = v1 * float(w[lid]) * inv_rms;
  float normed_b = v2 * float(w[lid + half_dim]) * inv_rms;

  float theta = pos * inv_freqs[lid];
  float cos_t = metal::fast::cos(theta);
  float sin_t = metal::fast::sin(theta);

  device T* o_row = out + gid * size_t(axis_size);
  o_row[lid] = static_cast<T>(normed_a * cos_t - normed_b * sin_t);
  o_row[lid + half_dim] = static_cast<T>(normed_a * sin_t + normed_b * cos_t);
}

#define instantiate_rms_norm_rope(name, type)                    \
  template [[host_name("rms_norm_rope_" #name)]] [[kernel]] void \
  rms_norm_rope<type>(                                           \
      const device type*,                                        \
      const device type*,                                        \
      const device float*,                                       \
      device type*,                                              \
      constant float&,                                           \
      constant uint&,                                            \
      constant int&,                                             \
      constant int&,                                             \
      constant int&,                                             \
      uint,                                                      \
      uint,                                                      \
      uint,                                                      \
      uint);

instantiate_rms_norm_rope(float32, float)
    instantiate_rms_norm_rope(float16, half)
        instantiate_rms_norm_rope(bfloat16, bfloat16_t)
