// Copyright © 2026 Eric Kryski.

// Fused RMSNorm + Residual Add kernel
// Computes: out = residual + rmsNorm(x, weight, eps)
// Combines RMS normalization with residual addition in a single dispatch.
// Saves one Metal kernel launch per call vs separate rms_norm + add.
//
// Used at every post-attention and post-FFN norm+residual site in Gemma4
// (3 calls/layer × 30 layers = 90 dispatches saved per token).

#include <metal_common>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

/// Fused RMSNorm + Residual Add (looped for large axis_size).
///
/// Input: x (to normalize), residual (skip connection), weight (norm scale).
/// Output: out = residual + weight * x * inv_rms
///
/// Each threadgroup processes one row. Threads loop over elements when
/// axis_size exceeds threadgroup size (e.g., hidden_size=2816 >
/// max_threads=1024).
///
/// Phase 1: Each thread accumulates sum(x^2) over its elements via strided
/// loop.
///          SIMD + threadgroup reduction to get total sum.
/// Phase 2: Each thread applies weight * x * inv_rms + residual over same
/// elements.
template <typename T>
[[kernel]] void rms_norm_residual(
    const device T* x [[buffer(0)]],
    const device T* residual [[buffer(1)]],
    const device T* w [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& axis_size [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;

  const device T* x_row = x + gid * size_t(axis_size);
  const device T* r_row = residual + gid * size_t(axis_size);
  device T* o_row = out + gid * size_t(axis_size);

  // Phase 1: Strided loop to accumulate sum(x^2)
  float sum_sq = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    float val = float(x_row[i]);
    sum_sq += val * val;
  }

  // SIMD reduction within each simdgroup
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

  float inv_rms = local_inv_rms[0];

  // Phase 2: Strided loop — apply weight * x * inv_rms + residual
  for (uint i = lid; i < axis_size; i += tg_size) {
    float val = float(x_row[i]);
    float normed = val * float(w[i]) * inv_rms;
    float res = float(r_row[i]);
    o_row[i] = static_cast<T>(res + normed);
  }
}

#define instantiate_rms_norm_residual(name, type)                    \
  template [[host_name("rms_norm_residual_" #name)]] [[kernel]] void \
  rms_norm_residual<type>(                                           \
      const device type*,                                            \
      const device type*,                                            \
      const device type*,                                            \
      device type*,                                                  \
      constant float&,                                               \
      constant uint&,                                                \
      uint,                                                          \
      uint,                                                          \
      uint,                                                          \
      uint,                                                          \
      uint);

instantiate_rms_norm_residual(float32, float)
    instantiate_rms_norm_residual(float16, half)
        instantiate_rms_norm_residual(bfloat16, bfloat16_t)
