// Copyright © 2026 Eric Kryski.
// Fused RMSNorm + Quantized GEMV for decode (T=1) inference.
//
// Eliminates global memory round-trip between RMSNorm and quantized matmul:
// 1. Load input x into shared memory ONCE
// 2. Compute RMSNorm factor via SIMD reduction (stays in register)
// 3. GEMV inner loop reads normed x from shared memory, not global
//
// Self-contained: includes only the 4-bit qdot logic inline (no quantized.h
// dependency to avoid Steel/MMA includes).

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Inlined 4-bit quantized dot product (from quantized.h qdot specialization)
// Computes: scale * Σ(w_i * x_i) + bias * Σ(x_i)
// where w_i are 4-bit packed in uint16_t pairs
template <int values_per_thread>
inline float qdot_4bit(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * float(ws[i] & 0x000f) +
         x_thread[4 * i + 1] * float(ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * float(ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * float(ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

// Safe version with bounds checking
template <int values_per_thread>
inline float qdot_4bit_safe(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum,
    int remaining) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    int base = 4 * i;
    if (base < remaining)
      accum += x_thread[base] * float(ws[i] & 0x000f);
    if (base + 1 < remaining)
      accum += x_thread[base + 1] * float(ws[i] & 0x00f0);
    if (base + 2 < remaining)
      accum += x_thread[base + 2] * float(ws[i] & 0x0f00);
    if (base + 3 < remaining)
      accum += x_thread[base + 3] * float(ws[i] & 0xf000);
  }
  return scale * accum + sum * bias;
}

// ============================================================================
// Fused RMSNorm + 4-bit Quantized GEMV
// ============================================================================
template <typename T, int group_size>
[[kernel]] void rms_norm_qgemv(
    const device T* x [[buffer(0)]],
    const device T* norm_weight [[buffer(1)]],
    const device uint32_t* w [[buffer(2)]],
    const device T* scales [[buffer(3)]],
    const device T* biases [[buffer(4)]],
    device T* y [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant int& in_vec_size [[buffer(7)]],
    constant int& out_vec_size [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int bits = 4;
  constexpr int SIMD_SIZE = 32;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = 32 / bits; // values per uint32 = 8 at bits=4
  constexpr int values_per_thread = pack_factor; // 8
  constexpr int block_size = values_per_thread * SIMD_SIZE; // 256
  constexpr int bytes_per_pack = 4; // uint32 = 4 bytes
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  // ======================================================================
  // Phase 1: Load x into shared memory + compute RMSNorm
  // ======================================================================
  threadgroup T shared_x[4096];
  threadgroup float shared_inv_rms[1];

  uint total_threads = num_simdgroups * SIMD_SIZE;
  uint thread_id = simd_gid * SIMD_SIZE + simd_lid;

  for (uint i = thread_id; i < uint(in_vec_size); i += total_threads) {
    shared_x[i] = x[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compute sum(x²) for RMSNorm (weight applied AFTER norm, not inside)
  float sum_sq = 0;
  for (uint i = thread_id; i < uint(in_vec_size); i += total_threads) {
    float v = float(shared_x[i]);
    sum_sq += v * v;
  }

  sum_sq = simd_sum(sum_sq);

  threadgroup float simd_sums[2];
  if (simd_lid == 0)
    simd_sums[simd_gid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_gid == 0 && simd_lid == 0) {
    float total = simd_sums[0] + simd_sums[1];
    shared_inv_rms[0] = metal::precise::rsqrt(total / float(in_vec_size) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float inv_rms = shared_inv_rms[0];

  // ======================================================================
  // Phase 2: Quantized GEMV with normed x from shared memory
  // ======================================================================
  const device uint8_t* ws = (const device uint8_t*)w;
  thread float x_thread[values_per_thread];
  thread float result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size)
    return;

  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  ws += used_out_row * in_vec_size_w + simd_lid * bytes_per_pack;
  scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  y += used_out_row;

  uint x_offset = simd_lid * values_per_thread;
  int k = 0;

  // 4-bit pre-scaling: matches load_vector<T,U,8,4> from quantized.h.
  // qdot_4bit multiplies x_thread by un-shifted uint16 mask bits.
  // Masks 0x000f, 0x00f0, 0x0f00, 0xf000 produce values ×1, ×16, ×256, ×4096.
  // x_thread[i] is pre-divided to compensate, repeating per uint16 pack.
  constexpr float qdot_prescale[8] = {
      1.0f,
      1.0f / 16.0f,
      1.0f / 256.0f,
      1.0f / 4096.0f,
      1.0f,
      1.0f / 16.0f,
      1.0f / 256.0f,
      1.0f / 4096.0f};

  for (; k < in_vec_size - block_size; k += block_size) {
    // Load normed x from shared memory with qdot pre-scaling
    float sum = 0;
    for (int i = 0; i < values_per_thread; i++) {
      float raw = float(shared_x[x_offset + i]);
      float nw = float(norm_weight[x_offset + i]);
      float normed = raw * nw * inv_rms;
      sum += normed;
      x_thread[i] = normed * qdot_prescale[i];
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      float s = float(scales[row * in_vec_size_g]);
      float b = float(biases[row * in_vec_size_g]);
      result[row] += qdot_4bit<values_per_thread>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x_offset += block_size;
  }

  // Handle remaining elements
  const int remaining = clamp(
      int(in_vec_size) - k - int(simd_lid * values_per_thread),
      0,
      values_per_thread);
  if (remaining > 0) {
    float sum = 0;
    for (int i = 0; i < values_per_thread; i++) {
      if (i < remaining) {
        float raw = float(shared_x[x_offset + i]);
        float nw = float(norm_weight[x_offset + i]);
        float normed = raw * nw * inv_rms;
        sum += normed;
        x_thread[i] = normed * qdot_prescale[i];
      } else {
        x_thread[i] = 0;
      }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      float s = float(scales[row * in_vec_size_g]);
      float b = float(biases[row * in_vec_size_g]);
      result[row] +=
          qdot_4bit_safe<values_per_thread>(wl, x_thread, s, b, sum, remaining);
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && used_out_row + row < out_vec_size) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

// ============================================================================
// Instantiation
// ============================================================================
#define instantiate_rms_norm_qgemv(type, tname, gs)                          \
  template [[host_name("rms_norm_qgemv_" #tname "_gs" #gs)]] [[kernel]] void \
  rms_norm_qgemv<type, gs>(                                                  \
      const device type*,                                                    \
      const device type*,                                                    \
      const device uint32_t*,                                                \
      const device type*,                                                    \
      const device type*,                                                    \
      device type*,                                                          \
      constant float&,                                                       \
      constant int&,                                                         \
      constant int&,                                                         \
      uint3,                                                                 \
      uint,                                                                  \
      uint);

instantiate_rms_norm_qgemv(half, float16, 64)
    instantiate_rms_norm_qgemv(bfloat16_t, bfloat16, 64)
        instantiate_rms_norm_qgemv(half, float16, 128)
            instantiate_rms_norm_qgemv(bfloat16_t, bfloat16, 128)
