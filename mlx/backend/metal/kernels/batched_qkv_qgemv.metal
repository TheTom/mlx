// Batched QKV Quantized GEMV for decode (T=1) inference.
//
// Fuses 3 independent Q/K/V 4-bit quantized GEMVs into a single Metal dispatch.
// Uses tid.z to select which matrix (Q=0, K=1, V=2). Each z-slice loads x
// independently to shared memory (threadgroups can't share across z-slices),
// then computes its GEMV with the original per-matrix tile size.
//
// Output: single contiguous buffer [N_q + N_k + N_v] with Q, K, V concatenated.
//
// Grid: (1, ceil(max(N_q, N_k, N_v) / 8), 3)
//   tid.y: output row tile (8 rows per threadgroup)
//   tid.z: matrix selector (0=Q, 1=K, 2=V)

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

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
// Main kernel: tid.z selects matrix, each z-slice is a standard GEMV
// ============================================================================
template <typename T, int group_size>
[[kernel]] void batched_qkv_qgemv(
    const device T* x [[buffer(0)]],
    // Q projection weights
    const device uint32_t* w_q [[buffer(1)]],
    const device T* scales_q [[buffer(2)]],
    const device T* biases_q [[buffer(3)]],
    // K projection weights
    const device uint32_t* w_k [[buffer(4)]],
    const device T* scales_k [[buffer(5)]],
    const device T* biases_k [[buffer(6)]],
    // V projection weights
    const device uint32_t* w_v [[buffer(7)]],
    const device T* scales_v [[buffer(8)]],
    const device T* biases_v [[buffer(9)]],
    // Output: single contiguous [N_q + N_k + N_v]
    device T* y [[buffer(10)]],
    // Dimensions
    constant int& out_vec_size_q [[buffer(11)]],
    constant int& out_vec_size_k [[buffer(12)]],
    constant int& out_vec_size_v [[buffer(13)]],
    constant int& in_vec_size [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int SIMD_SIZE = 32;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = 8;
  constexpr int values_per_thread = pack_factor;
  constexpr int block_size = values_per_thread * SIMD_SIZE; // 256
  constexpr int bytes_per_pack = 4;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  // Select matrix based on tid.z
  const device uint8_t* ws;
  const device T* sc;
  const device T* bi;
  device T* out;
  int out_vec_size;

  switch (tid.z) {
    case 0:
      ws = (const device uint8_t*)w_q;
      sc = scales_q;
      bi = biases_q;
      out = y;
      out_vec_size = out_vec_size_q;
      break;
    case 1:
      ws = (const device uint8_t*)w_k;
      sc = scales_k;
      bi = biases_k;
      out = y + out_vec_size_q;
      out_vec_size = out_vec_size_k;
      break;
    default:
      ws = (const device uint8_t*)w_v;
      sc = scales_v;
      bi = biases_v;
      out = y + out_vec_size_q + out_vec_size_k;
      out_vec_size = out_vec_size_v;
      break;
  }

  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  if (out_row >= out_vec_size)
    return;

  // Load x into shared memory (each z-slice loads independently)
  threadgroup T shared_x[4096];
  uint total_threads = num_simdgroups * SIMD_SIZE;
  uint thread_id = simd_gid * SIMD_SIZE + simd_lid;
  for (uint i = thread_id; i < uint(in_vec_size); i += total_threads) {
    shared_x[i] = x[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Standard GEMV from shared memory
  thread float x_thread[values_per_thread];
  thread float result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  ws += used_out_row * in_vec_size_w + simd_lid * bytes_per_pack;
  sc += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  bi += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  out += used_out_row;

  uint x_offset = simd_lid * values_per_thread;
  int k = 0;

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
    float sum = 0;
    for (int i = 0; i < values_per_thread; i++) {
      float val = float(shared_x[x_offset + i]);
      sum += val;
      x_thread[i] = val * qdot_prescale[i];
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      float s = float(sc[row * in_vec_size_g]);
      float b = float(bi[row * in_vec_size_g]);
      result[row] += qdot_4bit<values_per_thread>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bi += block_size / group_size;
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
        float val = float(shared_x[x_offset + i]);
        sum += val;
        x_thread[i] = val * qdot_prescale[i];
      } else {
        x_thread[i] = 0;
      }
    }

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      float s = float(sc[row * in_vec_size_g]);
      float b = float(bi[row * in_vec_size_g]);
      result[row] +=
          qdot_4bit_safe<values_per_thread>(wl, x_thread, s, b, sum, remaining);
    }
  }

  // SIMD reduction + write output
  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0 && used_out_row + row < out_vec_size) {
      out[row] = static_cast<T>(result[row]);
    }
  }
}

// ============================================================================
// Instantiation
// ============================================================================
#define instantiate_batched_qkv_qgemv(type, tname, gs)                     \
  template                                                                 \
      [[host_name("batched_qkv_qgemv_" #tname "_gs" #gs)]] [[kernel]] void \
      batched_qkv_qgemv<type, gs>(                                         \
          const device type*,                                              \
          const device uint32_t*,                                          \
          const device type*,                                              \
          const device type*,                                              \
          const device uint32_t*,                                          \
          const device type*,                                              \
          const device type*,                                              \
          const device uint32_t*,                                          \
          const device type*,                                              \
          const device type*,                                              \
          device type*,                                                    \
          constant int&,                                                   \
          constant int&,                                                   \
          constant int&,                                                   \
          constant int&,                                                   \
          uint3,                                                           \
          uint,                                                            \
          uint);

instantiate_batched_qkv_qgemv(half, float16, 64)
    instantiate_batched_qkv_qgemv(bfloat16_t, bfloat16, 64)
        instantiate_batched_qkv_qgemv(half, float16, 128)
            instantiate_batched_qkv_qgemv(bfloat16_t, bfloat16, 128)
