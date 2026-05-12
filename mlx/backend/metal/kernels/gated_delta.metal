// Copyright © 2026 Eric Kryski. GatedDeltaNet Metal kernels.
//
// Framework-level compiled kernels for the GatedDelta recurrence:
//   S_t = g_t * S_{t-1} + β_t * k_t * (v_t - k_t^T * S_{t-1})^T
//
// Two variants:
// 1. Standard: pre-computed q, k (normalized), g, beta
// 2. Fused: raw q, k (computes rmsNorm, g from aLog/a/dtBias, beta from
// sigmoid(b) internally)

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Function constant for mask selection (avoids separate kernel variants)
constant bool has_mask [[function_constant(10)]];

// ============================================================================
// Standard GatedDelta kernel
// ============================================================================
template <typename T, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_step(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device T* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    const device T* state_in [[buffer(5)]],
    const device bool* mask [[buffer(6)]],
    device T* y [[buffer(7)]],
    device T* state_out [[buffer(8)]],
    constant int& T_val [[buffer(9)]],
    uint3 thread_pos [[thread_position_in_grid]],
    uint3 tg_pos [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr int n_per_t = Dk / 32;

  auto n = thread_pos.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto q_ = q + b_idx * T_val * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T_val * Hk * Dk + hk_idx * Dk;
  auto v_ = v + b_idx * T_val * Hv * Dv + hv_idx * Dv;
  auto y_ = y + b_idx * T_val * Hv * Dv + hv_idx * Dv;

  auto dk_idx = tg_pos.x;
  auto dv_idx = thread_pos.y;

  auto g_ = g + b_idx * T_val * Hv;
  auto beta_ = beta + b_idx * T_val * Hv;

  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
  }

  for (int t = 0; t < T_val; ++t) {
    bool process = has_mask ? mask[b_idx * T_val + t] : true;
    if (process) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * static_cast<float>(g_[hv_idx]);
        kv_mem += state[i] * static_cast<float>(k_[s_idx]);
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) *
          static_cast<float>(beta_[hv_idx]);

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
        out += state[i] * static_cast<float>(q_[s_idx]);
      }
      out = simd_sum(out);
      if (simd_lane == 0) {
        y_[dv_idx] = static_cast<T>(out);
      }
    }
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y_ += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    o_state[n_per_t * dk_idx + i] = static_cast<T>(state[i]);
  }
}

// ============================================================================
// Fused GatedDelta kernel (absorbs rmsNorm + g + beta computation)
// ============================================================================
template <typename T, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_step_fused(
    const device T* q_raw [[buffer(0)]],
    const device T* k_raw [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device T* a [[buffer(3)]],
    const device T* b_input [[buffer(4)]],
    const device T* a_log [[buffer(5)]],
    const device T* dt_bias_arr [[buffer(6)]],
    const device T* state_in [[buffer(7)]],
    const device bool* mask [[buffer(8)]],
    device T* y [[buffer(9)]],
    device T* state_out [[buffer(10)]],
    constant int& T_val [[buffer(11)]],
    uint3 thread_pos [[thread_position_in_grid]],
    uint3 tg_pos [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
  constexpr int n_per_t = Dk / 32;
  constexpr float inv_scale_sq = 1.0f / float(Dk);
  float inv_scale_single = rsqrt(float(Dk));

  auto n = thread_pos.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto q_ = q_raw + b_idx * T_val * Hk * Dk + hk_idx * Dk;
  auto k_ = k_raw + b_idx * T_val * Hk * Dk + hk_idx * Dk;
  auto v_ = v + b_idx * T_val * Hv * Dv + hv_idx * Dv;
  auto y_ = y + b_idx * T_val * Hv * Dv + hv_idx * Dv;

  auto dk_idx = tg_pos.x;
  auto dv_idx = thread_pos.y;

  auto a_ = a + b_idx * T_val * Hv;
  auto b_ = b_input + b_idx * T_val * Hv;

  float exp_aLog = exp(static_cast<float>(a_log[hv_idx]));
  float dt_bias = static_cast<float>(dt_bias_arr[hv_idx]);

  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
  }

  for (int t = 0; t < T_val; ++t) {
    bool process = has_mask ? mask[b_idx * T_val + t] : true;
    if (process) {
      // Fused rmsNorm(q)
      float q_sum_sq = 0.0f;
      float q_vals[n_per_t];
      for (int i = 0; i < n_per_t; ++i) {
        q_vals[i] = static_cast<float>(q_[n_per_t * dk_idx + i]);
        q_sum_sq += q_vals[i] * q_vals[i];
      }
      q_sum_sq = simd_sum(q_sum_sq);
      float q_rms = rsqrt(q_sum_sq / float(Dk) + 1e-6f);
      for (int i = 0; i < n_per_t; ++i) {
        q_vals[i] = q_vals[i] * q_rms * inv_scale_sq;
      }

      // Fused rmsNorm(k)
      float k_sum_sq = 0.0f;
      float k_vals[n_per_t];
      for (int i = 0; i < n_per_t; ++i) {
        k_vals[i] = static_cast<float>(k_[n_per_t * dk_idx + i]);
        k_sum_sq += k_vals[i] * k_vals[i];
      }
      k_sum_sq = simd_sum(k_sum_sq);
      float k_rms = rsqrt(k_sum_sq / float(Dk) + 1e-6f);
      for (int i = 0; i < n_per_t; ++i) {
        k_vals[i] = k_vals[i] * k_rms * inv_scale_single;
      }

      // Fused g = exp(-exp(aLog) * softplus(a + dtBias))
      float a_val = static_cast<float>(a_[hv_idx]);
      float dt = a_val + dt_bias;
      float sp = dt > 20.0f ? dt : log(1.0f + exp(dt));
      float g_val = exp(-exp_aLog * sp);

      // Fused beta = sigmoid(b)
      float b_val = static_cast<float>(b_[hv_idx]);
      float beta_val = 1.0f / (1.0f + exp(-b_val));

      // State update
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] * g_val;
        kv_mem += state[i] * k_vals[i];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_val;

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] + k_vals[i] * delta;
        out += state[i] * q_vals[i];
      }
      out = simd_sum(out);
      if (simd_lane == 0) {
        y_[dv_idx] = static_cast<T>(out);
      }
    }
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    y_ += Hv * Dv;
    a_ += Hv;
    b_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    o_state[n_per_t * dk_idx + i] = static_cast<T>(state[i]);
  }
}

// ============================================================================
// Instantiation
// ============================================================================
// Qwen3.5 dimensions: Dk=192, Dv=128, Hk=4, Hv=4 (per-layer varies by model)
// Template params Hk and Hv must be known at compile time since they determine
// pointer strides. Common configs for real models:

#define instantiate_gdn(type, tname, dk, dv, hk, hv)           \
  template [[host_name(                                        \
      "gated_delta_step_" #tname "_" #dk "_" #dv "_" #hk       \
      "_" #hv)]] [[kernel]] void                               \
  gated_delta_step<type, dk, dv, hk, hv>(                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device bool*,                                      \
      device type*,                                            \
      device type*,                                            \
      constant int&,                                           \
      uint3,                                                   \
      uint3,                                                   \
      uint);                                                   \
  template [[host_name(                                        \
      "gated_delta_step_fused_" #tname "_" #dk "_" #dv "_" #hk \
      "_" #hv)]] [[kernel]] void                               \
  gated_delta_step_fused<type, dk, dv, hk, hv>(                \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device bool*,                                      \
      device type*,                                            \
      device type*,                                            \
      constant int&,                                           \
      uint3,                                                   \
      uint3,                                                   \
      uint);

// Qwen3.5-A3B: Dk=192, Dv=128, Hk=4, Hv=4
instantiate_gdn(half, float16, 192, 128, 4, 4)
    instantiate_gdn(bfloat16_t, bfloat16, 192, 128, 4, 4)

    // Qwen3.5 larger variants may have different dims
    instantiate_gdn(half, float16, 128, 128, 8, 8)
        instantiate_gdn(bfloat16_t, bfloat16, 128, 128, 8, 8)
            instantiate_gdn(half, float16, 64, 64, 8, 8)
                instantiate_gdn(bfloat16_t, bfloat16, 64, 64, 8, 8)

    // Qwen3.5-35B: Dk=128, Dv=128, numHeads=16, numKVHeads=2
    instantiate_gdn(half, float16, 128, 128, 16, 32)
        instantiate_gdn(bfloat16_t, bfloat16, 128, 128, 16, 32)

    // Qwen3.5 dense models: Dk=128, Dv=128, Hk=16, Hv=16 (0.8B-9B), Hv=48 (27B)
    instantiate_gdn(half, float16, 128, 128, 16, 16)
        instantiate_gdn(bfloat16_t, bfloat16, 128, 128, 16, 16)
            instantiate_gdn(half, float16, 128, 128, 16, 48)
                instantiate_gdn(bfloat16_t, bfloat16, 128, 128, 16, 48)
