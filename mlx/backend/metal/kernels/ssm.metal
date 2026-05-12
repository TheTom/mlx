// Copyright (C) 2026 Eric Kryski.
// SSM (Selective State Space Model) step kernel for Mamba / Nemotron.
//
// Computes one discrete SSM step:
//   dA  = exp(A * dt)
//   dBx = x * dt * B
//   state_out = dA * state_in + dBx
//   out = C^T * state_out + D * x
//
// Grid: (32, Dh, H * batch)  ThreadGroup: (32, 8, 1)
// Each simdgroup cooperatively reduces over the Ds state dimension.

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

template <typename T, int Dh, int Ds, int H, int G>
[[kernel]] void ssm_step(
    const device T* X [[buffer(0)]],
    const device T* A_log [[buffer(1)]],
    const device T* B [[buffer(2)]],
    const device T* C [[buffer(3)]],
    const device T* D [[buffer(4)]],
    const device T* dt [[buffer(5)]],
    const device T* state_in [[buffer(6)]],
    device T* out [[buffer(7)]],
    device T* state_out [[buffer(8)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr int n_per_t = Ds / 32;

  auto n = thread_position_in_grid.z;
  auto h_idx = n % H;
  auto g_idx = n / G;

  auto x = X + n * Dh;
  out += n * Dh;
  auto i_state = state_in + n * Dh * Ds;
  auto o_state = state_out + n * Dh * Ds;

  // C and B have shape [batch, group, state_dim]
  // C and B need to be offset by group size
  auto C_ = C + g_idx * Ds;
  auto B_ = B + g_idx * Ds;

  auto ds_idx = thread_position_in_threadgroup.x;
  auto d_idx = thread_position_in_grid.y;

  auto dt_ = static_cast<float>(dt[n]);
  auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
  auto dA = fast::exp(A * dt_);

  float acc = 0.0;
  auto x_ = static_cast<float>(x[d_idx]);

  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * ds_idx + i;
    auto idx = d_idx * Ds + s_idx;
    auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
    auto state = dA * i_state[idx] + dB_by_x;
    o_state[idx] = static_cast<T>(state);
    acc += state * C_[s_idx];
  }
  acc = simd_sum(acc);
  if (thread_index_in_simdgroup == 0) {
    out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
  }
}

// ============================================================================
// Instantiations for Nemotron and common SSM dimensions
// ============================================================================
#define instantiate_ssm(type, tname, dh, ds, h, g)                        \
  template [[host_name(                                                   \
      "ssm_step_" #tname "_" #dh "_" #ds "_" #h "_" #g)]] [[kernel]] void \
  ssm_step<type, dh, ds, h, g>(                                           \
      const device type*,                                                 \
      const device type*,                                                 \
      const device type*,                                                 \
      const device type*,                                                 \
      const device type*,                                                 \
      const device type*,                                                 \
      const device type*,                                                 \
      device type*,                                                       \
      device type*,                                                       \
      uint3,                                                              \
      uint3,                                                              \
      uint);

// Nemotron: Dh=64, Ds=64, various H and G
instantiate_ssm(half, float16, 64, 64, 16, 1) instantiate_ssm(
    half,
    float16,
    64,
    64,
    16,
    2) instantiate_ssm(half, float16, 64, 64, 16, 4)
    instantiate_ssm(half, float16, 64, 64, 16, 8) instantiate_ssm(
        half,
        float16,
        64,
        64,
        32,
        1) instantiate_ssm(half, float16, 64, 64, 32, 2)
        instantiate_ssm(half, float16, 64, 64, 32, 4) instantiate_ssm(
            half,
            float16,
            64,
            64,
            32,
            8) instantiate_ssm(half, float16, 64, 64, 48, 1)
            instantiate_ssm(half, float16, 64, 64, 48, 2) instantiate_ssm(
                half,
                float16,
                64,
                64,
                48,
                4) instantiate_ssm(half, float16, 64, 64, 48, 8)

                instantiate_ssm(
                    bfloat16_t,
                    bfloat16,
                    64,
                    64,
                    16,
                    1) instantiate_ssm(bfloat16_t, bfloat16, 64, 64, 16, 2)
                    instantiate_ssm(
                        bfloat16_t,
                        bfloat16,
                        64,
                        64,
                        16,
                        4) instantiate_ssm(bfloat16_t, bfloat16, 64, 64, 16, 8)
                        instantiate_ssm(bfloat16_t, bfloat16, 64, 64, 32, 1)
                            instantiate_ssm(bfloat16_t, bfloat16, 64, 64, 32, 2)
                                instantiate_ssm(
                                    bfloat16_t,
                                    bfloat16,
                                    64,
                                    64,
                                    32,
                                    4)
                                    instantiate_ssm(
                                        bfloat16_t,
                                        bfloat16,
                                        64,
                                        64,
                                        32,
                                        8)
                                        instantiate_ssm(
                                            bfloat16_t,
                                            bfloat16,
                                            64,
                                            64,
                                            48,
                                            1)
                                            instantiate_ssm(
                                                bfloat16_t,
                                                bfloat16,
                                                64,
                                                64,
                                                48,
                                                2)
                                                instantiate_ssm(
                                                    bfloat16_t,
                                                    bfloat16,
                                                    64,
                                                    64,
                                                    48,
                                                    4)
                                                    instantiate_ssm(
                                                        bfloat16_t,
                                                        bfloat16,
                                                        64,
                                                        64,
                                                        48,
                                                        8)

    // Nemotron: Dh=64, Ds=128
    instantiate_ssm(half, float16, 64, 128, 16, 1) instantiate_ssm(
        half,
        float16,
        64,
        128,
        16,
        2) instantiate_ssm(half, float16, 64, 128, 16, 4)
        instantiate_ssm(half, float16, 64, 128, 16, 8) instantiate_ssm(
            half,
            float16,
            64,
            128,
            32,
            1) instantiate_ssm(half, float16, 64, 128, 32, 2)
            instantiate_ssm(half, float16, 64, 128, 32, 4) instantiate_ssm(
                half,
                float16,
                64,
                128,
                32,
                8) instantiate_ssm(half, float16, 64, 128, 48, 1)
                instantiate_ssm(half, float16, 64, 128, 48, 2) instantiate_ssm(
                    half,
                    float16,
                    64,
                    128,
                    48,
                    4) instantiate_ssm(half, float16, 64, 128, 48, 8)

                    instantiate_ssm(
                        bfloat16_t,
                        bfloat16,
                        64,
                        128,
                        16,
                        1) instantiate_ssm(bfloat16_t, bfloat16, 64, 128, 16, 2)
                        instantiate_ssm(bfloat16_t, bfloat16, 64, 128, 16, 4)
                            instantiate_ssm(
                                bfloat16_t,
                                bfloat16,
                                64,
                                128,
                                16,
                                8)
                                instantiate_ssm(
                                    bfloat16_t,
                                    bfloat16,
                                    64,
                                    128,
                                    32,
                                    1)
                                    instantiate_ssm(
                                        bfloat16_t,
                                        bfloat16,
                                        64,
                                        128,
                                        32,
                                        2)
                                        instantiate_ssm(
                                            bfloat16_t,
                                            bfloat16,
                                            64,
                                            128,
                                            32,
                                            4)
                                            instantiate_ssm(
                                                bfloat16_t,
                                                bfloat16,
                                                64,
                                                128,
                                                32,
                                                8)
                                                instantiate_ssm(
                                                    bfloat16_t,
                                                    bfloat16,
                                                    64,
                                                    128,
                                                    48,
                                                    1)
                                                    instantiate_ssm(
                                                        bfloat16_t,
                                                        bfloat16,
                                                        64,
                                                        128,
                                                        48,
                                                        2)
                                                        instantiate_ssm(
                                                            bfloat16_t,
                                                            bfloat16,
                                                            64,
                                                            128,
                                                            48,
                                                            4)
                                                            instantiate_ssm(
                                                                bfloat16_t,
                                                                bfloat16,
                                                                64,
                                                                128,
                                                                48,
                                                                8)

    // Dh=128 variants
    instantiate_ssm(half, float16, 128, 64, 16, 1) instantiate_ssm(
        half,
        float16,
        128,
        64,
        16,
        2) instantiate_ssm(half, float16, 128, 64, 16, 4)
        instantiate_ssm(half, float16, 128, 64, 16, 8) instantiate_ssm(
            half,
            float16,
            128,
            64,
            32,
            1) instantiate_ssm(half, float16, 128, 64, 32, 2)
            instantiate_ssm(half, float16, 128, 64, 32, 4) instantiate_ssm(
                half,
                float16,
                128,
                64,
                32,
                8) instantiate_ssm(half, float16, 128, 64, 48, 1)
                instantiate_ssm(half, float16, 128, 64, 48, 2) instantiate_ssm(
                    half,
                    float16,
                    128,
                    64,
                    48,
                    4) instantiate_ssm(half, float16, 128, 64, 48, 8)

                    instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 16, 1) instantiate_ssm(
                        bfloat16_t,
                        bfloat16,
                        128,
                        64,
                        16,
                        2) instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 16, 4)
                        instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 16, 8) instantiate_ssm(
                            bfloat16_t,
                            bfloat16,
                            128,
                            64,
                            32,
                            1) instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 32, 2)
                            instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 32, 4) instantiate_ssm(
                                bfloat16_t,
                                bfloat16,
                                128,
                                64,
                                32,
                                8) instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 48, 1)
                                instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 48, 2) instantiate_ssm(
                                    bfloat16_t,
                                    bfloat16,
                                    128,
                                    64,
                                    48,
                                    4) instantiate_ssm(bfloat16_t, bfloat16, 128, 64, 48, 8)

                                    instantiate_ssm(half, float16, 128, 128, 16, 1) instantiate_ssm(
                                        half,
                                        float16,
                                        128,
                                        128,
                                        16,
                                        2) instantiate_ssm(half, float16, 128, 128, 16, 4)
                                        instantiate_ssm(half, float16, 128, 128, 16, 8) instantiate_ssm(
                                            half,
                                            float16,
                                            128,
                                            128,
                                            32,
                                            1) instantiate_ssm(half, float16, 128, 128, 32, 2)
                                            instantiate_ssm(half, float16, 128, 128, 32, 4) instantiate_ssm(
                                                half,
                                                float16,
                                                128,
                                                128,
                                                32,
                                                8) instantiate_ssm(half, float16, 128, 128, 48, 1)
                                                instantiate_ssm(half, float16, 128, 128, 48, 2) instantiate_ssm(
                                                    half,
                                                    float16,
                                                    128,
                                                    128,
                                                    48,
                                                    4) instantiate_ssm(half, float16, 128, 128, 48, 8)

                                                    instantiate_ssm(bfloat16_t, bfloat16, 128, 128, 16, 1) instantiate_ssm(
                                                        bfloat16_t,
                                                        bfloat16,
                                                        128,
                                                        128,
                                                        16,
                                                        2) instantiate_ssm(bfloat16_t, bfloat16, 128, 128, 16, 4)
                                                        instantiate_ssm(
                                                            bfloat16_t,
                                                            bfloat16,
                                                            128,
                                                            128,
                                                            16,
                                                            8)
                                                            instantiate_ssm(
                                                                bfloat16_t,
                                                                bfloat16,
                                                                128,
                                                                128,
                                                                32,
                                                                1)
                                                                instantiate_ssm(
                                                                    bfloat16_t,
                                                                    bfloat16,
                                                                    128,
                                                                    128,
                                                                    32,
                                                                    2)
                                                                    instantiate_ssm(
                                                                        bfloat16_t,
                                                                        bfloat16,
                                                                        128,
                                                                        128,
                                                                        32,
                                                                        4)
                                                                        instantiate_ssm(
                                                                            bfloat16_t,
                                                                            bfloat16,
                                                                            128,
                                                                            128,
                                                                            32,
                                                                            8)
                                                                            instantiate_ssm(
                                                                                bfloat16_t,
                                                                                bfloat16,
                                                                                128,
                                                                                128,
                                                                                48,
                                                                                1)
                                                                                instantiate_ssm(
                                                                                    bfloat16_t,
                                                                                    bfloat16,
                                                                                    128,
                                                                                    128,
                                                                                    48,
                                                                                    2)
                                                                                    instantiate_ssm(
                                                                                        bfloat16_t,
                                                                                        bfloat16,
                                                                                        128,
                                                                                        128,
                                                                                        48,
                                                                                        4)
                                                                                        instantiate_ssm(
                                                                                            bfloat16_t,
                                                                                            bfloat16,
                                                                                            128,
                                                                                            128,
                                                                                            48,
                                                                                            8)
