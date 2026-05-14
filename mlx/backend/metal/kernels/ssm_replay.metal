// Copyright © 2026 Eric Kryski. Spec 040 — Mamba / Mamba 2 state replay.
//
// Two kernels:
//   1. `ssm_step_record` — sequential forward over T steps capturing per-step
//      `(dA, dBx)` into delta logs alongside the standard `(y, state_out)`.
//   2. `ssm_replay` — re-fold the first `k` log entries onto a recurrent
//      state snapshot to recover state-after-k.
//
// Both kernels share the threadgroup shape from `ssm.metal`:
//   Grid: (32, Dh, H * batch)  ThreadGroup: (32, 8, 1)
//   - thread_position_in_grid.x → ds_lane (0..31)
//   - thread_position_in_grid.y → dh
//   - thread_position_in_grid.z → batch * H + h
//   Each simdgroup cooperatively reduces over the Ds state dimension.

#include <metal_common>
#include <metal_math>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

constant bool ssm_record_has_mask [[function_constant(50)]];
constant bool ssm_replay_has_mask [[function_constant(51)]];

// ============================================================================
// SSM step record — sequential T-loop with delta capture.
// ============================================================================
template <typename T, int Dh, int Ds, int H, int G>
[[kernel]] void ssm_step_record(
    const device T* x [[buffer(0)]], // [B, T, H, Dh]
    const device T* A_log [[buffer(1)]], // [H]
    const device T* B [[buffer(2)]], // [B, T, G, Ds]
    const device T* C [[buffer(3)]], // [B, T, G, Ds]
    const device T* D [[buffer(4)]], // [H]
    const device T* dt [[buffer(5)]], // [B, T, H]
    const device T* state_in [[buffer(6)]], // [B, H, Dh, Ds]
    device T* y [[buffer(7)]], // [B, T, H, Dh]
    device T* state_out [[buffer(8)]], // [B, H, Dh, Ds]
    device T* dA_log [[buffer(9)]], // [B, T, H, Ds]
    device T* dBx_log [[buffer(10)]], // [B, T, H, Dh, Ds]
    const device bool* mask
    [[buffer(11),
      function_constant(ssm_record_has_mask)]], // [B, T]
    const constant int& T_total [[buffer(12)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr int n_per_t = Ds / 32;

  auto n = thread_position_in_grid.z; // batch * H + h
  auto h_idx = n % H;
  auto b_idx = n / H;
  // Group routing mirrors ssm.metal: g_idx = n / G. Kept for consistency
  // with the existing dispatch but functionally we re-derive via b_idx + h_idx
  // below for the per-step pointer math (B/C are [B, T, G, Ds]).
  uint repeats = H / G;
  uint g_idx_in_batch = h_idx / repeats;

  auto d_idx = thread_position_in_grid.y; // dh
  auto ds_lane = thread_position_in_threadgroup.x; // 0..31

  // Pre-load state into registers (per-(dh, ds_lane) slice of n_per_t entries).
  float local_state[n_per_t];
  auto i_state_base = state_in + n * Dh * Ds + d_idx * Ds;
  auto o_state_base = state_out + n * Dh * Ds + d_idx * Ds;
  for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * ds_lane + i;
    local_state[i] = static_cast<float>(i_state_base[s_idx]);
  }

  float A = -fast::exp(static_cast<float>(A_log[h_idx]));

  // Sequential T loop.
  for (int t = 0; t < T_total; ++t) {
    // Per-(b, t, h) indices
    uint bt_h = (b_idx * uint(T_total) + uint(t)) * H + h_idx;
    uint bt_g = (b_idx * uint(T_total) + uint(t)) * G + g_idx_in_batch;
    uint bt = b_idx * uint(T_total) + uint(t);

    // Mask check: skip recurrence (dA=1, dBx=0) on masked timesteps so
    // rollback past a masked t is identity-preserving.
    bool active = true;
    if (ssm_record_has_mask) {
      active = mask[bt];
    }

    auto dt_v = static_cast<float>(dt[bt_h]);
    float dA;
    if (active) {
      dA = fast::exp(A * dt_v);
    } else {
      dA = 1.0f;
      dt_v = 0.0f; // zeros the innovation contribution
    }

    // Capture dA (per-Ds-lane redundant scalar; written once per ds slot).
    // Layout dA_log[B, T, H, Ds]; each ds slot gets the same scalar dA so
    // the replay kernel can read element-wise without a broadcast.
    device T* dA_log_t = dA_log + bt_h * Ds;
    for (int i = 0; i < n_per_t; ++i) {
      int s_idx = n_per_t * ds_lane + i;
      dA_log_t[s_idx] = static_cast<T>(dA);
    }

    auto x_v = static_cast<float>(x[bt_h * Dh + d_idx]);
    auto B_ = B + bt_g * Ds;
    auto C_ = C + bt_g * Ds;

    device T* dBx_log_t = dBx_log + (((bt_h * Dh) + d_idx) * Ds);

    float y_acc = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
      int s_idx = n_per_t * ds_lane + i;
      float B_v = static_cast<float>(B_[s_idx]);
      float dBx = active ? (x_v * dt_v * B_v) : 0.0f;
      dBx_log_t[s_idx] = static_cast<T>(dBx);

      // Update register state
      local_state[i] = dA * local_state[i] + dBx;

      // y_t = C · state + D · x
      y_acc += local_state[i] * static_cast<float>(C_[s_idx]);
    }
    float y_sum = simd_sum(y_acc);
    if (thread_index_in_simdgroup == 0) {
      y[bt_h * Dh + d_idx] =
          static_cast<T>(y_sum + x_v * static_cast<float>(D[h_idx]));
    }
  }

  // Final state write-back.
  for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * ds_lane + i;
    o_state_base[s_idx] = static_cast<T>(local_state[i]);
  }
}

// ============================================================================
// SSM replay — re-fold first `k` log entries onto state snapshot.
// ============================================================================
template <typename T, int Dh, int Ds, int H>
[[kernel]] void ssm_replay(
    const device T* state_snapshot [[buffer(0)]], // [B, H, Dh, Ds]
    const device T* dA_log [[buffer(1)]], // [B, T, H, Ds]
    const device T* dBx_log [[buffer(2)]], // [B, T, H, Dh, Ds]
    device T* state_after_k [[buffer(3)]], // [B, H, Dh, Ds]
    const device bool* mask
    [[buffer(4),
      function_constant(ssm_replay_has_mask)]], // [B, T]
    const constant int& k [[buffer(5)]],
    const constant int& T_total [[buffer(6)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
  constexpr int n_per_t = Ds / 32;

  auto n = thread_position_in_grid.z;
  auto h_idx = n % H;
  auto b_idx = n / H;

  auto d_idx = thread_position_in_grid.y;
  auto ds_lane = thread_position_in_threadgroup.x;

  // Load snapshot into registers.
  float local_state[n_per_t];
  auto src_base = state_snapshot + n * Dh * Ds + d_idx * Ds;
  for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * ds_lane + i;
    local_state[i] = static_cast<float>(src_base[s_idx]);
  }

  // Replay first k entries.
  for (int t = 0; t < k; ++t) {
    uint bt = b_idx * uint(T_total) + uint(t);
    uint bt_h = bt * H + h_idx;

    bool active = true;
    if (ssm_replay_has_mask) {
      active = mask[bt];
    }
    if (!active) {
      // dA=1, dBx=0 — state unchanged.
      continue;
    }

    auto dA_t = dA_log + bt_h * Ds;
    auto dBx_t = dBx_log + (((bt_h * Dh) + d_idx) * Ds);

    for (int i = 0; i < n_per_t; ++i) {
      int s_idx = n_per_t * ds_lane + i;
      local_state[i] = static_cast<float>(dA_t[s_idx]) * local_state[i] +
          static_cast<float>(dBx_t[s_idx]);
    }
  }

  // Write result.
  auto dst_base = state_after_k + n * Dh * Ds + d_idx * Ds;
  for (int i = 0; i < n_per_t; ++i) {
    int s_idx = n_per_t * ds_lane + i;
    dst_base[s_idx] = static_cast<T>(local_state[i]);
  }
}

// ============================================================================
// Instantiations for Nemotron / Jamba / Granite / FalconH1 shapes.
// Matches ssm.metal's coverage. (Dh, Ds, H, G) combinations.
// ============================================================================
#define instantiate_ssm_step_record(type, tname, dh, ds, h, g) \
  template [[host_name(                                        \
      "ssm_step_record_" #tname "_" #dh "_" #ds "_" #h         \
      "_" #g)]] [[kernel]] void                                \
  ssm_step_record<type, dh, ds, h, g>(                         \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      const device type*,                                      \
      device type*,                                            \
      device type*,                                            \
      device type*,                                            \
      device type*,                                            \
      const device bool*,                                      \
      const constant int&,                                     \
      uint3,                                                   \
      uint3,                                                   \
      uint);

#define instantiate_ssm_replay(type, tname, dh, ds, h)               \
  template [[host_name(                                              \
      "ssm_replay_" #tname "_" #dh "_" #ds "_" #h)]] [[kernel]] void \
  ssm_replay<type, dh, ds, h>(                                       \
      const device type*,                                            \
      const device type*,                                            \
      const device type*,                                            \
      device type*,                                                  \
      const device bool*,                                            \
      const constant int&,                                           \
      const constant int&,                                           \
      uint3,                                                         \
      uint3);

// Nemotron + family: Dh=64, Ds=64; common H=16/32/48, G=1/2/4/8.
#define instantiate_ssm_record_dh64_ds64_for(type, tname)                      \
  instantiate_ssm_step_record(                                                 \
      type,                                                                    \
      tname,                                                                   \
      64,                                                                      \
      64,                                                                      \
      16,                                                                      \
      1) instantiate_ssm_step_record(type, tname, 64, 64, 16, 2)               \
      instantiate_ssm_step_record(type, tname, 64, 64, 16, 4)                  \
          instantiate_ssm_step_record(type, tname, 64, 64, 16, 8)              \
              instantiate_ssm_step_record(type, tname, 64, 64, 32, 1)          \
                  instantiate_ssm_step_record(type, tname, 64, 64, 32, 2)      \
                      instantiate_ssm_step_record(type, tname, 64, 64, 32, 4)  \
                          instantiate_ssm_step_record(                         \
                              type, tname, 64, 64, 32, 8)                      \
                              instantiate_ssm_step_record(                     \
                                  type, tname, 64, 64, 48, 1)                  \
                                  instantiate_ssm_step_record(                 \
                                      type, tname, 64, 64, 48, 2)              \
                                      instantiate_ssm_step_record(             \
                                          type, tname, 64, 64, 48, 4)          \
                                          instantiate_ssm_step_record(         \
                                              type, tname, 64, 64, 48, 8)      \
                                              instantiate_ssm_replay(          \
                                                  type, tname, 64, 64, 16)     \
                                                  instantiate_ssm_replay(      \
                                                      type, tname, 64, 64, 32) \
                                                      instantiate_ssm_replay(  \
                                                          type,                \
                                                          tname,               \
                                                          64,                  \
                                                          64,                  \
                                                          48)

// Mamba 2 wider state: Dh=128, Ds=128.
#define instantiate_ssm_record_dh128_ds128_for(type, tname)                   \
  instantiate_ssm_step_record(                                                \
      type,                                                                   \
      tname,                                                                  \
      128,                                                                    \
      128,                                                                    \
      16,                                                                     \
      1) instantiate_ssm_step_record(type, tname, 128, 128, 16, 2)            \
      instantiate_ssm_step_record(type, tname, 128, 128, 16, 4)               \
          instantiate_ssm_step_record(type, tname, 128, 128, 16, 8)           \
              instantiate_ssm_step_record(type, tname, 128, 128, 32, 1)       \
                  instantiate_ssm_step_record(type, tname, 128, 128, 32, 2)   \
                      instantiate_ssm_step_record(                            \
                          type, tname, 128, 128, 32, 4)                       \
                          instantiate_ssm_step_record(                        \
                              type, tname, 128, 128, 32, 8)                   \
                              instantiate_ssm_step_record(                    \
                                  type, tname, 128, 128, 48, 1)               \
                                  instantiate_ssm_step_record(                \
                                      type, tname, 128, 128, 48, 2)           \
                                      instantiate_ssm_step_record(            \
                                          type, tname, 128, 128, 48, 4)       \
                                          instantiate_ssm_step_record(        \
                                              type, tname, 128, 128, 48, 8)   \
                                              instantiate_ssm_replay(         \
                                                  type, tname, 128, 128, 16)  \
                                                  instantiate_ssm_replay(     \
                                                      type,                   \
                                                      tname,                  \
                                                      128,                    \
                                                      128,                    \
                                                      32)                     \
                                                      instantiate_ssm_replay( \
                                                          type,               \
                                                          tname,              \
                                                          128,                \
                                                          128,                \
                                                          48)

instantiate_ssm_record_dh64_ds64_for(half, float16)
    instantiate_ssm_record_dh64_ds64_for(bfloat16_t, bfloat16)
        instantiate_ssm_record_dh128_ds128_for(half, float16)
            instantiate_ssm_record_dh128_ds128_for(bfloat16_t, bfloat16)
