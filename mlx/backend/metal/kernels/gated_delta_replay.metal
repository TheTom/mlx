// Copyright © 2026 Eric Kryski. GatedDeltaNet tape-capture + tape-replay
// Metal kernels for innovation-tape rollback (spec 020 phase 2).
//
// Companion to `gated_delta.metal`. Two kernels:
//
//   1. `gated_delta_step_record` — forward variant of `gated_delta_step`
//      that ALSO writes per-step `delta_t` to a tape output buffer.
//      Used by speculative-decoder verify forwards on hybrid GDN+Attention
//      models (Qwen 3.5 / 3.6). The cache records the per-step
//      `(delta_t, k_t, g_t)` triple during verify; the layer hands in k_t
//      and g_t directly, so this kernel only needs to surface delta_t.
//
//   2. `state_replay` — re-folds the accepted prefix of an innovation tape
//      from a pre-record snapshot. Used by
//      `SSMStateCache.rollback(acceptedPrefix:)` on partial-accept rounds.
//      Adopts upstream dflash-mlx correctness patterns from day 1:
//        - Masked-timestep correctness fix (`3217e15`): save `old_state`
//          before each step, restore via `metal::select` on masked
//          positions.
//        - Branchless pattern (`c9f992e`): `metal::select(old, new,
//          do_step)` instead of `if(do_step)` guards — better SIMD
//          occupancy when timestep masks are non-uniform within a SIMD
//          group.
//
// Same template parameter cells as `gated_delta.metal` since Qwen 3.5 and
// Qwen 3.6 share the same GDN config structure.

#include <metal_common>
#include <metal_simdgroup>
#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Function constants for masked vs non-masked variants.
constant bool gdn_record_has_mask [[function_constant(10)]];
constant bool state_replay_has_mask [[function_constant(20)]];

// ============================================================================
// Standard GatedDelta kernel with tape capture
// ============================================================================
//
// Body is byte-identical to `gated_delta_step` (in `gated_delta.metal`)
// except: inside the per-step loop, after `delta` is computed, one lane per
// SIMD group writes `delta` to the `delta_log` output buffer.
//
// Masked timesteps don't write the tape — the replay kernel reads its own
// mask buffer and skips those positions via `metal::select`, so tape
// entries for masked steps are never read.

template <typename T, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_step_record(
    const device T* q [[buffer(0)]],
    const device T* k [[buffer(1)]],
    const device T* v [[buffer(2)]],
    const device T* g [[buffer(3)]],
    const device T* beta [[buffer(4)]],
    const device T* state_in [[buffer(5)]],
    const device bool* mask [[buffer(6)]],
    device T* y [[buffer(7)]],
    device T* state_out [[buffer(8)]],
    device T* delta_log [[buffer(9)]],
    constant int& T_val [[buffer(10)]],
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
  auto td_ = delta_log + b_idx * T_val * Hv * Dv + hv_idx * Dv;

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
    bool process = gdn_record_has_mask ? mask[b_idx * T_val + t] : true;
    if (process) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * static_cast<float>(g_[hv_idx]);
        kv_mem += state[i] * static_cast<float>(k_[s_idx]);
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                   * static_cast<float>(beta_[hv_idx]);

      // Tape write: one lane per SIMD group writes delta. All SIMD lanes
      // hold the same value post-simd_sum, so any lane suffices. Same
      // discipline as the y write below.
      if (simd_lane == 0) {
        td_[dv_idx] = static_cast<T>(delta);
      }

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
    td_ += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }
  for (int i = 0; i < n_per_t; ++i) {
    o_state[n_per_t * dk_idx + i] = static_cast<T>(state[i]);
  }
}

// ============================================================================
// Tape-replay kernel
// ============================================================================
//
// Re-folds the accepted prefix `[0, accepted)` of an innovation tape onto a
// pre-record state snapshot. The tape carries per-step `(delta_t, k_t,
// g_t)` triples; the cache stores k AFTER GQA expansion, so the kernel's
// `k_log` stride is `Hv * Dk` (not `Hk * Dk` as in the forward kernel).
//
// Body:
//   for t in [0, T_log):
//     do_step = (t < accepted) && (mask[t] || !has_mask)
//     new_state = state * g[t] + k[t] * delta[t]
//     state    = do_step ? new_state : state    (via metal::select)

template <typename T, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void state_replay(
    const device T*    delta_log [[buffer(0)]],
    const device T*    k_log     [[buffer(1)]],
    const device T*    g_log     [[buffer(2)]],
    const device T*    state_in   [[buffer(3)]],
    const device bool* mask       [[buffer(4)]],
    device T*          state_out  [[buffer(5)]],
    constant int&      T_log     [[buffer(6)]],
    constant int&      accepted   [[buffer(7)]],
    uint3              thread_pos [[thread_position_in_grid]],
    uint3              tg_pos     [[thread_position_in_threadgroup]],
    uint               simd_lane  [[thread_index_in_simdgroup]]) {

  constexpr int n_per_t = Dk / 32;

  auto n      = thread_pos.z;
  auto b_idx  = n / Hv;
  auto hv_idx = n % Hv;

  auto dk_idx = tg_pos.x;
  auto dv_idx = thread_pos.y;

  // delta_log: [B, T_log, Hv, Dv]
  // k_log:     [B, T_log, Hv, Dk]  (already GQA-expanded by the cache)
  // g_log:     [B, T_log, Hv]
  auto delta_ = delta_log + b_idx * T_log * Hv * Dv + hv_idx * Dv;
  auto k_     = k_log     + b_idx * T_log * Hv * Dk + hv_idx * Dk;
  auto g_     = g_log     + b_idx * T_log * Hv;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
  }

  for (int t = 0; t < T_log; ++t) {
    bool within_accepted = (t < accepted);
    bool mask_passes     = state_replay_has_mask
                              ? mask[b_idx * T_log + t]
                              : true;
    bool do_step         = within_accepted && mask_passes;

    // Save old_state for masked / out-of-range positions. Stack-allocated;
    // the compiler holds these in registers alongside `state[i]`.
    float old_state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      old_state[i] = state[i];
    }

    float g_val = static_cast<float>(g_[hv_idx]);
    float d_val = static_cast<float>(delta_[dv_idx]);

    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      float new_val = state[i] * g_val
                    + static_cast<float>(k_[s_idx]) * d_val;
      state[i] = metal::select(old_state[i], new_val, do_step);
    }

    // Advance pointers regardless of do_step so the indexing stays in
    // lockstep with the tape layout.
    delta_ += Hv * Dv;
    k_     += Hv * Dk;
    g_     += Hv;
  }

  for (int i = 0; i < n_per_t; ++i) {
    o_state[n_per_t * dk_idx + i] = static_cast<T>(state[i]);
  }
}

// ============================================================================
// Instantiation
// ============================================================================
//
// Same template cells as `gated_delta.metal` since Qwen 3.5 and Qwen 3.6
// share the same GDN config structure. If new model variants land with
// different (Dk, Dv, Hk, Hv) tuples, add them here and to `gated_delta.metal`
// to keep parity.

#define instantiate_gdn_tape(type, tname, dk, dv, hk, hv) \
  template [[host_name("gated_delta_step_record_" #tname "_" #dk "_" #dv "_" #hk "_" #hv)]] \
  [[kernel]] void gated_delta_step_record<type, dk, dv, hk, hv>( \
    const device type*, const device type*, const device type*, \
    const device type*, const device type*, const device type*, \
    const device bool*, device type*, device type*, device type*, \
    constant int&, uint3, uint3, uint); \
  template [[host_name("state_replay_" #tname "_" #dk "_" #dv "_" #hk "_" #hv)]] \
  [[kernel]] void state_replay<type, dk, dv, hk, hv>( \
    const device type*, const device type*, const device type*, \
    const device type*, const device bool*, device type*, \
    constant int&, constant int&, uint3, uint3, uint);

// float32 instantiations are needed because `gatedDeltaUpdate` in
// `MLXLLM/Models/GatedDelta.swift` forces state to fp32 for prefill/verify
// (S > 1) precision, and the snapshot read by `SSMStateCache.rollback`
// carries that fp32 state. fp16/bf16 instantiations cover the decode-path
// `fusedGatedDeltaUpdate` (S == 1, state in q dtype) — though the
// recording path never goes through that, the parity keeps the kernel
// surface symmetric with `gated_delta.metal`.

// Qwen 3.5 / 3.6 A3B: Dk=192, Dv=128, Hk=4, Hv=4
instantiate_gdn_tape(half,        float16,  192, 128, 4, 4)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 192, 128, 4, 4)
instantiate_gdn_tape(float,       float32,  192, 128, 4, 4)

// Qwen 3.5 / 3.6 larger variants
instantiate_gdn_tape(half,        float16,  128, 128, 8, 8)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 128, 128, 8, 8)
instantiate_gdn_tape(float,       float32,  128, 128, 8, 8)
instantiate_gdn_tape(half,        float16,  64,  64,  8, 8)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 64,  64,  8, 8)
instantiate_gdn_tape(float,       float32,  64,  64,  8, 8)

// Qwen 3.5 / 3.6-35B: Dk=128, Dv=128, numHeads=16, numKVHeads=2 → Hk=16 Hv=32
instantiate_gdn_tape(half,        float16,  128, 128, 16, 32)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 128, 128, 16, 32)
instantiate_gdn_tape(float,       float32,  128, 128, 16, 32)

// Qwen 3.5 / 3.6 dense models: Dk=128, Dv=128, Hk=16, Hv=16 (0.8B–9B)
instantiate_gdn_tape(half,        float16,  128, 128, 16, 16)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 128, 128, 16, 16)
instantiate_gdn_tape(float,       float32,  128, 128, 16, 16)

// Qwen 3.5 / 3.6 dense 27B: Hv=48
instantiate_gdn_tape(half,        float16,  128, 128, 16, 48)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 128, 128, 16, 48)
instantiate_gdn_tape(float,       float32,  128, 128, 16, 48)

// Small-cell coverage for unit tests (matches mlx-swift-lm's
// `SSMStateCacheStateReplayTests` fixture: B=1, Hv=2, Hk=2, Dk=64, Dv=32).
instantiate_gdn_tape(half,        float16,  64,  32,  2, 2)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 64,  32,  2, 2)
instantiate_gdn_tape(float,       float32,  64,  32,  2, 2)

// Hk == Hv variants needed by the **replay kernel** specifically. The
// state-replay rollback consumes the cache's GQA-expanded k log, which
// has Hk_effective = Hv (not the layer's original Hk). The dispatcher
// in `MLXLMCommon/StateReplayKernels.swift::stateReplayUpdate` sets
// `let Hk = Hv`, so we instantiate every (Dk, Dv, Hv, Hv) cell that
// matches a real Qwen 3.5 / 3.6 variant.
//
// The forward kernel `gated_delta_step_record` still uses the layer's
// original Hk, so the GQA-asymmetric cells above continue to apply on
// the record path. Both surfaces share the same .metal source +
// template — only the dispatcher decides which (Hk, Hv) tuple it asks
// for, and both must be instantiated.

// Qwen 3.5 / 3.6 35B: replay Hk=Hv=32 (forward uses Hk=16, Hv=32)
instantiate_gdn_tape(half,        float16,  128, 128, 32, 32)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 128, 128, 32, 32)
instantiate_gdn_tape(float,       float32,  128, 128, 32, 32)

// Qwen 3.5 / 3.6 dense 27B: replay Hk=Hv=48 (forward uses Hk=16, Hv=48)
instantiate_gdn_tape(half,        float16,  128, 128, 48, 48)
instantiate_gdn_tape(bfloat16_t,  bfloat16, 128, 128, 48, 48)
instantiate_gdn_tape(float,       float32,  128, 128, 48, 48)
