// Copyright (c) 2026 Eric Kryski. Spec 040 — C++ doctest coverage for the
// Mamba state-replay primitives: `ssm_step_record` and `ssm_replay`.

#include <cmath>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

float max_abs_diff(const array& a, const array& b) {
  auto diff = abs(subtract(a, b));
  auto m = max(diff, /*keepdims=*/false);
  return astype(m, float32, Device::cpu).item<float>();
}

} // namespace

// Note on shapes (cf. mlx/backend/metal/kernels/ssm_replay.metal):
//   x:        [B, T, H, Dh]
//   A_log:    [H]
//   B, C:     [B, T, G, Ds]   (G = key/value groups; H is a multiple of G)
//   D:        [H]
//   dt:       [B, T, H]
//   state:    [B, H, Dh, Ds]
// Available kernel instantiations cover H in {16, 32, 48}, G in {1, 2, 4, 8},
// and (Dh, Ds) in {(64, 64), (128, 128)}. H=4 is not instantiated, so these
// tests use H=16, G=1 (the smallest available shape) instead.

TEST_CASE("ssm step record returns four arrays") {
  const int B = 1, T = 4, H = 16, G = 1, Dh = 64, Ds = 64;

  auto x = random::normal({B, T, H, Dh}, float16, std::nullopt, Device::gpu);
  auto A_log = random::normal({H}, float16, std::nullopt, Device::gpu);
  auto B_arr =
      random::normal({B, T, G, Ds}, float16, std::nullopt, Device::gpu);
  auto C_arr =
      random::normal({B, T, G, Ds}, float16, std::nullopt, Device::gpu);
  auto D_arr = random::normal({H}, float16, std::nullopt, Device::gpu);
  // Small dt magnitudes keep dA = exp(-exp(A_log)*dt) bounded — otherwise
  // negative-tailed random dt with positive A_log overflows to inf/nan.
  auto dt = multiply(
      random::normal({B, T, H}, float16, std::nullopt, Device::gpu),
      array(0.1f, float16),
      Device::gpu);
  auto state = zeros({B, H, Dh, Ds}, float16, Device::gpu);

  auto outs = fast::ssm_step_record(
      x, A_log, B_arr, C_arr, D_arr, dt, state, std::nullopt, Device::gpu);
  eval(outs);

  CHECK_EQ(outs.size(), size_t{4});
  // y:        [B, T, H, Dh]
  CHECK_EQ(outs[0].shape(), Shape{B, T, H, Dh});
  // state_out: [B, H, Dh, Ds]
  CHECK_EQ(outs[1].shape(), Shape{B, H, Dh, Ds});
  // dA_log:   [B, T, H, Ds]
  CHECK_EQ(outs[2].shape(), Shape{B, T, H, Ds});
  // dBx_log:  [B, T, H, Dh, Ds]
  CHECK_EQ(outs[3].shape(), Shape{B, T, H, Dh, Ds});
  CHECK(all(isfinite(outs[0])).item<bool>());
  CHECK(all(isfinite(outs[1])).item<bool>());
}

TEST_CASE("ssm replay round trip") {
  // Record a forward pass, then replay the FULL log onto the initial
  // snapshot. The replayed state should match `state_out` from the record
  // call within tight tolerance (fp16 accumulation noise only).
  const int B = 1, T = 4, H = 16, G = 1, Dh = 64, Ds = 64;

  auto x = random::normal({B, T, H, Dh}, float16, std::nullopt, Device::gpu);
  auto A_log = random::normal({H}, float16, std::nullopt, Device::gpu);
  auto B_arr =
      random::normal({B, T, G, Ds}, float16, std::nullopt, Device::gpu);
  auto C_arr =
      random::normal({B, T, G, Ds}, float16, std::nullopt, Device::gpu);
  auto D_arr = random::normal({H}, float16, std::nullopt, Device::gpu);
  // Small dt magnitudes keep dA = exp(-exp(A_log) * dt) numerically tame and
  // bounded away from underflow over T steps.
  auto dt = multiply(
      random::normal({B, T, H}, float16, std::nullopt, Device::gpu),
      array(0.1f, float16),
      Device::gpu);
  auto initial = zeros({B, H, Dh, Ds}, float16, Device::gpu);

  auto outs = fast::ssm_step_record(
      x, A_log, B_arr, C_arr, D_arr, dt, initial, std::nullopt, Device::gpu);
  auto& state_out = outs[1];
  auto& dA_log = outs[2];
  auto& dBx_log = outs[3];

  auto replayed = fast::ssm_replay(
      initial,
      dA_log,
      dBx_log,
      /*accepted_prefix=*/T,
      std::nullopt,
      Device::gpu);
  eval(state_out, replayed);

  CHECK_EQ(replayed.shape(), state_out.shape());
  // Tolerance: 5e-3 covers fp16 mantissa noise (~1e-3 per step) accumulated
  // over T=4 sequential dA-scaled updates. Empirically ~2e-3 in practice.
  CHECK(max_abs_diff(replayed, state_out) < 5e-3f);
}

TEST_CASE("ssm replay partial accept differs from full") {
  // Record T=8 steps, then replay only the first 4. The partial-replay state
  // should differ from BOTH the initial snapshot AND the full-T state_out:
  // proving the kernel is honouring `accepted_prefix` rather than always
  // replaying the entire log.
  const int B = 1, T = 8, H = 16, G = 1, Dh = 64, Ds = 64;

  auto x = random::normal({B, T, H, Dh}, float16, std::nullopt, Device::gpu);
  auto A_log = random::normal({H}, float16, std::nullopt, Device::gpu);
  auto B_arr =
      random::normal({B, T, G, Ds}, float16, std::nullopt, Device::gpu);
  auto C_arr =
      random::normal({B, T, G, Ds}, float16, std::nullopt, Device::gpu);
  auto D_arr = random::normal({H}, float16, std::nullopt, Device::gpu);
  auto dt = multiply(
      random::normal({B, T, H}, float16, std::nullopt, Device::gpu),
      array(0.1f, float16),
      Device::gpu);
  auto initial = zeros({B, H, Dh, Ds}, float16, Device::gpu);

  auto outs = fast::ssm_step_record(
      x, A_log, B_arr, C_arr, D_arr, dt, initial, std::nullopt, Device::gpu);
  auto& state_full = outs[1];
  auto& dA_log = outs[2];
  auto& dBx_log = outs[3];

  auto state_partial = fast::ssm_replay(
      initial,
      dA_log,
      dBx_log,
      /*accepted_prefix=*/4,
      std::nullopt,
      Device::gpu);
  eval(state_full, state_partial);

  CHECK(max_abs_diff(state_partial, initial) > 1e-4f);
  CHECK(max_abs_diff(state_partial, state_full) > 1e-4f);
}
