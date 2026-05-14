// Copyright (c) 2026 Eric Kryski. Spec 041 phase 1.1 — C++ doctest coverage
// for the fused flash quantized SDPA primitive.

#include <cmath>
#include <limits>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

// Affine-quantize the last dim of `x` at the given bits / group_size and
// return {packed, scales, biases} matching the layout `flash_quantized_sdpa`
// expects on its k_* / v_* inputs.
struct Quantized {
  array packed;
  array scales;
  array biases;
};

Quantized affine_quant(const array& x, int bits, int group_size) {
  auto qout = quantize(x, group_size, bits, "affine");
  return {qout[0], qout[1], qout[2]};
}

// Convenience: max(abs(a - b)) reduced to a scalar float on CPU.
float max_abs_diff(const array& a, const array& b) {
  auto diff = abs(subtract(a, b));
  auto m = max(diff, /*keepdims=*/false);
  // Cast through float32 in case inputs are fp16/bf16.
  return astype(m, float32, Device::cpu).item<float>();
}

float max_abs(const array& a) {
  auto m = max(abs(a), /*keepdims=*/false);
  return astype(m, float32, Device::cpu).item<float>();
}

} // namespace

TEST_CASE("flash quantized sdpa output shape and finite") {
  // Q [B=1, n_q=4, T_q=1, D=128], K/V [B=1, n_kv=4, T_kv=256, D=128].
  const int B = 1, n_h = 4, T_q = 1, T_kv = 256, D = 128;
  const int bits = 4, group_size = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  auto q = random::normal({B, n_h, T_q, D}, float16, std::nullopt, Device::gpu);
  auto k =
      random::normal({B, n_h, T_kv, D}, float16, std::nullopt, Device::gpu);
  auto v =
      random::normal({B, n_h, T_kv, D}, float16, std::nullopt, Device::gpu);

  auto kq = affine_quant(k, bits, group_size);
  auto vq = affine_quant(v, bits, group_size);

  auto out = fast::flash_quantized_sdpa(
      q,
      kq.packed,
      kq.scales,
      kq.biases,
      vq.packed,
      vq.scales,
      vq.biases,
      scale,
      bits,
      group_size,
      /*mask_mode=*/"",
      /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt,
      /*window_size=*/-1,
      Device::gpu);
  eval(out);

  CHECK_EQ(out.shape(), Shape{B, n_h, T_q, D});
  CHECK(all(isfinite(out)).item<bool>());
}

TEST_CASE("flash quantized sdpa matches dequant then sdpa reference") {
  // Same layout as the shape test. Reference: dequantize K/V then call the
  // unquantized `fast::scaled_dot_product_attention`.
  const int B = 1, n_h = 4, T_q = 1, T_kv = 256, D = 128;
  const int bits = 4, group_size = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  auto q = random::normal({B, n_h, T_q, D}, float16, std::nullopt, Device::gpu);
  auto k =
      random::normal({B, n_h, T_kv, D}, float16, std::nullopt, Device::gpu);
  auto v =
      random::normal({B, n_h, T_kv, D}, float16, std::nullopt, Device::gpu);

  auto kq = affine_quant(k, bits, group_size);
  auto vq = affine_quant(v, bits, group_size);

  auto out_flash = fast::flash_quantized_sdpa(
      q,
      kq.packed,
      kq.scales,
      kq.biases,
      vq.packed,
      vq.scales,
      vq.biases,
      scale,
      bits,
      group_size,
      /*mask_mode=*/"",
      /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt,
      /*window_size=*/-1,
      Device::gpu);

  // Reference: dequantize then run the standard fast SDPA.
  auto k_dq = dequantize(
      kq.packed,
      kq.scales,
      kq.biases,
      group_size,
      bits,
      "affine",
      std::nullopt,
      float16,
      Device::gpu);
  auto v_dq = dequantize(
      vq.packed,
      vq.scales,
      vq.biases,
      group_size,
      bits,
      "affine",
      std::nullopt,
      float16,
      Device::gpu);
  auto out_ref = fast::scaled_dot_product_attention(
      q, k_dq, v_dq, scale, "", std::nullopt, std::nullopt, Device::gpu);
  eval(out_flash, out_ref);

  // Tolerance: affine 4-bit quant + tiled online-softmax adds noise that is
  // both relative (per-element magnitude) and absolute (small drift on
  // near-zero outputs). 5% relative + 5e-3 absolute is well above what we've
  // observed in practice (~1-2%) but tight enough to catch real bugs.
  float ref_max = max_abs(out_ref);
  float tol = 0.05f * ref_max + 5e-3f;
  float diff = max_abs_diff(out_flash, out_ref);
  CHECK(diff < tol);
}

TEST_CASE("flash quantized sdpa with causal mask") {
  // Square prefill: T_q == T_kv == 8 so causal masking is observable. The
  // no-mask output should differ from the causal output because the
  // upper-triangular K positions get masked out for early Q tokens.
  const int B = 1, n_h = 4, T = 8, D = 128;
  const int bits = 4, group_size = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  auto q = random::normal({B, n_h, T, D}, float16, std::nullopt, Device::gpu);
  auto k = random::normal({B, n_h, T, D}, float16, std::nullopt, Device::gpu);
  auto v = random::normal({B, n_h, T, D}, float16, std::nullopt, Device::gpu);

  auto kq = affine_quant(k, bits, group_size);
  auto vq = affine_quant(v, bits, group_size);

  auto out_causal = fast::flash_quantized_sdpa(
      q,
      kq.packed,
      kq.scales,
      kq.biases,
      vq.packed,
      vq.scales,
      vq.biases,
      scale,
      bits,
      group_size,
      /*mask_mode=*/"causal",
      /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt,
      /*window_size=*/-1,
      Device::gpu);
  auto out_nomask = fast::flash_quantized_sdpa(
      q,
      kq.packed,
      kq.scales,
      kq.biases,
      vq.packed,
      vq.scales,
      vq.biases,
      scale,
      bits,
      group_size,
      /*mask_mode=*/"",
      /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt,
      /*window_size=*/-1,
      Device::gpu);
  eval(out_causal, out_nomask);

  CHECK_EQ(out_causal.shape(), Shape{B, n_h, T, D});
  CHECK(all(isfinite(out_causal)).item<bool>());
  // Sanity: causal mask must change the output (otherwise the mask flag
  // wasn't honoured).
  CHECK(max_abs_diff(out_causal, out_nomask) > 1e-3f);
}

TEST_CASE("flash quantized sdpa bits 8") {
  // 8-bit affine quant: tighter agreement with the dequant-then-SDPA
  // reference than the 4-bit test.
  const int B = 1, n_h = 4, T_q = 1, T_kv = 256, D = 128;
  const int bits = 8, group_size = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  auto q = random::normal({B, n_h, T_q, D}, float16, std::nullopt, Device::gpu);
  auto k =
      random::normal({B, n_h, T_kv, D}, float16, std::nullopt, Device::gpu);
  auto v =
      random::normal({B, n_h, T_kv, D}, float16, std::nullopt, Device::gpu);

  auto kq = affine_quant(k, bits, group_size);
  auto vq = affine_quant(v, bits, group_size);

  auto out_flash = fast::flash_quantized_sdpa(
      q,
      kq.packed,
      kq.scales,
      kq.biases,
      vq.packed,
      vq.scales,
      vq.biases,
      scale,
      bits,
      group_size,
      /*mask_mode=*/"",
      /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt,
      /*window_size=*/-1,
      Device::gpu);

  auto k_dq = dequantize(
      kq.packed,
      kq.scales,
      kq.biases,
      group_size,
      bits,
      "affine",
      std::nullopt,
      float16,
      Device::gpu);
  auto v_dq = dequantize(
      vq.packed,
      vq.scales,
      vq.biases,
      group_size,
      bits,
      "affine",
      std::nullopt,
      float16,
      Device::gpu);
  auto out_ref = fast::scaled_dot_product_attention(
      q, k_dq, v_dq, scale, "", std::nullopt, std::nullopt, Device::gpu);
  eval(out_flash, out_ref);

  // 8-bit affine quant noise is ~1/16 of 4-bit. Use a tighter (still
  // generous) 2% relative + 2e-3 absolute envelope.
  float ref_max = max_abs(out_ref);
  float tol = 0.02f * ref_max + 2e-3f;
  float diff = max_abs_diff(out_flash, out_ref);
  CHECK(diff < tol);
}

TEST_CASE("flash quantized sdpa with sliding window") {
  // Phase 1.2: window_size > 0 narrows the causal triangle to a diagonal
  // band of `window_size` keys per query. Reference path: dequantize K/V,
  // build an explicit additive sliding-window mask, and run the regular
  // SDPA with that array mask.
  const int B = 1, n_h = 4, T = 8, D = 128, W = 4;
  const int bits = 4, group_size = 64;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));

  auto q = random::normal({B, n_h, T, D}, float16, std::nullopt, Device::gpu);
  auto k = random::normal({B, n_h, T, D}, float16, std::nullopt, Device::gpu);
  auto v = random::normal({B, n_h, T, D}, float16, std::nullopt, Device::gpu);

  auto kq = affine_quant(k, bits, group_size);
  auto vq = affine_quant(v, bits, group_size);

  auto out_window = fast::flash_quantized_sdpa(
      q,
      kq.packed,
      kq.scales,
      kq.biases,
      vq.packed,
      vq.scales,
      vq.biases,
      scale,
      bits,
      group_size,
      /*mask_mode=*/"causal",
      /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt,
      /*window_size=*/W,
      Device::gpu);

  // Build an additive sliding-window-causal mask of shape [T, T]:
  //   mask[q, k] = 0      if (k <= q) AND (k >= q - W + 1)
  //   mask[q, k] = -inf   otherwise
  // This is independent of the kernel's own window code path.
  auto q_idx =
      reshape(arange(T, int32, Device::gpu), {T, 1}, Device::gpu); // [T, 1]
  auto k_idx =
      reshape(arange(T, int32, Device::gpu), {1, T}, Device::gpu); // [1, T]
  auto allowed = logical_and(
      less_equal(k_idx, q_idx, Device::gpu),
      greater_equal(
          k_idx,
          subtract(q_idx, array(W - 1, int32), Device::gpu),
          Device::gpu),
      Device::gpu);
  auto neg_inf = array(-std::numeric_limits<float>::infinity(), float16);
  auto zero = array(0.0f, float16);
  auto add_mask = where(allowed, zero, neg_inf, Device::gpu); // [T, T] float16

  // Reference on dequanted K/V with explicit array mask.
  auto k_dq = dequantize(
      kq.packed,
      kq.scales,
      kq.biases,
      group_size,
      bits,
      "affine",
      std::nullopt,
      float16,
      Device::gpu);
  auto v_dq = dequantize(
      vq.packed,
      vq.scales,
      vq.biases,
      group_size,
      bits,
      "affine",
      std::nullopt,
      float16,
      Device::gpu);
  auto out_ref = fast::scaled_dot_product_attention(
      q,
      k_dq,
      v_dq,
      scale,
      "array",
      add_mask,
      std::nullopt,
      Device::gpu,
      /*window_size=*/-1);
  eval(out_window, out_ref);

  CHECK_EQ(out_window.shape(), Shape{B, n_h, T, D});
  CHECK(all(isfinite(out_window)).item<bool>());
  // Same 5% relative + 5e-3 absolute envelope as the 4-bit reference test.
  float ref_max = max_abs(out_ref);
  float tol = 0.05f * ref_max + 5e-3f;
  float diff = max_abs_diff(out_window, out_ref);
  CHECK(diff < tol);
}
