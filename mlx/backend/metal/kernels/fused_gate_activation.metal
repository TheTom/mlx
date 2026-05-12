// Copyright © 2026 Eric Kryski.

// Fused dense gate-activation kernel.
// Input:  gateUp of shape [rows, 2*hidden_dims]
// Output: out    of shape [rows, hidden_dims]
// Computes: out[r, i] = activation(gateUp[r, i]) * gateUp[r, hidden_dims + i]
//           (for silu and gelu_approx)
//
// Two specializations:
//   - `single_row`: axis <= threadgroup capacity (≈4096 for N_READS=4, max TG
//     1024). One threadgroup per row; each thread reads 4 gate + 4 up values.
//   - `looped`: axis > 4096. One threadgroup per row with max threads; each
//     thread loops through its slice in strided chunks of 4.
//
// Replaces Split + activation + Multiply (≥2 dispatches) with one dispatch.

#include <metal_common>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

constant int FUSED_GATE_N_READS = 4;

// ─── Activation functions ───────────────────────────────────────────────────

inline float silu_act(float x) {
  return x / (1.0f + metal::precise::exp(-x));
}

inline float gelu_approx_act(float x) {
  // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  constexpr float kSqrt2OverPi = 0.7978845608f;
  float x3 = x * x * x;
  float inner = kSqrt2OverPi * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + metal::precise::tanh(inner));
}

// Clipped swiglu (GPT-OSS). Both halves are clipped to [-7, 7] first; the
// gate side uses sigmoid(1.702·g); the up side has a +1 bias before the
// multiply. Matches the `swiglu_gate` helper in warp_moe_gate_up.metal.
inline float clipped_swiglu(float g, float u) {
  g = metal::clamp(g, -7.0f, 7.0f);
  u = metal::clamp(u, -7.0f, 7.0f);
  float sig = 1.0f / (1.0f + metal::precise::exp(-1.702f * g));
  return g * sig * (u + 1.0f);
}

template <int activation_type>
inline float apply_gate(float g, float u) {
  if (activation_type == 0) {
    return silu_act(g) * u;
  } else if (activation_type == 1) {
    return gelu_approx_act(g) * u;
  } else {
    return clipped_swiglu(g, u);
  }
}

// ─── Single-row kernel (axis <= tg_size * N_READS) ──────────────────────────

template <typename T, int activation_type, int N_READS = FUSED_GATE_N_READS>
[[kernel]] void fused_gate_activation_single_row(
    const device T* gateUp [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant uint& hidden_dims [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  const device T* gu_row = gateUp + gid * size_t(2 * hidden_dims);
  const device T* up_base = gu_row + hidden_dims;
  device T* o_row = out + gid * size_t(hidden_dims);

  uint i = lid * N_READS;
  if (i + N_READS <= hidden_dims) {
    // Unrolled fast path.
    T g_buf[N_READS];
    T u_buf[N_READS];
    for (int k = 0; k < N_READS; ++k) {
      g_buf[k] = gu_row[i + k];
      u_buf[k] = up_base[i + k];
    }
    for (int k = 0; k < N_READS; ++k) {
      o_row[i + k] = static_cast<T>(
          apply_gate<activation_type>(float(g_buf[k]), float(u_buf[k])));
    }
  } else {
    // Tail.
    for (int k = 0; k < N_READS; ++k) {
      uint ik = i + k;
      if (ik < hidden_dims) {
        o_row[ik] = static_cast<T>(
            apply_gate<activation_type>(float(gu_row[ik]), float(up_base[ik])));
      }
    }
  }
}

// ─── Looped kernel (axis > tg_size * N_READS) ───────────────────────────────

template <typename T, int activation_type, int N_READS = FUSED_GATE_N_READS>
[[kernel]] void fused_gate_activation_looped(
    const device T* gateUp [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant uint& hidden_dims [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  const device T* gu_row = gateUp + gid * size_t(2 * hidden_dims);
  const device T* up_base = gu_row + hidden_dims;
  device T* o_row = out + gid * size_t(hidden_dims);

  const uint stride = tg_size * N_READS;
  for (uint chunk = lid * N_READS; chunk < hidden_dims; chunk += stride) {
    if (chunk + N_READS <= hidden_dims) {
      T g_buf[N_READS];
      T u_buf[N_READS];
      for (int k = 0; k < N_READS; ++k) {
        g_buf[k] = gu_row[chunk + k];
        u_buf[k] = up_base[chunk + k];
      }
      for (int k = 0; k < N_READS; ++k) {
        o_row[chunk + k] = static_cast<T>(
            apply_gate<activation_type>(float(g_buf[k]), float(u_buf[k])));
      }
    } else {
      for (int k = 0; k < N_READS; ++k) {
        uint ik = chunk + k;
        if (ik < hidden_dims) {
          o_row[ik] = static_cast<T>(apply_gate<activation_type>(
              float(gu_row[ik]), float(up_base[ik])));
        }
      }
    }
  }
}

// ─── Instantiations ─────────────────────────────────────────────────────────

#define instantiate_fga_single(name, type, act)                                \
  template [[host_name(                                                        \
      "fused_gate_activation_single_row_" #name "_act" #act)]] [[kernel]] void \
  fused_gate_activation_single_row<type, act>(                                 \
      const device type*, device type*, constant uint&, uint, uint);

#define instantiate_fga_looped(name, type, act)                            \
  template [[host_name(                                                    \
      "fused_gate_activation_looped_" #name "_act" #act)]] [[kernel]] void \
  fused_gate_activation_looped<type, act>(                                 \
      const device type*, device type*, constant uint&, uint, uint, uint);

#define instantiate_fga(name, type)                                           \
  instantiate_fga_single(name, type, 0) instantiate_fga_single(name, type, 1) \
      instantiate_fga_single(name, type, 2)                                   \
          instantiate_fga_looped(name, type, 0)                               \
              instantiate_fga_looped(name, type, 1)                           \
                  instantiate_fga_looped(name, type, 2)

instantiate_fga(float32, float) instantiate_fga(float16, half)
    instantiate_fga(bfloat16, bfloat16_t)
