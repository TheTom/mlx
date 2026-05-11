// Copyright © 2023-2024 Apple Inc.
#include <cassert>
#include <numeric>

#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core::fast {

std::vector<array> Custom::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto [_, vjps] = mlx::core::vjp(fallback_, primals, cotangents);
  std::vector<array> vjp_outs;
  for (int i = 0, j = 0; i < vjps.size(); ++i) {
    if (j < argnums.size() && i == argnums[j]) {
      vjp_outs.push_back(vjps[i]);
      j++;
    }
  }
  return vjp_outs;
}

std::vector<array> Custom::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  std::vector<array> all_tangents;
  for (int i = 0, j = 0; i < primals.size(); i++) {
    if (j < argnums.size() && i == argnums[j]) {
      all_tangents.emplace_back(tangents[j++]);
    } else {
      all_tangents.emplace_back(zeros_like(primals[i]));
    }
  }
  auto [_, jvps] = mlx::core::jvp(fallback_, primals, all_tangents);
  return jvps;
}

std::pair<std::vector<array>, std::vector<int>> Custom::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto outputs = mlx::core::vmap(fallback_, axes)(inputs);
  auto out_axes = std::vector<int>(outputs.size(), 0);
  return {outputs, out_axes};
}

array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[rms_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[rms_norm] (*weight) must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (weight.has_value()) ? result_type(x, (*weight)) : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[rms_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback =
      [has_weight, eps, out_type, s](const std::vector<array>& inputs) {
        auto x = astype(inputs[0], float32, s);
        x = multiply(
            x,
            rsqrt(
                add(mean(square(x, s), -1, /* keepdims */ true, s),
                    array(eps, float32),
                    s),
                s),
            s);
        x = astype(x, out_type, s);

        if (has_weight) {
          x = multiply(x, inputs[1], s);
        }

        return std::vector<array>{x};
      };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);

  if (!RMSNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight});
  }
  return fallback({x, passed_weight})[0];
}

array rms_norm_residual(
    const array& x,
    const array& residual,
    const array& weight,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  if (x.ndim() < 1) {
    throw std::invalid_argument(
        "[rms_norm_residual] Input must have at least 1 dimension");
  }

  auto out_type = result_type(x, weight);
  if (!issubdtype(out_type, floating)) {
    throw std::invalid_argument(
        "[rms_norm_residual] Input must be floating point");
  }

  auto s = to_stream(s_);

  auto fallback = [eps, s](const std::vector<array>& inputs) {
    auto normed = rms_norm(inputs[0], inputs[2], eps, s);
    auto result = add(inputs[1], normed, s);
    return std::vector<array>{result};
  };

  if (!RMSNormResidual::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNormResidual>(s, fallback, eps),
        {astype(x, out_type, s),
         astype(residual, out_type, s),
         astype(weight, out_type, s)});
  }
  return fallback(
      {astype(x, out_type, s),
       astype(residual, out_type, s),
       astype(weight, out_type, s)})[0];
}

array fused_gate_activation(
    const array& gate_up,
    int hidden_dims,
    int activation_type,
    StreamOrDevice s_ /* = {} */) {
  if (gate_up.ndim() < 1) {
    throw std::invalid_argument(
        "[fused_gate_activation] Input must have at least 1 dimension");
  }
  if (gate_up.shape().back() != 2 * hidden_dims) {
    throw std::invalid_argument(
        "[fused_gate_activation] Last axis must equal 2 * hidden_dims");
  }
  if (activation_type < 0 || activation_type > 2) {
    throw std::invalid_argument(
        "[fused_gate_activation] activation_type must be 0 (silu), 1 (gelu_approx), or 2 (clipped swiglu)");
  }

  auto out_type = gate_up.dtype();
  if (!issubdtype(out_type, floating)) {
    throw std::invalid_argument(
        "[fused_gate_activation] Input must be floating point");
  }

  auto s = to_stream(s_);

  // Fallback: split, apply activation on gate, multiply by up.
  auto fallback = [hidden_dims, activation_type, s](
                      const std::vector<array>& inputs) {
    auto parts = split(inputs[0], 2, -1, s);
    const auto& gate = parts[0];
    const auto& up = parts[1];
    array activated = gate;
    if (activation_type == 0) {
      // silu(gate) = gate * sigmoid(gate)
      activated = multiply(gate, sigmoid(gate, s), s);
      return std::vector<array>{multiply(activated, up, s)};
    } else if (activation_type == 1) {
      // gelu_approx(gate) = 0.5 * gate * (1 + tanh(sqrt(2/pi) * (gate + 0.044715 * gate^3)))
      auto gate3 = multiply(multiply(gate, gate, s), gate, s);
      auto inner = multiply(
          array(0.7978845608f, gate.dtype()),
          add(gate, multiply(array(0.044715f, gate.dtype()), gate3, s), s),
          s);
      activated = multiply(
          array(0.5f, gate.dtype()),
          multiply(
              gate,
              add(array(1.0f, gate.dtype()), tanh(inner, s), s),
              s),
          s);
      return std::vector<array>{multiply(activated, up, s)};
    } else {
      // clipped swiglu (GPT-OSS): out = g·sigmoid(1.702·g)·(u + 1) with g, u clamped to [-7, 7]
      auto seven = array(7.0f, gate.dtype());
      auto neg_seven = array(-7.0f, gate.dtype());
      auto g = minimum(maximum(gate, neg_seven, s), seven, s);
      auto u = minimum(maximum(up, neg_seven, s), seven, s);
      auto scaled = multiply(array(1.702f, gate.dtype()), g, s);
      auto sig = sigmoid(scaled, s);
      auto one = array(1.0f, gate.dtype());
      return std::vector<array>{
          multiply(multiply(g, sig, s), add(u, one, s), s)};
    }
  };

  auto out_shape = gate_up.shape();
  out_shape.back() = hidden_dims;

  if (!FusedGateActivation::use_fallback(s)) {
    return array(
        std::move(out_shape),
        out_type,
        std::make_shared<FusedGateActivation>(
            s, fallback, hidden_dims, activation_type),
        {gate_up});
  }
  return fallback({gate_up})[0];
}

array rms_norm_rope(
    const array& x,
    const array& weight,
    const array& inv_freqs,
    float eps,
    int offset,
    int n_heads,
    int seq_len,
    StreamOrDevice s_ /* = {} */) {
  if (x.ndim() < 2) {
    throw std::invalid_argument(
        "[rms_norm_rope] Input must have at least 2 dimensions");
  }

  auto out_type = result_type(x, weight);
  if (!issubdtype(out_type, floating)) {
    throw std::invalid_argument(
        "[rms_norm_rope] Input must be floating point");
  }

  auto s = to_stream(s_);

  auto fallback = [eps, offset, n_heads, seq_len, s](
                      const std::vector<array>& inputs) {
    // Fallback: separate rms_norm + rope via MLX ops
    auto normed = rms_norm(inputs[0], inputs[1], eps, s);
    // Transpose [B, L, H, D] -> [B, H, L, D] for RoPE, then back
    auto transposed = transpose(normed, {0, 2, 1, 3}, s);
    // RoPE with custom frequencies
    auto rotated = rope(
        transposed,
        transposed.shape(-1),
        false,  // non-traditional
        0.0f,   // base (unused when freqs provided)
        1.0f,   // scale
        offset,
        inputs[2],  // inv_freqs as freqs (RoPE will invert them)
        s);
    auto result = transpose(rotated, {0, 2, 1, 3}, s);
    return std::vector<array>{result};
  };

  if (!RMSNormRoPE::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<RMSNormRoPE>(
            s, fallback, eps, n_heads, seq_len, offset),
        {astype(x, out_type, s),
         astype(weight, out_type, s),
         astype(inv_freqs, float32, s)});
  }
  return fallback(
      {astype(x, out_type, s),
       astype(weight, out_type, s),
       astype(inv_freqs, float32, s)})[0];
}

array rms_norm_qgemv(
    const array& x,
    const array& norm_weight,
    const array& w,
    const array& scales,
    const array& biases,
    float eps,
    int group_size,
    StreamOrDevice s_ /* = {} */) {
  if (x.ndim() < 1) {
    throw std::invalid_argument(
        "[rms_norm_qgemv] Input must have at least 1 dimension");
  }

  int K = x.shape().back();
  int N = w.shape(0);
  auto out_type = x.dtype();

  auto s = to_stream(s_);

  // Fallback: separate rms_norm + quantized_matmul
  auto fallback = [eps, group_size, s](const std::vector<array>& inputs) {
    auto normed = rms_norm(inputs[0], inputs[1], eps, s);
    auto result = quantized_matmul(
        normed, inputs[2], inputs[3], inputs[4],
        true, group_size, 4, "affine", s);
    return std::vector<array>{result};
  };

  // Output shape: replace last dim K with N
  auto out_shape = x.shape();
  out_shape.back() = N;

  if (!RMSNormQuantizedGEMV::use_fallback(s)) {
    return array(
        std::move(out_shape),
        out_type,
        std::make_shared<RMSNormQuantizedGEMV>(s, fallback, eps, group_size),
        {astype(x, out_type, s),
         astype(norm_weight, out_type, s),
         w,
         astype(scales, out_type, s),
         astype(biases, out_type, s)});
  }
  return fallback(
      {astype(x, out_type, s),
       astype(norm_weight, out_type, s),
       w,
       astype(scales, out_type, s),
       astype(biases, out_type, s)})[0];
}

array batched_qkv_qgemv(
    const array& x,
    const array& w_q,
    const array& scales_q,
    const array& biases_q,
    const array& w_k,
    const array& scales_k,
    const array& biases_k,
    const array& w_v,
    const array& scales_v,
    const array& biases_v,
    int group_size,
    StreamOrDevice s_) {

  int K = x.shape().back();
  int N_q = w_q.shape(0);
  int N_k = w_k.shape(0);
  int N_v = w_v.shape(0);
  auto out_type = x.dtype();
  auto s = to_stream(s_);

  // Fallback: 3 separate quantized_matmul, concatenated
  auto fallback = [group_size, N_q, N_k, N_v, s](const std::vector<array>& inputs) {
    auto q = quantized_matmul(
        inputs[0], inputs[1], inputs[2], inputs[3],
        true, group_size, 4, "affine", s);
    auto k = quantized_matmul(
        inputs[0], inputs[4], inputs[5], inputs[6],
        true, group_size, 4, "affine", s);
    auto v = quantized_matmul(
        inputs[0], inputs[7], inputs[8], inputs[9],
        true, group_size, 4, "affine", s);
    return std::vector<array>{concatenate({q, k, v}, -1, s)};
  };

  // Output shape: [..., N_q + N_k + N_v]
  auto out_shape = x.shape();
  out_shape.back() = N_q + N_k + N_v;

  if (!BatchedQKVQuantizedGEMV::use_fallback(s)) {
    return array(
        std::move(out_shape),
        out_type,
        std::make_shared<BatchedQKVQuantizedGEMV>(
            s, fallback, group_size, N_q, N_k, N_v),
        {astype(x, out_type, s),
         w_q, astype(scales_q, out_type, s), astype(biases_q, out_type, s),
         w_k, astype(scales_k, out_type, s), astype(biases_k, out_type, s),
         w_v, astype(scales_v, out_type, s), astype(biases_v, out_type, s)});
  }

  return fallback(
      {astype(x, out_type, s),
       w_q, astype(scales_q, out_type, s), astype(biases_q, out_type, s),
       w_k, astype(scales_k, out_type, s), astype(biases_k, out_type, s),
       w_v, astype(scales_v, out_type, s), astype(biases_v, out_type, s)})[0];
}

// ============================================================================
// Warp MoE Gate+Up (fused gate+up projection with activation)
// ============================================================================
array warp_moe_gate_up(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& indices,
    int group_size,
    int hidden_dims,
    int activation_type,
    StreamOrDevice s_) {

  auto out_type = x.dtype();
  auto s = to_stream(s_);
  int top_k = indices.shape(0);

  auto fallback = [group_size, hidden_dims, activation_type, s](
      const std::vector<array>& inputs) -> std::vector<array> {
    // Fallback: use gatherQuantizedMM + split + activation
    throw std::runtime_error("WarpMoeGateUp fallback NYI — use gatherQuantizedMM path");
  };

  Shape out_shape = {top_k, hidden_dims};

  if (!WarpMoeGateUp::use_fallback(s)) {
    return array(
        std::move(out_shape),
        out_type,
        std::make_shared<WarpMoeGateUp>(
            s, fallback, group_size, hidden_dims, activation_type),
        {flatten(astype(x, out_type, s)),
         w, astype(scales, out_type, s), astype(biases, out_type, s),
         astype(indices, int32, s)});
  }
  return fallback({flatten(astype(x, out_type, s)),
      w, astype(scales, out_type, s), astype(biases, out_type, s),
      astype(indices, int32, s)})[0];
}

// ============================================================================
// Warp MoE Down (fused down projection with routing weight folding)
// ============================================================================
array warp_moe_down(
    const array& activated,
    const array& w,
    const array& scales,
    const array& biases,
    const array& indices,
    const array& scores,
    int group_size,
    int hidden_dims,
    int out_dims,
    StreamOrDevice s_) {

  auto out_type = activated.dtype();
  auto s = to_stream(s_);

  auto fallback = [group_size, hidden_dims, out_dims, s](
      const std::vector<array>& inputs) -> std::vector<array> {
    throw std::runtime_error("WarpMoeDown fallback NYI — use gatherQuantizedMM path");
  };

  Shape out_shape = {out_dims};

  if (!WarpMoeDown::use_fallback(s)) {
    return array(
        std::move(out_shape),
        out_type,
        std::make_shared<WarpMoeDown>(
            s, fallback, group_size, hidden_dims, out_dims),
        {astype(activated, out_type, s),
         w, astype(scales, out_type, s), astype(biases, out_type, s),
         astype(indices, int32, s),
         astype(scores, out_type, s)});
  }
  return fallback({astype(activated, out_type, s),
      w, astype(scales, out_type, s), astype(biases, out_type, s),
      astype(indices, int32, s), astype(scores, out_type, s)})[0];
}

std::vector<array> RMSNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 2);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& g = inputs[2];

    std::vector<array> vjps;

    auto n = rsqrt(
        add(mean(square(x, s), /* axis= */ -1, /* keepdims= */ true, s),
            array(eps, x.dtype()),
            s),
        s);
    auto n3 = power(n, array(3, x.dtype()), s);

    // df/dx
    auto gw = multiply(g, w, s);
    auto t = mean(multiply(gw, x, s), /* axis= */ -1, /* keepdims= */ true, s);
    t = multiply(multiply(x, t, s), n3, s);
    vjps.push_back(subtract(multiply(gw, n, s), t, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x, n, s), s), axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape()},
      {primals[0].dtype(), primals[1].dtype()},
      std::make_shared<RMSNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool RMSNorm::is_equivalent(const Primitive& other) const {
  const RMSNorm& a_other = static_cast<const RMSNorm&>(other);
  return eps_ == a_other.eps_;
}

bool RMSNormVJP::is_equivalent(const Primitive& other) const {
  const RMSNormVJP& a_other = static_cast<const RMSNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s_ /* = {} */) {
  bool has_weight = weight.has_value();
  bool has_bias = bias.has_value();

  if (x.ndim() == 0) {
    std::ostringstream msg;
    msg << "[layer_norm] Input must have at least 1 dimension but got input with "
           "0 dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (has_weight) {
    if ((*weight).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have 1 dimension but has "
          << (*weight).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*weight).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] weight must have the same size as the last dimension of"
             " x but has "
          << (*weight).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (has_bias) {
    if ((*bias).ndim() != 1) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have 1 dimension but has "
          << (*bias).ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if ((*bias).size() != x.shape(-1)) {
      std::ostringstream msg;
      msg << "[layer_norm] bias must have the same size as the last dimension of"
             " x but has "
          << (*bias).size() << " elements.";
      throw std::invalid_argument(msg.str());
    }
  }

  auto out_type = (has_weight)
      ? ((has_bias) ? result_type(x, *weight, *bias) : result_type(x, *weight))
      : x.dtype();
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[layer_norm] Received unsupported type " << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto fallback = [has_weight, has_bias, eps, out_type, s](
                      const std::vector<array>& inputs) {
    auto x = astype(inputs[0], float32, s);

    auto mu = mean(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto xc = subtract(x, mu, s);
    auto v = mean(square(xc, s), /* axis= */ -1, /* keepdims= */ true, s);

    x = multiply(xc, rsqrt(add(v, array(eps, float32), s), s));
    x = astype(x, out_type, s);

    // If the LN is affine then transform x according to the weight and bias
    if (has_weight) {
      x = multiply(x, inputs[1], s);
    }
    if (has_bias) {
      x = add(x, inputs[2], s);
    }

    return std::vector<array>{x};
  };

  auto passed_weight =
      (has_weight) ? astype(*weight, out_type, s) : array(1, out_type);
  auto passed_bias =
      (has_bias) ? astype(*bias, out_type, s) : array(0, out_type);

  if (!LayerNorm::use_fallback(s)) {
    return array(
        x.shape(),
        out_type,
        std::make_shared<LayerNorm>(s, fallback, eps),
        {astype(x, out_type, s), passed_weight, passed_bias});
  }
  return fallback({x, passed_weight, passed_bias})[0];
}

std::vector<array> LayerNorm::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() == 3);
  assert(outputs.size() == 1);
  assert(cotangents.size() == 1);

  auto s = stream();
  auto fallback = [eps = eps_, s](const std::vector<array>& inputs) {
    auto& x = inputs[0];
    auto& w = inputs[1];
    auto& b = inputs[2];
    auto& g = inputs[3];

    std::vector<array> vjps;

    auto norm = number_of_elements(x, {-1}, true, x.dtype(), s);
    auto sumx = sum(x, /* axis= */ -1, /* keepdims= */ true, s);
    auto sumx2 = sum(square(x, s), /* axis= */ -1, /* keepdims= */ true, s);
    auto mu = multiply(sumx, norm, s);
    auto mu2 = multiply(sumx2, norm, s);
    auto var = subtract(mu2, square(mu, s), s);
    auto n = rsqrt(add(var, array(eps, x.dtype()), s));
    auto n3 = power(n, array(3, x.dtype()), s);
    auto x_c = subtract(x, mu, s);

    // df/dx
    auto wg = multiply(w, g, s);
    auto sumwg =
        multiply(sum(wg, /* axis= */ -1, /* keepdims= */ true, s), norm, s);
    auto sumwgxc = multiply(
        sum(multiply(wg, x_c, s), /* axis= */ -1, /* keepdims= */ true, s),
        norm,
        s);
    auto t1 = multiply(multiply(x_c, sumwgxc, s), n3, s);
    auto t2 = multiply(subtract(wg, sumwg, s), n, s);
    vjps.push_back(subtract(t2, t1, s));

    // df/dw
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    if (w.ndim() == 0) {
      vjps.push_back(zeros_like(w, s));
    } else {
      vjps.push_back(sum(
          multiply(g, multiply(x_c, n, s), s), axes, /* keepdims= */ false, s));
    }

    // df/db
    if (b.ndim() == 0) {
      vjps.push_back(zeros_like(b, s));
    } else {
      vjps.push_back(sum(g, axes, /* keepdims= */ false, s));
    }

    return vjps;
  };

  auto vjps = array::make_arrays(
      {primals[0].shape(), primals[1].shape(), primals[2].shape()},
      {primals[0].dtype(), primals[1].dtype(), primals[2].dtype()},
      std::make_shared<LayerNormVJP>(s, fallback, eps_),
      {primals[0], primals[1], primals[2], cotangents[0]});

  std::vector<array> returned_vjps;
  for (auto& arg : argnums) {
    returned_vjps.push_back(std::move(vjps[arg]));
  }

  return returned_vjps;
}

bool LayerNorm::is_equivalent(const Primitive& other) const {
  const LayerNorm& a_other = static_cast<const LayerNorm&>(other);
  return eps_ == a_other.eps_;
}

bool LayerNormVJP::is_equivalent(const Primitive& other) const {
  const LayerNormVJP& a_other = static_cast<const LayerNormVJP&>(other);
  return eps_ == a_other.eps_;
}

array rope(
    std::vector<array> inputs,
    int dims,
    bool traditional,
    float base,
    float scale,
    bool forward,
    StreamOrDevice s) {
  auto& x = inputs[0];
  auto& offset = inputs[1];
  if (x.ndim() < 3) {
    std::ostringstream msg;
    msg << "[rope] Input must have at least 3 dimensions but got input with "
        << x.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[rope] Input must be a floating type but got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.ndim() > 1) {
    std::ostringstream msg;
    msg << "[rope] offset must have at most one dimension but has shape "
        << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.size() != 1 && offset.size() != x.shape(0)) {
    std::ostringstream msg;
    msg << "[rope] offset must be a scalar or vector with " << x.shape(0)
        << " elements but has shape " << offset.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(offset.dtype(), integer)) {
    std::ostringstream msg;
    msg << "[rope] offset must be an integer but got type " << offset.dtype()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (offset.dtype().size() != 4) {
    inputs[1] = astype(offset, int32, s);
  }
  if (dims <= 0) {
    std::ostringstream msg;
    msg << "[rope] dims must be positive but got " << dims << ".";
    throw std::invalid_argument(msg.str());
  }
  if (dims % 2 != 0) {
    std::ostringstream msg;
    msg << "[rope] dims must be even but got " << dims << ".";
    throw std::invalid_argument(msg.str());
  }
  if (dims > x.shape(-1)) {
    std::ostringstream msg;
    msg << "[rope] dims must not exceed the input's last dimension ("
        << x.shape(-1) << ") but got " << dims << ".";
    throw std::invalid_argument(msg.str());
  }

  if (inputs.size() == 3 &&
      (inputs[2].ndim() != 1 || inputs[2].shape(0) != dims / 2)) {
    std::ostringstream msg;
    msg << "[rope] freqs must be one dimensional with size " << dims / 2
        << " but got shape " << inputs[2].shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto fallback = [dims, traditional, base, scale, forward, s](
                      std::vector<array> inputs) {
    auto x = inputs[0];
    auto shape = x.shape();
    if (x.ndim() == 3) {
      x = expand_dims(x, 1, s);
    } else if (x.ndim() > 4) {
      x = flatten(x, 1, 1 + (x.ndim() - 4), s);
    }

    auto B = x.shape(0);
    auto N = x.shape(1);
    auto T = x.shape(2);
    auto t = x.dtype();
    // Compute sines and cosines
    auto half_dims = dims / 2;
    auto offset = inputs[1];
    if (offset.size() > 1) {
      offset = expand_dims(offset, {-1, -2}, s);
    }
    auto positions = multiply(
        add(arange(x.shape(2), float32, s), offset, s),
        array(scale, float32),
        s);

    auto default_inv_freqs = [&s, base, half_dims]() {
      return exp(
          multiply(
              arange(0, -half_dims, -1, float32, s),
              array(std::log(base) / half_dims, float32),
              s),
          s);
    };

    auto inv_freqs =
        inputs.size() == 3 ? reciprocal(inputs[2], s) : default_inv_freqs();
    auto theta = multiply(expand_dims(positions, -1, s), inv_freqs, s);
    auto coss = astype(cos(theta, s), t, s);
    auto sins = astype(sin(theta, s), t, s);

    auto apply_rope = [forward, s](
                          const array& x1,
                          const array& x2,
                          const array& coss,
                          const array& sins) {
      std::vector<array> outs;
      if (forward) {
        outs.push_back(
            subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
        outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
      } else {
        outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
        outs.push_back(
            subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
      }
      return outs;
    };

    if (traditional) {
      auto x1 = slice(x, {0, 0, 0, 0}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto x2 = slice(x, {0, 0, 0, 1}, {B, N, T, dims}, {1, 1, 1, 2}, s);
      auto outs = apply_rope(x1, x2, coss, sins);
      for (auto& o : outs) {
        o = expand_dims(o, -1, s);
      }
      auto out = reshape(concatenate(outs, -1, s), {B, N, T, dims}, s);
      if (dims < x.shape(-1)) {
        out =
            concatenate({out, slice(x, {0, 0, 0, dims}, x.shape(), s)}, -1, s);
      }
      return std::vector<array>{reshape(out, shape, s)};
    } else {
      auto out_s = x.shape();
      out_s.back() = half_dims;
      auto x1 = slice(x, {0, 0, 0, 0}, out_s, s);
      out_s.back() = dims;
      auto x2 = slice(x, {0, 0, 0, half_dims}, out_s, s);

      auto outs = apply_rope(x1, x2, coss, sins);
      if (dims < x.shape(-1)) {
        outs.push_back(slice(x, {0, 0, 0, dims}, x.shape(), s));
      }
      return std::vector<array>{reshape(concatenate(outs, -1, s), shape, s)};
    }
  };
  auto stream = to_stream(s);
  if (!RoPE::use_fallback(stream)) {
    return array(
        x.shape(),
        x.dtype(),
        std::make_shared<RoPE>(
            stream, fallback, dims, traditional, base, scale, forward),
        std::move(inputs));
  }
  return fallback(std::move(inputs))[0];
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  std::vector<array> inputs = {x, offset};
  if (freqs) {
    inputs.push_back(astype(*freqs, float32, s));
    if (base) {
      throw std::invalid_argument(
          "[rope] Only one of base or freqs can have a value.");
    }
  } else if (!base) {
    throw std::invalid_argument("[rope] Neither base nor freqs has a value.");
  }
  return rope(
      std::move(inputs),
      dims,
      traditional,
      base.has_value() ? *base : 1.0,
      scale,
      true,
      s);
}

array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  return rope(
      x, dims, traditional, base, scale, array(offset, int32), freqs, s);
}

std::vector<array> RoPE::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  auto s = stream();
  auto fallback = [dims = dims_,
                   traditional = traditional_,
                   base = base_,
                   scale = scale_,
                   forward = forward_,
                   s](std::vector<array> inputs) {
    return std::vector<array>{
        rope(std::move(inputs), dims, traditional, base, scale, !forward, s)};
  };
  if (argnums.size() > 1 || argnums[0] != 0) {
    throw std::invalid_argument(
        "[RoPE::vjp] vjp for offset or frequencies not supported");
  }
  auto inputs = std::vector<array>{cotangents[0], primals[1]};
  if (primals.size() == 3) {
    inputs.push_back(primals[2]);
  }
  return {array(
      cotangents[0].shape(),
      cotangents[0].dtype(),
      std::make_shared<RoPE>(
          s, fallback, dims_, traditional_, base_, scale_, !forward_),
      std::move(inputs))};
}

bool RoPE::is_equivalent(const Primitive& other) const {
  const RoPE& a_other = static_cast<const RoPE&>(other);
  return (
      dims_ == a_other.dims_ && base_ == a_other.base_ &&
      scale_ == a_other.scale_ && traditional_ == a_other.traditional_ &&
      forward_ == a_other.forward_);
}

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode /* = "" */,
    std::optional<array> mask_arr /* = {} */,
    const std::optional<array>& sinks /* = {} */,
    StreamOrDevice s /* = {}*/,
    int window_size /* = -1 */) {
  for (const auto& tensor : {queries, keys, values}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] input with shape "
          << tensor.shape() << " expected to be rank 4";
      throw std::invalid_argument(msg.str());
    }
  }
  // Check valid mask
  if (mask_mode != "" && mask_mode != "causal" && mask_mode != "array") {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Invalid mask_mode " << mask_mode
        << ". mask_mode must be 'causal', 'array' or ''.";
    throw std::invalid_argument(msg.str());
  }

  bool do_causal = false;
  bool has_mask = false;
  bool has_arr_mask = false;
  bool has_bool_mask = false;

  if (mask_mode == "causal") {
    has_mask = true;
    do_causal = true;

    if (mask_arr) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Invalid mask_arr for mask_mode "
          << "'casusal'. No array mask should be passed.";
      throw std::invalid_argument(msg.str());
    }
  } else if (mask_arr) {
    has_mask = true;
    has_arr_mask = true;
    has_bool_mask = mask_arr->dtype() == bool_;
  }

  // Sliding-window currently requires the causal mask mode: it narrows the
  // causal triangle to a diagonal band of `window_size` keys per query.
  if (window_size > 0 && !do_causal) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] window_size > 0 requires "
        << "mask_mode == 'causal' (got '" << mask_mode << "').";
    throw std::invalid_argument(msg.str());
  }
  if (window_size == 0) {
    throw std::invalid_argument(
        "[scaled_dot_product_attention] window_size must be positive (or -1 to disable).");
  }

  if (has_arr_mask && mask_arr->ndim() > 4) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] the mask with shape "
        << mask_arr->shape() << " expected to have at most rank 4.";
    throw std::invalid_argument(msg.str());
  }

  const size_t batch_dim = queries.shape(0);
  for (const auto& tensor : {keys, values}) {
    if (tensor.shape(0) != batch_dim) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] mismatching batch dimension for input with shape "
          << tensor.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  // Q, K must have matching last dims (d_k aka 'head_dim');
  if (queries.shape(-1) != keys.shape(-1)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] query, keys expected to have matching last dimension; found query shape "
        << queries.shape() << " for keys shape " << keys.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // K, V must have matching number of heads (n_kv_heads);
  auto n_q_heads = queries.shape(-3);
  auto n_kv_heads = keys.shape(-3);

  if (keys.shape(-3) != values.shape(-3)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] keys, values expected to have matching n_kv_heads; found keys with n_heads "
        << keys.shape(-3) << " for values with n_heads " << values.shape(-3)
        << ".";
    throw std::invalid_argument(msg.str());
  }

  // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
  if (n_q_heads % n_kv_heads != 0) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] n_heads must be a multiple of n_kv_heads, found n_heads "
        << n_q_heads << " for n_kv_heads " << n_kv_heads << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type(queries, keys, values);
  if (!issubdtype(final_type, floating)) {
    std::ostringstream msg;
    msg << "[scaled_dot_product_attention] Received unsupported type "
        << final_type << ".";
    throw std::invalid_argument(msg.str());
  }
  bool has_sinks = sinks.has_value();

  auto q = astype(queries, final_type, s);
  auto k = astype(keys, final_type, s);
  auto v = astype(values, final_type, s);

  auto fallback = [scale,
                   n_q_heads,
                   n_kv_heads,
                   do_causal,
                   has_sinks,
                   has_arr_mask,
                   window_size,
                   s](const std::vector<array>& inputs) {
    auto q = multiply(array(scale, inputs[0].dtype()), inputs[0], s);
    int n_repeats = n_q_heads / n_kv_heads;
    auto k = inputs[1];
    auto v = inputs[2];
    if (n_repeats > 1) {
      q = unflatten(q, 1, {n_kv_heads, n_repeats}, s);
      k = expand_dims(k, 2, s);
      v = expand_dims(v, 2, s);
    }
    auto scores = matmul(q, swapaxes(k, -1, -2, s), s);
    if (has_arr_mask || do_causal) {
      // Mask must be broadcast-compatible with [B, n_q_heads, L_q, L_kv]
      auto make_or_fetch_mask = [&]() {
        if (do_causal) {
          int kL = k.shape(-2);
          int qL = q.shape(-2);
          int offset = kL - qL;
          auto q_idx = arange(offset, qL + offset, s);
          auto k_idx = arange(0, kL, s);
          q_idx = expand_dims(q_idx, 1, s);
          k_idx = expand_dims(k_idx, 0, s);
          auto causal_mask = greater_equal(q_idx, k_idx, s);
          if (window_size > 0) {
            // Also require k_idx > q_idx - window_size — i.e., keys
            // within the last `window_size` positions of each query.
            auto lower_bound =
                subtract(q_idx, array(window_size, q_idx.dtype()), s);
            auto within_window = greater(k_idx, lower_bound, s);
            return logical_and(causal_mask, within_window, s);
          }
          return causal_mask;
        }
        return inputs[3];
      };
      auto mask = make_or_fetch_mask();

      if (n_repeats > 1 && mask.ndim() >= 3) {
        if (mask.shape(-3) == 1) {
          mask = expand_dims(mask, -3, s);
        } else {
          mask = unflatten(mask, -3, {n_kv_heads, n_repeats}, s);
        }
      }
      if (mask.dtype() == bool_) {
        scores = where(
            mask, scores, array(finfo(scores.dtype()).min, scores.dtype()), s);
      } else {
        scores = add(scores, mask, s);
      }
    }
    if (has_sinks) {
      auto sinks = inputs.back();
      // scores has shape B N_q N_k L_q L_k
      sinks = expand_dims(sinks, {0, 2, 3}, s);
      if (scores.ndim() == 5) {
        sinks = unflatten(sinks, 1, {n_kv_heads, n_repeats}, s);
      }
      auto bsx_shape = scores.shape();
      bsx_shape.back() = 1;
      scores = concatenate({broadcast_to(sinks, bsx_shape, s), scores}, -1, s);
    }
    scores = softmax(scores, std::vector<int>{-1}, true, s);
    if (has_sinks) {
      // Slice off scores
      auto start = Shape(scores.ndim(), 0);
      start.back() = 1;
      auto stop = scores.shape();
      scores = slice(scores, std::move(start), std::move(stop), s);
    }
    auto out = matmul(scores, v, s);
    if (n_repeats > 1) {
      out = flatten(out, 1, 2, s);
    }
    return std::vector<array>{out};
  };

  auto stream = to_stream(s);
  std::vector<array> inputs = {q, k, v};
  if (has_arr_mask) {
    // Check type
    has_bool_mask = mask_arr->dtype() == bool_;
    if (promote_types(mask_arr->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Mask type must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    } else if (!has_bool_mask) {
      mask_arr = astype(*mask_arr, final_type, stream);
    }
    // Broadcast mask
    auto mask_shape = queries.shape();
    mask_shape.back() = keys.shape(-2);
    inputs.push_back(broadcast_to(*mask_arr, mask_shape, stream));
  }
  if (has_sinks) {
    if (promote_types(sinks->dtype(), final_type) != final_type) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Type of sinks must promote to output type "
          << final_type << ".";
      throw std::invalid_argument(msg.str());
    }
    if (sinks->ndim() != 1 || sinks->shape(0) != n_q_heads) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] Received invalid shape for sinks "
          << sinks->shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    inputs.push_back(astype(*sinks, final_type, stream));
  }

  bool is_training = detail::in_grad_tracing();
  bool has_fast_vjp = !ScaledDotProductAttentionVJP::use_fallback(q, stream);
  bool output_logsumexp = is_training && has_fast_vjp;
  if (!ScaledDotProductAttention::use_fallback(
          q,
          k,
          v,
          has_mask,
          has_arr_mask,
          do_causal,
          is_training,
          output_logsumexp,
          stream,
          window_size)) {
    if (has_bool_mask && !ScaledDotProductAttention::supports_bool_mask()) {
      // Convert bool mask to additive mask.
      float inf = std::numeric_limits<float>::infinity();
      array& mask = inputs[3];
      mask = where(
          mask,
          full_like(mask, 0, final_type, s),
          full_like(mask, -inf, final_type, s));
    }
    Shape out_shape{q.shape(0), q.shape(1), q.shape(2), v.shape(-1)};
    auto primitive = std::make_shared<ScaledDotProductAttention>(
        stream,
        fallback,
        scale,
        do_causal,
        has_sinks,
        output_logsumexp,
        window_size);
    if (output_logsumexp) {
      return array::make_arrays(
          {std::move(out_shape), Shape{q.shape(0), q.shape(1), q.shape(2), 1}},
          {final_type, float32},
          primitive,
          std::move(inputs))[0];
    } else {
      return array(
          std::move(out_shape), final_type, primitive, std::move(inputs));
    }
  }
  return fallback(std::move(inputs))[0];
}

std::vector<array> ScaledDotProductAttention::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {
  assert(primals.size() >= 3);
  assert(cotangents.size() == outputs.size());

  auto s = stream();
  if (ScaledDotProductAttentionVJP::use_fallback(primals[0], s)) {
    assert(outputs.size() == 1);
    return Custom::vjp(primals, cotangents, argnums, outputs);
  }

  auto fallback = [sdpa = fallback_, s](const std::vector<array>& inputs) {
    std::vector<array> primals(inputs.begin(), std::prev(inputs.end()));
    auto [_, vjps] = mlx::core::vjp(sdpa, primals, {inputs.back()});
    return vjps;
  };

  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  for (int i = 0; i < /* outputs size */ 3; ++i) {
    shapes.push_back(primals[i].shape());
    dtypes.push_back(primals[i].dtype());
  }
  auto primitive = std::make_shared<ScaledDotProductAttentionVJP>(
      s, fallback, scale_, do_causal_, has_sinks_);
  std::vector<array> inputs = primals;
  inputs.push_back(outputs[0]);
  inputs.push_back(outputs[1]);
  inputs.push_back(cotangents[0]);
  auto vjps = array::make_arrays(std::move(shapes), dtypes, primitive, inputs);

  std::vector<array> returned_vjps;
  for (int arg : argnums) {
    if (arg >= 3) {
      throw std::invalid_argument(
          "[scale_dot_product_attention] Does not support VJP with respect "
          " to mask or attention sinks.");
    }
    returned_vjps.push_back(std::move(vjps[arg]));
  }
  return returned_vjps;
}

bool ScaledDotProductAttention::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttention& a_other =
      static_cast<const ScaledDotProductAttention&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_ &&
      output_logsumexp_ == a_other.output_logsumexp_ &&
      window_size_ == a_other.window_size_;
}

bool ScaledDotProductAttentionVJP::is_equivalent(const Primitive& other) const {
  const ScaledDotProductAttentionVJP& a_other =
      static_cast<const ScaledDotProductAttentionVJP&>(other);
  return scale_ == a_other.scale_ && do_causal_ == a_other.do_causal_ &&
      has_sinks_ == a_other.has_sinks_;
}

bool Quantize::is_equivalent(const Primitive& other) const {
  const Quantize& p_other = static_cast<const Quantize&>(other);
  return (
      p_other.group_size_ == group_size_ && p_other.bits_ == bits_ &&
      p_other.mode_ == mode_ && p_other.dequantize_ == dequantize_);
}

std::vector<Shape> Quantize::output_shapes(const std::vector<array>& inputs) {
  auto& w = inputs[0];
  if (dequantize_) {
    auto out_size = w.shape(-1) * 32 / bits_;
    auto out_shape = w.shape();
    out_shape.back() = out_size;
    return {std::move(out_shape)};
  } else {
    auto wq_shape = w.shape();
    wq_shape.back() = w.shape(-1) * bits_ / 32;
    auto sshape = w.shape();
    sshape.back() = w.shape(-1) / group_size_;
    if (inputs.size() == 2) {
      return {std::move(wq_shape), std::move(sshape)};
    } else {
      auto bshape = sshape;
      return {std::move(wq_shape), std::move(sshape), std::move(bshape)};
    }
  }
}

bool ConvertFP8::is_equivalent(const Primitive& other) const {
  const ConvertFP8& a_other = static_cast<const ConvertFP8&>(other);
  return to_fp8_ == a_other.to_fp8_;
}

// ============================================================================
// TurboQuant API implementations
// ============================================================================

namespace {

inline int turbo_packed_width(int dim, int bits) {
  return (dim * bits + 31) / 32;
}

} // namespace

array turbo_score(
    const array& q_rot,
    const array& packed,
    const array& norms,
    const array& codebook,
    int token_count,
    int repeat_count,
    int bits,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_q = q_rot.shape(0);

  // Fallback: no CPU implementation
  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_score] Only runs on GPU");
  };

  return array(
      {total_q, token_count},
      float32,
      std::make_shared<TurboScore>(s, fallback, bits, dim),
      {astype(q_rot, float32, s),
       packed,
       astype(norms, float32, s),
       astype(codebook, float32, s)});
}

std::vector<array> turbo_encode(
    const array& input,
    const array& rotation,
    const array& boundaries,
    const array& codebook,
    int bits,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int num_rows = input.shape(0);
  int pw = turbo_packed_width(dim, bits);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_encode] Only runs on GPU");
  };

  return array::make_arrays(
      {{num_rows, pw}, {num_rows}},
      {uint32, float32},
      std::make_shared<TurboEncode>(s, fallback, bits, dim, /*use_wht=*/false),
      {astype(input, float32, s),
       astype(rotation, float32, s),
       astype(boundaries, float32, s),
       astype(codebook, float32, s)});
}

std::vector<array> turbo_encode_wht(
    const array& input,
    const array& wht_signs,
    const array& boundaries,
    int bits,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int num_rows = input.shape(0);
  int pw = turbo_packed_width(dim, bits);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_encode_wht] Only runs on GPU");
  };

  return array::make_arrays(
      {{num_rows, pw}, {num_rows}},
      {uint32, float32},
      std::make_shared<TurboEncode>(s, fallback, bits, dim, /*use_wht=*/true),
      {astype(input, float32, s),
       astype(wht_signs, float32, s),
       astype(boundaries, float32, s)});
}

std::vector<array> turbo_flash_pass1(
    const array& q_rot,
    const array& key_packed,
    const array& key_norms,
    const array& key_codebook,
    const array& val_packed,
    const array& val_norms,
    const array& val_codebook,
    int token_count,
    int repeat_count,
    int num_blocks,
    int block_size,
    int key_bits,
    int value_bits,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_q = q_rot.shape(0);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_flash_pass1] Only runs on GPU");
  };

  // Pack scalar constants as 0-d int32 arrays so they become graph inputs
  return array::make_arrays(
      {{total_q * num_blocks, dim},
       {total_q, num_blocks},
       {total_q, num_blocks}},
      {float32, float32, float32},
      std::make_shared<TurboFlashPass1>(
          s, fallback, key_bits, value_bits, dim, /*causal=*/false),
      {astype(q_rot, float32, s),
       key_packed,
       astype(key_norms, float32, s),
       astype(key_codebook, float32, s),
       val_packed,
       astype(val_norms, float32, s),
       astype(val_codebook, float32, s),
       array(token_count, int32),
       array(repeat_count, int32),
       array(num_blocks, int32),
       array(block_size, int32)});
}

std::vector<array> turbo_flash_pass1_causal(
    const array& q_rot,
    const array& key_packed,
    const array& key_norms,
    const array& key_codebook,
    const array& val_packed,
    const array& val_norms,
    const array& val_codebook,
    int token_count,
    int repeat_count,
    int num_blocks,
    int block_size,
    int L,
    int q_offset,
    int key_bits,
    int value_bits,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_q = q_rot.shape(0);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_flash_pass1_causal] Only runs on GPU");
  };

  return array::make_arrays(
      {{total_q * num_blocks, dim},
       {total_q, num_blocks},
       {total_q, num_blocks}},
      {float32, float32, float32},
      std::make_shared<TurboFlashPass1>(
          s, fallback, key_bits, value_bits, dim, /*causal=*/true),
      {astype(q_rot, float32, s),
       key_packed,
       astype(key_norms, float32, s),
       astype(key_codebook, float32, s),
       val_packed,
       astype(val_norms, float32, s),
       astype(val_codebook, float32, s),
       array(token_count, int32),
       array(repeat_count, int32),
       array(num_blocks, int32),
       array(block_size, int32),
       array(L, int32),
       array(q_offset, int32)});
}

std::vector<array> turbo_flash_pass1_nr0(
    const array& q_rot,
    const array& key_packed,
    const array& key_norms,
    const array& key_codebook,
    const array& val_packed,
    const array& val_norms,
    const array& val_codebook,
    int token_count,
    int repeat_count,
    int num_blocks,
    int block_size,
    int key_bits,
    int value_bits,
    int dim,
    int nr0,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_q = q_rot.shape(0);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_flash_pass1_nr0] Only runs on GPU");
  };

  return array::make_arrays(
      {{total_q * num_blocks, dim},
       {total_q, num_blocks},
       {total_q, num_blocks}},
      {float32, float32, float32},
      std::make_shared<TurboFlashPass1NR0>(
          s, fallback, key_bits, value_bits, dim, nr0, /*causal=*/false),
      {astype(q_rot, float32, s),
       key_packed,
       astype(key_norms, float32, s),
       astype(key_codebook, float32, s),
       val_packed,
       astype(val_norms, float32, s),
       astype(val_codebook, float32, s),
       array(token_count, int32),
       array(repeat_count, int32),
       array(num_blocks, int32),
       array(block_size, int32)});
}

std::vector<array> turbo_flash_pass1_nr0_causal(
    const array& q_rot,
    const array& key_packed,
    const array& key_norms,
    const array& key_codebook,
    const array& val_packed,
    const array& val_norms,
    const array& val_codebook,
    int token_count,
    int repeat_count,
    int num_blocks,
    int block_size,
    int L,
    int q_offset,
    int key_bits,
    int value_bits,
    int dim,
    int nr0,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_q = q_rot.shape(0);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_flash_pass1_nr0_causal] Only runs on GPU");
  };

  return array::make_arrays(
      {{total_q * num_blocks, dim},
       {total_q, num_blocks},
       {total_q, num_blocks}},
      {float32, float32, float32},
      std::make_shared<TurboFlashPass1NR0>(
          s, fallback, key_bits, value_bits, dim, nr0, /*causal=*/true),
      {astype(q_rot, float32, s),
       key_packed,
       astype(key_norms, float32, s),
       astype(key_codebook, float32, s),
       val_packed,
       astype(val_norms, float32, s),
       astype(val_codebook, float32, s),
       array(token_count, int32),
       array(repeat_count, int32),
       array(num_blocks, int32),
       array(block_size, int32),
       array(L, int32),
       array(q_offset, int32)});
}

// Output is bfloat16 — matches model dtype for all current MLX LMs and avoids
// an fp32→bf16 conversion in the consumer's graph. Kernel accumulators stay
// fp32; only the final write narrows. This dtype change is also what fixes
// upstream MLX graph-compile fusion bug (#87/#92) that produced `!!!!!`
// decoding on certain shapes — see kernel comment in
// backend/metal/kernels/turbo_quant.metal.
array turbo_flash_pass2(
    const array& o_partials,
    const array& m_partials,
    const array& l_partials,
    int num_blocks,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_q = m_partials.shape(0);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_flash_pass2] Only runs on GPU");
  };

  return array(
      {total_q, dim},
      bfloat16,
      std::make_shared<TurboFlashPass2>(
          s, fallback, dim, /*fused_rotation=*/false),
      {astype(o_partials, float32, s),
       astype(m_partials, float32, s),
       astype(l_partials, float32, s)});
}

array turbo_flash_pass2_fused(
    const array& o_partials,
    const array& m_partials,
    const array& l_partials,
    const array& val_rotation,
    int num_blocks,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_q = m_partials.shape(0);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_flash_pass2_fused] Only runs on GPU");
  };

  return array(
      {total_q, dim},
      bfloat16,
      std::make_shared<TurboFlashPass2>(
          s, fallback, dim, /*fused_rotation=*/true),
      {astype(o_partials, float32, s),
       astype(m_partials, float32, s),
       astype(l_partials, float32, s),
       astype(val_rotation, float32, s)});
}

array turbo_value(
    const array& weights,
    const array& packed,
    const array& norms,
    const array& codebook,
    int token_count,
    int repeat_count,
    float sparse_threshold,
    int bits,
    int dim,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  int total_heads = weights.shape(0);

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_value] Only runs on GPU");
  };

  return array(
      {total_heads, dim},
      float32,
      std::make_shared<TurboValue>(s, fallback, bits, dim),
      {astype(weights, float32, s),
       packed,
       astype(norms, float32, s),
       astype(codebook, float32, s),
       array(sparse_threshold, float32)});
}

array turbo_bulk_dequant_rotated(
    const array& packed,
    const array& norms,
    const array& codebook,
    int bits,
    int dim,
    Dtype output_dtype,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  if (s.device == Device::cpu) {
    throw std::runtime_error("[turbo_bulk_dequant_rotated] Only runs on GPU");
  }
  if (output_dtype != bfloat16 && output_dtype != float16) {
    throw std::invalid_argument(
        "[turbo_bulk_dequant_rotated] output dtype must be bfloat16 or float16");
  }
  if (packed.ndim() != 4) {
    throw std::invalid_argument(
        "[turbo_bulk_dequant_rotated] packed must be [B, H, T, PackedWidth]");
  }

  int B = static_cast<int>(packed.shape(0));
  int H = static_cast<int>(packed.shape(1));
  int T = static_cast<int>(packed.shape(2));

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[turbo_bulk_dequant_rotated] Only runs on GPU");
  };

  return array(
      {B, H, T, dim},
      output_dtype,
      std::make_shared<TurboBulkDequantRotated>(
          s, fallback, bits, dim, output_dtype),
      {packed,
       astype(norms, float32, s),
       astype(codebook, float32, s)});
}

// ============================================================================
// GatedDeltaNet / SSM recurrence primitives
// ============================================================================

std::vector<array> gated_delta_step(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state,
    const std::optional<array>& mask,
    int T,
    bool fused,
    int Dk,
    int Dv,
    int Hk,
    int Hv,
    StreamOrDevice s_ /* = {} */) {
  auto s = to_stream(s_);
  auto out_type = q.dtype();
  bool has_mask = mask.has_value();

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[gated_delta_step] Only runs on GPU");
  };

  // Compute output shapes
  // y: [B, T, Hv, Dv]
  int B = static_cast<int>(state.shape(0));
  Shape y_shape = {B, T, Hv, Dv};
  // state_out: [B, Hv, Dv, Dk]
  Shape state_shape = state.shape();

  // Build inputs vector matching Metal buffer order.
  //
  // Standard path (fused=false):
  //   API args map 1:1 -> q(0), k(1), v(2), g(3), beta(4), state_in(5),
  //                        [mask(6)]
  //
  // Fused path (fused=true):
  //   The fused kernel expects 8 input arrays:
  //     q_raw(0), k_raw(1), v(2), a(3), b_input(4), a_log(5), dt_bias(6),
  //     state_in(7), [mask(8)]
  //   This API re-purposes the 6 positional args as:
  //     q -> q_raw, k -> k_raw, v -> v, g -> a, beta -> b_input
  //     state -> state_in
  //   a_log and dt_bias are model constants that the Swift caller must
  //   inject by constructing the GatedDeltaStep primitive directly with
  //   all 8 inputs. This API function only supports the standard path.
  if (fused) {
    throw std::invalid_argument(
        "[gated_delta_step] Fused variant requires 8 input arrays. "
        "Construct the GatedDeltaStep primitive directly.");
  }

  std::vector<array> inputs = {
      astype(q, out_type, s),
      astype(k, out_type, s),
      astype(v, out_type, s),
      astype(g, out_type, s),
      astype(beta, out_type, s),
      astype(state, out_type, s)};
  if (has_mask) {
    inputs.push_back(*mask);
  }

  auto outputs = array::make_arrays(
      {std::move(y_shape), std::move(state_shape)},
      {out_type, out_type},
      std::make_shared<GatedDeltaStep>(
          s, fallback, fused, has_mask, T, Dk, Dv, Hk, Hv),
      std::move(inputs));
  // Break sibling bond: state_out pins T-proportional y, causing
  // peak ∝ layers × context.  Async eval detaches both outputs
  // without blocking the CPU.
  async_eval(outputs);
  return outputs;
}

std::vector<array> gated_delta_step_fused(
    const array& q_raw,
    const array& k_raw,
    const array& v,
    const array& a,
    const array& b_input,
    const array& a_log,
    const array& dt_bias,
    const array& state,
    const std::optional<array>& mask,
    int T,
    int Dk,
    int Dv,
    int Hk,
    int Hv,
    StreamOrDevice s_ /* = {} */) {
  auto s = to_stream(s_);
  auto out_type = q_raw.dtype();
  bool has_mask = mask.has_value();

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[gated_delta_step_fused] Only runs on GPU");
  };

  int B = static_cast<int>(state.shape(0));
  Shape y_shape = {B, T, Hv, Dv};
  Shape state_shape = state.shape();

  // Fused path: q_raw, k_raw, v, a, b_input, a_log, dt_bias, state [, mask]
  std::vector<array> inputs = {
      astype(q_raw, out_type, s),
      astype(k_raw, out_type, s),
      astype(v, out_type, s),
      astype(a, out_type, s),
      astype(b_input, out_type, s),
      astype(a_log, out_type, s),
      astype(dt_bias, out_type, s),
      astype(state, out_type, s)};
  if (has_mask) {
    inputs.push_back(*mask);
  }

  auto outputs = array::make_arrays(
      {std::move(y_shape), std::move(state_shape)},
      {out_type, out_type},
      std::make_shared<GatedDeltaStep>(
          s, fallback, /*fused=*/true, has_mask, T, Dk, Dv, Hk, Hv),
      std::move(inputs));
  // Break sibling bond (see gated_delta_step comment above).
  async_eval(outputs);
  return outputs;
}

std::vector<array> gated_delta_step_record(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state,
    const std::optional<array>& mask,
    int T,
    int Dk,
    int Dv,
    int Hk,
    int Hv,
    StreamOrDevice s_ /* = {} */) {
  auto s = to_stream(s_);
  auto out_type = q.dtype();
  bool has_mask = mask.has_value();

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[gated_delta_step_record] Only runs on GPU");
  };

  // Output shapes
  // y: [B, T, Hv, Dv]
  int B = static_cast<int>(state.shape(0));
  Shape y_shape = {B, T, Hv, Dv};
  // state_out: [B, Hv, Dv, Dk]
  Shape state_shape = state.shape();
  // delta_log: [B, T, Hv, Dv]
  Shape tape_shape = {B, T, Hv, Dv};

  // Input buffer order:
  //   0=q, 1=k, 2=v, 3=g, 4=beta, 5=state_in, [6=mask]
  std::vector<array> inputs = {
      astype(q, out_type, s),
      astype(k, out_type, s),
      astype(v, out_type, s),
      astype(g, out_type, s),
      astype(beta, out_type, s),
      astype(state, out_type, s)};
  if (has_mask) {
    inputs.push_back(*mask);
  }

  auto outputs = array::make_arrays(
      {std::move(y_shape), std::move(state_shape), std::move(tape_shape)},
      {out_type, out_type, out_type},
      std::make_shared<GatedDeltaStepRecord>(
          s, fallback, has_mask, T, Dk, Dv, Hk, Hv),
      std::move(inputs));
  // Break sibling bond (see gated_delta_step comment above).
  async_eval(outputs);
  return outputs;
}

std::vector<array> state_replay(
    const array& delta_log,
    const array& k_log,
    const array& g_log,
    const array& state,
    const std::optional<array>& mask,
    int T_log,
    int accepted,
    int Dk,
    int Dv,
    int Hk,
    int Hv,
    StreamOrDevice s_ /* = {} */) {
  auto s = to_stream(s_);
  auto out_type = state.dtype();
  bool has_mask = mask.has_value();

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[state_replay] Only runs on GPU");
  };

  // Output shape: state_out [B, Hv, Dv, Dk]
  Shape state_shape = state.shape();

  // Input buffer order:
  //   0=delta_log, 1=k_log, 2=g_log, 3=state_in, [4=mask]
  // Cast tape entries to state dtype for kernel uniformity.
  std::vector<array> inputs = {
      astype(delta_log, out_type, s),
      astype(k_log, out_type, s),
      astype(g_log, out_type, s),
      astype(state, out_type, s)};
  if (has_mask) {
    inputs.push_back(*mask);
  }

  auto outputs = array::make_arrays(
      {std::move(state_shape)},
      {out_type},
      std::make_shared<TapeReplay>(
          s, fallback, has_mask, T_log, accepted, Dk, Dv, Hk, Hv),
      std::move(inputs));
  return outputs;
}

std::vector<array> ssm_step(
    const array& X,
    const array& A_log,
    const array& B,
    const array& C,
    const array& D,
    const array& dt,
    const array& state,
    int Dh,
    int Ds,
    int H,
    int G,
    StreamOrDevice s_ /* = {} */) {
  auto s = to_stream(s_);
  auto out_type = X.dtype();

  auto fallback = [](const std::vector<array>&) -> std::vector<array> {
    throw std::runtime_error("[ssm_step] Only runs on GPU");
  };

  // Output shapes
  // out: same shape as X [N, Dh]
  Shape out_shape = X.shape();
  // state_out: same shape as state [N, Dh, Ds]
  Shape state_shape = state.shape();

  std::vector<array> inputs = {
      astype(X, out_type, s),
      astype(A_log, out_type, s),
      astype(B, out_type, s),
      astype(C, out_type, s),
      astype(D, out_type, s),
      astype(dt, out_type, s),
      astype(state, out_type, s)};

  auto outputs = array::make_arrays(
      {std::move(out_shape), std::move(state_shape)},
      {out_type, out_type},
      std::make_shared<SSMStep>(s, fallback, Dh, Ds, H, G),
      std::move(inputs));
  // Break sibling bond (see gated_delta_step comment above).
  async_eval(outputs);
  return outputs;
}

} // namespace mlx::core::fast
