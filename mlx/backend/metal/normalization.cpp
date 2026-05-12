// Copyright © 2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back() || x.shape(-2) == 1);
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "rms";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(out);
  auto& compute_encoder = metal::get_command_encoder(s);
  {
    auto kernel = d.get_kernel(op_name);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_output_array(out, 2);
    compute_encoder.set_bytes(eps_, 3);
    compute_encoder.set_bytes(axis_size, 4);
    compute_encoder.set_bytes(w_stride, 5);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

bool RMSNormQuantizedGEMV::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNormQuantizedGEMV::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  const array& x = inputs[0];
  const array& norm_weight = inputs[1];
  const array& w = inputs[2];
  const array& scales = inputs[3];
  const array& biases = inputs[4];

  auto K = static_cast<int>(x.shape().back());
  auto N = static_cast<int>(out.shape().back());

  out.set_data(allocator::malloc(out.nbytes()));

  // Kernel name: rms_norm_qgemv_<type>_gs<group_size>
  std::string kname = "rms_norm_qgemv_" + type_to_name(out) + "_gs" +
      std::to_string(group_size_);
  auto kernel = d.get_kernel(kname);

  // Same grid as qmv_impl: 8 outputs per threadgroup (2 simdgroups × 4 rows)
  constexpr int outputs_per_tg = 8;
  int n_tg_y = (N + outputs_per_tg - 1) / outputs_per_tg;

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(norm_weight, 1);
  compute_encoder.set_input_array(w, 2);
  compute_encoder.set_input_array(scales, 3);
  compute_encoder.set_input_array(biases, 4);
  compute_encoder.set_output_array(out, 5);
  compute_encoder.set_bytes(eps_, 6);
  compute_encoder.set_bytes(K, 7);
  compute_encoder.set_bytes(N, 8);
  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, n_tg_y, 1),
      MTL::Size(64, 1, 1)); // 2 simdgroups × 32 threads
}

bool RMSNormQuantizedGEMV::is_equivalent(const Primitive& other) const {
  const RMSNormQuantizedGEMV& r =
      static_cast<const RMSNormQuantizedGEMV&>(other);
  return eps_ == r.eps_ && group_size_ == r.group_size_;
}

// ============================================================================
// BatchedQKVQuantizedGEMV
// ============================================================================

bool BatchedQKVQuantizedGEMV::use_fallback(Stream stream) {
  return stream.device == Device::cpu;
}

void BatchedQKVQuantizedGEMV::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Inputs: x, w_q, scales_q, biases_q, w_k, scales_k, biases_k, w_v, scales_v,
  // biases_v
  const array& x = inputs[0];
  const array& w_q = inputs[1];
  const array& scales_q = inputs[2];
  const array& biases_q = inputs[3];
  const array& w_k = inputs[4];
  const array& scales_k = inputs[5];
  const array& biases_k = inputs[6];
  const array& w_v = inputs[7];
  const array& scales_v = inputs[8];
  const array& biases_v = inputs[9];

  // Single output: concatenated [N_q + N_k + N_v]
  auto& out = outputs[0];
  auto K = static_cast<int>(x.shape().back());

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname = "batched_qkv_qgemv_" + type_to_name(out) + "_gs" +
      std::to_string(group_size_);
  auto kernel = d.get_kernel(kname);

  // Grid covers the largest output dim across Q/K/V
  constexpr int outputs_per_tg = 8;
  int max_n = std::max({n_q_, n_k_, n_v_});
  int n_tg_y = (max_n + outputs_per_tg - 1) / outputs_per_tg;

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(x, 0);
  // Q weights
  compute_encoder.set_input_array(w_q, 1);
  compute_encoder.set_input_array(scales_q, 2);
  compute_encoder.set_input_array(biases_q, 3);
  // K weights
  compute_encoder.set_input_array(w_k, 4);
  compute_encoder.set_input_array(scales_k, 5);
  compute_encoder.set_input_array(biases_k, 6);
  // V weights
  compute_encoder.set_input_array(w_v, 7);
  compute_encoder.set_input_array(scales_v, 8);
  compute_encoder.set_input_array(biases_v, 9);
  // Output (single contiguous buffer)
  compute_encoder.set_output_array(out, 10);
  // Dimensions
  compute_encoder.set_bytes(n_q_, 11);
  compute_encoder.set_bytes(n_k_, 12);
  compute_encoder.set_bytes(n_v_, 13);
  compute_encoder.set_bytes(K, 14);

  // z=3: one z-slice per matrix (Q, K, V), all run in parallel
  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, n_tg_y, 3), MTL::Size(64, 1, 1));
}

bool BatchedQKVQuantizedGEMV::is_equivalent(const Primitive& other) const {
  const BatchedQKVQuantizedGEMV& r =
      static_cast<const BatchedQKVQuantizedGEMV&>(other);
  return group_size_ == r.group_size_ && n_q_ == r.n_q_ && n_k_ == r.n_k_ &&
      n_v_ == r.n_v_;
}

// ============================================================================
// WarpMoeGateUp
// ============================================================================

bool WarpMoeGateUp::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void WarpMoeGateUp::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Inputs: x, w, scales, biases, indices
  const array& x = inputs[0]; // [inputDims] flattened
  const array& w = inputs[1]; // [numExperts, 2*H, K_packed]
  const array& sc = inputs[2]; // [numExperts, 2*H, K/gs]
  const array& bi = inputs[3]; // [numExperts, 2*H, K/gs]
  const array& indices = inputs[4]; // [topK]

  auto& out = outputs[0]; // [topK, hiddenDims]
  out.set_data(allocator::malloc(out.nbytes()));

  int in_vec_size = static_cast<int>(x.size());
  int top_k = static_cast<int>(indices.size());

  std::string kname = "warp_moe_gate_up_" + type_to_name(out) + "_gs" +
      std::to_string(group_size_) + "_act" + std::to_string(activation_type_);
  auto kernel = d.get_kernel(kname);

  constexpr int outputs_per_tg = 8;
  int n_tg_y = (hidden_dims_ + outputs_per_tg - 1) / outputs_per_tg;

  auto& ce = metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(x, 0);
  ce.set_input_array(w, 1);
  ce.set_input_array(sc, 2);
  ce.set_input_array(bi, 3);
  ce.set_input_array(indices, 4);
  ce.set_output_array(out, 5);
  ce.set_bytes(in_vec_size, 6);
  ce.set_bytes(hidden_dims_, 7);
  ce.set_bytes(top_k, 8);

  ce.dispatch_threadgroups(
      MTL::Size(1, n_tg_y, top_k), // z = topK experts in parallel
      MTL::Size(64, 1, 1));
}

bool WarpMoeGateUp::is_equivalent(const Primitive& other) const {
  const WarpMoeGateUp& r = static_cast<const WarpMoeGateUp&>(other);
  return group_size_ == r.group_size_ && hidden_dims_ == r.hidden_dims_ &&
      activation_type_ == r.activation_type_;
}

// ============================================================================
// WarpMoeDown
// ============================================================================

bool WarpMoeDown::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void WarpMoeDown::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Inputs: activated, w, scales, biases, indices, scores
  const array& activated = inputs[0]; // [topK, hiddenDims]
  const array& w = inputs[1]; // [numExperts, outDims, H_packed]
  const array& sc = inputs[2];
  const array& bi = inputs[3];
  const array& indices = inputs[4]; // [topK]
  const array& scores = inputs[5]; // [topK]

  auto& out = outputs[0]; // [outDims]
  out.set_data(allocator::malloc(out.nbytes()));

  int top_k = static_cast<int>(indices.size());

  std::string kname = "warp_moe_down_" + type_to_name(out) + "_gs" +
      std::to_string(group_size_);
  auto kernel = d.get_kernel(kname);

  constexpr int outputs_per_tg = 8;
  int n_tg_y = (out_dims_ + outputs_per_tg - 1) / outputs_per_tg;

  auto& ce = metal::get_command_encoder(s);
  ce.set_compute_pipeline_state(kernel);
  ce.set_input_array(activated, 0);
  ce.set_input_array(w, 1);
  ce.set_input_array(sc, 2);
  ce.set_input_array(bi, 3);
  ce.set_input_array(indices, 4);
  ce.set_input_array(scores, 5);
  ce.set_output_array(out, 6);
  ce.set_bytes(hidden_dims_, 7);
  ce.set_bytes(out_dims_, 8);
  ce.set_bytes(top_k, 9);

  ce.dispatch_threadgroups(MTL::Size(1, n_tg_y, 1), MTL::Size(64, 1, 1));
}

bool WarpMoeDown::is_equivalent(const Primitive& other) const {
  const WarpMoeDown& r = static_cast<const WarpMoeDown&>(other);
  return group_size_ == r.group_size_ && hidden_dims_ == r.hidden_dims_ &&
      out_dims_ == r.out_dims_;
}

bool RMSNormResidual::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNormResidual::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Copies of inputs that needed contiguous_copy_gpu — registered via
  // add_temporaries so the underlying MTL buffers stay alive until the
  // command buffer completes (mirrors SDPA's pattern in
  // scaled_dot_product_attention.cpp).
  std::vector<array> copies;
  copies.reserve(2);

  // Ensure x is contiguous in last dim
  const array& x_in = inputs[0];
  bool no_copy =
      x_in.flags().contiguous && x_in.strides()[x_in.ndim() - 1] == 1;
  if (no_copy && x_in.ndim() > 1) {
    auto st = x_in.strides()[x_in.ndim() - 2];
    no_copy &= (st == 0 || st == x_in.shape().back() || x_in.shape(-2) == 1);
  }
  array x = x_in;
  if (!no_copy) {
    x = contiguous_copy_gpu(x_in, s);
    copies.push_back(x);
  }

  // Ensure residual is contiguous in last dim
  const array& r_in = inputs[1];
  bool r_no_copy =
      r_in.flags().contiguous && r_in.strides()[r_in.ndim() - 1] == 1;
  if (r_no_copy && r_in.ndim() > 1) {
    auto st = r_in.strides()[r_in.ndim() - 2];
    r_no_copy &= (st == 0 || st == r_in.shape().back() || r_in.shape(-2) == 1);
  }
  array residual = r_in;
  if (!r_no_copy) {
    residual = contiguous_copy_gpu(r_in, s);
    copies.push_back(residual);
  }

  // Output allocation — must be fresh because kernel reads x and residual
  // while writing output (in-place would cause read-after-write hazard).
  out.set_data(
      allocator::malloc(x.data_size() * x.itemsize()),
      x.data_size(),
      x.strides(),
      x.flags());

  const array& w = inputs[2];

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  std::string kname = "rms_norm_residual_" + type_to_name(out);
  auto kernel = d.get_kernel(kname);

  // Use max threadgroup size — kernel loops over elements when axis_size >
  // threadgroup
  size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
  size_t n_threads = n_rows * threadgroup_size;

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(residual, 1);
  compute_encoder.set_input_array(w, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(eps_, 4);
  compute_encoder.set_bytes(axis_size, 5);
  compute_encoder.dispatch_threads(
      MTL::Size(n_threads, 1, 1), MTL::Size(threadgroup_size, 1, 1));
  compute_encoder.add_temporaries(std::move(copies));
}

bool RMSNormResidual::is_equivalent(const Primitive& other) const {
  const RMSNormResidual& r = static_cast<const RMSNormResidual&>(other);
  return eps_ == r.eps_;
}

bool FusedGateActivation::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void FusedGateActivation::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // See RMSNormResidual::eval_gpu for the add_temporaries rationale.
  std::vector<array> copies;
  copies.reserve(1);

  // Ensure gate_up is contiguous in last dim.
  const array& gu_in = inputs[0];
  bool no_copy =
      gu_in.flags().contiguous && gu_in.strides()[gu_in.ndim() - 1] == 1;
  if (no_copy && gu_in.ndim() > 1) {
    auto st = gu_in.strides()[gu_in.ndim() - 2];
    no_copy &= (st == 0 || st == gu_in.shape().back() || gu_in.shape(-2) == 1);
  }
  array gate_up = gu_in;
  if (!no_copy) {
    gate_up = contiguous_copy_gpu(gu_in, s);
    copies.push_back(gate_up);
  }

  // Output buffer — distinct from input (output has half the size in the
  // last axis). Donation is not possible because the allocation sizes
  // differ.
  out.set_data(allocator::malloc(out.nbytes()));

  uint32_t hidden = static_cast<uint32_t>(hidden_dims_);
  int n_rows = gate_up.data_size() / (2 * hidden);

  constexpr int simd_size = 32;
  constexpr int n_reads = 4;

  // Pick single-row vs looped based on whether one threadgroup's worth of
  // threads (each handling n_reads elements) can cover hidden_dims.
  std::string variant;
  size_t threadgroup_size;
  bool use_looped = false;

  // Single-row requires ceil(hidden / n_reads) threads, rounded up to a
  // multiple of simd_size, and must fit in maxTotalThreadsPerThreadgroup.
  size_t threads_needed = (hidden + n_reads - 1) / n_reads;
  size_t simds_needed = (threads_needed + simd_size - 1) / simd_size;
  size_t single_row_tg = simd_size * simds_needed;

  std::string kname_base = "fused_gate_activation_" + std::string("") +
      type_to_name(out) + "_act" + std::to_string(activation_type_);

  // Probe a single-row kernel first; fall back to looped if TG exceeds max.
  std::string single_row_name = "fused_gate_activation_single_row_" +
      type_to_name(out) + "_act" + std::to_string(activation_type_);
  auto single_row_kernel = d.get_kernel(single_row_name);
  if (single_row_tg <= single_row_kernel->maxTotalThreadsPerThreadgroup()) {
    threadgroup_size = single_row_tg;
  } else {
    use_looped = true;
  }

  auto kernel = use_looped
      ? d.get_kernel(
            "fused_gate_activation_looped_" + type_to_name(out) + "_act" +
            std::to_string(activation_type_))
      : single_row_kernel;

  if (use_looped) {
    threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
  }

  size_t n_threads = static_cast<size_t>(n_rows) * threadgroup_size;

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(gate_up, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(hidden, 2);
  compute_encoder.dispatch_threads(
      MTL::Size(n_threads, 1, 1), MTL::Size(threadgroup_size, 1, 1));
  compute_encoder.add_temporaries(std::move(copies));
}

bool FusedGateActivation::is_equivalent(const Primitive& other) const {
  const FusedGateActivation& r = static_cast<const FusedGateActivation&>(other);
  return hidden_dims_ == r.hidden_dims_ &&
      activation_type_ == r.activation_type_;
}

bool RMSNormRoPE::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNormRoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Ensure input is contiguous in last dim
  const array& x_in = inputs[0];
  bool no_copy =
      x_in.flags().contiguous && x_in.strides()[x_in.ndim() - 1] == 1;
  if (no_copy && x_in.ndim() > 1) {
    auto st = x_in.strides()[x_in.ndim() - 2];
    no_copy &= (st == 0 || st == x_in.shape().back() || x_in.shape(-2) == 1);
  }
  array x = no_copy ? x_in : contiguous_copy_gpu(x_in, s);
  if (x.is_donatable()) {
    out.copy_shared_buffer(x);
  } else {
    out.set_data(
        allocator::malloc(x.data_size() * x.itemsize()),
        x.data_size(),
        x.strides(),
        x.flags());
  }

  const array& w = inputs[1];
  const array& inv_freqs = inputs[2];

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;
  uint32_t half_dim = axis_size / 2;

  // Thread count: one thread per rotation pair (half_dim threads per row)
  const int simd_size = 32;
  size_t threadgroup_needed = half_dim;
  size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
  size_t threadgroup_size = simd_size * simds_needed;

  std::string kname = "rms_norm_rope_" + type_to_name(out);
  auto kernel = d.get_kernel(kname);

  assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
  size_t n_threads = n_rows * threadgroup_size;

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(inv_freqs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(eps_, 4);
  compute_encoder.set_bytes(axis_size, 5);
  compute_encoder.set_bytes(offset_, 6);
  compute_encoder.set_bytes(n_heads_, 7);
  compute_encoder.set_bytes(seq_len_, 8);
  compute_encoder.dispatch_threads(
      MTL::Size(n_threads, 1, 1), MTL::Size(threadgroup_size, 1, 1));
}

bool RMSNormRoPE::is_equivalent(const Primitive& other) const {
  const RMSNormRoPE& r = static_cast<const RMSNormRoPE&>(other);
  return eps_ == r.eps_ && n_heads_ == r.n_heads_ && seq_len_ == r.seq_len_ &&
      offset_ == r.offset_;
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  auto check_input = [&s](const array& x) -> std::pair<array, bool> {
    if (x.flags().row_contiguous) {
      return {x, false};
    }
    array x_copy = contiguous_copy_gpu(x, s);
    return {x_copy, true};
  };
  bool donate_g = inputs[2].is_donatable();
  auto [x, copied] = check_input(inputs[0]);
  const array& w = inputs[1];
  auto [g, g_copied] = check_input(inputs[2]);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];

  // Check whether we had a weight
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs
  bool g_in_gx = false;
  if (x.is_donatable()) {
    gx.copy_shared_buffer(x);
  } else if (g.is_donatable()) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  if (g_copied && !g_in_gx) {
    compute_encoder.add_temporary(g);
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate the gradient accumulator gw and a temporary to store the
  // gradients before they are accumulated.
  array gw_temp =
      (has_w) ? array({n_rows, x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
      compute_encoder.add_temporary(gw_temp);
    }
  }
  gw.set_data(allocator::malloc(gw.nbytes()));

  const int simd_size = 32;
  const int n_reads = RMS_N_READS;
  const int looped_limit = RMS_LOOPED_LIMIT;
  std::string op_name = "vjp_rms";
  if (axis_size > looped_limit) {
    op_name += "_looped";
  }
  op_name += type_to_name(gx);

  std::string hash_name = op_name + ((has_w) ? "_w" : "_now");
  metal::MTLFCList func_consts = {
      {&has_w, MTL::DataType::DataTypeBool, 20},
  };

  {
    auto kernel = d.get_kernel(op_name, hash_name, func_consts);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      assert(threadgroup_size <= kernel->maxTotalThreadsPerThreadgroup());
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(g, 2);
    compute_encoder.set_output_array(gx, 3);
    compute_encoder.set_output_array(gw_temp, 4);
    compute_encoder.set_bytes(eps_, 5);
    compute_encoder.set_bytes(axis_size, 6);
    compute_encoder.set_bytes(w_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);
  }
}

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back() || x.shape(-2) == 1);
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  int simd_size = 32;
  int n_reads = 8;
  int looped_limit = 6656;
  std::string op_name = "layer_norm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
    n_reads = 4;
  }
  op_name += type_to_name(out);
  auto& compute_encoder = metal::get_command_encoder(s);
  {
    auto kernel = d.get_kernel(op_name);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      if (threadgroup_size > kernel->maxTotalThreadsPerThreadgroup()) {
        std::ostringstream msg;
        msg << "[layer_norm] Threadgroup size " << threadgroup_size
            << " is larger than the maximum allowed threadgroup size "
            << kernel->maxTotalThreadsPerThreadgroup();
        throw std::runtime_error(msg.str());
      }
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    uint32_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(b, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(eps_, 4);
    compute_encoder.set_bytes(axis_size, 5);
    compute_encoder.set_bytes(w_stride, 6);
    compute_encoder.set_bytes(b_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  auto check_input = [&s](const array& x) -> std::pair<array, bool> {
    if (x.flags().row_contiguous) {
      return {x, false};
    }
    array x_copy = contiguous_copy_gpu(x, s);
    return {x_copy, true};
  };
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[3].is_donatable();
  auto [x, copied] = check_input(inputs[0]);
  donate_x |= copied;
  const array& w = inputs[1];
  auto [g, g_copied] = check_input(inputs[3]);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

  // Check whether we had a weight
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs
  bool g_in_gx = false;
  if (donate_x) {
    gx.copy_shared_buffer(x);
  } else if (donate_g) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  if (g_copied && !g_in_gx) {
    compute_encoder.add_temporary(g);
  }

  auto axis_size = static_cast<uint32_t>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Allocate a temporary to store the gradients for w and allocate the output
  // gradient accumulators.
  array gw_temp =
      (has_w) ? array({n_rows, x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
      compute_encoder.add_temporary(gw_temp);
    }
  }
  gw.set_data(allocator::malloc(gw.nbytes()));
  gb.set_data(allocator::malloc(gb.nbytes()));

  // Finish with the gradient for b in case we had a b
  if (gb.ndim() == 1 && gb.size() == axis_size) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        g, gb, "sum", plan, {0}, compute_encoder, d, s);
  }

  int simd_size = 32;
  int n_reads = 8;
  int looped_limit = 8192;
  std::string op_name = "vjp_layer_norm";
  if (axis_size > looped_limit) {
    op_name += "_looped";
    n_reads = 4;
  }
  op_name += type_to_name(gx);

  std::string hash_name = op_name + ((has_w) ? "_w" : "_now");
  metal::MTLFCList func_consts = {
      {&has_w, MTL::DataType::DataTypeBool, 20},
  };

  {
    auto kernel = d.get_kernel(op_name, hash_name, func_consts);

    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      if (threadgroup_size > kernel->maxTotalThreadsPerThreadgroup()) {
        std::ostringstream msg;
        msg << "[vjp_layer_norm] Threadgroup size " << threadgroup_size
            << " is larger than the maximum allowed threadgroup size "
            << kernel->maxTotalThreadsPerThreadgroup();
        throw std::runtime_error(msg.str());
      }
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(w, 1);
    compute_encoder.set_input_array(g, 2);
    compute_encoder.set_output_array(gx, 3);
    compute_encoder.set_output_array(gw_temp, 4);
    compute_encoder.set_bytes(eps_, 5);
    compute_encoder.set_bytes(axis_size, 6);
    compute_encoder.set_bytes(w_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    strided_reduce_general_dispatch(
        gw_temp, gw, "sum", plan, {0}, compute_encoder, d, s);
  }
}

} // namespace mlx::core::fast
