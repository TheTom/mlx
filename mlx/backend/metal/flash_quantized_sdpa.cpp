// Copyright © 2026 Eric Kryski. Flash quantized SDPA — spec 041 phase 1.1.
//
// Dispatch for `FlashQuantizedSDPA::eval_gpu`. The kernel lives in
// `kernels/flash_quantized_sdpa.h` (template) +
// `kernels/flash_quantized_sdpa.metal` (instantiations).
//
// Inputs order matches `fast.cpp::flash_quantized_sdpa(...)`:
//   0: queries
//   1: k_packed   2: k_scales   3: k_biases
//   4: v_packed   5: v_scales   6: v_biases
//   [7]: mask (optional, present iff !do_causal && has_arr_mask)
//   [next]: sinks (optional)

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void FlashQuantizedSDPA::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  const array& q = inputs[0];
  const array& k_packed = inputs[1];
  const array& k_scales = inputs[2];
  const array& k_biases = inputs[3];
  const array& v_packed = inputs[4];
  const array& v_scales = inputs[5];
  const array& v_biases = inputs[6];

  int input_cursor = 7;
  const array* mask_arr = nullptr;
  // The fast.cpp wrapper appends mask only when has_arr_mask == true, so any
  // input past `v_biases` that exists before sinks is the mask. We can detect
  // it by checking remaining input count vs `has_sinks_`.
  int remaining = static_cast<int>(inputs.size()) - input_cursor;
  bool has_array_mask = (remaining > (has_sinks_ ? 1 : 0));
  if (has_array_mask) {
    mask_arr = &inputs[input_cursor++];
  }
  const array* sinks_arr = nullptr;
  if (has_sinks_) {
    sinks_arr = &inputs[input_cursor++];
  }

  int D = q.shape(-1);
  int V = out.shape(-1);
  int gqa_factor = n_q_heads_ / n_kv_heads_;
  int N = k_packed.shape(-2);

  // Stride helpers (in elements). After `ensureRowContiguous`, strides
  // follow the canonical [B, n_kv_heads, T, *] layout, so head_stride is
  // T * row_size and seq_stride is row_size.
  int packed_per_token = D / (32 / bits_);
  int v_packed_per_token = V / (32 / bits_);
  int scale_per_token = D / group_size_;
  int v_scale_per_token = V / group_size_;

  size_t k_head_stride_packed = static_cast<size_t>(N) * packed_per_token;
  size_t k_seq_stride_packed = packed_per_token;
  size_t k_head_stride_scale = static_cast<size_t>(N) * scale_per_token;
  size_t k_seq_stride_scale = scale_per_token;
  size_t v_head_stride_packed = static_cast<size_t>(N) * v_packed_per_token;
  size_t v_seq_stride_packed = v_packed_per_token;
  size_t v_head_stride_scale = static_cast<size_t>(N) * v_scale_per_token;
  size_t v_seq_stride_scale = v_scale_per_token;

  // Function constants — match kernel-side IDs (40-46).
  bool has_mask_b = has_array_mask;
  bool query_transposed_b = !q.flags().row_contiguous;
  bool do_causal_b = do_causal_;
  bool bool_mask_b = has_array_mask && mask_arr->dtype() == bool_;
  bool float_mask_b = has_array_mask && !bool_mask_b;
  bool has_sinks_b = has_sinks_;
  // Phase 1.2: sliding-window mask. `do_sliding == true` implies the
  // kernel rejects keys at `i <= q_pos - window_size` in addition to the
  // causal upper bound.
  bool do_sliding_b = window_size_ > 0;

  metal::MTLFCList func_consts = {
      {&has_mask_b, MTL::DataType::DataTypeBool, 40},
      {&query_transposed_b, MTL::DataType::DataTypeBool, 41},
      {&do_causal_b, MTL::DataType::DataTypeBool, 42},
      {&bool_mask_b, MTL::DataType::DataTypeBool, 43},
      {&float_mask_b, MTL::DataType::DataTypeBool, 44},
      {&has_sinks_b, MTL::DataType::DataTypeBool, 45},
      {&do_sliding_b, MTL::DataType::DataTypeBool, 46},
  };

  // Kernel name: flash_quantized_sdpa_{type}_{D}_{V}_{bits}_{group_size}.
  // Use `get_type_string` (returns "float" / "float16_t" / "bfloat16_t")
  // to match the Metal-side instantiations in `flash_quantized_sdpa.metal`.
  std::string tname = mlx::core::get_type_string(q.dtype());
  std::string kname = "flash_quantized_sdpa_" + tname + "_" +
      std::to_string(D) + "_" + std::to_string(V) + "_" +
      std::to_string(bits_) + "_" + std::to_string(group_size_);
  std::string hash_name = kname +
      (has_array_mask ? (bool_mask_b ? "_boolmask" : "_floatmask")
                      : "_nomask") +
      (query_transposed_b ? "_qt" : "_qnt") + (do_causal_b ? "_c" : "_nc") +
      (has_sinks_b ? "_sinks" : "_nosinks") +
      (do_sliding_b ? "_sliding" : "_nosliding");

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Bind buffers
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k_packed, 1);
  compute_encoder.set_input_array(k_scales, 2);
  compute_encoder.set_input_array(k_biases, 3);
  compute_encoder.set_input_array(v_packed, 4);
  compute_encoder.set_input_array(v_scales, 5);
  compute_encoder.set_input_array(v_biases, 6);
  compute_encoder.set_output_array(out, 7);
  compute_encoder.set_bytes(gqa_factor, 8);
  compute_encoder.set_bytes(N, 9);
  compute_encoder.set_bytes(k_head_stride_packed, 10);
  compute_encoder.set_bytes(k_seq_stride_packed, 11);
  compute_encoder.set_bytes(k_head_stride_scale, 12);
  compute_encoder.set_bytes(k_seq_stride_scale, 13);
  compute_encoder.set_bytes(v_head_stride_packed, 14);
  compute_encoder.set_bytes(v_seq_stride_packed, 15);
  compute_encoder.set_bytes(v_head_stride_scale, 16);
  compute_encoder.set_bytes(v_seq_stride_scale, 17);
  compute_encoder.set_bytes(scale_, 18);

  if (has_array_mask) {
    compute_encoder.set_input_array(*mask_arr, 19 + (float_mask_b ? 1 : 0));
    int32_t mask_kv_seq_stride =
        mask_arr->shape(3) > 1 ? mask_arr->strides(3) : 0;
    int32_t mask_q_seq_stride =
        mask_arr->shape(2) > 1 ? mask_arr->strides(2) : 0;
    int32_t mask_head_stride = mask_arr->shape(1) > 1
        ? mask_arr->strides(1)
        : (mask_arr->shape(0) > 1 ? mask_arr->strides(0) : 0);
    compute_encoder.set_bytes(mask_kv_seq_stride, 21);
    compute_encoder.set_bytes(mask_q_seq_stride, 22);
    compute_encoder.set_bytes(mask_head_stride, 23);
  }
  if (has_sinks_b) {
    compute_encoder.set_input_array(*sinks_arr, 24);
    compute_encoder.set_bytes(n_q_heads_, 25);
  }
  if (do_sliding_b) {
    compute_encoder.set_bytes(window_size_, 26);
  }

  // Grid: (B * n_q_heads, T_q, 1). Threadgroup: (1024, 1, 1) = 32 simdgroups.
  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(q.shape(0) * q.shape(1), q.shape(2), 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

bool FlashQuantizedSDPA::is_equivalent(const Primitive& other) const {
  const FlashQuantizedSDPA& o = static_cast<const FlashQuantizedSDPA&>(other);
  return scale_ == o.scale_ && do_causal_ == o.do_causal_ &&
      has_sinks_ == o.has_sinks_ && bits_ == o.bits_ &&
      group_size_ == o.group_size_ && n_q_heads_ == o.n_q_heads_ &&
      n_kv_heads_ == o.n_kv_heads_ && window_size_ == o.window_size_;
}

std::vector<Shape> FlashQuantizedSDPA::output_shapes(
    const std::vector<array>& inputs) {
  const array& q = inputs[0];
  const array& v_scales = inputs[5];
  int V = v_scales.shape(-1) * group_size_;
  return {{q.shape(0), q.shape(1), q.shape(2), V}};
}

} // namespace mlx::core::fast
