// Copyright (C) 2026 Eric Kryski.
// TurboQuant Metal dispatch layer for compressed-domain attention kernels.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

namespace {

// PackedWidth = ceil(Dim * Bits / 32)
inline int packed_width(int dim, int bits) {
  return (dim * bits + 31) / 32;
}

// Ensure array is row-contiguous; copy if not. The copy (if any) is appended
// to `copies` so the caller can pass them to `add_temporaries` and keep the
// underlying MTL buffers alive until the command buffer completes. Without
// that registration the temporary `array::Data` shared_ptr drops at the end
// of `eval_gpu` and `BufferCache::reuse_from_cache` can hand the buffer to
// another allocation while the CB is still pending → "Invalid Resource"
// (or, depending on which thread observes it first, the Swift-level
// `MLXArray deallocated with non-zero retain count` warning).
//
// Mirrors the SDPA pattern in `scaled_dot_product_attention.cpp` (search for
// `add_temporaries(std::move(copies))`).
inline const array&
ensure_contiguous(const array& x, const Stream& s, std::vector<array>& copies) {
  if (x.flags().row_contiguous) {
    return x;
  }
  copies.push_back(contiguous_copy_gpu(x, s));
  return copies.back();
}

} // namespace

// ============================================================================
// TurboScore
// ============================================================================

void TurboScore::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  const auto& q_rot = ensure_contiguous(inputs[0], s, copies);
  const auto& packed = ensure_contiguous(inputs[1], s, copies);
  const auto& norms = ensure_contiguous(inputs[2], s, copies);
  const auto& codebook = ensure_contiguous(inputs[3], s, copies);

  out.set_data(allocator::malloc(out.nbytes()));

  int total_q = q_rot.shape(0);
  int token_count = static_cast<int>(packed.shape(1));
  int repeat_count = total_q / static_cast<int>(packed.shape(0));

  std::string kname =
      "turbo_score_" + std::to_string(bits_) + "_" + std::to_string(dim_);
  auto kernel = d.get_kernel(kname);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(q_rot, 0);
  compute_encoder.set_input_array(packed, 1);
  compute_encoder.set_input_array(norms, 2);
  compute_encoder.set_input_array(codebook, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(token_count, 5);
  compute_encoder.set_bytes(repeat_count, 6);

  auto grid = MTL::Size(32, total_q, token_count);
  auto group = MTL::Size(32, 1, 1);
  compute_encoder.dispatch_threads(grid, group);
  compute_encoder.add_temporaries(std::move(copies));
}

bool TurboScore::is_equivalent(const Primitive& other) const {
  const TurboScore& o = static_cast<const TurboScore&>(other);
  return bits_ == o.bits_ && dim_ == o.dim_;
}

// ============================================================================
// TurboEncode
// ============================================================================

void TurboEncode::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& packed_out = outputs[0];
  auto& norms_out = outputs[1];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  const auto& input = ensure_contiguous(inputs[0], s, copies);
  int num_rows = input.shape(0);

  // Allocate outputs — fresh buffers always.
  // No zero-fill needed: the kernel initializes threadgroup shared_packed to 0
  // internally, then writes the result to device memory.
  packed_out.set_data(allocator::malloc(packed_out.nbytes()));
  norms_out.set_data(allocator::malloc(norms_out.nbytes()));

  std::string kname;
  if (use_wht_) {
    // WHT variant: inputs are [input, wht_signs, boundaries]
    const auto& wht_signs = ensure_contiguous(inputs[1], s, copies);
    const auto& boundaries = ensure_contiguous(inputs[2], s, copies);

    kname = "turbo_fused_encode_wht_" + std::to_string(bits_) + "_" +
        std::to_string(dim_);
    auto kernel = d.get_kernel(kname);

    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(input, 0);
    compute_encoder.set_input_array(wht_signs, 1);
    compute_encoder.set_input_array(boundaries, 2);
    compute_encoder.set_output_array(packed_out, 3);
    compute_encoder.set_output_array(norms_out, 4);

    // Kernel uses thread_position_in_threadgroup (d) and
    // threadgroup_position_in_grid (row), so dispatch by threadgroups.
    auto grid = MTL::Size(num_rows, 1, 1);
    auto group = MTL::Size(dim_, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, group);
    compute_encoder.add_temporaries(std::move(copies));
  } else {
    // Dense rotation variant: inputs are [input, rotation, boundaries,
    // codebook]
    const auto& rotation = ensure_contiguous(inputs[1], s, copies);
    const auto& boundaries = ensure_contiguous(inputs[2], s, copies);
    const auto& codebook = ensure_contiguous(inputs[3], s, copies);

    kname = "turbo_fused_encode_" + std::to_string(bits_) + "_" +
        std::to_string(dim_);
    auto kernel = d.get_kernel(kname);

    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(input, 0);
    compute_encoder.set_input_array(rotation, 1);
    compute_encoder.set_input_array(boundaries, 2);
    compute_encoder.set_input_array(codebook, 3);
    compute_encoder.set_output_array(packed_out, 4);
    compute_encoder.set_output_array(norms_out, 5);

    auto grid = MTL::Size(num_rows, 1, 1);
    auto group = MTL::Size(dim_, 1, 1);
    compute_encoder.dispatch_threadgroups(grid, group);
    compute_encoder.add_temporaries(std::move(copies));
  }
}

bool TurboEncode::is_equivalent(const Primitive& other) const {
  const TurboEncode& o = static_cast<const TurboEncode&>(other);
  return bits_ == o.bits_ && dim_ == o.dim_ && use_wht_ == o.use_wht_;
}

std::vector<Shape> TurboEncode::output_shapes(
    const std::vector<array>& inputs) {
  int num_rows = inputs[0].shape(0);
  int pw = packed_width(dim_, bits_);
  return {{num_rows, pw}, {num_rows}};
}

// ============================================================================
// TurboFlashPass1
// ============================================================================

void TurboFlashPass1::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& o_partials = outputs[0];
  auto& m_partials = outputs[1];
  auto& l_partials = outputs[2];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  const auto& q_rot = ensure_contiguous(inputs[0], s, copies);
  const auto& key_packed = ensure_contiguous(inputs[1], s, copies);
  const auto& key_norms = ensure_contiguous(inputs[2], s, copies);
  const auto& key_codebook = ensure_contiguous(inputs[3], s, copies);
  const auto& val_packed = ensure_contiguous(inputs[4], s, copies);
  const auto& val_norms = ensure_contiguous(inputs[5], s, copies);
  const auto& val_codebook = ensure_contiguous(inputs[6], s, copies);

  int total_q = q_rot.shape(0);

  // Constants passed via inputs[7..] as scalar arrays
  int token_count = inputs[7].item<int>();
  int repeat_count = inputs[8].item<int>();
  int num_blocks = inputs[9].item<int>();
  int block_size = inputs[10].item<int>();

  // Allocate outputs
  o_partials.set_data(allocator::malloc(o_partials.nbytes()));
  m_partials.set_data(allocator::malloc(m_partials.nbytes()));
  l_partials.set_data(allocator::malloc(l_partials.nbytes()));

  std::string kname;
  if (causal_) {
    kname = "turbo_flash_p1_causal_" + std::to_string(key_bits_) + "_" +
        std::to_string(value_bits_) + "_" + std::to_string(dim_);
  } else {
    kname = "turbo_flash_p1_" + std::to_string(key_bits_) + "_" +
        std::to_string(value_bits_) + "_" + std::to_string(dim_);
  }
  auto kernel = d.get_kernel(kname);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(q_rot, 0);
  compute_encoder.set_input_array(key_packed, 1);
  compute_encoder.set_input_array(key_norms, 2);
  compute_encoder.set_input_array(key_codebook, 3);
  compute_encoder.set_input_array(val_packed, 4);
  compute_encoder.set_input_array(val_norms, 5);
  compute_encoder.set_input_array(val_codebook, 6);
  compute_encoder.set_output_array(o_partials, 7);
  compute_encoder.set_output_array(m_partials, 8);
  compute_encoder.set_output_array(l_partials, 9);
  compute_encoder.set_bytes(token_count, 10);
  compute_encoder.set_bytes(repeat_count, 11);
  compute_encoder.set_bytes(num_blocks, 12);
  compute_encoder.set_bytes(block_size, 13);

  if (causal_) {
    int L = inputs[11].item<int>();
    int q_offset = inputs[12].item<int>();
    compute_encoder.set_bytes(L, 14);
    compute_encoder.set_bytes(q_offset, 15);
  }

  auto grid = MTL::Size(32, total_q, num_blocks);
  auto group = MTL::Size(32, 1, 1);
  compute_encoder.dispatch_threads(grid, group);
  compute_encoder.add_temporaries(std::move(copies));
}

bool TurboFlashPass1::is_equivalent(const Primitive& other) const {
  const TurboFlashPass1& o = static_cast<const TurboFlashPass1&>(other);
  return key_bits_ == o.key_bits_ && value_bits_ == o.value_bits_ &&
      dim_ == o.dim_ && causal_ == o.causal_;
}

std::vector<Shape> TurboFlashPass1::output_shapes(
    const std::vector<array>& inputs) {
  int total_q = inputs[0].shape(0);
  int num_blocks = inputs[9].item<int>();
  return {
      {total_q * num_blocks, dim_},
      {total_q, num_blocks},
      {total_q, num_blocks}};
}

// ============================================================================
// TurboFlashPass1NR0
// ============================================================================

void TurboFlashPass1NR0::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& o_partials = outputs[0];
  auto& m_partials = outputs[1];
  auto& l_partials = outputs[2];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  const auto& q_rot = ensure_contiguous(inputs[0], s, copies);
  const auto& key_packed = ensure_contiguous(inputs[1], s, copies);
  const auto& key_norms = ensure_contiguous(inputs[2], s, copies);
  const auto& key_codebook = ensure_contiguous(inputs[3], s, copies);
  const auto& val_packed = ensure_contiguous(inputs[4], s, copies);
  const auto& val_norms = ensure_contiguous(inputs[5], s, copies);
  const auto& val_codebook = ensure_contiguous(inputs[6], s, copies);

  int total_q = q_rot.shape(0);

  // Constants passed via inputs[7..] as scalar arrays
  int token_count = inputs[7].item<int>();
  int repeat_count = inputs[8].item<int>();
  int num_blocks = inputs[9].item<int>();
  int block_size = inputs[10].item<int>();

  // Allocate outputs
  o_partials.set_data(allocator::malloc(o_partials.nbytes()));
  m_partials.set_data(allocator::malloc(m_partials.nbytes()));
  l_partials.set_data(allocator::malloc(l_partials.nbytes()));

  std::string kname;
  if (causal_) {
    kname = "turbo_flash_p1_nr0_causal_" + std::to_string(key_bits_) + "_" +
        std::to_string(value_bits_) + "_" + std::to_string(dim_) + "_" +
        std::to_string(nr0_);
  } else {
    kname = "turbo_flash_p1_nr0_" + std::to_string(key_bits_) + "_" +
        std::to_string(value_bits_) + "_" + std::to_string(dim_) + "_" +
        std::to_string(nr0_);
  }
  auto kernel = d.get_kernel(kname);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(q_rot, 0);
  compute_encoder.set_input_array(key_packed, 1);
  compute_encoder.set_input_array(key_norms, 2);
  compute_encoder.set_input_array(key_codebook, 3);
  compute_encoder.set_input_array(val_packed, 4);
  compute_encoder.set_input_array(val_norms, 5);
  compute_encoder.set_input_array(val_codebook, 6);
  compute_encoder.set_output_array(o_partials, 7);
  compute_encoder.set_output_array(m_partials, 8);
  compute_encoder.set_output_array(l_partials, 9);
  compute_encoder.set_bytes(token_count, 10);
  compute_encoder.set_bytes(repeat_count, 11);
  compute_encoder.set_bytes(num_blocks, 12);
  compute_encoder.set_bytes(block_size, 13);

  if (causal_) {
    int L = inputs[11].item<int>();
    int q_offset = inputs[12].item<int>();
    compute_encoder.set_bytes(L, 14);
    compute_encoder.set_bytes(q_offset, 15);
  }

  // Grid Y: totalQ / nr0 (each threadgroup processes NR0 queries)
  auto grid = MTL::Size(32, total_q / nr0_, num_blocks);
  auto group = MTL::Size(32, 1, 1);
  compute_encoder.dispatch_threads(grid, group);
  compute_encoder.add_temporaries(std::move(copies));
}

bool TurboFlashPass1NR0::is_equivalent(const Primitive& other) const {
  const TurboFlashPass1NR0& o = static_cast<const TurboFlashPass1NR0&>(other);
  return key_bits_ == o.key_bits_ && value_bits_ == o.value_bits_ &&
      dim_ == o.dim_ && nr0_ == o.nr0_ && causal_ == o.causal_;
}

std::vector<Shape> TurboFlashPass1NR0::output_shapes(
    const std::vector<array>& inputs) {
  int total_q = inputs[0].shape(0);
  int num_blocks = inputs[9].item<int>();
  return {
      {total_q * num_blocks, dim_},
      {total_q, num_blocks},
      {total_q, num_blocks}};
}

// ============================================================================
// TurboFlashPass2
// ============================================================================

void TurboFlashPass2::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  const auto& o_partials = ensure_contiguous(inputs[0], s, copies);
  const auto& m_partials = ensure_contiguous(inputs[1], s, copies);
  const auto& l_partials = ensure_contiguous(inputs[2], s, copies);

  int total_q = m_partials.shape(0);
  int num_blocks = m_partials.shape(1);

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname;
  if (fused_rotation_) {
    const auto& val_rotation = ensure_contiguous(inputs[3], s, copies);

    kname = "turbo_flash_p2_fused_" + std::to_string(dim_);
    auto kernel = d.get_kernel(kname);

    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(o_partials, 0);
    compute_encoder.set_input_array(m_partials, 1);
    compute_encoder.set_input_array(l_partials, 2);
    compute_encoder.set_input_array(val_rotation, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(num_blocks, 5);

    auto grid = MTL::Size(32, total_q, 1);
    auto group = MTL::Size(32, 1, 1);
    compute_encoder.dispatch_threads(grid, group);
    compute_encoder.add_temporaries(std::move(copies));
  } else {
    kname = "turbo_flash_p2_" + std::to_string(dim_);
    auto kernel = d.get_kernel(kname);

    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(o_partials, 0);
    compute_encoder.set_input_array(m_partials, 1);
    compute_encoder.set_input_array(l_partials, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(num_blocks, 4);

    auto grid = MTL::Size(32, total_q, 1);
    auto group = MTL::Size(32, 1, 1);
    compute_encoder.dispatch_threads(grid, group);
    compute_encoder.add_temporaries(std::move(copies));
  }
}

bool TurboFlashPass2::is_equivalent(const Primitive& other) const {
  const TurboFlashPass2& o = static_cast<const TurboFlashPass2&>(other);
  return dim_ == o.dim_ && fused_rotation_ == o.fused_rotation_;
}

// ============================================================================
// TurboValue
// ============================================================================

void TurboValue::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];

  std::vector<array> copies;
  copies.reserve(inputs.size());
  const auto& weights = ensure_contiguous(inputs[0], s, copies);
  const auto& packed = ensure_contiguous(inputs[1], s, copies);
  const auto& norms = ensure_contiguous(inputs[2], s, copies);
  const auto& codebook = ensure_contiguous(inputs[3], s, copies);

  int total_heads = weights.shape(0);
  int token_count = weights.shape(1);
  int repeat_count = total_heads / static_cast<int>(packed.shape(0));

  // Read sparse_threshold from inputs[4] (scalar float array)
  float sparse_threshold = inputs[4].item<float>();

  out.set_data(allocator::malloc(out.nbytes()));

  std::string kname =
      "turbo_value_" + std::to_string(bits_) + "_" + std::to_string(dim_);
  auto kernel = d.get_kernel(kname);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(weights, 0);
  compute_encoder.set_input_array(packed, 1);
  compute_encoder.set_input_array(norms, 2);
  compute_encoder.set_input_array(codebook, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(token_count, 5);
  compute_encoder.set_bytes(repeat_count, 6);
  compute_encoder.set_bytes(sparse_threshold, 7);

  int dim_blocks = (dim_ + 31) / 32;
  auto grid = MTL::Size(32, total_heads, dim_blocks);
  auto group = MTL::Size(32, 1, 1);
  compute_encoder.dispatch_threads(grid, group);
  compute_encoder.add_temporaries(std::move(copies));
}

bool TurboValue::is_equivalent(const Primitive& other) const {
  const TurboValue& o = static_cast<const TurboValue&>(other);
  return bits_ == o.bits_ && dim_ == o.dim_;
}

} // namespace mlx::core::fast
