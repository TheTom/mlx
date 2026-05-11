// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array rms_norm_residual(
    const array& x,
    const array& residual,
    const array& weight,
    float eps,
    StreamOrDevice s = {});

/// Fused dense gate+activation kernel.
///
/// Input `gate_up` has shape `[..., 2 * hidden_dims]` (concat of gate and up).
/// Output has shape `[..., hidden_dims]` and equals `activation(gate) * up`.
/// `activation_type`: 0 = silu, 1 = gelu_approx.
///
/// Collapses split + activation + multiply (4 Metal dispatches) into a
/// single dispatch. Metal-only fast path.
MLX_API array fused_gate_activation(
    const array& gate_up,
    int hidden_dims,
    int activation_type,
    StreamOrDevice s = {});

MLX_API array rms_norm_rope(
    const array& x,
    const array& weight,
    const array& inv_freqs,
    float eps,
    int offset,
    int n_heads,
    int seq_len,
    StreamOrDevice s = {});

MLX_API array rms_norm_qgemv(
    const array& x,
    const array& norm_weight,
    const array& w,
    const array& scales,
    const array& biases,
    float eps,
    int group_size,
    StreamOrDevice s = {});

MLX_API array batched_qkv_qgemv(
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
    StreamOrDevice s = {});

MLX_API array warp_moe_gate_up(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& indices,
    int group_size,
    int hidden_dims,
    int activation_type,
    StreamOrDevice s = {});

MLX_API array warp_moe_down(
    const array& activated,
    const array& w,
    const array& scales,
    const array& biases,
    const array& indices,
    const array& scores,
    int group_size,
    int hidden_dims,
    int out_dims,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {},
    int window_size = -1);

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

// ============================================================================
// TurboQuant: compressed-domain attention primitives
// ============================================================================

/// Compute Q*K scores from packed codebook-quantized keys.
MLX_API array turbo_score(
    const array& q_rot,
    const array& packed,
    const array& norms,
    const array& codebook,
    int token_count,
    int repeat_count,
    int bits,
    int dim,
    StreamOrDevice s = {});

/// Fused norm+rotate+quantize+pack (dense rotation variant).
/// Returns {packed_out, norms_out}.
MLX_API std::vector<array> turbo_encode(
    const array& input,
    const array& rotation,
    const array& boundaries,
    const array& codebook,
    int bits,
    int dim,
    StreamOrDevice s = {});

/// Fused norm+WHT+quantize+pack (Walsh-Hadamard variant).
/// Returns {packed_out, norms_out}.
MLX_API std::vector<array> turbo_encode_wht(
    const array& input,
    const array& wht_signs,
    const array& boundaries,
    int bits,
    int dim,
    StreamOrDevice s = {});

/// TurboFlash attention pass 1 (non-causal, single decode token).
/// Returns {o_partials, m_partials, l_partials}.
MLX_API std::vector<array> turbo_flash_pass1(
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
    StreamOrDevice s = {});

/// TurboFlash attention pass 1 (causal, L>1 prefill).
/// Returns {o_partials, m_partials, l_partials}.
MLX_API std::vector<array> turbo_flash_pass1_causal(
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
    StreamOrDevice s = {});

/// TurboFlash attention pass 1 NR0 (non-causal, multi-row amortized KV dequant).
/// Returns {o_partials, m_partials, l_partials}.
MLX_API std::vector<array> turbo_flash_pass1_nr0(
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
    StreamOrDevice s = {});

/// TurboFlash attention pass 1 NR0 (causal, multi-row amortized KV dequant).
/// Returns {o_partials, m_partials, l_partials}.
MLX_API std::vector<array> turbo_flash_pass1_nr0_causal(
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
    StreamOrDevice s = {});

/// TurboFlash attention pass 2: cross-block online softmax reduction.
MLX_API array turbo_flash_pass2(
    const array& o_partials,
    const array& m_partials,
    const array& l_partials,
    int num_blocks,
    int dim,
    StreamOrDevice s = {});

/// TurboFlash attention pass 2 with fused output rotation.
MLX_API array turbo_flash_pass2_fused(
    const array& o_partials,
    const array& m_partials,
    const array& l_partials,
    const array& val_rotation,
    int num_blocks,
    int dim,
    StreamOrDevice s = {});

/// Weighted sum of codebook-quantized values (V aggregation).
MLX_API array turbo_value(
    const array& weights,
    const array& packed,
    const array& norms,
    const array& codebook,
    int token_count,
    int repeat_count,
    float sparse_threshold,
    int bits,
    int dim,
    StreamOrDevice s = {});

/// Bulk-dequantize a packed [B, H, T, PackedWidth] codec buffer back to
/// FP16/BF16 [B, H, T, dim] in rotated codec space. One Metal dispatch.
///
/// `packed` must be uint32; `norms` and `codebook` must be float32.
/// `output_dtype` selects between bfloat16 and float16 outputs.
MLX_API array turbo_bulk_dequant_rotated(
    const array& packed,
    const array& norms,
    const array& codebook,
    int bits,
    int dim,
    Dtype output_dtype,
    StreamOrDevice s = {});

// ============================================================================
// GatedDeltaNet / SSM recurrence primitives
// ============================================================================

/// GatedDelta recurrence step (standard variant).
/// Returns {y [B, T, Hv, Dv], state_out [B, Hv, Dv, Dk]}.
MLX_API std::vector<array> gated_delta_step(
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
    StreamOrDevice s = {});

/// GatedDelta fused recurrence step (fused norm+gate+beta inside kernel).
/// Returns {y [B, T, Hv, Dv], state_out [B, Hv, Dv, Dk]}.
MLX_API std::vector<array> gated_delta_step_fused(
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
    StreamOrDevice s = {});

/// GatedDelta forward recurrence step with per-step `delta_t` tape capture.
/// Used by speculative-decoder verify forwards on hybrid GDN+Attention
/// models — captures innovations for possible rollback via `state_replay`.
/// Returns {y [B, T, Hv, Dv], state_out [B, Hv, Dv, Dk], delta_log [B, T, Hv, Dv]}.
MLX_API std::vector<array> gated_delta_step_record(
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
    StreamOrDevice s = {});

/// Tape-replay rollback. Re-folds the accepted prefix `[0, accepted)` of an
/// innovation tape (per-step `(delta_t, k_t, g_t)` triples) onto a
/// pre-record state snapshot. k_log carries GQA-expanded keys so the
/// stride is `Hv * Dk` (not `Hk * Dk`).
/// Returns {state_out [B, Hv, Dv, Dk]}.
MLX_API std::vector<array> state_replay(
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
    StreamOrDevice s = {});

/// SSM (Selective State Space Model) recurrence step.
/// Returns {out [N, Dh], state_out [N, Dh, Ds]}.
MLX_API std::vector<array> ssm_step(
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
    StreamOrDevice s = {});

} // namespace mlx::core::fast
