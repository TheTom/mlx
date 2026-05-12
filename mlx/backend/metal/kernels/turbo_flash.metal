// Copyright © 2026 Eric Kryski. TurboFlash Pass 1 kernels for compressed-domain
// attention.
//
// These are the hottest TurboQuant kernels — run every decode token.
// Runtime-varying params (token_count, num_blocks, repeat_count, BlockSize)
// are buffer arguments to avoid per-token pipeline recompilation.

#include <metal_common>
#include <metal_simdgroup>

#include "utils.h"

using namespace metal;

// ============================================================================
// TurboFlash Pass 1: Standard (non-causal, L=1 decode)
// ============================================================================
template <
    int KeyBits,
    int ValueBits,
    int Dim,
    int KeyPackedWidth,
    int ValuePackedWidth>
[[kernel]] void turbo_flash_p1(
    const device float* q_rot [[buffer(0)]],
    const device uint32_t* key_packed [[buffer(1)]],
    const device float* key_norms [[buffer(2)]],
    const device float* key_codebook [[buffer(3)]],
    const device uint32_t* val_packed [[buffer(4)]],
    const device float* val_norms [[buffer(5)]],
    const device float* val_codebook [[buffer(6)]],
    device float* o_partials [[buffer(7)]],
    device float* m_partials [[buffer(8)]],
    device float* l_partials [[buffer(9)]],
    constant int& token_count [[buffer(10)]],
    constant int& repeat_count [[buffer(11)]],
    constant int& num_blocks [[buffer(12)]],
    constant int& block_size [[buffer(13)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
  constexpr uint KEY_LEVELS = 1u << KeyBits;
  constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
  constexpr uint VAL_LEVELS = 1u << ValueBits;
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

  uint lane = pos.x;
  uint q_idx = pos.y;
  uint block_idx = pos.z;
  uint kv_idx = q_idx / uint(repeat_count);

  uint t_start = block_idx * uint(block_size);
  uint t_end = t_start + uint(block_size);
  if (t_end > uint(token_count))
    t_end = uint(token_count);

  float key_cb[KEY_LEVELS];
  for (uint i = 0; i < KEY_LEVELS; i++)
    key_cb[i] = key_codebook[i];
  float val_cb[VAL_LEVELS];
  for (uint i = 0; i < VAL_LEVELS; i++)
    val_cb[i] = val_codebook[i];

  float q_vals[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    q_vals[i] = (d < uint(Dim)) ? q_rot[q_idx * Dim + d] : 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  float o[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++)
    o[i] = 0.0f;

  for (uint t = t_start; t < t_end; t++) {
    const device uint32_t* k_packed_ptr = key_packed +
        kv_idx * uint(token_count) * KeyPackedWidth + t * KeyPackedWidth;
    float k_norm = key_norms[kv_idx * uint(token_count) + t];

    float dot_partial = 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim))
        break;
      uint k_bit_offset = d * KeyBits;
      uint k_word_idx = k_bit_offset / 32;
      uint k_shift = k_bit_offset % 32;
      uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
      int k_spill = (int)k_shift + (int)KeyBits - 32;
      if (k_spill > 0)
        k_value |=
            (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
      k_value &= KEY_MASK;
      dot_partial += q_vals[i] * key_cb[k_value];
    }
    float score = simd_sum(dot_partial) * k_norm;

    float new_m = max(m, score);
    float exp_diff = exp(m - new_m);
    float exp_score = exp(score - new_m);

    const device uint32_t* v_packed_ptr = val_packed +
        kv_idx * uint(token_count) * ValuePackedWidth + t * ValuePackedWidth;
    float v_norm = val_norms[kv_idx * uint(token_count) + t];

    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim))
        break;
      uint v_bit_offset = d * ValueBits;
      uint v_word_idx = v_bit_offset / 32;
      uint v_shift = v_bit_offset % 32;
      uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
      int v_spill = (int)v_shift + (int)ValueBits - 32;
      if (v_spill > 0)
        v_value |=
            (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
      v_value &= VAL_MASK;
      o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
    }
    l = l * exp_diff + exp_score;
    m = new_m;
  }

  uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    if (d < uint(Dim))
      o_partials[partial_base + d] = o[i];
  }
  if (lane == 0) {
    uint ml_idx = q_idx * uint(num_blocks) + block_idx;
    m_partials[ml_idx] = m;
    l_partials[ml_idx] = l;
  }
}

// ============================================================================
// TurboFlash Pass 1 Causal: With per-query causal masking for L>1 prefill
// ============================================================================
template <
    int KeyBits,
    int ValueBits,
    int Dim,
    int KeyPackedWidth,
    int ValuePackedWidth>
[[kernel]] void turbo_flash_p1_causal(
    const device float* q_rot [[buffer(0)]],
    const device uint32_t* key_packed [[buffer(1)]],
    const device float* key_norms [[buffer(2)]],
    const device float* key_codebook [[buffer(3)]],
    const device uint32_t* val_packed [[buffer(4)]],
    const device float* val_norms [[buffer(5)]],
    const device float* val_codebook [[buffer(6)]],
    device float* o_partials [[buffer(7)]],
    device float* m_partials [[buffer(8)]],
    device float* l_partials [[buffer(9)]],
    constant int& token_count [[buffer(10)]],
    constant int& repeat_count [[buffer(11)]],
    constant int& num_blocks [[buffer(12)]],
    constant int& block_size [[buffer(13)]],
    constant int& L [[buffer(14)]],
    constant int& q_offset [[buffer(15)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
  constexpr uint KEY_LEVELS = 1u << KeyBits;
  constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
  constexpr uint VAL_LEVELS = 1u << ValueBits;
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

  uint lane = pos.x;
  uint q_idx = pos.y;
  uint block_idx = pos.z;

  uint q_within_L = q_idx % uint(L);
  uint q_head_idx = q_idx / uint(L);
  uint kv_idx = q_head_idx / uint(repeat_count);
  uint q_abs = uint(q_offset) + q_within_L;

  uint t_start = block_idx * uint(block_size);
  uint t_end = t_start + uint(block_size);
  if (t_end > uint(token_count))
    t_end = uint(token_count);

  // Early exit: entire block is future-masked
  if (t_start > q_abs) {
    uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d < uint(Dim))
        o_partials[partial_base + d] = 0.0f;
    }
    if (lane == 0) {
      uint ml_idx = q_idx * uint(num_blocks) + block_idx;
      m_partials[ml_idx] = -INFINITY;
      l_partials[ml_idx] = 0.0f;
    }
    return;
  }
  if (t_end > q_abs + 1)
    t_end = q_abs + 1;

  float key_cb[KEY_LEVELS];
  for (uint i = 0; i < KEY_LEVELS; i++)
    key_cb[i] = key_codebook[i];
  float val_cb[VAL_LEVELS];
  for (uint i = 0; i < VAL_LEVELS; i++)
    val_cb[i] = val_codebook[i];

  float q_vals[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    q_vals[i] = (d < uint(Dim)) ? q_rot[q_idx * Dim + d] : 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  float o[DIMS_PER_LANE];
  for (uint i = 0; i < DIMS_PER_LANE; i++)
    o[i] = 0.0f;

  for (uint t = t_start; t < t_end; t++) {
    const device uint32_t* k_packed_ptr = key_packed +
        kv_idx * uint(token_count) * KeyPackedWidth + t * KeyPackedWidth;
    float k_norm = key_norms[kv_idx * uint(token_count) + t];

    float dot_partial = 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim))
        break;
      uint k_bit_offset = d * KeyBits;
      uint k_word_idx = k_bit_offset / 32;
      uint k_shift = k_bit_offset % 32;
      uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
      int k_spill = (int)k_shift + (int)KeyBits - 32;
      if (k_spill > 0)
        k_value |=
            (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
      k_value &= KEY_MASK;
      dot_partial += q_vals[i] * key_cb[k_value];
    }
    float score = simd_sum(dot_partial) * k_norm;

    float new_m = max(m, score);
    float exp_diff = exp(m - new_m);
    float exp_score = exp(score - new_m);

    const device uint32_t* v_packed_ptr = val_packed +
        kv_idx * uint(token_count) * ValuePackedWidth + t * ValuePackedWidth;
    float v_norm = val_norms[kv_idx * uint(token_count) + t];

    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= uint(Dim))
        break;
      uint v_bit_offset = d * ValueBits;
      uint v_word_idx = v_bit_offset / 32;
      uint v_shift = v_bit_offset % 32;
      uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
      int v_spill = (int)v_shift + (int)ValueBits - 32;
      if (v_spill > 0)
        v_value |=
            (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
      v_value &= VAL_MASK;
      o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
    }
    l = l * exp_diff + exp_score;
    m = new_m;
  }

  uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
  for (uint i = 0; i < DIMS_PER_LANE; i++) {
    uint d = lane + i * 32;
    if (d < uint(Dim))
      o_partials[partial_base + d] = o[i];
  }
  if (lane == 0) {
    uint ml_idx = q_idx * uint(num_blocks) + block_idx;
    m_partials[ml_idx] = m;
    l_partials[ml_idx] = l;
  }
}

// ============================================================================
// TurboFlash Pass 1 NR0: Multi-row amortized KV dequant (non-causal)
// ============================================================================
template <
    int KeyBits,
    int ValueBits,
    int Dim,
    int KeyPackedWidth,
    int ValuePackedWidth,
    int NR0>
[[kernel]] void turbo_flash_p1_nr0(
    const device float* q_rot [[buffer(0)]],
    const device uint32_t* key_packed [[buffer(1)]],
    const device float* key_norms [[buffer(2)]],
    const device float* key_codebook [[buffer(3)]],
    const device uint32_t* val_packed [[buffer(4)]],
    const device float* val_norms [[buffer(5)]],
    const device float* val_codebook [[buffer(6)]],
    device float* o_partials [[buffer(7)]],
    device float* m_partials [[buffer(8)]],
    device float* l_partials [[buffer(9)]],
    constant int& token_count [[buffer(10)]],
    constant int& repeat_count [[buffer(11)]],
    constant int& num_blocks [[buffer(12)]],
    constant int& block_size [[buffer(13)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
  constexpr uint KEY_LEVELS = 1u << KeyBits;
  constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
  constexpr uint VAL_LEVELS = 1u << ValueBits;
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

  uint lane = pos.x; // SIMD lane (0-31)
  uint query_group = pos.y; // which group of NR0 queries
  uint block_idx = pos.z; // which KV block

  // Token range for this block
  uint t_start = block_idx * uint(block_size);
  uint t_end = t_start + uint(block_size);
  if (t_end > uint(token_count))
    t_end = uint(token_count);

  // Load key codebook into registers (shared across all NR0 queries)
  float key_cb[KEY_LEVELS];
  for (uint i = 0; i < KEY_LEVELS; i++) {
    key_cb[i] = key_codebook[i];
  }

  // Load value codebook into registers (shared across all NR0 queries)
  float val_cb[VAL_LEVELS];
  for (uint i = 0; i < VAL_LEVELS; i++) {
    val_cb[i] = val_codebook[i];
  }

  // Load query values for ALL NR0 rows — each row's dims interleaved in
  // registers
  float q_vals[NR0 * DIMS_PER_LANE];
  for (uint r = 0; r < NR0; r++) {
    uint q_idx = query_group * NR0 + r;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      q_vals[r * DIMS_PER_LANE + i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
    }
  }

  // Per-query KV head mapping (for GQA — each query may map to different KV
  // head)
  uint kv_indices[NR0];
  for (uint r = 0; r < NR0; r++) {
    kv_indices[r] = (query_group * NR0 + r) / uint(repeat_count);
  }

  // Online softmax state — NR0 independent streams, all in registers
  float m_state[NR0];
  float l_state[NR0];
  float o_state[NR0 * DIMS_PER_LANE];
  for (uint r = 0; r < NR0; r++) {
    m_state[r] = -INFINITY;
    l_state[r] = 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      o_state[r * DIMS_PER_LANE + i] = 0.0f;
    }
  }

  // Process tokens in this block — KV dequant done ONCE, reused across NR0
  // queries
  for (uint t = t_start; t < t_end; t++) {
    // --- Dequant K for this token ONCE (amortized across NR0 queries) ---
    float k_decoded[DIMS_PER_LANE];
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= Dim) {
        k_decoded[i] = 0.0f;
        continue;
      }

      uint k_bit_offset = d * KeyBits;
      uint k_word_idx = k_bit_offset / 32;
      uint k_shift = k_bit_offset % 32;

      const device uint32_t* k_packed_ptr = key_packed +
          kv_indices[0] * uint(token_count) * KeyPackedWidth +
          t * KeyPackedWidth;

      uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
      int k_spill = (int)k_shift + (int)KeyBits - 32;
      if (k_spill > 0) {
        k_value |=
            (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
      }
      k_value &= KEY_MASK;
      k_decoded[i] = key_cb[k_value];
    }
    float k_norm = key_norms[kv_indices[0] * uint(token_count) + t];

    // --- Dequant V for this token ONCE ---
    float v_decoded[DIMS_PER_LANE];
    const device uint32_t* v_packed_ptr = val_packed +
        kv_indices[0] * uint(token_count) * ValuePackedWidth +
        t * ValuePackedWidth;
    float v_norm = val_norms[kv_indices[0] * uint(token_count) + t];
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= Dim) {
        v_decoded[i] = 0.0f;
        continue;
      }

      uint v_bit_offset = d * ValueBits;
      uint v_word_idx = v_bit_offset / 32;
      uint v_shift = v_bit_offset % 32;
      uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
      int v_spill = (int)v_shift + (int)ValueBits - 32;
      if (v_spill > 0) {
        v_value |=
            (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
      }
      v_value &= VAL_MASK;
      v_decoded[i] = val_cb[v_value] * v_norm;
    }

    // --- Score + softmax + V accumulate for each of NR0 queries ---
    for (uint r = 0; r < NR0; r++) {
      float dot_partial = 0.0f;
      for (uint i = 0; i < DIMS_PER_LANE; i++) {
        dot_partial += q_vals[r * DIMS_PER_LANE + i] * k_decoded[i];
      }
      float score = simd_sum(dot_partial) * k_norm;

      float new_m = max(m_state[r], score);
      float exp_diff = exp(m_state[r] - new_m);
      float exp_score = exp(score - new_m);

      for (uint i = 0; i < DIMS_PER_LANE; i++) {
        o_state[r * DIMS_PER_LANE + i] =
            o_state[r * DIMS_PER_LANE + i] * exp_diff +
            exp_score * v_decoded[i];
      }

      l_state[r] = l_state[r] * exp_diff + exp_score;
      m_state[r] = new_m;
    }
  }

  // Write partial results for all NR0 queries
  for (uint r = 0; r < NR0; r++) {
    uint q_idx = query_group * NR0 + r;
    uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d < Dim) {
        o_partials[partial_base + d] = o_state[r * DIMS_PER_LANE + i];
      }
    }
    if (lane == 0) {
      uint ml_idx = q_idx * uint(num_blocks) + block_idx;
      m_partials[ml_idx] = m_state[r];
      l_partials[ml_idx] = l_state[r];
    }
  }
}

// ============================================================================
// TurboFlash Pass 1 NR0 Causal: Multi-row with per-query causal masking
// ============================================================================
template <
    int KeyBits,
    int ValueBits,
    int Dim,
    int KeyPackedWidth,
    int ValuePackedWidth,
    int NR0>
[[kernel]] void turbo_flash_p1_nr0_causal(
    const device float* q_rot [[buffer(0)]],
    const device uint32_t* key_packed [[buffer(1)]],
    const device float* key_norms [[buffer(2)]],
    const device float* key_codebook [[buffer(3)]],
    const device uint32_t* val_packed [[buffer(4)]],
    const device float* val_norms [[buffer(5)]],
    const device float* val_codebook [[buffer(6)]],
    device float* o_partials [[buffer(7)]],
    device float* m_partials [[buffer(8)]],
    device float* l_partials [[buffer(9)]],
    constant int& token_count [[buffer(10)]],
    constant int& repeat_count [[buffer(11)]],
    constant int& num_blocks [[buffer(12)]],
    constant int& block_size [[buffer(13)]],
    constant int& L [[buffer(14)]],
    constant int& q_offset [[buffer(15)]],
    uint3 pos [[thread_position_in_grid]]) {
  constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
  constexpr uint KEY_LEVELS = 1u << KeyBits;
  constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
  constexpr uint VAL_LEVELS = 1u << ValueBits;
  constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

  uint lane = pos.x;
  uint query_group = pos.y;
  uint block_idx = pos.z;

  // Token range for this block
  uint t_start = block_idx * uint(block_size);
  uint t_end = t_start + uint(block_size);
  if (t_end > uint(token_count))
    t_end = uint(token_count);

  // Compute per-row causal boundaries and find the maximum (most permissive)
  uint q_abs[NR0];
  uint max_q_abs = 0;
  for (uint r = 0; r < NR0; r++) {
    uint q_idx = query_group * NR0 + r;
    uint q_within_L = q_idx % uint(L);
    q_abs[r] = uint(q_offset) + q_within_L;
    if (q_abs[r] > max_q_abs)
      max_q_abs = q_abs[r];
  }

  // Early exit: entire block is future-masked for ALL NR0 queries
  if (t_start > max_q_abs) {
    for (uint r = 0; r < NR0; r++) {
      uint q_idx = query_group * NR0 + r;
      uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
      for (uint i = 0; i < DIMS_PER_LANE; i++) {
        uint d = lane + i * 32;
        if (d < Dim)
          o_partials[partial_base + d] = 0.0f;
      }
      if (lane == 0) {
        uint ml_idx = q_idx * uint(num_blocks) + block_idx;
        m_partials[ml_idx] = -INFINITY;
        l_partials[ml_idx] = 0.0f;
      }
    }
    return;
  }

  // Clamp t_end to the most permissive causal boundary
  if (t_end > max_q_abs + 1)
    t_end = max_q_abs + 1;

  // Load codebooks (shared across all NR0 queries)
  float key_cb[KEY_LEVELS];
  for (uint i = 0; i < KEY_LEVELS; i++)
    key_cb[i] = key_codebook[i];
  float val_cb[VAL_LEVELS];
  for (uint i = 0; i < VAL_LEVELS; i++)
    val_cb[i] = val_codebook[i];

  // Load query values for all NR0 rows
  float q_vals[NR0 * DIMS_PER_LANE];
  for (uint r = 0; r < NR0; r++) {
    uint q_idx = query_group * NR0 + r;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      q_vals[r * DIMS_PER_LANE + i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
    }
  }

  // KV head mapping (use first query's head — same assumption as non-causal
  // NR0)
  uint q_head_idx_0 = (query_group * NR0) / uint(L);
  uint kv_idx = q_head_idx_0 / uint(repeat_count);

  // Online softmax state — NR0 independent streams
  float m_state[NR0];
  float l_state[NR0];
  float o_state[NR0 * DIMS_PER_LANE];
  for (uint r = 0; r < NR0; r++) {
    m_state[r] = -INFINITY;
    l_state[r] = 0.0f;
    for (uint i = 0; i < DIMS_PER_LANE; i++)
      o_state[r * DIMS_PER_LANE + i] = 0.0f;
  }

  // Process tokens — KV dequant once, score per-row with causal mask
  for (uint t = t_start; t < t_end; t++) {
    // Dequant K once
    float k_decoded[DIMS_PER_LANE];
    const device uint32_t* k_packed_ptr = key_packed +
        kv_idx * uint(token_count) * KeyPackedWidth + t * KeyPackedWidth;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= Dim) {
        k_decoded[i] = 0.0f;
        continue;
      }
      uint k_bit_offset = d * KeyBits;
      uint k_word_idx = k_bit_offset / 32;
      uint k_shift = k_bit_offset % 32;
      uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
      int k_spill = (int)k_shift + (int)KeyBits - 32;
      if (k_spill > 0) {
        k_value |=
            (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
      }
      k_value &= KEY_MASK;
      k_decoded[i] = key_cb[k_value];
    }
    float k_norm = key_norms[kv_idx * uint(token_count) + t];

    // Dequant V once
    float v_decoded[DIMS_PER_LANE];
    const device uint32_t* v_packed_ptr = val_packed +
        kv_idx * uint(token_count) * ValuePackedWidth + t * ValuePackedWidth;
    float v_norm = val_norms[kv_idx * uint(token_count) + t];
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d >= Dim) {
        v_decoded[i] = 0.0f;
        continue;
      }
      uint v_bit_offset = d * ValueBits;
      uint v_word_idx = v_bit_offset / 32;
      uint v_shift = v_bit_offset % 32;
      uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
      int v_spill = (int)v_shift + (int)ValueBits - 32;
      if (v_spill > 0) {
        v_value |=
            (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
      }
      v_value &= VAL_MASK;
      v_decoded[i] = val_cb[v_value] * v_norm;
    }

    // Score + softmax + V for each query row (with per-row causal mask)
    for (uint r = 0; r < NR0; r++) {
      // Per-row causal: skip if this token is future for this specific query
      if (t > q_abs[r])
        continue;

      float dot_partial = 0.0f;
      for (uint i = 0; i < DIMS_PER_LANE; i++) {
        dot_partial += q_vals[r * DIMS_PER_LANE + i] * k_decoded[i];
      }
      float score = simd_sum(dot_partial) * k_norm;

      float new_m = max(m_state[r], score);
      float exp_diff = exp(m_state[r] - new_m);
      float exp_score = exp(score - new_m);

      for (uint i = 0; i < DIMS_PER_LANE; i++) {
        o_state[r * DIMS_PER_LANE + i] =
            o_state[r * DIMS_PER_LANE + i] * exp_diff +
            exp_score * v_decoded[i];
      }
      l_state[r] = l_state[r] * exp_diff + exp_score;
      m_state[r] = new_m;
    }
  }

  // Write partial results for all NR0 queries
  for (uint r = 0; r < NR0; r++) {
    uint q_idx = query_group * NR0 + r;
    uint partial_base = (q_idx * uint(num_blocks) + block_idx) * Dim;
    for (uint i = 0; i < DIMS_PER_LANE; i++) {
      uint d = lane + i * 32;
      if (d < Dim)
        o_partials[partial_base + d] = o_state[r * DIMS_PER_LANE + i];
    }
    if (lane == 0) {
      uint ml_idx = q_idx * uint(num_blocks) + block_idx;
      m_partials[ml_idx] = m_state[r];
      l_partials[ml_idx] = l_state[r];
    }
  }
}

// ============================================================================
// Instantiation macros for TurboFlash Pass 1
// ============================================================================
#define TQ_PW(dim, bits) (((dim) * (bits) + 31) / 32)

#define instantiate_turbo_flash_p1(kb, vb, dim)                             \
  template                                                                  \
      [[host_name("turbo_flash_p1_" #kb "_" #vb "_" #dim)]] [[kernel]] void \
      turbo_flash_p1<kb, vb, dim, TQ_PW(dim, kb), TQ_PW(dim, vb)>(          \
          const device float*,                                              \
          const device uint32_t*,                                           \
          const device float*,                                              \
          const device float*,                                              \
          const device uint32_t*,                                           \
          const device float*,                                              \
          const device float*,                                              \
          device float*,                                                    \
          device float*,                                                    \
          device float*,                                                    \
          constant int&,                                                    \
          constant int&,                                                    \
          constant int&,                                                    \
          constant int&,                                                    \
          uint3);                                                           \
  template [[host_name(                                                     \
      "turbo_flash_p1_causal_" #kb "_" #vb "_" #dim)]] [[kernel]] void      \
  turbo_flash_p1_causal<kb, vb, dim, TQ_PW(dim, kb), TQ_PW(dim, vb)>(       \
      const device float*,                                                  \
      const device uint32_t*,                                               \
      const device float*,                                                  \
      const device float*,                                                  \
      const device uint32_t*,                                               \
      const device float*,                                                  \
      const device float*,                                                  \
      device float*,                                                        \
      device float*,                                                        \
      device float*,                                                        \
      constant int&,                                                        \
      constant int&,                                                        \
      constant int&,                                                        \
      constant int&,                                                        \
      constant int&,                                                        \
      constant int&,                                                        \
      uint3);

// Common asymmetric (KeyBits, ValueBits) combinations
#define instantiate_flash_for_dim(dim)                                        \
  instantiate_turbo_flash_p1(4, 4, dim) instantiate_turbo_flash_p1(4, 2, dim) \
      instantiate_turbo_flash_p1(4, 3, dim)                                   \
          instantiate_turbo_flash_p1(3, 2, dim)                               \
              instantiate_turbo_flash_p1(3, 3, dim)                           \
                  instantiate_turbo_flash_p1(8, 4, dim)                       \
                      instantiate_turbo_flash_p1(8, 2, dim)                   \
                          instantiate_turbo_flash_p1(8, 8, dim)

instantiate_flash_for_dim(64) instantiate_flash_for_dim(80)
    instantiate_flash_for_dim(96) instantiate_flash_for_dim(128)
        instantiate_flash_for_dim(256)
            instantiate_flash_for_dim(512) // Gemma 4 family (E2B, 26B-A4B, 31B)

// ============================================================================
// Instantiation macros for TurboFlash Pass 1 NR0 (multi-row)
// ============================================================================
#define instantiate_turbo_flash_p1_nr0(kb, vb, dim, nr0)                       \
  template [[host_name(                                                        \
      "turbo_flash_p1_nr0_" #kb "_" #vb "_" #dim "_" #nr0)]] [[kernel]] void   \
  turbo_flash_p1_nr0<kb, vb, dim, TQ_PW(dim, kb), TQ_PW(dim, vb), nr0>(        \
      const device float*,                                                     \
      const device uint32_t*,                                                  \
      const device float*,                                                     \
      const device float*,                                                     \
      const device uint32_t*,                                                  \
      const device float*,                                                     \
      const device float*,                                                     \
      device float*,                                                           \
      device float*,                                                           \
      device float*,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      uint3);                                                                  \
  template [[host_name(                                                        \
      "turbo_flash_p1_nr0_causal_" #kb "_" #vb "_" #dim                        \
      "_" #nr0)]] [[kernel]] void                                              \
  turbo_flash_p1_nr0_causal<kb, vb, dim, TQ_PW(dim, kb), TQ_PW(dim, vb), nr0>( \
      const device float*,                                                     \
      const device uint32_t*,                                                  \
      const device float*,                                                     \
      const device float*,                                                     \
      const device uint32_t*,                                                  \
      const device float*,                                                     \
      const device float*,                                                     \
      device float*,                                                           \
      device float*,                                                           \
      device float*,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      constant int&,                                                           \
      uint3);

// NR0=2 for all common bit/dim combos
#define instantiate_nr0_for_dim(dim)                                   \
  instantiate_turbo_flash_p1_nr0(4, 4, dim, 2)                         \
      instantiate_turbo_flash_p1_nr0(4, 2, dim, 2)                     \
          instantiate_turbo_flash_p1_nr0(4, 3, dim, 2)                 \
              instantiate_turbo_flash_p1_nr0(3, 2, dim, 2)             \
                  instantiate_turbo_flash_p1_nr0(3, 3, dim, 2)         \
                      instantiate_turbo_flash_p1_nr0(8, 4, dim, 2)     \
                          instantiate_turbo_flash_p1_nr0(8, 2, dim, 2) \
                              instantiate_turbo_flash_p1_nr0(8, 8, dim, 2)

    instantiate_nr0_for_dim(64) instantiate_nr0_for_dim(80)
        instantiate_nr0_for_dim(96) instantiate_nr0_for_dim(128)
            instantiate_nr0_for_dim(256)
                instantiate_nr0_for_dim(512) // Gemma 4 family
