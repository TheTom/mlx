// Copyright © 2026 Eric Kryski. Flash quantized SDPA — spec 041 phase 1.1.
//
// Affine-quantized variant of `sdpa_vector`. Same online-softmax loop, but
// K and V are dequantised inline per-thread from packed-indices + scale +
// bias triples (the same layout `quantizedMM` consumes).
//
// Layout assumptions (row-contiguous; passed through `ensureRowContiguous`):
//   - queries: [B, n_q_heads, T_q, D] T
//   - k_packed: [B, n_kv_heads, T_kv, D / (32/Bits)] uint32
//   - k_scales: [B, n_kv_heads, T_kv, D / GroupSize] T
//   - k_biases: [B, n_kv_heads, T_kv, D / GroupSize] T   (affine has bias;
//   mxfp4 sets bias=0)
//   - v_packed: [B, n_kv_heads, T_kv, V / (32/Bits)] uint32
//   - v_scales / v_biases: same shape rule as k.
//   - out:     [B, n_q_heads, T_q, V] T
//
// Threadgroup geometry: same as sdpa_vector (BN=BD=32, one threadgroup per
// (batch*query_head, query_pos), 32 simdgroups distributing K positions).

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask_fq [[function_constant(40)]];
constant bool query_transposed_fq [[function_constant(41)]];
constant bool do_causal_fq [[function_constant(42)]];
constant bool bool_mask_fq [[function_constant(43)]];
constant bool float_mask_fq [[function_constant(44)]];
constant bool has_sinks_fq [[function_constant(45)]];

template <typename T, int D, int V, int Bits, int GroupSize>
[[kernel]] void flash_quantized_sdpa(
    const device T* queries [[buffer(0)]],
    const device uint32_t* k_packed [[buffer(1)]],
    const device T* k_scales [[buffer(2)]],
    const device T* k_biases [[buffer(3)]],
    const device uint32_t* v_packed [[buffer(4)]],
    const device T* v_scales [[buffer(5)]],
    const device T* v_biases [[buffer(6)]],
    device T* out [[buffer(7)]],
    const constant int& gqa_factor [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant size_t& k_head_stride_packed [[buffer(10)]],
    const constant size_t& k_seq_stride_packed [[buffer(11)]],
    const constant size_t& k_head_stride_scale [[buffer(12)]],
    const constant size_t& k_seq_stride_scale [[buffer(13)]],
    const constant size_t& v_head_stride_packed [[buffer(14)]],
    const constant size_t& v_seq_stride_packed [[buffer(15)]],
    const constant size_t& v_head_stride_scale [[buffer(16)]],
    const constant size_t& v_seq_stride_scale [[buffer(17)]],
    const constant float& scale [[buffer(18)]],
    const device bool* bmask [[buffer(19), function_constant(bool_mask_fq)]],
    const device T* fmask [[buffer(20), function_constant(float_mask_fq)]],
    const constant int& mask_kv_seq_stride
    [[buffer(21), function_constant(has_mask_fq)]],
    const constant int& mask_q_seq_stride
    [[buffer(22), function_constant(has_mask_fq)]],
    const constant int& mask_head_stride
    [[buffer(23), function_constant(has_mask_fq)]],
    const device T* sinks [[buffer(24), function_constant(has_sinks_fq)]],
    const constant int& num_q_heads
    [[buffer(25), function_constant(has_sinks_fq)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  constexpr int pack_factor = 32 / Bits;
  constexpr uint mask_bits = (1u << Bits) - 1u;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U v[v_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Position calc — matches sdpa_vector exactly.
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed_fq ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;

  // Query slice: each thread reads `qk_per_thread` consecutive dims.
  const device T* queries_thread =
      queries + q_offset * D + simd_lid * qk_per_thread;

  // Mask offsets (same pattern as sdpa_vector).
  if (bool_mask_fq) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask_fq) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  // K base pointers for this kv_head, this simdgroup's first key position.
  const device uint32_t* k_packed_head =
      k_packed + kv_head_idx * k_head_stride_packed;
  const device T* k_scales_head = k_scales + kv_head_idx * k_head_stride_scale;
  const device T* k_biases_head = k_biases + kv_head_idx * k_head_stride_scale;

  const device uint32_t* v_packed_head =
      v_packed + kv_head_idx * v_head_stride_packed;
  const device T* v_scales_head = v_scales + kv_head_idx * v_head_stride_scale;
  const device T* v_biases_head = v_biases + kv_head_idx * v_head_stride_scale;

  // Pre-load query (apply scale).
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * static_cast<U>(queries_thread[i]);
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (has_sinks_fq && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // Iterate K positions in chunks of BN across simdgroups.
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal_fq) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask_fq) {
      use_key = bmask[0];
    } else if (float_mask_fq) {
      use_key = (fmask[0] >= -INFINITY);
    }

    if (use_key) {
      // Inline dequant K[i] for this thread's qk_per_thread dim slice.
      const device uint32_t* k_packed_t =
          k_packed_head + i * k_seq_stride_packed;
      const device T* k_scales_t = k_scales_head + i * k_seq_stride_scale;
      const device T* k_biases_t = k_biases_head + i * k_seq_stride_scale;

      for (int j = 0; j < qk_per_thread; j++) {
        int d = simd_lid * qk_per_thread + j;
        int word_idx = d / pack_factor;
        int shift = (d % pack_factor) * Bits;
        int group_idx = d / GroupSize;
        uint val = (k_packed_t[word_idx] >> shift) & mask_bits;
        k[j] = static_cast<U>(k_scales_t[group_idx]) * U(val) +
            static_cast<U>(k_biases_t[group_idx]);
      }

      // Compute the i-th score across qk_per_thread dims, then simd_sum.
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask_fq) {
        score += static_cast<U>(fmask[0]);
      }

      // Online softmax update.
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Inline dequant V[i] for this thread's v_per_thread slice.
      const device uint32_t* v_packed_t =
          v_packed_head + i * v_seq_stride_packed;
      const device T* v_scales_t = v_scales_head + i * v_seq_stride_scale;
      const device T* v_biases_t = v_biases_head + i * v_seq_stride_scale;

      for (int j = 0; j < v_per_thread; j++) {
        int d = simd_lid * v_per_thread + j;
        int word_idx = d / pack_factor;
        int shift = (d % pack_factor) * Bits;
        int group_idx = d / GroupSize;
        uint val = (v_packed_t[word_idx] >> shift) & mask_bits;
        v[j] = static_cast<U>(v_scales_t[group_idx]) * U(val) +
            static_cast<U>(v_biases_t[group_idx]);
      }

      // Output accumulator update.
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * v[j];
      }
    }

    if (bool_mask_fq) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask_fq) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Cross-simdgroup max + sum_exp reduction (same shape as sdpa_vector).
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Aggregate outputs across simdgroups.
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output.
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[o_offset * V + simd_gid * v_per_thread + i] = static_cast<T>(o[i]);
    }
  }
}
