// Copyright © 2026 Eric Kryski. TurboQuant fused single-pass SDPA with sinks
// — spec 041 phase 1.1 follow-up for sinks-using models (GPT-OSS family).
//
// MSE-codec variant of `flash_quantized_sdpa.h`. Single-pass online softmax
// over compressed K and V — no pass1/pass2 split — to side-step the
// pass2-sinks-fold graph-fusion incoherence that previous β-with-sinks
// drafts hit on GPT-OSS-20B.
//
// Layout (matches `turbo_flash.metal` pass1 conventions):
//   - queries:     [B*nQ, D] T (rotated by WHT codec before this call)
//   - k_packed:    [B*nKV, N, KeyPackedWidth] uint32
//   - k_norms:     [B*nKV, N] float
//   - k_codebook:  [2^KeyBits] float
//   - v_packed:    [B*nKV, N, ValuePackedWidth] uint32
//   - v_norms:     [B*nKV, N] float
//   - v_codebook:  [2^ValueBits] float
//   - sinks?:      [nQ] T per-head sink logits (optional)
//   - out:         [B*nQ, D] T (in rotated V space — caller applies Π_v^T)

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

constant bool tf_has_sinks [[function_constant(60)]];
constant bool tf_do_causal [[function_constant(61)]];

template <int KeyBits, int ValueBits, int Dim>
[[kernel]] void turbo_flash_sdpa_v(
    const device float* queries [[buffer(0)]],
    const device uint32_t* k_packed [[buffer(1)]],
    const device float* k_norms [[buffer(2)]],
    const device float* k_codebook [[buffer(3)]],
    const device uint32_t* v_packed [[buffer(4)]],
    const device float* v_norms [[buffer(5)]],
    const device float* v_codebook [[buffer(6)]],
    device bfloat* out [[buffer(7)]],
    const constant int& token_count [[buffer(8)]],
    const constant int& repeat_count [[buffer(9)]],
    const device bfloat* sinks [[buffer(10), function_constant(tf_has_sinks)]],
    const constant int& num_q_heads
    [[buffer(11), function_constant(tf_has_sinks)]],
    const constant int& window_size
    [[buffer(12), function_constant(tf_do_causal)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = (Dim + BD - 1) / BD;
  constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
  constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
  // PackedWidth uses *bit-contiguous* packing (`(Dim * bits + 31) / 32`) to
  // match the Swift encoder (`TurboQuantPacking.packedWidth`). The earlier
  // `(Dim + PACK_FACTOR - 1) / PACK_FACTOR` form silently dropped the
  // boundary-spanning values for bits ∉ {2, 4, 8, 16} — values where a 32-bit
  // word doesn't hold an integer number of indices (3, 5, 6, 7). For those
  // widths the encoder packs across word boundaries, and the unpack below
  // (mirroring `turbo_flash.metal`) handles the spill.
  constexpr int KEY_PACKED_WIDTH = (Dim * KeyBits + 31) / 32;
  constexpr int VAL_PACKED_WIDTH = (Dim * ValueBits + 31) / 32;
  constexpr uint KEY_LEVELS = 1u << KeyBits;
  constexpr uint VAL_LEVELS = 1u << ValueBits;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U v[qk_per_thread];
  thread U o[qk_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Threadgroup-local codebook caches — affine's "single-group hoist"
  // equivalent for TurboQuant. The codebook is small (≤ 256 floats for
  // 8-bit) and constant per call; hoist into TG memory at kernel start
  // so each thread reads from L1 instead of device memory.
  threadgroup U tg_key_codebook[KEY_LEVELS];
  threadgroup U tg_val_codebook[VAL_LEVELS];
  uint linear_tid = simd_gid * BD + simd_lid;
  for (uint i = linear_tid; i < KEY_LEVELS; i += BN * BD) {
    tg_key_codebook[i] = static_cast<U>(k_codebook[i]);
  }
  for (uint i = linear_tid; i < VAL_LEVELS; i += BN * BD) {
    tg_val_codebook[i] = static_cast<U>(v_codebook[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Position calc — matches sdpa_vector. One threadgroup per query
  // position (B * nQ); 32 simdgroups distribute K positions.
  const int q_batch_head_idx = tid.x;
  const int kv_head_idx = q_batch_head_idx / repeat_count;
  const int o_offset = q_batch_head_idx;
  const int q_offset = o_offset;

  // Query slice (rotated + scaled by caller already).
  const device float* queries_thread =
      queries + q_offset * Dim + simd_lid * qk_per_thread;
#pragma clang loop unroll(full)
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = queries_thread[i];
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < qk_per_thread; i++) {
    o[i] = 0;
  }

  // K / V base pointers for this kv_head.
  const device uint32_t* k_packed_head =
      k_packed + kv_head_idx * uint(token_count) * KEY_PACKED_WIDTH;
  const device float* k_norms_head = k_norms + kv_head_idx * uint(token_count);
  const device uint32_t* v_packed_head =
      v_packed + kv_head_idx * uint(token_count) * VAL_PACKED_WIDTH;
  const device float* v_norms_head = v_norms + kv_head_idx * uint(token_count);

  // Sinks init — simdgroup 0 starts with max=sink, sum=exp(0)=1.
  // Other simdgroups start from -INF / 0 (standard online softmax).
  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (tf_has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % uint(num_q_heads)]);
    sum_exp_score = 1;
  }

  // Sliding window upper / lower bounds (causal mask).
  int causal_upper = token_count - 1; // L=1 decode case: last K position.
  int sliding_lower = -1;
  if (tf_do_causal && window_size > 0) {
    sliding_lower = causal_upper - window_size;
  }

  // Iterate K positions in chunks of BN across simdgroups.
  for (int i = simd_gid; i < token_count; i += BN) {
    bool use_key = true;
    if (tf_do_causal && window_size > 0) {
      use_key = (i > sliding_lower);
    }

    if (use_key) {
      // Inline dequant K[i] for this thread's qk_per_thread dim slice.
      const device uint32_t* k_packed_t = k_packed_head + i * KEY_PACKED_WIDTH;
      U k_norm = static_cast<U>(k_norms_head[i]);

#pragma clang loop unroll(full)
      for (int j = 0; j < qk_per_thread; j++) {
        int d = simd_lid * qk_per_thread + j;
        if (d >= Dim) {
          k[j] = 0;
          continue;
        }
        // Bit-contiguous unpacking — matches the encoder
        // (`TurboQuantPacking.packLowBit`) and `turbo_flash.metal`. For
        // bits ∈ {3, 5, 6, 7} values span word boundaries; the spill
        // branch stitches the low bits in.
        uint bit_offset = (uint)(d * KeyBits);
        uint word_idx = bit_offset / 32u;
        uint shift = bit_offset % 32u;
        uint val_idx = (k_packed_t[word_idx] >> shift);
        int spill = (int)shift + (int)KeyBits - 32;
        if (spill > 0) {
          val_idx |=
              (k_packed_t[word_idx + 1] << ((uint)KeyBits - (uint)spill));
        }
        val_idx &= KEY_MASK;
        k[j] = tg_key_codebook[val_idx] * k_norm;
      }

      // Score = q · k (Q already pre-scaled and pre-rotated).
      U score = 0;
#pragma clang loop unroll(full)
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Online softmax update.
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Inline dequant V[i] for this thread's dim slice.
      if (exp_score > 1e-20) {
        const device uint32_t* v_packed_t =
            v_packed_head + i * VAL_PACKED_WIDTH;
        U v_norm = static_cast<U>(v_norms_head[i]);

#pragma clang loop unroll(full)
        for (int j = 0; j < qk_per_thread; j++) {
          int d = simd_lid * qk_per_thread + j;
          if (d >= Dim) {
            v[j] = 0;
            continue;
          }
          // Same bit-contiguous unpacking as K above.
          uint bit_offset = (uint)(d * ValueBits);
          uint word_idx = bit_offset / 32u;
          uint shift = bit_offset % 32u;
          uint val_idx = (v_packed_t[word_idx] >> shift);
          int spill = (int)shift + (int)ValueBits - 32;
          if (spill > 0) {
            val_idx |=
                (v_packed_t[word_idx + 1] << ((uint)ValueBits - (uint)spill));
          }
          val_idx &= VAL_MASK;
          v[j] = tg_val_codebook[val_idx] * v_norm;
        }

#pragma clang loop unroll(full)
        for (int j = 0; j < qk_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * v[j];
        }
      } else {
#pragma clang loop unroll(full)
        for (int j = 0; j < qk_per_thread; j++) {
          o[j] = o[j] * factor;
        }
      }
    }
  }

  // Cross-simdgroup max + sum reduction.
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Aggregate o across simdgroups.
  for (int i = 0; i < qk_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output (bfloat — matches turbo_flash pass2 output dtype).
  if (simd_lid == 0) {
    for (int i = 0; i < qk_per_thread; i++) {
      int d = simd_gid * qk_per_thread + i;
      if (d < Dim) {
        out[o_offset * Dim + d] = static_cast<bfloat>(o[i]);
      }
    }
  }
}
