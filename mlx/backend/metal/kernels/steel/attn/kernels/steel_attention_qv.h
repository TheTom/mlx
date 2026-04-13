// Copyright © 2025 Apple Inc. / TurboQuant+ (Tom Turney) / Shiyang "Landon" Yue
//
// Steel flash attention with quantized V (mx.quantize scalar format).
// K stays fp16 for scoring. V dequantized inline from 4-bit packed uint32
// into threadgroup memory as fp16, then standard steel MMA for softmax(S)@V.
//
// Extends sdpa_vector_qv (decode L=1) to prefill (L>1) via steel tiling.
// One matmul per KV block (no QJL since K is fp16).

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];
constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];

struct MaxOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct MulOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
struct DivOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x / y; }
};

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    int BITS = 4,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_qv(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    // Quantized V: packed 4-bit + per-group scales + biases
    const device uint* qv_data [[buffer(2)]],
    const device float* qv_scales [[buffer(3)]],
    const device float* qv_biases [[buffer(4)]],
    device T* O [[buffer(5)]],
    const constant AttnParams* params [[buffer(6)]],
    // V strides
    const constant size_t& qv_data_head_stride [[buffer(7)]],
    const constant size_t& qv_group_head_stride [[buffer(8)]],
    const constant int& qv_group_size [[buffer(9)]],
    // Thread indices
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  // Elements per uint32 word depends on bit width
  constexpr int elems_per_word = 32 / BITS;  // 8 for 4-bit, 4 for 8-bit, 16 for 2-bit
  constexpr int packed_per_row = (BD * BITS + 31) / 32;  // uint32 words per V row
  constexpr uint V_MASK = (1u << BITS) - 1u;
  const int grp_per_row = BD / qv_group_size;

  ulong3 tidl{tid.x, tid.y, tid.z};

  // Q and K use standard strides from AttnParams
  Q += tidl.z * params->Q_strides[0] +
      tidl.y * params->Q_strides[1] +
      tidl.x * BQ * params->Q_strides[2];

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;

  K += tidl.z * params->K_strides[0] +
      kv_head_idx * params->K_strides[1];

  // V quantized data layout: (B, n_kv_heads, T, packed_per_row)
  // qv_data_head_stride = T * packed_per_row (per-head in uint32 words)
  // qv_group_head_stride = T * groups_per_row (per-head in floats)
  ulong n_kv_heads = params->H / params->gqa_factor;
  ulong vd_batch_stride = n_kv_heads * qv_data_head_stride;
  ulong vg_batch_stride = n_kv_heads * qv_group_head_stride;
  const device uint* vd_base = qv_data + tidl.z * vd_batch_stride + kv_head_idx * qv_data_head_stride;
  const device float* vs_base = qv_scales + tidl.z * vg_batch_stride + kv_head_idx * qv_group_head_stride;
  const device float* vb_base = qv_biases + tidl.z * vg_batch_stride + kv_head_idx * qv_group_head_stride;

  O += tidl.z * params->O_strides[0] +
      tidl.y * params->O_strides[1] +
      tidl.x * BQ * params->O_strides[2];

  // Threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * BD;      // K transposed
  constexpr short tgp_mem_1 = BK * (BD + padV);       // V
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * LDQ_tgp];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Standard Q and K loaders (fp16) — must match steel_attention.h exactly
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ LDQ_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  // MMA tiles
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  // Softmax accumulators
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const int thread_idx = simd_group_id * 32 + simd_lane_id;
  const int total_threads = WM * WN * 32;

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;
  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = min(params->NK, (q_max + BK - 1) / BK);
    int q_min = max(0, int(tid.x) * BQ + params->qL_off);
    kb_min_causal = q_min / BK;
  }

  // KV block loop
  for (int kb = 0; kb < kb_lim; kb++) {
    // Load K tile (fp16, standard loader, transposed)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // S = Q @ K^T
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Apply scale
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out invalid tokens
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) Stile.frag_at(i, j)[jj] = neg_inf;
          }
        }
      }
    }

    // Causal mask
    if (do_causal && kb >= kb_min_causal) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) Stile.frag_at(i, j)[jj] = neg_inf;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V tile: vectorized dequant — read uint32, unpack BITS-wide elements
    {
      const int kv_seq_start = kb * BK;
      const int kv_seq_end = min(kv_seq_start + BK, params->kL);
      const int valid_tokens = kv_seq_end - kv_seq_start;

      // Each uint32 holds elems_per_word elements at BITS width
      const int words_in_row = packed_per_row;
      const int total_words = BK * words_in_row;
      const int words_per_thread = (total_words + total_threads - 1) / total_threads;

      for (int w = 0; w < words_per_thread; w++) {
        int flat_idx = thread_idx * words_per_thread + w;
        if (flat_idx >= total_words) break;

        int t_local = flat_idx / words_in_row;
        int word_in_row = flat_idx % words_in_row;
        int t_global = kv_seq_start + t_local;
        int d_base = word_in_row * elems_per_word;

        if (t_local < valid_tokens && t_global < params->kL) {
          uint packed = vd_base[t_global * packed_per_row + word_in_row];

          int grp_base = d_base / qv_group_size;
          int grp_off = t_global * grp_per_row;
          float s0 = vs_base[grp_off + grp_base];
          float b0 = vb_base[grp_off + grp_base];

          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < elems_per_word; i++) {
            int d = d_base + i;
            if (d >= BD) break;
            uint raw = (packed >> (i * BITS)) & V_MASK;
            int grp = d / qv_group_size;
            float sc = (grp == grp_base) ? s0 : vs_base[grp_off + grp];
            float bi = (grp == grp_base) ? b0 : vb_base[grp_off + grp];
            Vs[t_local * LDV_tgp + d] = T(float(raw) * sc + bi);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < elems_per_word; i++) {
            int d = d_base + i;
            if (d >= BD) break;
            Vs[t_local * LDV_tgp + d] = T(0);
          }
        }
      }
    }

    // Online softmax
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<MulOp>(factor);

    // O += softmax(S) @ V
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
          if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id), Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0), Otile.frag_at(iq, id));
        }
      }
    }

    loader_k.next();
  }

  // Normalize and store
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * params->O_strides[2] + sn;
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) return;
    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}


// ============================================================================
// Codebook variant: V stored as TurboQuant codebook indices + norms
// Dequant: codebook[unpack_bits(packed, d)] * norm[t]
// Supports 2, 3, 4-bit (any bit width with simple bit-shift packing)
// ============================================================================

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    int BITS = 4,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_qv_cb(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    // Codebook-quantized V: packed indices + per-token norms + codebook
    const device uint* v_packed [[buffer(2)]],
    const device half* v_norms [[buffer(3)]],
    const device float* v_codebook [[buffer(4)]],
    device T* O [[buffer(5)]],
    const constant AttnParams* params [[buffer(6)]],
    const constant size_t& v_data_head_stride [[buffer(7)]],
    const constant size_t& v_norms_head_stride [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  constexpr int elems_per_word = 32 / BITS;
  constexpr int packed_per_row = (BD * BITS + 31) / 32;
  constexpr uint V_MASK = (1u << BITS) - 1u;

  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] +
      tidl.y * params->Q_strides[1] +
      tidl.x * BQ * params->Q_strides[2];

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;

  K += tidl.z * params->K_strides[0] +
      kv_head_idx * params->K_strides[1];

  // V codebook data: (B, n_kv_heads, T, packed_per_row)
  ulong n_kv_heads = params->H / params->gqa_factor;
  ulong vd_batch_stride = n_kv_heads * v_data_head_stride;
  ulong vn_batch_stride = n_kv_heads * v_norms_head_stride;
  const device uint* vd_base = v_packed + tidl.z * vd_batch_stride + kv_head_idx * v_data_head_stride;
  const device half* vn_base = v_norms + tidl.z * vn_batch_stride + kv_head_idx * v_norms_head_stride;

  O += tidl.z * params->O_strides[0] +
      tidl.y * params->O_strides[1] +
      tidl.x * BQ * params->O_strides[2];

  // Threadgroup memory
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * BD;
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * LDQ_tgp];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;

  QBlockLoader loader_q(Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const int thread_idx = simd_group_id * 32 + simd_lane_id;
  const int total_threads = WM * WN * 32;

  for (int kb = 0; kb < params->NK; kb++) {
    // Load K (standard fp16)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // S = Q @ K^T
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) Stile.frag_at(i, j)[jj] = neg_inf;
          }
        }
      }
    }

    if (do_causal) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) Stile.frag_at(i, j)[jj] = neg_inf;
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load V tile: codebook dequant → fp16 in threadgroup memory
    {
      const int kv_seq_start = kb * BK;
      const int kv_seq_end = min(kv_seq_start + BK, params->kL);
      const int valid_tokens = kv_seq_end - kv_seq_start;

      const int words_in_row = packed_per_row;
      const int total_words = BK * words_in_row;
      const int words_per_thread = (total_words + total_threads - 1) / total_threads;

      for (int w = 0; w < words_per_thread; w++) {
        int flat_idx = thread_idx * words_per_thread + w;
        if (flat_idx >= total_words) break;

        int t_local = flat_idx / words_in_row;
        int word_in_row = flat_idx % words_in_row;
        int t_global = kv_seq_start + t_local;
        int d_base = word_in_row * elems_per_word;

        if (t_local < valid_tokens && t_global < params->kL) {
          uint packed = vd_base[t_global * packed_per_row + word_in_row];
          float vnorm = float(vn_base[t_global]);

          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < elems_per_word; i++) {
            int d = d_base + i;
            if (d >= BD) break;
            uint raw = (packed >> (i * BITS)) & V_MASK;
            Vs[t_local * LDV_tgp + d] = T(v_codebook[raw] * vnorm);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < elems_per_word; i++) {
            int d = d_base + i;
            if (d >= BD) break;
            Vs[t_local * LDV_tgp + d] = T(0);
          }
        }
      }
    }

    // Online softmax
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<MulOp>(factor);

    // O += softmax(S) @ V
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + ik * kFragSize * LDV_tgp + id * kFragSize]);
          if constexpr (BD == 128) simdgroup_barrier(mem_flags::mem_none);
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id), Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0), Otile.frag_at(iq, id));
        }
      }
    }

    loader_k.next();
  }

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * params->O_strides[2] + sn;
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) return;
    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}
