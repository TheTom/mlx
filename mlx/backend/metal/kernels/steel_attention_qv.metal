// Copyright © 2025 Shiyang "Landon" Yue
// Steel flash attention with quantized V (prefill L>1)
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_qv.h"

using namespace metal;

#define instantiate_attention_qv(type, bq, bk, bd, wm, wn, bits) \
  instantiate_kernel(                                             \
      "attention_qv_" #type "_bq" #bq "_bk" #bk "_bd" #bd       \
      "_wm" #wm "_wn" #wn "_b" #bits,                            \
      attention_qv, type, bq, bk, bd, wm, wn, bits, float)

// Power-of-2 bit widths: 2-bit, 4-bit, 8-bit
// 3-bit uses non-standard packing in mx.quantize (not simple bit shift)
#define instantiate_attention_qv_bits(type, bq, bk, bd, wm, wn) \
  instantiate_attention_qv(type, bq, bk, bd, wm, wn, 2)         \
  instantiate_attention_qv(type, bq, bk, bd, wm, wn, 4)         \
  instantiate_attention_qv(type, bq, bk, bd, wm, wn, 8)

#define instantiate_attention_qv_heads(type)            \
  instantiate_attention_qv_bits(type, 32, 32, 64, 4, 1)  \
  instantiate_attention_qv_bits(type, 32, 32, 128, 4, 1) \
  instantiate_attention_qv_bits(type, 32, 16, 256, 4, 1)

instantiate_attention_qv_heads(float16_t)

// Codebook variant: TurboQuant V (codebook[index] * norm)
#define instantiate_attention_qv_cb(type, bq, bk, bd, wm, wn, bits) \
  instantiate_kernel(                                                 \
      "attention_qv_cb_" #type "_bq" #bq "_bk" #bk "_bd" #bd        \
      "_wm" #wm "_wn" #wn "_b" #bits,                                \
      attention_qv_cb, type, bq, bk, bd, wm, wn, bits, float)

#define instantiate_attention_qv_cb_bits(type, bq, bk, bd, wm, wn) \
  instantiate_attention_qv_cb(type, bq, bk, bd, wm, wn, 2)         \
  instantiate_attention_qv_cb(type, bq, bk, bd, wm, wn, 3)         \
  instantiate_attention_qv_cb(type, bq, bk, bd, wm, wn, 4)

#define instantiate_attention_qv_cb_heads(type)            \
  instantiate_attention_qv_cb_bits(type, 32, 32, 64, 4, 1)  \
  instantiate_attention_qv_cb_bits(type, 32, 32, 128, 4, 1) \
  instantiate_attention_qv_cb_bits(type, 32, 16, 256, 4, 1)

instantiate_attention_qv_cb_heads(float16_t)
// clang-format on
