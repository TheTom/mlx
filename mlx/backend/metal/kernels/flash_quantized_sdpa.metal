// Copyright © 2026 Eric Kryski. Flash quantized SDPA — spec 041 phase 1.1.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/flash_quantized_sdpa.h"

using namespace metal;

#define instantiate_flash_quantized_sdpa(type, head_dim, value_dim, bits, group_size) \
  instantiate_kernel(                                                                  \
      "flash_quantized_sdpa_" #type "_" #head_dim "_" #value_dim "_" #bits "_" #group_size, \
      flash_quantized_sdpa,                                                            \
      type,                                                                            \
      head_dim,                                                                        \
      value_dim,                                                                       \
      bits,                                                                            \
      group_size)

// Phase 1: bits ∈ {4, 8}, groupSize = 64. Head dims cover common shapes:
//   D=64 (Qwen 3.5-0.8B), D=128 (Qwen / Gemma 4 E2B / E4B), D=256 (Gemma 4 31B sliding),
//   D=512 (Gemma 4 31B global, GPT-OSS-20B).
#define instantiate_flash_quantized_sdpa_for_head_dim(type, bits, group_size) \
  instantiate_flash_quantized_sdpa(type, 64,  64,  bits, group_size)          \
  instantiate_flash_quantized_sdpa(type, 96,  96,  bits, group_size)          \
  instantiate_flash_quantized_sdpa(type, 128, 128, bits, group_size)          \
  instantiate_flash_quantized_sdpa(type, 256, 256, bits, group_size)          \
  instantiate_flash_quantized_sdpa(type, 512, 512, bits, group_size)

// Phase 2 bit-widths added alongside Phase 1 — MLX dequantize supports them all.
#define instantiate_flash_quantized_sdpa_for_type(type, group_size) \
  instantiate_flash_quantized_sdpa_for_head_dim(type, 2, group_size) \
  instantiate_flash_quantized_sdpa_for_head_dim(type, 3, group_size) \
  instantiate_flash_quantized_sdpa_for_head_dim(type, 4, group_size) \
  instantiate_flash_quantized_sdpa_for_head_dim(type, 6, group_size) \
  instantiate_flash_quantized_sdpa_for_head_dim(type, 8, group_size)

instantiate_flash_quantized_sdpa_for_type(float, 64)
instantiate_flash_quantized_sdpa_for_type(bfloat16_t, 64)
instantiate_flash_quantized_sdpa_for_type(float16_t, 64)
    // clang-format on
