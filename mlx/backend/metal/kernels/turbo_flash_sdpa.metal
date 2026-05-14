// Copyright © 2026 Eric Kryski. TurboQuant fused single-pass SDPA — spec 041
// phase 1.1 follow-up for sinks-using models.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/turbo_flash_sdpa.h"

using namespace metal;

#define instantiate_turbo_flash_sdpa_v(kb, vb, dim)                       \
  template [[host_name("turbo_flash_sdpa_v_" #kb "_" #vb "_" #dim)]]      \
  [[kernel]] void                                                          \
  turbo_flash_sdpa_v<kb, vb, dim>(                                         \
      const device float*,                                                 \
      const device uint32_t*,                                              \
      const device float*,                                                 \
      const device float*,                                                 \
      const device uint32_t*,                                              \
      const device float*,                                                 \
      const device float*,                                                 \
      device bfloat*,                                                      \
      const constant int&,                                                 \
      const constant int&,                                                 \
      const device bfloat*,                                                \
      const constant int&,                                                 \
      const constant int&,                                                 \
      uint3, uint3, uint, uint);

// Cover the common GPT-OSS / Qwen / Gemma 4 head dims × the four bench
// turbo-cache (keyBits, valueBits) combos.
#define instantiate_turbo_flash_sdpa_v_for_kv(kb, vb)                      \
  instantiate_turbo_flash_sdpa_v(kb, vb, 64)                               \
  instantiate_turbo_flash_sdpa_v(kb, vb, 96)                               \
  instantiate_turbo_flash_sdpa_v(kb, vb, 128)                              \
  instantiate_turbo_flash_sdpa_v(kb, vb, 256)                              \
  instantiate_turbo_flash_sdpa_v(kb, vb, 512)

// Phase 1.1 supported (kb, vb) combos: cover all the parseTurboScheme
// strings the bench exposes.
instantiate_turbo_flash_sdpa_v_for_kv(4, 4)
instantiate_turbo_flash_sdpa_v_for_kv(4, 2)
instantiate_turbo_flash_sdpa_v_for_kv(4, 3)
instantiate_turbo_flash_sdpa_v_for_kv(3, 2)
instantiate_turbo_flash_sdpa_v_for_kv(3, 3)
instantiate_turbo_flash_sdpa_v_for_kv(8, 2)
instantiate_turbo_flash_sdpa_v_for_kv(8, 3)
instantiate_turbo_flash_sdpa_v_for_kv(8, 4)
instantiate_turbo_flash_sdpa_v_for_kv(8, 8)
instantiate_turbo_flash_sdpa_v_for_kv(2, 2)
    // clang-format on
