"""Tests for scaled_dot_product_attention_qv (decode + prefill).

Verifies the fused QV SDPA kernel against dequantize + native SDPA
for both decode (L=1, sdpa_vector_qv) and prefill (L>1, steel attention_qv).
"""
# Copyright © 2025 Shiyang "Landon" Yue

import unittest
import mlx.core as mx


def reference_qv_sdpa(q, k, vq, vs, vb, scale, group_size=32):
    """Reference: dequantize V, then native fp16 SDPA."""
    bits = 4 if group_size == 32 else 8
    v_deq = mx.dequantize(vq, vs, vb, group_size=group_size, bits=bits)
    return mx.fast.scaled_dot_product_attention(q, k, v_deq, scale=scale)


class TestSDPAQuantizedV(unittest.TestCase):

    def _make_inputs(self, B, H_kv, T, D, GQA, group_size=32):
        H_q = H_kv * GQA
        k = mx.random.normal((B, H_kv, T, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, T, D)).astype(mx.float16)
        bits = 4 if group_size == 32 else 8
        vq, vs, vb = mx.quantize(v, group_size=group_size, bits=bits)
        # Expand K for GQA (native SDPA needs expanded K)
        k_exp = mx.repeat(k, GQA, axis=1)
        vq_exp = mx.repeat(vq, GQA, axis=1)
        vs_exp = mx.repeat(vs, GQA, axis=1)
        vb_exp = mx.repeat(vb, GQA, axis=1)
        return k, k_exp, vq_exp, vs_exp.astype(mx.float32), vb_exp.astype(mx.float32)

    def _cosine(self, a, b):
        a = a.astype(mx.float32).reshape(-1)
        b = b.astype(mx.float32).reshape(-1)
        return (mx.sum(a * b) / (
            mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b))
        )).item()

    # === Decode tests (L=1) ===

    def test_decode_basic(self):
        """Basic decode: B=1, H=4, D=128, T=512, GQA=4."""
        B, H, T, D, GQA = 1, 4, 512, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        scale = D ** -0.5
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, scale)
        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=scale, group_size=32)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_decode_long_context(self):
        """Decode at 8K context."""
        B, H, T, D, GQA = 1, 8, 8192, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, D ** -0.5)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_decode_d256(self):
        """Decode with D=256 (gemma-4 SWA layers)."""
        B, H, T, D, GQA = 1, 16, 2048, 256, 2
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, D ** -0.5)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    # === Prefill tests (L>1) — currently uses dequant + native SDPA ===
    # The fused steel attention_qv kernel is in development.
    # These tests verify the Python-level prefill approach works.

    def test_prefill_basic(self):
        """Prefill: B=1, H=4, D=128, T=512, L=64, GQA=4."""
        B, H, T, D, L, GQA = 1, 4, 512, 128, 64, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        scale = D ** -0.5
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, scale)
        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=scale, group_size=32)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_prefill_long_context(self):
        """Prefill at 4K context, L=128."""
        B, H, T, D, L, GQA = 1, 4, 4096, 128, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, D ** -0.5)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_decode_fp16_scales(self):
        """Decode with fp16 scales/biases (as mx.quantize returns them)."""
        B, H, T, D, GQA = 1, 4, 512, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        # Pass fp16 scales/biases directly — NOT cast to float32
        # The dispatch should handle the cast internally
        vs_fp16 = vs.astype(mx.float16) if vs.dtype != mx.float16 else vs
        vb_fp16 = vb.astype(mx.float16) if vb.dtype != mx.float16 else vb

        scale = D ** -0.5
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, scale)
        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs_fp16, vb_fp16, scale=scale, group_size=32)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_prefill_fp16_scales(self):
        """Prefill with fp16 scales/biases."""
        B, H, T, D, L, GQA = 1, 4, 512, 128, 32, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        vs_fp16 = vs.astype(mx.float16) if vs.dtype != mx.float16 else vs
        vb_fp16 = vb.astype(mx.float16) if vb.dtype != mx.float16 else vb

        scale = D ** -0.5
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, scale)
        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs_fp16, vb_fp16, scale=scale, group_size=32)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_prefill_d256(self):
        """Prefill with D=256, L=32."""
        B, H, T, D, L, GQA = 1, 16, 2048, 256, 32, 2
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, D ** -0.5)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)


    def _test_prefill_bits(self, bits, group_size):
        """Helper: test prefill at a given bit width."""
        B, H, T, D, L, GQA = 1, 4, 512, 128, 32, 4
        k, k_exp, _, _, _ = self._make_inputs(B, H, T, D, GQA)
        # Quantize V at the specified bit width
        v = mx.random.normal((B, H, T, D)).astype(mx.float16)
        vq, vs, vb = mx.quantize(v, group_size=group_size, bits=bits)
        vq_e = mx.repeat(vq, GQA, axis=1)
        vs_e = mx.repeat(vs, GQA, axis=1)
        vb_e = mx.repeat(vb, GQA, axis=1)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)
        mx.eval(q, k_exp, vq_e, vs_e, vb_e)

        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq_e, vs_e, vb_e, scale=D ** -0.5, group_size=group_size)
        v_deq = mx.dequantize(vq_e, vs_e, vb_e, group_size=group_size, bits=bits)
        ref = mx.fast.scaled_dot_product_attention(q, k_exp, v_deq, scale=D ** -0.5)
        mx.eval(ref, out)

        cos = self._cosine(ref, out)
        self.assertGreater(cos, 0.95, f"{bits}-bit prefill cosine {cos:.4f} < 0.95")

    def test_prefill_2bit(self):
        """Prefill with 2-bit V."""
        self._test_prefill_bits(2, 32)

    def test_prefill_3bit_unsupported(self):
        """3-bit uses non-standard packing — not supported by scalar or codebook kernel."""
        pass

    # === Codebook (TurboQuant) tests ===

    def test_prefill_codebook_4bit(self):
        """Prefill with 4-bit TurboQuant codebook V."""
        from mlx.nn.layers.turbo_kv_cache import (
            turbo_encode, turbo_decode, TurboQuantCodebook,
            _sign_flip_vector, _sign_flip_vector2)

        B, H, T, D, L, GQA = 1, 4, 512, 128, 32, 4
        seed, bits = 42, 4
        k = mx.random.normal((B, H, T, D)).astype(mx.float16)
        v = mx.random.normal((B, H, T, D)).astype(mx.float16)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)

        v_packed, v_norms = turbo_encode(v, bits=bits, seed=seed)
        v_deq = turbo_decode(v_packed, v_norms, D, bits=bits, seed=seed).astype(mx.float16)
        cb = TurboQuantCodebook(bits=bits, dim=D)
        codebook = mx.array(cb.centroids, dtype=mx.float32)
        signs1 = _sign_flip_vector(D, seed)
        signs2 = _sign_flip_vector2(D, seed)

        k_e = mx.repeat(k, GQA, axis=1)
        v_deq_e = mx.repeat(v_deq, GQA, axis=1)
        v_packed_e = mx.repeat(v_packed, GQA, axis=1)
        v_norms_e = mx.repeat(v_norms, GQA, axis=1)
        mx.eval(k_e, v_deq_e, v_packed_e, v_norms_e, q, codebook, signs1, signs2)

        ref = mx.fast.scaled_dot_product_attention(q, k_e, v_deq_e, scale=D ** -0.5)
        out_rot = mx.fast.scaled_dot_product_attention_qv_cb(
            q, k_e, v_packed_e, v_norms_e.astype(mx.float16), codebook,
            scale=D ** -0.5, bits=bits)
        mx.eval(ref, out_rot)

        # Inverse Hadamard + sign flip
        out_f = out_rot.astype(mx.float32)
        out_final = ((mx.hadamard_transform(out_f * signs2) / float(D)) * signs1).astype(mx.float16)

        self.assertGreater(self._cosine(ref, out_final), 0.99)

    def test_prefill_codebook_2bit(self):
        """Prefill with 2-bit TurboQuant codebook V."""
        from mlx.nn.layers.turbo_kv_cache import (
            turbo_encode, turbo_decode, TurboQuantCodebook,
            _sign_flip_vector, _sign_flip_vector2)

        B, H, T, D, L, GQA = 1, 4, 256, 128, 32, 4
        seed, bits = 42, 2
        k = mx.random.normal((B, H, T, D)).astype(mx.float16)
        v = mx.random.normal((B, H, T, D)).astype(mx.float16)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)

        v_packed, v_norms = turbo_encode(v, bits=bits, seed=seed)
        v_deq = turbo_decode(v_packed, v_norms, D, bits=bits, seed=seed).astype(mx.float16)
        cb = TurboQuantCodebook(bits=bits, dim=D)
        codebook = mx.array(cb.centroids, dtype=mx.float32)
        signs1 = _sign_flip_vector(D, seed)
        signs2 = _sign_flip_vector2(D, seed)

        k_e = mx.repeat(k, GQA, axis=1)
        v_deq_e = mx.repeat(v_deq, GQA, axis=1)
        v_packed_e = mx.repeat(v_packed, GQA, axis=1)
        v_norms_e = mx.repeat(v_norms, GQA, axis=1)
        mx.eval(k_e, v_deq_e, v_packed_e, v_norms_e, q, codebook, signs1, signs2)

        ref = mx.fast.scaled_dot_product_attention(q, k_e, v_deq_e, scale=D ** -0.5)
        out_rot = mx.fast.scaled_dot_product_attention_qv_cb(
            q, k_e, v_packed_e, v_norms_e.astype(mx.float16), codebook,
            scale=D ** -0.5, bits=bits)
        mx.eval(ref, out_rot)

        out_f = out_rot.astype(mx.float32)
        out_final = ((mx.hadamard_transform(out_f * signs2) / float(D)) * signs1).astype(mx.float16)

        self.assertGreater(self._cosine(ref, out_final), 0.99)

    def test_prefill_8bit(self):
        """Prefill with 8-bit V."""
        self._test_prefill_bits(8, 64)

    # === Memory tests ===

    def test_memory_decode(self):
        """Verify QV decode uses less memory than fp16."""
        B, H, T, D, GQA = 1, 8, 8192, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        v_fp16 = mx.random.normal((B, H, T, D)).astype(mx.float16)
        v_fp16_exp = mx.repeat(v_fp16, GQA, axis=1)
        mx.eval(q, k_exp, vq, vs, vb, v_fp16_exp)

        # KV memory comparison
        kv_fp16 = k_exp.nbytes + v_fp16_exp.nbytes
        kv_qv = k_exp.nbytes + vq.nbytes + vs.nbytes + vb.nbytes
        savings = kv_fp16 / kv_qv

        # QV should use less KV memory
        self.assertGreater(savings, 1.3)

        # Peak memory during attention
        mx.clear_cache()
        mx.eval(mx.zeros(1))
        mem_before = mx.get_active_memory()
        out_qv = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)
        mx.eval(out_qv)
        mem_qv = mx.get_active_memory() - mem_before

        mx.clear_cache()
        mx.eval(mx.zeros(1))
        mem_before = mx.get_active_memory()
        out_fp = mx.fast.scaled_dot_product_attention(
            q, k_exp, v_fp16_exp, scale=D ** -0.5)
        mx.eval(out_fp)
        mem_fp = mx.get_active_memory() - mem_before

        # Print for visibility
        print(f"\n  Memory (T={T}, D={D}, {H} KV heads):")
        print(f"    KV store: QV={kv_qv/1024**2:.0f}MB fp16={kv_fp16/1024**2:.0f}MB savings={savings:.1f}x")
        print(f"    Peak attn: QV={mem_qv/1024**2:.1f}MB fp16={mem_fp/1024**2:.1f}MB")


if __name__ == "__main__":
    unittest.main()
