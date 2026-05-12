// Copyright © 2024 Apple Inc.

#include <optional>
#include <variant>

#include "mlx/primitives.h"

namespace mlx::core::fast {

// Custom primitive accepts a fallback function which it uses for
// transformations. Transformations are virtual so that derived classes may
// override the default behavior.
class Custom : public Primitive {
 public:
  explicit Custom(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback)
      : Primitive(stream), fallback_(std::move(fallback)) {}

  virtual std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  virtual std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  virtual std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

 protected:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
};

class RMSNorm : public Custom {
 public:
  RMSNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_NAME(RMSNorm)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class RMSNormQuantizedGEMV : public Custom {
 public:
  RMSNormQuantizedGEMV(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps,
      int group_size)
      : Custom(stream, std::move(fallback)),
        eps_(eps),
        group_size_(group_size) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(RMSNormQuantizedGEMV)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(nullptr, eps_, group_size_);
  }

 private:
  float eps_;
  int group_size_;
};

class BatchedQKVQuantizedGEMV : public Custom {
 public:
  BatchedQKVQuantizedGEMV(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int group_size,
      int n_q,
      int n_k,
      int n_v)
      : Custom(stream, std::move(fallback)),
        group_size_(group_size),
        n_q_(n_q),
        n_k_(n_k),
        n_v_(n_v) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(BatchedQKVQuantizedGEMV)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(nullptr, group_size_, n_q_, n_k_, n_v_);
  }

 private:
  int group_size_;
  int n_q_, n_k_, n_v_;
};

class WarpMoeGateUp : public Custom {
 public:
  WarpMoeGateUp(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int group_size,
      int hidden_dims,
      int activation_type)
      : Custom(stream, std::move(fallback)),
        group_size_(group_size),
        hidden_dims_(hidden_dims),
        activation_type_(activation_type) {}

  static bool use_fallback(Stream stream);
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  DEFINE_NAME(WarpMoeGateUp)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(
        nullptr, group_size_, hidden_dims_, activation_type_);
  }

 private:
  int group_size_;
  int hidden_dims_;
  int activation_type_;
};

class WarpMoeDown : public Custom {
 public:
  WarpMoeDown(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int group_size,
      int hidden_dims,
      int out_dims)
      : Custom(stream, std::move(fallback)),
        group_size_(group_size),
        hidden_dims_(hidden_dims),
        out_dims_(out_dims) {}

  static bool use_fallback(Stream stream);
  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  DEFINE_NAME(WarpMoeDown)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, group_size_, hidden_dims_, out_dims_);
  }

 private:
  int group_size_;
  int hidden_dims_;
  int out_dims_;
};

class RMSNormResidual : public Custom {
 public:
  RMSNormResidual(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(RMSNormResidual)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(nullptr, eps_);
  }

 private:
  float eps_;
};

class FusedGateActivation : public Custom {
 public:
  FusedGateActivation(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int hidden_dims,
      int activation_type)
      : Custom(stream, std::move(fallback)),
        hidden_dims_(hidden_dims),
        activation_type_(activation_type) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(FusedGateActivation)
  bool is_equivalent(const Primitive& other) const override;

  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = hidden_dims_;
    return {shape};
  }

  auto state() const {
    return std::make_tuple(nullptr, hidden_dims_, activation_type_);
  }

 private:
  int hidden_dims_;
  int activation_type_;
};

class RMSNormRoPE : public Custom {
 public:
  RMSNormRoPE(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps,
      int n_heads,
      int seq_len,
      int offset)
      : Custom(stream, std::move(fallback)),
        eps_(eps),
        n_heads_(n_heads),
        seq_len_(seq_len),
        offset_(offset) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(RMSNormRoPE)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(nullptr, eps_, n_heads_, seq_len_, offset_);
  }

 private:
  float eps_;
  int n_heads_;
  int seq_len_;
  int offset_;
};

class RMSNormVJP : public Custom {
 public:
  RMSNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(RMSNormVJP)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class LayerNorm : public Custom {
 public:
  LayerNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  static bool use_fallback(Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_NAME(LayerNorm)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class LayerNormVJP : public Custom {
 public:
  LayerNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, std::move(fallback)), eps_(eps) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(LayerNormVJP)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class RoPE : public Custom {
 public:
  RoPE(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int dims,
      bool traditional,
      float base,
      float scale,
      bool forward)
      : Custom(stream, std::move(fallback)),
        dims_(dims),
        traditional_(traditional),
        base_(base),
        scale_(scale),
        forward_(forward) {}

  static bool use_fallback(Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_NAME(RoPE)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(
        nullptr, dims_, traditional_, base_, scale_, forward_);
  }

 private:
  int dims_;
  bool traditional_;
  float base_;
  float scale_;
  bool forward_;
};

class ScaledDotProductAttention : public Custom {
 public:
  ScaledDotProductAttention(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float scale,
      bool do_causal,
      bool has_sinks,
      bool output_logsumexp,
      int window_size = -1)
      : Custom(stream, std::move(fallback)),
        scale_(scale),
        do_causal_(do_causal),
        has_sinks_(has_sinks),
        output_logsumexp_(output_logsumexp),
        window_size_(window_size) {}

  static bool use_fallback(
      const array& q,
      const array& k,
      const array& v,
      bool has_mask,
      bool has_arr_mask,
      bool do_causal,
      bool is_training,
      bool output_logsumexp,
      Stream s,
      int window_size = -1);
  static bool supports_bool_mask();

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  bool is_equivalent(const Primitive& other) const override;

  DEFINE_NAME(ScaledDotProductAttention);
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(
        nullptr,
        scale_,
        do_causal_,
        has_sinks_,
        output_logsumexp_,
        window_size_);
  }

 private:
  float scale_;
  bool do_causal_;
  bool has_sinks_;
  bool output_logsumexp_;
  int window_size_;
};

class ScaledDotProductAttentionVJP : public Custom {
 public:
  ScaledDotProductAttentionVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float scale,
      bool do_causal,
      bool has_sinks)
      : Custom(stream, std::move(fallback)),
        scale_(scale),
        do_causal_(do_causal),
        has_sinks_(has_sinks) {}

  static bool use_fallback(const array& q, Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(ScaledDotProductAttentionVJP);
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_tuple(nullptr, scale_, do_causal_, has_sinks_);
  }

 private:
  float scale_;
  bool do_causal_;
  bool has_sinks_;
};

class ConvertFP8 : public Primitive {
 public:
  explicit ConvertFP8(Stream stream, bool to_fp8)
      : Primitive(stream), to_fp8_(to_fp8) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  const char* name() const override {
    if (to_fp8_) {
      return "ToFP8";
    } else {
      return "FromFP8";
    }
  }
  bool state() const {
    return to_fp8_;
  };

  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE();

 private:
  bool to_fp8_;
};

class Quantize : public Custom {
 public:
  explicit Quantize(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int group_size,
      int bits,
      QuantizationMode mode,
      bool dequantize)
      : Custom(stream, std::move(fallback)),
        group_size_(group_size),
        bits_(bits),
        mode_(mode),
        dequantize_(dequantize) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(Quantize);

  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(nullptr, group_size_, bits_, mode_, dequantize_);
  }

 private:
  int group_size_;
  int bits_;
  QuantizationMode mode_;
  bool dequantize_;
};

using ScalarArg = std::variant<bool, int, float>;

class CustomKernel : public Primitive {
 public:
  CustomKernel(
      Stream stream,
      std::string name,
      std::string source,
      std::tuple<int, int, int> grid,
      std::tuple<int, int, int> threadgroup,
      std::vector<std::tuple<bool, bool, bool>> shape_infos,
      bool ensure_row_contiguous,
      std::optional<float> init_value,
      std::vector<ScalarArg> scalar_arguments,
      bool is_precompiled,
      int shared_memory)
      : Primitive(stream),
        name_(std::move(name)),
        source_(std::move(source)),
        grid_(grid),
        threadgroup_(threadgroup),
        shape_infos_(std::move(shape_infos)),
        ensure_row_contiguous_(ensure_row_contiguous),
        init_value_(init_value),
        scalar_arguments_(std::move(scalar_arguments)),
        is_precompiled_(is_precompiled),
        shared_memory_(shared_memory) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("Custom kernels only run on GPU.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(CustomKernel);
  auto state() const {
    return std::make_tuple(
        name_,
        source_,
        grid_,
        threadgroup_,
        shape_infos_,
        ensure_row_contiguous_,
        init_value_,
        scalar_arguments_,
        is_precompiled_,
        shared_memory_);
  }

 private:
  std::string name_;
  std::string source_;
  std::tuple<int, int, int> grid_;
  std::tuple<int, int, int> threadgroup_;
  std::vector<std::tuple<bool, bool, bool>> shape_infos_;
  bool ensure_row_contiguous_;
  std::optional<float> init_value_;
  std::vector<ScalarArg> scalar_arguments_;
  bool is_precompiled_;
  int shared_memory_;
};

// ============================================================================
// TurboQuant primitives — compressed-domain attention kernels
// ============================================================================

class TurboScore : public Custom {
 public:
  TurboScore(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int bits,
      int dim)
      : Custom(stream, std::move(fallback)), bits_(bits), dim_(dim) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("TurboScore only runs on GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TurboScore)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, bits_, dim_);
  }

 private:
  int bits_;
  int dim_;
};

class TurboEncode : public Custom {
 public:
  TurboEncode(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int bits,
      int dim,
      bool use_wht)
      : Custom(stream, std::move(fallback)),
        bits_(bits),
        dim_(dim),
        use_wht_(use_wht) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("TurboEncode only runs on GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TurboEncode)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(nullptr, bits_, dim_, use_wht_);
  }

 private:
  int bits_;
  int dim_;
  bool use_wht_;
};

class TurboFlashPass1 : public Custom {
 public:
  TurboFlashPass1(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int key_bits,
      int value_bits,
      int dim,
      bool causal)
      : Custom(stream, std::move(fallback)),
        key_bits_(key_bits),
        value_bits_(value_bits),
        dim_(dim),
        causal_(causal) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("TurboFlashPass1 only runs on GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TurboFlashPass1)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(nullptr, key_bits_, value_bits_, dim_, causal_);
  }

 private:
  int key_bits_;
  int value_bits_;
  int dim_;
  bool causal_;
};

class TurboFlashPass1NR0 : public Custom {
 public:
  TurboFlashPass1NR0(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int key_bits,
      int value_bits,
      int dim,
      int nr0,
      bool causal)
      : Custom(stream, std::move(fallback)),
        key_bits_(key_bits),
        value_bits_(value_bits),
        dim_(dim),
        nr0_(nr0),
        causal_(causal) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("TurboFlashPass1NR0 only runs on GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TurboFlashPass1NR0)
  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(
        nullptr, key_bits_, value_bits_, dim_, nr0_, causal_);
  }

 private:
  int key_bits_;
  int value_bits_;
  int dim_;
  int nr0_;
  bool causal_;
};

class TurboFlashPass2 : public Custom {
 public:
  TurboFlashPass2(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int dim,
      bool fused_rotation)
      : Custom(stream, std::move(fallback)),
        dim_(dim),
        fused_rotation_(fused_rotation) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("TurboFlashPass2 only runs on GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TurboFlashPass2)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, dim_, fused_rotation_);
  }

 private:
  int dim_;
  bool fused_rotation_;
};

class TurboValue : public Custom {
 public:
  TurboValue(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int bits,
      int dim)
      : Custom(stream, std::move(fallback)), bits_(bits), dim_(dim) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("TurboValue only runs on GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TurboValue)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, bits_, dim_);
  }

 private:
  int bits_;
  int dim_;
};

// Bulk-dequant of a packed [B, H, T, PackedWidth] codec buffer back to
// FP16/BF16 [B, H, T, dim] in rotated codec space. One Metal dispatch
// (vs the per-token bit-unpack inside TurboFlash). Output dtype is
// templated through the host-name suffix so callers can target their
// model dtype directly without an extra cast.
class TurboBulkDequantRotated : public Custom {
 public:
  TurboBulkDequantRotated(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int bits,
      int dim,
      Dtype output_dtype)
      : Custom(stream, std::move(fallback)),
        bits_(bits),
        dim_(dim),
        output_dtype_(output_dtype) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("TurboBulkDequantRotated only runs on GPU");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TurboBulkDequantRotated)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, bits_, dim_, output_dtype_);
  }

 private:
  int bits_;
  int dim_;
  Dtype output_dtype_;
};

// ============================================================================
// GatedDeltaNet / SSM recurrence primitives
// ============================================================================

class GatedDeltaStep : public Custom {
 public:
  GatedDeltaStep(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      bool fused,
      bool has_mask,
      int T,
      int Dk,
      int Dv,
      int Hk,
      int Hv)
      : Custom(stream, std::move(fallback)),
        fused_(fused),
        has_mask_(has_mask),
        T_(T),
        Dk_(Dk),
        Dv_(Dv),
        Hk_(Hk),
        Hv_(Hv) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(GatedDeltaStep)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(nullptr, fused_, has_mask_, T_, Dk_, Dv_, Hk_, Hv_);
  }

 private:
  bool fused_;
  bool has_mask_;
  int T_;
  int Dk_;
  int Dv_;
  int Hk_;
  int Hv_;
};

/// GatedDelta forward kernel with per-step `delta_t` tape capture.
/// Used by speculative-decoder verify forwards on hybrid GDN+Attention
/// models (Qwen 3.5 / 3.6) — captures innovations for possible
/// rollback via the paired `TapeReplay` primitive. Outputs:
///   {y [B, T, Hv, Dv], state_out [B, Hv, Dv, Dk], delta_log [B, T, Hv, Dv]}
class GatedDeltaStepRecord : public Custom {
 public:
  GatedDeltaStepRecord(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      bool has_mask,
      int T,
      int Dk,
      int Dv,
      int Hk,
      int Hv)
      : Custom(stream, std::move(fallback)),
        has_mask_(has_mask),
        T_(T),
        Dk_(Dk),
        Dv_(Dv),
        Hk_(Hk),
        Hv_(Hv) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(GatedDeltaStepRecord)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(nullptr, has_mask_, T_, Dk_, Dv_, Hk_, Hv_);
  }

 private:
  bool has_mask_;
  int T_;
  int Dk_;
  int Dv_;
  int Hk_;
  int Hv_;
};

/// Tape-replay rollback kernel. Re-folds the accepted prefix
/// `[0, accepted)` of an innovation tape (per-step `(delta_t, k_t, g_t)`
/// triples) onto a pre-record state snapshot. Adopts upstream dflash-mlx
/// correctness patterns from day 1 (masked-timestep fix + branchless
/// `metal::select`). Output: {state_out [B, Hv, Dv, Dk]}.
class TapeReplay : public Custom {
 public:
  TapeReplay(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      bool has_mask,
      int T_log,
      int accepted,
      int Dk,
      int Dv,
      int Hk,
      int Hv)
      : Custom(stream, std::move(fallback)),
        has_mask_(has_mask),
        T_log_(T_log),
        accepted_(accepted),
        Dk_(Dk),
        Dv_(Dv),
        Hk_(Hk),
        Hv_(Hv) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(TapeReplay)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(
        nullptr, has_mask_, T_log_, accepted_, Dk_, Dv_, Hk_, Hv_);
  }

 private:
  bool has_mask_;
  int T_log_;
  int accepted_;
  int Dk_;
  int Dv_;
  int Hk_;
  int Hv_;
};

class SSMStep : public Custom {
 public:
  SSMStep(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int Dh,
      int Ds,
      int H,
      int G)
      : Custom(stream, std::move(fallback)), Dh_(Dh), Ds_(Ds), H_(H), G_(G) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_NAME(SSMStep)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_tuple(nullptr, Dh_, Ds_, H_, G_);
  }

 private:
  int Dh_;
  int Ds_;
  int H_;
  int G_;
};

} // namespace mlx::core::fast
