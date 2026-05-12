// Copyright (C) 2026 Eric Kryski.
// GatedDeltaStep Metal primitive dispatch.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

bool GatedDeltaStep::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void GatedDeltaStep::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& y = outputs[0];
  auto& state_out = outputs[1];

  // Allocate y output
  y.set_data(allocator::malloc(y.nbytes()));

  // The kernel loads state_in fully into registers before the T-loop
  // and writes state_out from registers after it, so the two can safely
  // alias the same device buffer.  Donate state_in when possible to
  // avoid a second state-sized allocation per layer per token.
  const auto& state_in = inputs[fused_ ? 7 : 5];
  if (state_in.is_donatable() && state_in.flags().row_contiguous &&
      state_in.size() == state_out.size()) {
    state_out.copy_shared_buffer(state_in);
  } else {
    state_out.set_data(allocator::malloc(state_out.nbytes()));
  }

  // Build kernel name from template parameters
  std::string tname = type_to_name(y.dtype());
  std::string kname;
  if (fused_) {
    kname = "gated_delta_step_fused_" + tname + "_" + std::to_string(Dk_) +
        "_" + std::to_string(Dv_) + "_" + std::to_string(Hk_) + "_" +
        std::to_string(Hv_);
  } else {
    kname = "gated_delta_step_" + tname + "_" + std::to_string(Dk_) + "_" +
        std::to_string(Dv_) + "_" + std::to_string(Hk_) + "_" +
        std::to_string(Hv_);
  }

  // Function constant for mask selection (index 10 in the .metal source)
  std::string hash_name = kname + (has_mask_ ? "_mask" : "_nomask");
  metal::MTLFCList func_consts = {
      {&has_mask_, MTL::DataType::DataTypeBool, 10},
  };

  auto kernel = d.get_kernel(kname, hash_name, func_consts);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int T_val = T_;

  if (fused_) {
    // Fused variant buffer layout:
    //   0=q_raw, 1=k_raw, 2=v, 3=a, 4=b, 5=aLog, 6=dtBias,
    //   7=state_in, 8=mask, 9=y, 10=state_out, 11=T_val
    const auto& q_raw = inputs[0];
    const auto& k_raw = inputs[1];
    const auto& v = inputs[2];
    const auto& a = inputs[3];
    const auto& b_input = inputs[4];
    const auto& a_log = inputs[5];
    const auto& dt_bias = inputs[6];
    const auto& state_in = inputs[7];

    compute_encoder.set_input_array(q_raw, 0);
    compute_encoder.set_input_array(k_raw, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(a, 3);
    compute_encoder.set_input_array(b_input, 4);
    compute_encoder.set_input_array(a_log, 5);
    compute_encoder.set_input_array(dt_bias, 6);
    compute_encoder.set_input_array(state_in, 7);

    if (has_mask_) {
      const auto& mask = inputs[8];
      compute_encoder.set_input_array(mask, 8);
    } else {
      // Bind a dummy buffer at index 8 (mask slot) -- use state_in
      compute_encoder.set_input_array(state_in, 8);
    }

    compute_encoder.set_output_array(y, 9);
    compute_encoder.set_output_array(state_out, 10);
    compute_encoder.set_bytes(T_val, 11);
  } else {
    // Standard variant buffer layout:
    //   0=q, 1=k, 2=v, 3=g, 4=beta, 5=state_in,
    //   6=mask, 7=y, 8=state_out, 9=T_val
    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    const auto& g = inputs[3];
    const auto& beta = inputs[4];
    const auto& state_in = inputs[5];

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_input_array(g, 3);
    compute_encoder.set_input_array(beta, 4);
    compute_encoder.set_input_array(state_in, 5);

    if (has_mask_) {
      const auto& mask = inputs[6];
      compute_encoder.set_input_array(mask, 6);
    } else {
      // Bind a dummy buffer at index 6 (mask slot) -- use state_in
      compute_encoder.set_input_array(state_in, 6);
    }

    compute_encoder.set_output_array(y, 7);
    compute_encoder.set_output_array(state_out, 8);
    compute_encoder.set_bytes(T_val, 9);
  }

  // Determine batch size from output shapes
  // y shape: [B, T, Hv, Dv], state_out shape: [B, Hv, Dv, Dk]
  int B = static_cast<int>(state_out.shape(0));

  // Grid: (32, Dv, B * Hv)  ThreadGroup: (32, 4, 1)
  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, Dv_ / 4, B * Hv_), MTL::Size(32, 4, 1));
}

bool GatedDeltaStep::is_equivalent(const Primitive& other) const {
  const GatedDeltaStep& o = static_cast<const GatedDeltaStep&>(other);
  return fused_ == o.fused_ && has_mask_ == o.has_mask_ && T_ == o.T_ &&
      Dk_ == o.Dk_ && Dv_ == o.Dv_ && Hk_ == o.Hk_ && Hv_ == o.Hv_;
}

} // namespace mlx::core::fast
