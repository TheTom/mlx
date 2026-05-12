// Copyright © 2026 Eric Kryski.
// GatedDeltaStepRecord + TapeReplay Metal primitive dispatch.
//
// Companion to `gated_delta.cpp`. Two primitives for spec 020 phase 2
// innovation-tape rollback on hybrid GDN+Attention models (Qwen 3.5 / 3.6).

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

// ============================================================================
// GatedDeltaStepRecord — forward + tape capture
// ============================================================================

bool GatedDeltaStepRecord::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void GatedDeltaStepRecord::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& y = outputs[0];
  auto& state_out = outputs[1];
  auto& delta_log = outputs[2];

  // Allocate y + delta_log outputs.
  y.set_data(allocator::malloc(y.nbytes()));
  delta_log.set_data(allocator::malloc(delta_log.nbytes()));

  // Same state_in donation trick as `gated_delta_step` — the kernel loads
  // state_in into registers before the T-loop and writes state_out from
  // registers after it, so the two can alias the same buffer.
  const auto& state_in = inputs[5];
  if (state_in.is_donatable() && state_in.flags().row_contiguous &&
      state_in.size() == state_out.size()) {
    state_out.copy_shared_buffer(state_in);
  } else {
    state_out.set_data(allocator::malloc(state_out.nbytes()));
  }

  // Build kernel name: gated_delta_step_record_<dtype>_<Dk>_<Dv>_<Hk>_<Hv>
  std::string tname = type_to_name(y.dtype());
  std::string kname = "gated_delta_step_record_" + tname + "_" +
      std::to_string(Dk_) + "_" + std::to_string(Dv_) + "_" +
      std::to_string(Hk_) + "_" + std::to_string(Hv_);

  // Function constant for mask selection (matches index 10 in the .metal).
  std::string hash_name = kname + (has_mask_ ? "_mask" : "_nomask");
  metal::MTLFCList func_consts = {
      {&has_mask_, MTL::DataType::DataTypeBool, 10},
  };

  auto kernel = d.get_kernel(kname, hash_name, func_consts);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int T_val = T_;

  // Buffer layout matches `.metal` declaration:
  //   0=q, 1=k, 2=v, 3=g, 4=beta, 5=state_in, 6=mask, 7=y, 8=state_out,
  //   9=delta_log, 10=T_val
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];
  const auto& g = inputs[3];
  const auto& beta = inputs[4];

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
    // Bind a dummy buffer at the mask slot — pattern from gated_delta.cpp.
    compute_encoder.set_input_array(state_in, 6);
  }

  compute_encoder.set_output_array(y, 7);
  compute_encoder.set_output_array(state_out, 8);
  compute_encoder.set_output_array(delta_log, 9);
  compute_encoder.set_bytes(T_val, 10);

  // Grid: (32, Dv, B * Hv)  ThreadGroup: (32, 4, 1) — same as
  // gated_delta_step.
  int B = static_cast<int>(state_out.shape(0));
  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, Dv_ / 4, B * Hv_), MTL::Size(32, 4, 1));
}

bool GatedDeltaStepRecord::is_equivalent(const Primitive& other) const {
  const GatedDeltaStepRecord& o =
      static_cast<const GatedDeltaStepRecord&>(other);
  return has_mask_ == o.has_mask_ && T_ == o.T_ && Dk_ == o.Dk_ &&
      Dv_ == o.Dv_ && Hk_ == o.Hk_ && Hv_ == o.Hv_;
}

// ============================================================================
// TapeReplay — re-fold accepted prefix onto pre-record snapshot
// ============================================================================

bool TapeReplay::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void TapeReplay::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& state_out = outputs[0];

  // Donate state_in if eligible (same load-into-registers-then-write
  // pattern as `gated_delta_step` allows aliasing).
  const auto& state_in = inputs[3];
  if (state_in.is_donatable() && state_in.flags().row_contiguous &&
      state_in.size() == state_out.size()) {
    state_out.copy_shared_buffer(state_in);
  } else {
    state_out.set_data(allocator::malloc(state_out.nbytes()));
  }

  // Build kernel name: state_replay_<dtype>_<Dk>_<Dv>_<Hk>_<Hv>
  std::string tname = type_to_name(state_out.dtype());
  std::string kname = "state_replay_" + tname + "_" + std::to_string(Dk_) +
      "_" + std::to_string(Dv_) + "_" + std::to_string(Hk_) + "_" +
      std::to_string(Hv_);

  // Function constant for mask selection (matches index 20 in the .metal).
  std::string hash_name = kname + (has_mask_ ? "_mask" : "_nomask");
  metal::MTLFCList func_consts = {
      {&has_mask_, MTL::DataType::DataTypeBool, 20},
  };

  auto kernel = d.get_kernel(kname, hash_name, func_consts);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int T_log_val = T_log_;
  int accepted_val = accepted_;

  // Buffer layout matches `.metal` declaration:
  //   0=delta_log, 1=k_log, 2=g_log, 3=state_in, 4=mask, 5=state_out,
  //   6=T_log, 7=accepted
  const auto& delta_log = inputs[0];
  const auto& k_log = inputs[1];
  const auto& g_log = inputs[2];

  compute_encoder.set_input_array(delta_log, 0);
  compute_encoder.set_input_array(k_log, 1);
  compute_encoder.set_input_array(g_log, 2);
  compute_encoder.set_input_array(state_in, 3);

  if (has_mask_) {
    const auto& mask = inputs[4];
    compute_encoder.set_input_array(mask, 4);
  } else {
    compute_encoder.set_input_array(state_in, 4);
  }

  compute_encoder.set_output_array(state_out, 5);
  compute_encoder.set_bytes(T_log_val, 6);
  compute_encoder.set_bytes(accepted_val, 7);

  // Grid: (32, Dv, B * Hv)  ThreadGroup: (32, 4, 1)
  int B = static_cast<int>(state_out.shape(0));
  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, Dv_ / 4, B * Hv_), MTL::Size(32, 4, 1));
}

bool TapeReplay::is_equivalent(const Primitive& other) const {
  const TapeReplay& o = static_cast<const TapeReplay&>(other);
  return has_mask_ == o.has_mask_ && T_log_ == o.T_log_ &&
      accepted_ == o.accepted_ && Dk_ == o.Dk_ && Dv_ == o.Dv_ &&
      Hk_ == o.Hk_ && Hv_ == o.Hv_;
}

} // namespace mlx::core::fast
