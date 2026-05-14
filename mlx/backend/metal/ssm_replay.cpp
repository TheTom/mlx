// Copyright © 2026 Eric Kryski. Spec 040 — Mamba state-replay dispatch.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

// ============================================================================
// SSMStepRecord: sequential T-loop with delta capture.
// ============================================================================
void SSMStepRecord::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& y_out = outputs[0];
  auto& state_out = outputs[1];
  auto& dA_log_out = outputs[2];
  auto& dBx_log_out = outputs[3];
  y_out.set_data(allocator::malloc(y_out.nbytes()));
  state_out.set_data(allocator::malloc(state_out.nbytes()));
  dA_log_out.set_data(allocator::malloc(dA_log_out.nbytes()));
  dBx_log_out.set_data(allocator::malloc(dBx_log_out.nbytes()));

  const array& x = inputs[0];
  const array& A_log = inputs[1];
  const array& B = inputs[2];
  const array& C = inputs[3];
  const array& D_arr = inputs[4];
  const array& dt = inputs[5];
  const array& state_in = inputs[6];
  const array* mask = has_mask_ ? &inputs[7] : nullptr;

  int B_dim = x.shape(0);
  int T_total = x.shape(1);

  std::string tname = type_to_name(y_out.dtype());
  std::string kname = "ssm_step_record_" + tname + "_" + std::to_string(Dh_) +
      "_" + std::to_string(Ds_) + "_" + std::to_string(H_) + "_" +
      std::to_string(G_);

  metal::MTLFCList func_consts = {
      {&has_mask_, MTL::DataType::DataTypeBool, 50},
  };
  std::string hash_name = kname + (has_mask_ ? "_mask" : "_nomask");

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(A_log, 1);
  compute_encoder.set_input_array(B, 2);
  compute_encoder.set_input_array(C, 3);
  compute_encoder.set_input_array(D_arr, 4);
  compute_encoder.set_input_array(dt, 5);
  compute_encoder.set_input_array(state_in, 6);
  compute_encoder.set_output_array(y_out, 7);
  compute_encoder.set_output_array(state_out, 8);
  compute_encoder.set_output_array(dA_log_out, 9);
  compute_encoder.set_output_array(dBx_log_out, 10);
  if (has_mask_) {
    compute_encoder.set_input_array(*mask, 11);
  }
  compute_encoder.set_bytes(T_total, 12);

  // Grid: (32, Dh, B * H). ThreadGroup: (32, 8, 1). One simdgroup per (b, h,
  // dh).
  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, Dh_ / 8, H_ * B_dim), MTL::Size(32, 8, 1));
}

bool SSMStepRecord::is_equivalent(const Primitive& other) const {
  const SSMStepRecord& o = static_cast<const SSMStepRecord&>(other);
  return Dh_ == o.Dh_ && Ds_ == o.Ds_ && H_ == o.H_ && G_ == o.G_ &&
      has_mask_ == o.has_mask_;
}

std::vector<Shape> SSMStepRecord::output_shapes(
    const std::vector<array>& inputs) {
  const array& x = inputs[0];
  const array& B = inputs[2];
  const array& state_in = inputs[6];
  int B_dim = x.shape(0);
  int T = x.shape(1);
  int H = x.shape(2);
  int dh = x.shape(3);
  int ds = B.shape(3);
  return {
      {B_dim, T, H, dh}, // y
      state_in.shape(), // state_out
      {B_dim, T, H, ds}, // dA_log
      {B_dim, T, H, dh, ds} // dBx_log
  };
}

// ============================================================================
// SSMReplay: replay first k log entries onto state snapshot.
// ============================================================================
void SSMReplay::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  const array& state_snapshot = inputs[0];
  const array& dA_log = inputs[1];
  const array& dBx_log = inputs[2];
  const array* mask = has_mask_ ? &inputs[3] : nullptr;

  int B_dim = state_snapshot.shape(0);
  int T_total = dA_log.shape(1);

  std::string tname = type_to_name(out.dtype());
  std::string kname = "ssm_replay_" + tname + "_" + std::to_string(Dh_) + "_" +
      std::to_string(Ds_) + "_" + std::to_string(H_);

  metal::MTLFCList func_consts = {
      {&has_mask_, MTL::DataType::DataTypeBool, 51},
  };
  std::string hash_name = kname + (has_mask_ ? "_mask" : "_nomask");

  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(state_snapshot, 0);
  compute_encoder.set_input_array(dA_log, 1);
  compute_encoder.set_input_array(dBx_log, 2);
  compute_encoder.set_output_array(out, 3);
  if (has_mask_) {
    compute_encoder.set_input_array(*mask, 4);
  }
  compute_encoder.set_bytes(accepted_prefix_, 5);
  compute_encoder.set_bytes(T_total, 6);

  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, Dh_ / 8, H_ * B_dim), MTL::Size(32, 8, 1));
}

bool SSMReplay::is_equivalent(const Primitive& other) const {
  const SSMReplay& o = static_cast<const SSMReplay&>(other);
  return Dh_ == o.Dh_ && Ds_ == o.Ds_ && H_ == o.H_ &&
      accepted_prefix_ == o.accepted_prefix_ && has_mask_ == o.has_mask_;
}

std::vector<Shape> SSMReplay::output_shapes(const std::vector<array>& inputs) {
  return {inputs[0].shape()};
}

} // namespace mlx::core::fast
