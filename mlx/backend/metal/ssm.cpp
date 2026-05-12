// Copyright (C) 2026 Eric Kryski.
// SSMStep Metal primitive dispatch.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

bool SSMStep::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void SSMStep::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& out = outputs[0];
  auto& state_out = outputs[1];

  // Allocate main output
  out.set_data(allocator::malloc(out.nbytes()));

  // The kernel reads i_state[idx] then writes o_state[idx] per element
  // with unique indices per thread, so the two can safely alias.
  // Donate state_in's buffer to avoid a second allocation.
  const auto& state_in = inputs[6];
  if (state_in.is_donatable() && state_in.flags().row_contiguous &&
      state_in.size() == state_out.size()) {
    state_out.copy_shared_buffer(state_in);
  } else {
    state_out.set_data(allocator::malloc(state_out.nbytes()));
  }

  // Build kernel name: ssm_step_{type}_{Dh}_{Ds}_{H}_{G}
  std::string tname = type_to_name(out.dtype());
  std::string kname = "ssm_step_" + tname + "_" + std::to_string(Dh_) + "_" +
      std::to_string(Ds_) + "_" + std::to_string(H_) + "_" + std::to_string(G_);

  auto kernel = d.get_kernel(kname);

  // Buffer layout:
  //   0=X, 1=A_log, 2=B, 3=C, 4=D, 5=dt, 6=state_in,
  //   7=out, 8=state_out
  const auto& X = inputs[0];
  const auto& A_log = inputs[1];
  const auto& B = inputs[2];
  const auto& C = inputs[3];
  const auto& D_arr = inputs[4];
  const auto& dt = inputs[5];

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  compute_encoder.set_input_array(X, 0);
  compute_encoder.set_input_array(A_log, 1);
  compute_encoder.set_input_array(B, 2);
  compute_encoder.set_input_array(C, 3);
  compute_encoder.set_input_array(D_arr, 4);
  compute_encoder.set_input_array(dt, 5);
  compute_encoder.set_input_array(state_in, 6);
  compute_encoder.set_output_array(out, 7);
  compute_encoder.set_output_array(state_out, 8);

  // Determine batch from X shape: X is [N, Dh] where N = batch * H
  // state_in is [N, Dh, Ds] where N = batch * H
  int N = static_cast<int>(X.shape(0));
  int batch = N / H_;

  // Grid: (32, Dh, H * batch)  ThreadGroup: (32, 8, 1)
  // threadgroups: (1, Dh/8, H * batch)
  compute_encoder.dispatch_threadgroups(
      MTL::Size(1, Dh_ / 8, H_ * batch), MTL::Size(32, 8, 1));
}

bool SSMStep::is_equivalent(const Primitive& other) const {
  const SSMStep& o = static_cast<const SSMStep&>(other);
  return Dh_ == o.Dh_ && Ds_ == o.Ds_ && H_ == o.H_ && G_ == o.G_;
}

} // namespace mlx::core::fast
