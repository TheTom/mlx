// Copyright © 2023-2024 Apple Inc.

#include <os/log.h>
#include <os/signpost.h>
#include <cstdlib>
#include <sstream>

#include <fmt/format.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace std {

// Required for putting the pointer in unordered_set.
template <class T>
struct hash<NS::SharedPtr<T>> {
  size_t operator()(const NS::SharedPtr<T>& p) const {
    return std::hash<T*>{}(p.get());
  }
};

} // namespace std

namespace mlx::core::metal {

// ─────────────────────────────────────────────────────────────────────
// os_signpost — per-kernel-dispatch tracing (opt-in)
// ─────────────────────────────────────────────────────────────────────
//
// Gated on `MLX_METAL_PROFILE=1`. When unset, `signpost_log()` returns
// `OS_LOG_DISABLED` — `os_signpost_*` calls against that handle
// short-circuit on a kernel-level atomic check and never touch the
// trace buffer. Overhead when off: one branch on `OS_LOG_DISABLED`.
//
// Subsystem `ai.mlx.metal`, category `PointsOfInterest`. Each kernel
// dispatch produces an interval labelled with the compute pipeline's
// `label()`, which Instruments Metal System Trace cross-references
// against the GPU-side kernel track so you can read CPU encoding and
// GPU execution on the same timeline.
os_log_t signpost_log() {
  static os_log_t g_log = []() -> os_log_t {
    const char* v = std::getenv("MLX_METAL_PROFILE");
    if (v && v[0] == '1') {
      return os_log_create("ai.mlx.metal", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    }
    return OS_LOG_DISABLED;
  }();
  return g_log;
}

bool signposts_enabled() {
  static bool g_enabled = []() {
    const char* v = std::getenv("MLX_METAL_PROFILE");
    return v && v[0] == '1';
  }();
  return g_enabled;
}

namespace {

constexpr const char* default_mtllib_path = METAL_PATH;

auto get_metal_version() {
  auto get_metal_version_ = []() {
    if (__builtin_available(macOS 26, iOS 26, tvOS 26, visionOS 26, *)) {
      return MTL::LanguageVersion4_0;
    } else if (__builtin_available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)) {
      return MTL::LanguageVersion3_2;
    } else {
      return MTL::LanguageVersion3_1;
    }
  };
  static auto metal_version_ = get_metal_version_();
  return metal_version_;
}

NS::SharedPtr<MTL::Device> load_device() {
  auto pool = new_scoped_memory_pool();
  auto devices = NS::TransferPtr(MTL::CopyAllDevices());
  // In headless, sandboxed, or virtualized macOS sessions CopyAllDevices()
  // returns an empty NSArray. Indexing object(0) on an empty array raises
  // an unrecoverable NSRangeException, so guard the access and fall back to
  // CreateSystemDefaultDevice (which can also return null).
  MTL::Device* first = nullptr;
  if (devices && devices->count() > 0) {
    first = static_cast<MTL::Device*>(devices->object(0));
  }
  auto device = first ? NS::RetainPtr(first)
                      : NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    throw std::runtime_error(
        "[metal::load_device] No Metal device available. This typically "
        "occurs in headless, sandboxed, or virtualized macOS sessions "
        "where the GPU is not accessible.");
  }
  return device;
}

std::pair<MTL::Library*, NS::Error*> load_library_from_path(
    MTL::Device* device,
    const char* path) {
  auto library = NS::String::string(path, NS::UTF8StringEncoding);
  NS::Error* error;
  auto lib = device->newLibrary(library, &error);

  return std::make_pair(lib, error);
}

#ifdef SWIFTPM_BUNDLE
MTL::Library* try_load_bundle(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string bundle_path = std::string(url->fileSystemRepresentation()) + "/" +
      SWIFTPM_BUNDLE + ".bundle";
  auto bundle = NS::Bundle::alloc()->init(
      NS::String::string(bundle_path.c_str(), NS::UTF8StringEncoding));
  if (bundle != nullptr) {
    std::string resource_path =
        std::string(bundle->resourceURL()->fileSystemRepresentation()) + "/" +
        lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, resource_path.c_str());
    if (lib) {
      return lib;
    }
  }
  return nullptr;
}

MTL::Library* try_load_framework(
    MTL::Device* device,
    NS::URL* url,
    const std::string& lib_name) {
  std::string resource_path = std::string(url->fileSystemRepresentation()) +
      "/" + lib_name + ".metallib";
  auto [lib, error] = load_library_from_path(device, resource_path.c_str());
  if (lib) {
    return lib;
  }
  return nullptr;
}
#endif

// Firstly, search for the metallib in the same path as this binary
std::pair<MTL::Library*, NS::Error*> load_colocated_library(
    MTL::Device* device,
    const std::string& relative_path) {
  auto path = current_binary_dir() / relative_path;
  if (!path.has_extension()) {
    path.replace_extension(".metallib");
  }

  return load_library_from_path(device, path.c_str());
}

std::pair<MTL::Library*, NS::Error*> load_swiftpm_library(
    MTL::Device* device,
    const std::string& lib_name) {
#ifdef SWIFTPM_BUNDLE
  MTL::Library* library =
      try_load_bundle(device, NS::Bundle::mainBundle()->bundleURL(), lib_name);
  if (library != nullptr) {
    return {library, nullptr};
  }
  auto bundles = NS::Bundle::allBundles();
  for (int i = 0, c = (int)bundles->count(); i < c; i++) {
    auto bundle = reinterpret_cast<NS::Bundle*>(bundles->object(i));
    library = try_load_bundle(device, bundle->resourceURL(), lib_name);
    if (library != nullptr) {
      return {library, nullptr};
    }
  }
  // if SWIFTPM_BUNDLE is a framework identifier, try loading from that
  auto frameworks = NS::Bundle::allFrameworks();
  for (int i = 0, c = (int)frameworks->count(); i < c; i++) {
    const auto bundle = reinterpret_cast<NS::Bundle*>(frameworks->object(i));
    const auto identifier = bundle->bundleIdentifier();
    if (identifier != nullptr &&
        !strcmp(identifier->utf8String(), SWIFTPM_BUNDLE)) {
      library = try_load_framework(device, bundle->resourceURL(), lib_name);
      if (library != nullptr) {
        return {library, nullptr};
      }
    }
  }
#endif
  return {nullptr, nullptr};
}

MTL::Library* load_default_library(MTL::Device* device) {
  NS::Error* error[5];
  MTL::Library* lib;
  // First try the colocated mlx.metallib
  std::tie(lib, error[0]) = load_colocated_library(device, "mlx");
  if (lib) {
    return lib;
  }

  std::tie(lib, error[1]) = load_colocated_library(device, "Resources/mlx");
  if (lib) {
    return lib;
  }

  // Then try default.metallib in a SwiftPM bundle if we have one
  std::tie(lib, error[2]) = load_swiftpm_library(device, "default");
  if (lib) {
    return lib;
  }

  // Try lo load resources from Framework resources if SwiftPM wrapped as a
  // dynamic framework.
  std::tie(lib, error[3]) = load_colocated_library(device, "Resources/default");
  if (lib) {
    return lib;
  }

  // Finally try default_mtllib_path
  std::tie(lib, error[4]) = load_library_from_path(device, default_mtllib_path);
  if (!lib) {
    std::ostringstream msg;
    msg << "Failed to load the default metallib. ";
    for (int i = 0; i < 5; i++) {
      if (error[i] != nullptr) {
        msg << error[i]->localizedDescription()->utf8String() << " ";
      }
    }
    throw std::runtime_error(msg.str());
  }
  return lib;
}

MTL::Library* load_library(
    MTL::Device* device,
    const std::string& lib_name,
    const std::string& lib_path) {
  // We have been given a path that ends in metallib so try to load it
  if (lib_path.size() > 9 &&
      std::equal(lib_path.end() - 9, lib_path.end(), ".metallib")) {
    auto [lib, error] = load_library_from_path(device, lib_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << lib_path << "> with error "
          << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // We have been given a path so try to load from lib_path / lib_name.metallib
  if (lib_path.size() > 0) {
    std::string full_path = lib_path + "/" + lib_name + ".metallib";
    auto [lib, error] = load_library_from_path(device, full_path.c_str());
    if (!lib) {
      std::ostringstream msg;
      msg << "Failed to load the metallib from <" << full_path
          << "> with error " << error->localizedDescription()->utf8String();
      throw std::runtime_error(msg.str());
    }
    return lib;
  }

  // Try to load the colocated library
  {
    auto [lib, error] = load_colocated_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  // Try to load the library from swiftpm
  {
    auto [lib, error] = load_swiftpm_library(device, lib_name);
    if (lib) {
      return lib;
    }
  }

  std::ostringstream msg;
  msg << "Failed to load the metallib " << lib_name << ".metallib. "
      << "We attempted to load it from <" << current_binary_dir() << "/"
      << lib_name << ".metallib>";
#ifdef SWIFTPM_BUNDLE
  msg << " and from the Swift PM bundle.";
#endif
  throw std::runtime_error(msg.str());
}

} // namespace

CommandEncoder::CommandEncoder(
    Device& d,
    int index,
    ResidencySet& residency_set)
    : device_(d) {
  auto pool = new_scoped_memory_pool();
  queue_ = NS::TransferPtr(device_.mtl_device()->newCommandQueue());
  if (!queue_) {
    throw std::runtime_error(
        "[metal::CommandEncoder] Failed to make new command queue.");
  }
  if (residency_set.mtl_residency_set()) {
    queue_->addResidencySet(residency_set.mtl_residency_set());
  }
  debug_set_stream_queue_label(queue_.get(), index);
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
}

CommandEncoder::~CommandEncoder() {
  exiting_ = true;
  synchronize();
  auto pool = new_scoped_memory_pool();
  buffer_.reset();
  queue_.reset();
}

void CommandEncoder::set_buffer(
    const MTL::Buffer* buf,
    int idx,
    int64_t offset /* = 0 */) {
  // Record as both input and output to ensure synchronization between command
  // buffers. First sighting in this CB triggers an explicit retain; the
  // all_inputs_ set already dedupes so we won't double-retain. Required by
  // Apple's MTLResourceHazardTrackingModeUntracked +
  // commandBufferWithUnretainedReferences contract.
  bool first = all_inputs_.insert((void*)buf).second;
  all_outputs_.insert((void*)buf);
  if (first && env::metal_retain_bound_buffers() && buf) {
    auto* mut_buf = const_cast<MTL::Buffer*>(buf);
    mut_buf->retain();
    retained_buffers_.push_back(mut_buf);
  }
  get_command_encoder()->setBuffer(buf, offset, idx);
}

void CommandEncoder::set_input_array(
    const array& a,
    int idx,
    int64_t offset /* = 0 */) {
  bool first = all_inputs_.insert(a.buffer().ptr()).second;
  if (first) {
    buffer_input_sizes_ += a.data_size();
    // See note in set_buffer above re: untracked-hazard / unretained-CB
    // contract. Retain on first sighting only.
    if (env::metal_retain_bound_buffers() && a.buffer().ptr()) {
      auto* mut_buf =
          static_cast<MTL::Buffer*>(const_cast<void*>(a.buffer().ptr()));
      mut_buf->retain();
      retained_buffers_.push_back(mut_buf);
    }
  }
  auto r_buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  needs_barrier_ =
      needs_barrier_ | (prev_outputs_.find(r_buf) != prev_outputs_.end());
  auto a_buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  get_command_encoder()->setBuffer(a_buf, a.offset() + offset, idx);
}

std::vector<MTL::Buffer*> CommandEncoder::take_retained_buffers() {
  std::vector<MTL::Buffer*> out;
  out.swap(retained_buffers_);
  return out;
}

void CommandEncoder::set_output_array(
    array& a,
    int idx,
    int64_t offset /* = 0 */) {
  // Add barriers before adding the output to the output set
  set_input_array(a, idx, offset);
  register_output_array(a);
}

void CommandEncoder::register_output_array(const array& a) {
  // Track ALL output bytes (not deduplicated): pooled allocators may hand out
  // the same buffer pointer repeatedly within one command buffer, and the
  // input-side dedup hides that re-use. Counting every output materialization
  // gives the commit heuristic a truer picture of in-flight memory pressure
  // than the input-only counter alone.
  if (all_outputs_.insert(a.buffer().ptr()).second) {
    buffer_output_sizes_ += a.data_size();
  }

  auto buf = static_cast<MTL::Resource*>(const_cast<void*>(a.buffer().ptr()));
  if (concurrent_) {
    concurrent_outputs_.insert(buf);
  } else {
    next_outputs_.insert(buf);
  }
}

void CommandEncoder::add_temporary(array arr) {
  temporaries_.push_back(std::move(arr));
}

void CommandEncoder::add_temporaries(std::vector<array> arrays) {
  temporaries_.insert(
      temporaries_.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

void CommandEncoder::maybeInsertBarrier() {
  if (needs_barrier_) {
    get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
    needs_barrier_ = false;
    prev_outputs_ = std::move(next_outputs_);
  } else {
    prev_outputs_.insert(next_outputs_.begin(), next_outputs_.end());
  }
  next_outputs_.clear();
}

void CommandEncoder::set_compute_pipeline_state(
    MTL::ComputePipelineState* kernel) {
  get_command_encoder()->setComputePipelineState(kernel);
  if (signposts_enabled()) {
    // Open a `kernel_dispatch` interval for this CPU encoding
    // window (set_compute_pipeline_state → set_buffer(s) →
    // dispatch_*). The dispatch_* call closes it. Label comes from
    // the pipeline state's debug name — set by `get_kernel` when
    // the PSO was first compiled, so every dispatch is self-
    // identifying in Instruments.
    auto log = signpost_log();
    cur_dispatch_signpost_id_ = os_signpost_id_generate(log);
    const char* name = device_.pso_name(kernel);
    os_signpost_interval_begin(
        log, cur_dispatch_signpost_id_, "kernel_dispatch", "%{public}s", name);
  }
}

void CommandEncoder::dispatch_threadgroups(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  maybeInsertBarrier();
  buffer_ops_++;
  total_dispatches_++;
  get_command_encoder()->dispatchThreadgroups(grid_dims, group_dims);
  if (cur_dispatch_signpost_id_ != OS_SIGNPOST_ID_NULL) {
    os_signpost_interval_end(
        signpost_log(), cur_dispatch_signpost_id_, "kernel_dispatch");
    cur_dispatch_signpost_id_ = OS_SIGNPOST_ID_NULL;
  }
}

void CommandEncoder::dispatch_threads(
    MTL::Size grid_dims,
    MTL::Size group_dims) {
  maybeInsertBarrier();
  buffer_ops_++;
  total_dispatches_++;
  get_command_encoder()->dispatchThreads(grid_dims, group_dims);
  if (cur_dispatch_signpost_id_ != OS_SIGNPOST_ID_NULL) {
    os_signpost_interval_end(
        signpost_log(), cur_dispatch_signpost_id_, "kernel_dispatch");
    cur_dispatch_signpost_id_ = OS_SIGNPOST_ID_NULL;
  }
}

void CommandEncoder::reset_dispatch_counter() {
  total_dispatches_ = 0;
}

uint64_t CommandEncoder::total_dispatches() const {
  return total_dispatches_;
}

void CommandEncoder::barrier() {
  get_command_encoder()->memoryBarrier(MTL::BarrierScopeBuffers);
}

void CommandEncoder::end_encoding() {
  // Each command encoder has a unique fence. We also store a map of
  // all previous outputs of command encoders to their corresponding fence.
  // - The command encoder records its inputs and outputs.
  // - Wait on a fence if any inputs in the encoder are outputs of a previous
  //   encoder.
  // - Update the map of outputs to include this command encoder's outputs.
  // - Always signal this command encoders fence.
  // - Add a completion handler for this command encoder that removes outputs
  //   from the map to limit the growth of the map and avoid unnecessary waits
  // - Temporaries are a special case as they do not cross command encoder
  //   boundaries. These can be removed early from the encoders inputs and
  //   outputs since they don't need synchronization.
  if (!encoder_) {
    return;
  }
  // Signpost the full fence/completion-handler lifecycle. Fires on
  // every compute-encoder boundary — barrier injection, command-
  // buffer rotation (`needs_commit`), the `synchronize` call after
  // a decode step, etc. Dispatch count varies by workload; typical
  // decode pattern commits a new buffer every ~200 ops (M-series
  // default `max_ops_per_buffer`), so end_encoding is called
  // several times per token.
  os_signpost_id_t sid = OS_SIGNPOST_ID_NULL;
  if (signposts_enabled()) {
    sid = os_signpost_id_generate(signpost_log());
    os_signpost_interval_begin(signpost_log(), sid, "end_encoding");
  }

  // Remove temporaries from inputs and outputs.
  for (auto& t : temporaries_) {
    all_outputs_.erase(t.buffer().ptr());
    all_inputs_.erase(t.buffer().ptr());
  }

  // Keep references to the fences we waited on and put them in the completion
  // handler so they are not prematurely released.
  std::unordered_set<NS::SharedPtr<MTL::Fence>> waiting_on;
  {
    std::lock_guard lk(outputs_mtx_);
    for (auto& in : all_inputs_) {
      if (auto it = prev_ce_outputs_.find(in); it != prev_ce_outputs_.end()) {
        // If we've already waited on a fence, don't wait on it again.
        if (waiting_on.find(it->second) == waiting_on.end()) {
          encoder_->waitForFence(it->second.get());
          waiting_on.insert(it->second);
        }
      }
    }
    for (auto& out : all_outputs_) {
      prev_ce_outputs_[out] = fence_;
    }
  }

  encoder_->updateFence(fence_.get());
  buffer_->addCompletedHandler([this,
                                fence = std::move(fence_),
                                temporaries = std::move(temporaries_),
                                all_outputs = std::move(all_outputs_),
                                waiting_on = std::move(waiting_on)](
                                   MTL::CommandBuffer*) mutable {
    std::lock_guard lk(outputs_mtx_);
    for (auto& o : all_outputs) {
      if (auto it = prev_ce_outputs_.find(o); it != prev_ce_outputs_.end()) {
        if (it->second == fence) {
          prev_ce_outputs_.erase(it);
        }
      }
    }
  });

  encoder_->endEncoding();
  encoder_.reset();
  needs_barrier_ = false;
  concurrent_ = false;
  prev_outputs_.clear();
  next_outputs_.clear();
  concurrent_outputs_.clear();
  all_inputs_.clear();

  if (sid != OS_SIGNPOST_ID_NULL) {
    os_signpost_interval_end(signpost_log(), sid, "end_encoding");
  }
}

bool CommandEncoder::needs_commit() const {
  auto [max_ops, max_mb] = device_.get_max_ops_mb_per_buffer();
  // Sum input + output bytes: inputs dominate for dense matmul-heavy ops,
  // outputs for large intermediate allocations (attention score matrices,
  // big prefill activations). Either alone undercounts memory pressure.
  size_t total_bytes = buffer_input_sizes_ + buffer_output_sizes_;
  return (buffer_ops_ > max_ops) || ((total_bytes >> 20) > max_mb);
}

void CommandEncoder::commit() {
  // Each `commit` submits the current MTLCommandBuffer to the queue
  // and allocates a fresh one. On M-series hardware this is where
  // Metal does internal command-buffer validation + scheduling —
  // the cost is small but non-trivial (low tens of µs per call),
  // and the per-token commit count is `ceil(dispatches / 200)` at
  // the default `max_ops_per_buffer`, so it adds up on models with
  // >500 dispatches per step.
  os_signpost_id_t sid = OS_SIGNPOST_ID_NULL;
  if (signposts_enabled()) {
    sid = os_signpost_id_generate(signpost_log());
    os_signpost_interval_begin(signpost_log(), sid, "commit");
  }
  buffer_->commit();
  buffer_ = NS::RetainPtr(queue_->commandBufferWithUnretainedReferences());
  buffer_ops_ = 0;
  buffer_input_sizes_ = 0;
  buffer_output_sizes_ = 0;
  if (sid != OS_SIGNPOST_ID_NULL) {
    os_signpost_interval_end(signpost_log(), sid, "commit");
  }
}

void CommandEncoder::synchronize() {
  // Wraps the CPU side of a full stream sync: end the current
  // encoder, commit its command buffer, then block on
  // `waitUntilCompleted`. The bulk of wall-clock time inside this
  // span is the wait — often 10s of ms per call during decode
  // loops that sync between steps. Visible under `ai.mlx.metal`
  // with nested `end_encoding` + `commit` intervals for attribution.
  os_signpost_id_t sid = OS_SIGNPOST_ID_NULL;
  if (signposts_enabled()) {
    sid = os_signpost_id_generate(signpost_log());
    os_signpost_interval_begin(signpost_log(), sid, "synchronize");
  }
  auto pool = new_scoped_memory_pool();
  auto cb = NS::RetainPtr(get_command_buffer());
  end_encoding();
  commit();
  cb->waitUntilCompleted();
  if (!exiting_) {
    if (cb->status() == MTL::CommandBufferStatusError) {
      throw std::runtime_error(
          fmt::format(
              "[METAL] Command buffer execution failed: {}.",
              cb->error()->localizedDescription()->utf8String()));
    }
  }
  if (sid != OS_SIGNPOST_ID_NULL) {
    os_signpost_interval_end(signpost_log(), sid, "synchronize");
  }
}

MTL::ComputeCommandEncoder* CommandEncoder::get_command_encoder() {
  if (!encoder_) {
    encoder_ = NS::RetainPtr(
        buffer_->computeCommandEncoder(MTL::DispatchTypeConcurrent));
    fence_ = NS::TransferPtr(device_.mtl_device()->newFence());
  }
  return encoder_.get();
}

Device::Device() : device_(load_device()), residency_set_(device_.get()) {
  auto pool = new_scoped_memory_pool();
  default_library_ = NS::TransferPtr(load_default_library(device_.get()));
  arch_ = env::metal_gpu_arch();
  if (arch_.empty()) {
    arch_ = std::string(device_->architecture()->name()->utf8String());
  }
  int ag_tens = 0;
  int ag_ones = 0;
  if (arch_.size() >= 3) {
    ag_tens = arch_[arch_.size() - 3] - '0';
    ag_ones = arch_[arch_.size() - 2] - '0';
    ag_tens = (ag_tens < 10 && ag_tens >= 0) ? ag_tens : 0;
    ag_ones = (ag_ones < 10 && ag_ones >= 0) ? ag_ones : 0;
  }
  arch_gen_ = ag_tens * 10 + ag_ones;
  auto arch = arch_.back();
  switch (arch) {
    case 'p': // phone
      max_ops_per_buffer_ = 20;
      max_mb_per_buffer_ = 40;
      break;
    case 'g': // base, pro
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
    case 's': // max
      // With the input+output byte tracking in needs_commit(), `max_ops`
      // can sit well above historical defaults without blowing up prefill
      // memory — the MB limit kicks in first when large intermediates
      // accumulate (attention score matrices etc).
      //
      // Benchmark evidence (Gemma4 E2B 4-bit turbo4v2, M1 Max):
      //   ops=200, mb=50  → decode 95.2 tok/s  @ peak 3.47 GB (old default)
      //   ops=500, mb=100 → decode 99.8 tok/s  @ peak 4.32 GB (+4.8% / +0.85GB)
      //   ops=500, mb=200 → decode 99.6 tok/s  @ peak 5.79 GB (+4.6% / +2.3GB)
      //
      // mb=100 is the sweet spot: decode wins materialize, memory bounded.
      max_ops_per_buffer_ = 500;
      max_mb_per_buffer_ = 100;
      break;
    case 'd': // ultra
      // Ultra has more memory headroom — raise mb too.
      max_ops_per_buffer_ = 500;
      max_mb_per_buffer_ = 200;
      break;
    default: // default to medium
      max_ops_per_buffer_ = 40;
      max_mb_per_buffer_ = 40;
      break;
  }
  max_ops_per_buffer_ = env::max_ops_per_buffer(max_ops_per_buffer_);
  max_mb_per_buffer_ = env::max_mb_per_buffer(max_mb_per_buffer_);
}

Device::~Device() = default;

MTL::Library* Device::get_library(
    const std::string& name,
    const std::string& path /* = "" */) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto new_lib = load_library(device_.get(), name, path.c_str());
  library_map_.insert({name, NS::TransferPtr(new_lib)});
  return new_lib;
}

NS::SharedPtr<MTL::Library> Device::build_library_(
    const std::string& source_string) {
  auto pool = new_scoped_memory_pool();

  auto ns_code =
      NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);

  NS::Error* error = nullptr;
  auto options = MTL::CompileOptions::alloc()->init()->autorelease();
  options->setFastMathEnabled(false);
  options->setLanguageVersion(get_metal_version());
#ifndef NDEBUG
  if (options->languageVersion() >= MTL::LanguageVersion3_2) {
    options->setEnableLogging(true);
  }
#endif
  auto mtl_lib = NS::TransferPtr(device_->newLibrary(ns_code, options, &error));

  // Throw error if unable to compile library
  if (!mtl_lib) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to build metal library from source\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_lib;
}

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    MTL::Library* mtl_lib) {
  auto pool = new_scoped_memory_pool();
  // Pull kernel from library
  auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  return NS::TransferPtr(mtl_lib->newFunction(ns_name));
}

NS::SharedPtr<MTL::Function> Device::get_function_(
    const std::string& name,
    const std::string& specialized_name,
    const MTLFCList& func_consts,
    MTL::Library* mtl_lib) {
  if (func_consts.empty() && (specialized_name == name)) {
    return get_function_(name, mtl_lib);
  }

  auto pool = new_scoped_memory_pool();

  // Prepare function constants
  auto mtl_func_consts =
      MTL::FunctionConstantValues::alloc()->init()->autorelease();

  for (auto [value, type, index] : func_consts) {
    mtl_func_consts->setConstantValue(value, type, index);
  }

  // Prepare function desc
  auto desc = MTL::FunctionDescriptor::functionDescriptor();
  desc->setName(NS::String::string(name.c_str(), NS::ASCIIStringEncoding));
  desc->setSpecializedName(
      NS::String::string(specialized_name.c_str(), NS::ASCIIStringEncoding));
  desc->setConstantValues(mtl_func_consts);

  // Pull kernel from library
  NS::Error* error = nullptr;
  auto mtl_function = NS::TransferPtr(mtl_lib->newFunction(desc, &error));

  // Throw error if unable to build metal function
  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load function " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return mtl_function;
}

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function) {
  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> kernel;

  if (mtl_function) {
    kernel =
        NS::TransferPtr(device_->newComputePipelineState(mtl_function, &error));
  }

  // Throw error if unable to compile metal function
  if (!mtl_function || !kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

NS::SharedPtr<MTL::ComputePipelineState> Device::get_kernel_(
    const std::string& name,
    const MTL::Function* mtl_function,
    const MTL::LinkedFunctions* linked_functions) {
  // Check inputs
  if (!linked_functions) {
    return get_kernel_(name, mtl_function);
  }

  if (!mtl_function) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    throw std::runtime_error(msg.str());
  }

  auto pool = new_scoped_memory_pool();

  // Prepare compute pipeline state descriptor
  auto desc = MTL::ComputePipelineDescriptor::alloc()->init()->autorelease();
  desc->setComputeFunction(mtl_function);
  desc->setLinkedFunctions(linked_functions);

  // Compile kernel to compute pipeline
  NS::Error* error = nullptr;
  auto kernel = NS::TransferPtr(device_->newComputePipelineState(
      desc, MTL::PipelineOptionNone, nullptr, &error));

  // Throw error if unable to compile metal function
  if (!kernel) {
    std::ostringstream msg;
    msg << "[metal::Device] Unable to load kernel " << name << "\n";
    if (error) {
      msg << error->localizedDescription()->utf8String() << "\n";
    }
    throw std::runtime_error(msg.str());
  }

  return kernel;
}

MTL::Library* Device::get_library(
    const std::string& name,
    const std::function<std::string(void)>& builder) {
  {
    std::shared_lock rlock(library_mtx_);
    if (auto it = library_map_.find(name); it != library_map_.end()) {
      return it->second.get();
    }
  }

  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    return it->second.get();
  }

  auto mtl_lib = build_library_(builder());
  library_map_.insert({name, mtl_lib});
  return mtl_lib.get();
}

void Device::clear_library(const std::string& name) {
  std::unique_lock wlock(library_mtx_);
  if (auto it = library_map_.find(name); it != library_map_.end()) {
    library_kernels_.erase(it->second.get());
    library_map_.erase(it);
  }
}

NS::SharedPtr<MTL::LinkedFunctions> Device::get_linked_functions_(
    const std::vector<MTL::Function*>& funcs) {
  if (funcs.empty()) {
    return nullptr;
  }

  auto pool = new_scoped_memory_pool();
  auto lfuncs = NS::TransferPtr(MTL::LinkedFunctions::linkedFunctions());
  NS::Array* funcs_arr = NS::Array::array(
      reinterpret_cast<const NS::Object* const*>(funcs.data()), funcs.size());
  lfuncs->setPrivateFunctions(funcs_arr);
  return lfuncs;
}

MTL::ComputePipelineState* Device::get_kernel_(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  // Single writer allowed
  std::unique_lock wlock(kernel_mtx_);

  // Try loading again to avoid loading twice
  auto& kernel_map_ = library_kernels_[mtl_lib];
  if (auto it = kernel_map_.find(hash_name); it != kernel_map_.end()) {
    return it->second.get();
  }

  auto pool = new_scoped_memory_pool();

  // Pull kernel from library
  auto mtl_function = get_function_(base_name, hash_name, func_consts, mtl_lib);

  // Compile kernel to compute pipeline
  auto mtl_linked_funcs = get_linked_functions_(linked_functions);
  auto kernel =
      get_kernel_(hash_name, mtl_function.get(), mtl_linked_funcs.get());

  // Add kernel to cache
  kernel_map_.insert({hash_name, kernel});
  // Index the raw PSO pointer against its hash name so the
  // `os_signpost`-based kernel tracer in `CommandEncoder::
  // set_compute_pipeline_state` can recover a human-readable label
  // (metal-cpp exposes `label()` but no setter for PSOs, and the
  // underlying API only accepts labels via the pipeline descriptor
  // — which the simple `newComputePipelineState(function, ...)`
  // path used above does not plumb through).
  pso_names_[kernel.get()] = hash_name;

  return kernel.get();
}

const char* Device::pso_name(const MTL::ComputePipelineState* pso) const {
  std::shared_lock lock(const_cast<std::shared_mutex&>(kernel_mtx_));
  auto it = pso_names_.find(pso);
  if (it == pso_names_.end()) {
    return "?";
  }
  return it->second.c_str();
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name /* = "" */,
    const MTLFCList& func_consts /* = {} */,
    const std::vector<MTL::Function*>& linked_functions /* = {} */) {
  const auto& kname = hash_name.empty() ? base_name : hash_name;
  {
    // Multiple readers allowed
    std::shared_lock lock(kernel_mtx_);

    // Look for cached kernel
    auto& kernel_map_ = library_kernels_[mtl_lib];
    if (auto it = kernel_map_.find(kname); it != kernel_map_.end()) {
      return it->second.get();
    }
  }
  return get_kernel_(base_name, mtl_lib, kname, func_consts, linked_functions);
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& hash_name /*  = "" */,
    const MTLFCList& func_consts /*  = {} */,
    const std::vector<MTL::Function*>& linked_functions /*  = {} */) {
  return get_kernel(
      base_name,
      default_library_.get(),
      hash_name,
      func_consts,
      linked_functions);
}

Device& device(mlx::core::Device) {
  // Leak singleton device intentionally, to avoid cases where a compute kernel
  // returns and tries to access the object after it has been freed by the main
  // thread teardown.
  static Device* metal_device = new Device;
  return *metal_device;
}

CommandEncoder& get_command_encoder(Stream s) {
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    // Lazily initialize the command encoder for this stream on the current
    // thread. This handles Swift structured concurrency where Tasks can migrate
    // between threads, and the thread-local encoder map may not have the stream
    // registered.
    auto& d = device(s.device);
    it = encoders.try_emplace(s.index, d, s.index, d.residency_set()).first;
  }
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

NS::SharedPtr<NS::AutoreleasePool> new_scoped_memory_pool() {
  return NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
}

bool is_nax_available() {
#ifdef MLX_METAL_NO_NAX
  return false;
#else
  auto _check_nax = []() {
    bool can_use_nax = false;
    if (__builtin_available(
            macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
      can_use_nax = true;
    }
    auto& d = metal::device(mlx::core::Device::gpu);
    auto arch = d.get_architecture().back();
    auto gen = d.get_architecture_gen();
    can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
    return can_use_nax;
  };
  static bool is_nax_available_ = _check_nax();
  return is_nax_available_;
#endif
}

} // namespace mlx::core::metal
