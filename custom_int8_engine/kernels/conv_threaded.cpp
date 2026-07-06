#include "y26_k1x_threaded_conv.h"

#include "y26_k1x_activation.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxCluster0Threads = 4;

enum ThreadTaskKind {
    kThreadTaskConv = 0,
    kThreadTaskActivationRvvF32 = 1,
};

int output_dim(int input, int kernel, int stride, int pad) {
    return (input + 2 * pad - kernel) / stride + 1;
}

int output_h_for(const Y26Stage7ConvNodeConfig& cfg) {
    return output_dim(cfg.params.input_h, cfg.kernel_h, cfg.params.stride_h, cfg.params.pad_h);
}

int output_w_for(const Y26Stage7ConvNodeConfig& cfg) {
    return output_dim(cfg.params.input_w, cfg.kernel_w, cfg.params.stride_w, cfg.params.pad_w);
}

bool supported_config(const Y26Stage7ConvNodeConfig* cfg, int thread_count) {
    if (cfg == nullptr || thread_count < 1 || thread_count > kMaxCluster0Threads) {
        return false;
    }
    if (cfg->params.input_h <= 0 || cfg->params.input_w <= 0 || cfg->params.input_c <= 0 ||
        cfg->params.output_c <= 0 || cfg->params.stride_h != 1 || cfg->params.stride_w != 1 ||
        cfg->kernel_h != 3 || cfg->kernel_w != 3 || cfg->params.pad_h != 1 || cfg->params.pad_w != 1 ||
        cfg->weights_ohwi_s8 == nullptr || cfg->bias_i32 == nullptr || cfg->weight_scales == nullptr) {
        return false;
    }
    if (cfg->weight_scale_count < static_cast<std::size_t>(cfg->params.output_c) ||
        cfg->bias_count < static_cast<std::size_t>(cfg->params.output_c)) {
        return false;
    }
    const std::size_t expected_weight_count = static_cast<std::size_t>(cfg->params.output_c) *
                                              static_cast<std::size_t>(cfg->kernel_h) *
                                              static_cast<std::size_t>(cfg->kernel_w) *
                                              static_cast<std::size_t>(cfg->params.input_c);
    return cfg->weight_count >= expected_weight_count && output_h_for(*cfg) > 0 && output_w_for(*cfg) > 0;
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

bool pin_current_thread_to_cpu(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

int current_cpu() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

}  // namespace

struct ThreadWorker {
    Y26ThreadedConvWorkerPlan plan {};
    Y26Stage7ConvNodeConfig root_cfg {};
    Y26Stage7ConvNodeConfig cfg {};
    Y26PrepackedConvWeights* weights = nullptr;
    Y26ConvWorkspace* conv_workspace = nullptr;
    std::vector<std::int32_t> raw;
    std::vector<std::int32_t> corrected;
    std::thread thread;
    int status = Y26_CONV_STATUS_SUCCESS;
    int observed_cpu = -1;
    int affinity_set = 0;
    double total_us = 0.0;
    double correction_us = 0.0;
};

struct Y26ThreadedConvWorkspace {
    Y26Stage7ConvNodeConfig root_cfg {};
    Y26ThreadedConvPlan plan {};
    std::vector<ThreadWorker> workers;
    std::barrier<>* start_barrier = nullptr;
    std::barrier<>* done_barrier = nullptr;
    std::atomic<int> ready_count {0};
    std::mutex ready_mutex;
    std::condition_variable ready_cv;
    const std::int8_t* current_input = nullptr;
    std::int32_t* current_output = nullptr;
    int task_kind = kThreadTaskConv;
    Y26ActivationRequantParams activation_params {};
    const std::int32_t* activation_input_i32 = nullptr;
    const std::int8_t* activation_lut_s8 = nullptr;
    std::int8_t* activation_output_s8 = nullptr;
    bool stop = false;
    bool workers_started = false;
};

namespace {

void clear_worker(ThreadWorker& worker) {
    y26_prepacked_conv_weights_destroy(worker.weights);
    y26_conv_workspace_destroy(worker.conv_workspace);
    worker.weights = nullptr;
    worker.conv_workspace = nullptr;
}

void destroy_workspace(Y26ThreadedConvWorkspace* workspace) {
    if (workspace == nullptr) {
        return;
    }
    if (workspace->workers_started && workspace->start_barrier != nullptr) {
        workspace->stop = true;
        workspace->start_barrier->arrive_and_wait();
        for (ThreadWorker& worker : workspace->workers) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
    }
    for (ThreadWorker& worker : workspace->workers) {
        clear_worker(worker);
    }
    delete workspace->start_barrier;
    delete workspace->done_barrier;
    delete workspace;
}

void copy_worker_output(const ThreadWorker& worker, std::int32_t* output_i32) {
    const int output_w = worker.plan.output_rows_written > 0 ? worker.cfg.params.input_w : 0;
    const int output_c = worker.cfg.params.output_c;
    const std::size_t row_values = static_cast<std::size_t>(output_w) * static_cast<std::size_t>(output_c);
    const std::int32_t* src = worker.corrected.data() +
                              static_cast<std::size_t>(worker.plan.local_output_offset) * row_values;
    std::int32_t* dst = output_i32 + static_cast<std::size_t>(worker.plan.row_begin) * row_values;
    for (int row = 0; row < worker.plan.output_rows_written; ++row) {
        std::memcpy(dst + static_cast<std::size_t>(row) * row_values,
                    src + static_cast<std::size_t>(row) * row_values,
                    row_values * sizeof(std::int32_t));
    }
}

int run_worker_once(ThreadWorker& worker,
                    const std::int8_t* input_s8,
                    std::int32_t* output_i32) {
    const auto begin = Clock::now();
    const std::int8_t* local_input =
        input_s8 + static_cast<std::size_t>(worker.plan.input_row_begin) *
                       static_cast<std::size_t>(worker.root_cfg.params.input_w) *
                       static_cast<std::size_t>(worker.root_cfg.params.input_c);
    int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(local_input,
                                                          worker.weights,
                                                          worker.raw.data(),
                                                          worker.cfg.input_storage_zero_point_s8,
                                                          worker.conv_workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    const auto correction_begin = Clock::now();
    if (status == Y26_CONV_STATUS_SUCCESS) {
        const int local_output_m = worker.plan.local_output_h * worker.cfg.params.input_w;
        status = y26_conv2d_apply_u8_as_s8_correction_nhwc(worker.raw.data(),
                                                           worker.cfg.bias_i32,
                                                           y26_prepacked_conv_weights_sums(worker.weights),
                                                           worker.corrected.data(),
                                                           local_output_m,
                                                           worker.cfg.params.output_c,
                                                           worker.cfg.activation_zero_point_u8);
    }
    if (status == Y26_CONV_STATUS_SUCCESS) {
        copy_worker_output(worker, output_i32);
    }
    const auto end = Clock::now();
    worker.total_us = elapsed_us(begin, end);
    worker.correction_us = elapsed_us(correction_begin, end);
    return status;
}

int run_worker_activation_once(ThreadWorker& worker,
                               const Y26ActivationRequantParams& params,
                               const std::int32_t* input_i32,
                               const std::int8_t* lut_s8,
                               std::int8_t* output_s8) {
    const int output_w = worker.plan.output_rows_written > 0 ? worker.root_cfg.params.input_w : 0;
    const int channels = params.channels;
    if (worker.plan.output_rows_written == 0) {
        worker.total_us = 0.0;
        worker.correction_us = 0.0;
        return Y26_CONV_STATUS_SUCCESS;
    }
    const std::size_t row_values = static_cast<std::size_t>(output_w) * static_cast<std::size_t>(channels);
    Y26ActivationRequantParams local_params = params;
    local_params.element_count = static_cast<std::size_t>(worker.plan.output_rows_written) * row_values;
    const std::size_t offset = static_cast<std::size_t>(worker.plan.row_begin) * row_values;
    const auto begin = Clock::now();
    const int status = y26_activation_requant_silu_int8_lut_rvv_f32(
        &local_params, input_i32 + offset, lut_s8, output_s8 + offset);
    const auto end = Clock::now();
    worker.total_us = elapsed_us(begin, end);
    worker.correction_us = 0.0;
    return status;
}

void worker_loop(Y26ThreadedConvWorkspace* workspace, std::size_t worker_index) {
    ThreadWorker& worker = workspace->workers[worker_index];
    worker.affinity_set = pin_current_thread_to_cpu(worker.plan.cpu) ? 1 : 0;
    worker.observed_cpu = current_cpu();
    workspace->ready_count.fetch_add(1, std::memory_order_release);
    workspace->ready_cv.notify_one();
    for (;;) {
        workspace->start_barrier->arrive_and_wait();
        if (workspace->stop) {
            break;
        }
        if (workspace->task_kind == kThreadTaskActivationRvvF32) {
            worker.status = run_worker_activation_once(worker,
                                                       workspace->activation_params,
                                                       workspace->activation_input_i32,
                                                       workspace->activation_lut_s8,
                                                       workspace->activation_output_s8);
        } else {
            worker.status = run_worker_once(worker, workspace->current_input, workspace->current_output);
        }
        workspace->done_barrier->arrive_and_wait();
    }
}

bool configure_workers(Y26ThreadedConvWorkspace* workspace, const Y26Stage7ConvNodeConfig& cfg, int thread_count) {
    const int output_h = output_h_for(cfg);
    const int output_w = output_w_for(cfg);
    workspace->workers.reserve(static_cast<std::size_t>(thread_count));
    workspace->plan.thread_count = thread_count;
    workspace->plan.output_h = output_h;
    workspace->plan.output_w = output_w;
    workspace->plan.output_c = cfg.params.output_c;
    workspace->plan.kernel_h = cfg.kernel_h;
    workspace->plan.kernel_w = cfg.kernel_w;
    workspace->plan.input_c = cfg.params.input_c;
    const long long base_macs = static_cast<long long>(output_h) * output_w * cfg.params.output_c *
                                cfg.kernel_h * cfg.kernel_w * cfg.params.input_c;

    for (int tid = 0; tid < thread_count; ++tid) {
        const int row_begin = (output_h * tid) / thread_count;
        const int row_end = (output_h * (tid + 1)) / thread_count;
        const bool top_chunk = row_begin == 0;
        const bool bottom_chunk = row_end == output_h;
        const int input_row_begin = thread_count == 1 ? 0 : std::max(0, row_begin - 1);
        const int input_row_end = thread_count == 1 ? cfg.params.input_h : std::min(cfg.params.input_h, row_end + 1);
        const int symmetric_pad_h = top_chunk || bottom_chunk ? 1 : 0;

        ThreadWorker worker {};
        worker.root_cfg = cfg;
        worker.plan.cpu = tid;
        worker.plan.row_begin = row_begin;
        worker.plan.row_end = row_end;
        worker.plan.input_row_begin = input_row_begin;
        worker.plan.input_row_end = input_row_end;
        worker.plan.local_output_offset = (!top_chunk && bottom_chunk) ? 1 : 0;
        worker.plan.output_rows_written = row_end - row_begin;
        worker.cfg = cfg;
        worker.cfg.params.input_h = input_row_end - input_row_begin;
        worker.cfg.params.pad_h = symmetric_pad_h;
        worker.plan.local_output_h = output_h_for(worker.cfg);
        if (output_w_for(worker.cfg) != output_w ||
            worker.plan.local_output_h < worker.plan.output_rows_written + worker.plan.local_output_offset) {
            clear_worker(worker);
            return false;
        }
        worker.plan.discarded_rows = worker.plan.local_output_h - worker.plan.output_rows_written;
        worker.plan.overcomputed_rows = worker.plan.discarded_rows;
        worker.weights = y26_prepacked_conv_weights_create_mmt4d_s8(worker.cfg.weights_ohwi_s8,
                                                                    &worker.cfg.params,
                                                                    worker.cfg.kernel_h,
                                                                    worker.cfg.kernel_w,
                                                                    worker.cfg.node_name,
                                                                    worker.cfg.weight_scales);
        worker.conv_workspace =
            y26_conv_workspace_create(&worker.cfg.params, worker.cfg.kernel_h, worker.cfg.kernel_w);
        if (worker.weights == nullptr || worker.conv_workspace == nullptr) {
            clear_worker(worker);
            return false;
        }
        worker.plan.prepacked_bytes = y26_prepacked_conv_weights_total_bytes(worker.weights);
        worker.plan.workspace_bytes = y26_conv_workspace_bytes(worker.conv_workspace);
        worker.raw.resize(static_cast<std::size_t>(worker.plan.local_output_h) *
                          static_cast<std::size_t>(output_w) * static_cast<std::size_t>(cfg.params.output_c));
        worker.corrected.resize(worker.raw.size());
        workspace->plan.total_overcomputed_rows += worker.plan.overcomputed_rows;
        workspace->plan.total_discarded_rows += worker.plan.discarded_rows;
        workspace->plan.estimated_extra_macs += static_cast<long long>(worker.plan.overcomputed_rows) * output_w *
                                                cfg.params.output_c * cfg.kernel_h * cfg.kernel_w *
                                                cfg.params.input_c;
        workspace->plan.workers[tid] = worker.plan;
        workspace->workers.push_back(std::move(worker));
    }
    workspace->plan.estimated_extra_mac_pct =
        base_macs > 0 ? 100.0 * static_cast<double>(workspace->plan.estimated_extra_macs) /
                            static_cast<double>(base_macs)
                      : 0.0;
    return true;
}

}  // namespace

extern "C" Y26ThreadedConvWorkspace* y26_threaded_conv_create_spatial_rows(
    const Y26Stage7ConvNodeConfig* cfg,
    int thread_count) {
    if (!supported_config(cfg, thread_count)) {
        return nullptr;
    }
    Y26ThreadedConvWorkspace* workspace = new (std::nothrow) Y26ThreadedConvWorkspace();
    if (workspace == nullptr) {
        return nullptr;
    }
    workspace->root_cfg = *cfg;
    if (!configure_workers(workspace, *cfg, thread_count)) {
        destroy_workspace(workspace);
        return nullptr;
    }
    workspace->start_barrier = new (std::nothrow) std::barrier<>(thread_count + 1);
    workspace->done_barrier = new (std::nothrow) std::barrier<>(thread_count + 1);
    if (workspace->start_barrier == nullptr || workspace->done_barrier == nullptr) {
        destroy_workspace(workspace);
        return nullptr;
    }
    workspace->workers_started = true;
    for (std::size_t i = 0; i < workspace->workers.size(); ++i) {
        workspace->workers[i].thread = std::thread(worker_loop, workspace, i);
    }
    std::unique_lock<std::mutex> lock(workspace->ready_mutex);
    workspace->ready_cv.wait(lock, [&]() {
        return workspace->ready_count.load(std::memory_order_acquire) == thread_count;
    });
    return workspace;
}

extern "C" void y26_threaded_conv_destroy(Y26ThreadedConvWorkspace* workspace) {
    destroy_workspace(workspace);
}

extern "C" int y26_threaded_conv_run_ime_cluster0(const Y26ThreadedConvWorkspace* const_workspace,
                                                   const std::int8_t* input_nhwc_s8,
                                                   std::int32_t* corrected_output_nhwc,
                                                   Y26ThreadedConvTimingUs* timing) {
    if (const_workspace == nullptr || input_nhwc_s8 == nullptr || corrected_output_nhwc == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    }
    Y26ThreadedConvWorkspace* workspace = const_cast<Y26ThreadedConvWorkspace*>(const_workspace);
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
    }
    workspace->current_input = input_nhwc_s8;
    workspace->current_output = corrected_output_nhwc;
    workspace->task_kind = kThreadTaskConv;
    for (ThreadWorker& worker : workspace->workers) {
        worker.status = Y26_CONV_STATUS_SUCCESS;
        worker.total_us = 0.0;
        worker.correction_us = 0.0;
    }
    const auto begin = Clock::now();
    workspace->start_barrier->arrive_and_wait();
    workspace->done_barrier->arrive_and_wait();
    const auto end = Clock::now();

    int status = Y26_CONV_STATUS_SUCCESS;
    double worker_max = 0.0;
    double worker_min = workspace->workers.empty() ? 0.0 : workspace->workers.front().total_us;
    double correction_max = 0.0;
    for (const ThreadWorker& worker : workspace->workers) {
        if (worker.status != Y26_CONV_STATUS_SUCCESS && status == Y26_CONV_STATUS_SUCCESS) {
            status = worker.status;
        }
        worker_max = std::max(worker_max, worker.total_us);
        worker_min = std::min(worker_min, worker.total_us);
        correction_max = std::max(correction_max, worker.correction_us);
    }
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        timing->conv_us = timing->total_us;
        timing->worker_max_us = worker_max;
        timing->worker_min_us = worker_min;
        timing->correction_us = correction_max;
    }
    return status;
}

extern "C" int y26_threaded_conv_run_activation_rvv_f32_rows(
    const Y26ThreadedConvWorkspace* const_workspace,
    const Y26ActivationRequantParams* params,
    const std::int32_t* producer_i32,
    const std::int8_t* lut_256_s8,
    std::int8_t* consumer_input_s8,
    Y26ThreadedActivationTimingUs* timing) {
    if (const_workspace == nullptr || params == nullptr || producer_i32 == nullptr || lut_256_s8 == nullptr ||
        consumer_input_s8 == nullptr || params->channels <= 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    Y26ThreadedConvWorkspace* workspace = const_cast<Y26ThreadedConvWorkspace*>(const_workspace);
    const std::size_t expected = static_cast<std::size_t>(workspace->plan.output_h) *
                                 static_cast<std::size_t>(workspace->plan.output_w) *
                                 static_cast<std::size_t>(params->channels);
    if (params->element_count != expected) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
    }
    workspace->task_kind = kThreadTaskActivationRvvF32;
    workspace->activation_params = *params;
    workspace->activation_input_i32 = producer_i32;
    workspace->activation_lut_s8 = lut_256_s8;
    workspace->activation_output_s8 = consumer_input_s8;
    for (ThreadWorker& worker : workspace->workers) {
        worker.status = Y26_CONV_STATUS_SUCCESS;
        worker.total_us = 0.0;
        worker.correction_us = 0.0;
    }
    const auto begin = Clock::now();
    workspace->start_barrier->arrive_and_wait();
    workspace->done_barrier->arrive_and_wait();
    const auto end = Clock::now();

    int status = Y26_CONV_STATUS_SUCCESS;
    double worker_max = 0.0;
    double worker_min = workspace->workers.empty() ? 0.0 : workspace->workers.front().total_us;
    for (const ThreadWorker& worker : workspace->workers) {
        if (worker.status != Y26_CONV_STATUS_SUCCESS && status == Y26_CONV_STATUS_SUCCESS) {
            status = worker.status;
        }
        worker_max = std::max(worker_max, worker.total_us);
        worker_min = std::min(worker_min, worker.total_us);
    }
    if (timing != nullptr) {
        timing->total_us = elapsed_us(begin, end);
        timing->worker_max_us = worker_max;
        timing->worker_min_us = worker_min;
    }
    return status;
}

extern "C" int y26_threaded_conv_thread_count(const Y26ThreadedConvWorkspace* workspace) {
    return workspace != nullptr ? workspace->plan.thread_count : 0;
}

extern "C" int y26_threaded_conv_get_plan(const Y26ThreadedConvWorkspace* workspace,
                                           Y26ThreadedConvPlan* plan) {
    if (workspace == nullptr || plan == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    *plan = workspace->plan;
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" int y26_threaded_conv_worker_affinity_ok(const Y26ThreadedConvWorkspace* workspace) {
    if (workspace == nullptr) {
        return 0;
    }
    for (const ThreadWorker& worker : workspace->workers) {
        if (worker.affinity_set != 1 || worker.observed_cpu != worker.plan.cpu || worker.plan.cpu < 0 ||
            worker.plan.cpu > 3) {
            return 0;
        }
    }
    return 1;
}
