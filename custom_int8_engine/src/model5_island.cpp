#include "y26_k1x_model5_island.h"

#include "y26_k1x_activation.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kAlignment = 64;
constexpr std::uint32_t kWorkspaceMagic = 0x59323557U;
constexpr std::uint32_t kWorkspaceVersion = 1U;

bool workspace_initialized(const Y26Model5IslandWorkspace* workspace) {
    return workspace != nullptr && workspace->lifecycle_magic == kWorkspaceMagic &&
           workspace->lifecycle_version == kWorkspaceVersion;
}

void reset_initialized_workspace(Y26Model5IslandWorkspace* workspace) {
    std::memset(workspace, 0, sizeof(*workspace));
    workspace->lifecycle_magic = kWorkspaceMagic;
    workspace->lifecycle_version = kWorkspaceVersion;
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

void* allocate_aligned(std::size_t bytes) {
    return bytes == 0 ? nullptr : ::operator new(bytes, std::align_val_t(kAlignment));
}

void free_aligned(void* pointer) {
    ::operator delete(pointer, std::align_val_t(kAlignment));
}

int output_dim(int input, int kernel, int stride, int pad) {
    return (input + 2 * pad - kernel) / stride + 1;
}

bool config_valid(const Y26Model5IslandConfig* cfg) {
    if (cfg == nullptr || cfg->model5_conv.params.input_h <= 0 || cfg->model5_conv.params.input_w <= 0 ||
        cfg->model5_conv.params.input_c <= 0 || cfg->model5_conv.params.output_c <= 0 ||
        cfg->model5_conv.kernel_h != 3 || cfg->model5_conv.kernel_w != 3 ||
        cfg->model5_conv.params.stride_h != 2 || cfg->model5_conv.params.stride_w != 2 ||
        cfg->model5_conv.params.pad_h != 1 || cfg->model5_conv.params.pad_w != 1 ||
        cfg->model5_conv.weights_ohwi_s8 == nullptr || cfg->model5_conv.weight_scales == nullptr ||
        cfg->model5_conv.bias_i32 == nullptr || cfg->model4_preact_scale <= 0.0F ||
        cfg->model4_postact_scale <= 0.0F || cfg->model5_postact_scale <= 0.0F ||
        (cfg->ime_accumulator_groups != 4 && cfg->ime_accumulator_groups != 6) ||
        (cfg->dataflow_mode != Y26_MODEL5_DATAFLOW_STAGE43_R0 &&
         cfg->dataflow_mode != Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK)) {
        return false;
    }
    const auto valid_zp = [](int value) { return value >= 0 && value <= 255; };
    return valid_zp(cfg->model4_preact_zero_point_u8) && valid_zp(cfg->model4_postact_zero_point_u8) &&
           valid_zp(cfg->model5_postact_zero_point_u8) &&
           cfg->model5_conv.weight_scale_count >= static_cast<std::size_t>(cfg->model5_conv.params.output_c) &&
           cfg->model5_conv.bias_count >= static_cast<std::size_t>(cfg->model5_conv.params.output_c);
}

Y26Stage5Block0Config scalar_config(const Y26Model5IslandConfig& cfg) {
    return Y26Stage5Block0Config{
        "model5_contiguous_island",
        cfg.model5_conv.node_name,
        cfg.model5_conv.params,
        cfg.model5_conv.kernel_h,
        cfg.model5_conv.kernel_w,
        cfg.model5_conv.activation_zero_point_u8,
        cfg.model5_conv.input_storage_zero_point_s8,
        cfg.model5_conv.weights_ohwi_s8,
        cfg.model5_conv.weight_count,
        cfg.model5_conv.bias_i32,
        cfg.model5_conv.bias_count,
    };
}

Y26ActivationRequantParams activation_params(const Y26Model5IslandConfig& cfg,
                                             std::size_t element_count) {
    return Y26ActivationRequantParams{
        element_count,
        cfg.model5_conv.params.output_c,
        cfg.model5_conv.input_scale,
        cfg.model5_conv.weight_scales,
        cfg.model5_conv.output_scale,
        cfg.model5_conv.output_zero_point_u8,
        cfg.model5_postact_scale,
        cfg.model5_postact_zero_point_u8,
    };
}

int apply_direct_lut(const std::uint8_t* input_u8,
                     std::int8_t* output_s8,
                     std::size_t count,
                     const std::int8_t* lut_s8) {
    if (input_u8 == nullptr || output_s8 == nullptr || lut_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < count; ++index) {
        output_s8[index] = lut_s8[input_u8[index]];
    }
    return Y26_CONV_STATUS_SUCCESS;
}

void reset_timing(Y26Model5IslandTimingUs* timing) {
    if (timing != nullptr) {
        std::memset(timing, 0, sizeof(*timing));
    }
}

}  // namespace

extern "C" int y26_model5_island_workspace_init(Y26Model5IslandWorkspace* workspace) {
    if (workspace == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    reset_initialized_workspace(workspace);
    return Y26_CONV_STATUS_SUCCESS;
}

extern "C" void y26_model5_island_release(Y26Model5IslandWorkspace* workspace) {
    if (!workspace_initialized(workspace)) {
        return;
    }
    y26_stage5_block0_release(&workspace->scalar_conv);
    y26_threaded_conv_destroy(workspace->threaded_conv);
    free_aligned(workspace->model4_postact_nhwc_s8);
    free_aligned(workspace->model5_corrected_nhwc_i32);
    free_aligned(workspace->model5_fixed_requant);
    reset_initialized_workspace(workspace);
}

extern "C" int y26_model5_island_prepare(const Y26Model5IslandConfig* cfg,
                                           int thread_count,
                                           Y26Model5IslandWorkspace* workspace) {
    if (!config_valid(cfg) || !workspace_initialized(workspace) || thread_count < 1 || thread_count > 4) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    y26_model5_island_release(workspace);
    const int output_h = output_dim(cfg->model5_conv.params.input_h,
                                    cfg->model5_conv.kernel_h,
                                    cfg->model5_conv.params.stride_h,
                                    cfg->model5_conv.params.pad_h);
    const int output_w = output_dim(cfg->model5_conv.params.input_w,
                                    cfg->model5_conv.kernel_w,
                                    cfg->model5_conv.params.stride_w,
                                    cfg->model5_conv.params.pad_w);
    workspace->model4_element_count = static_cast<std::size_t>(cfg->model5_conv.params.input_h) *
                                      static_cast<std::size_t>(cfg->model5_conv.params.input_w) *
                                      static_cast<std::size_t>(cfg->model5_conv.params.input_c);
    workspace->model5_element_count = static_cast<std::size_t>(output_h) *
                                      static_cast<std::size_t>(output_w) *
                                      static_cast<std::size_t>(cfg->model5_conv.params.output_c);
    try {
        workspace->model4_postact_nhwc_s8 =
            static_cast<std::int8_t*>(allocate_aligned(workspace->model4_element_count));
        workspace->model5_corrected_nhwc_i32 = static_cast<std::int32_t*>(
            allocate_aligned(workspace->model5_element_count * sizeof(std::int32_t)));
        workspace->model5_fixed_requant = static_cast<Y26FixedRequantParams*>(
            allocate_aligned(static_cast<std::size_t>(cfg->model5_conv.params.output_c) *
                             sizeof(Y26FixedRequantParams)));
        const Y26Stage5Block0Config scalar_cfg = scalar_config(*cfg);
        int status = y26_stage5_block0_prepare(&scalar_cfg, &workspace->scalar_conv);
        if (status != Y26_CONV_STATUS_SUCCESS || workspace->model4_postact_nhwc_s8 == nullptr ||
            workspace->model5_corrected_nhwc_i32 == nullptr || workspace->model5_fixed_requant == nullptr) {
            y26_model5_island_release(workspace);
            return status == Y26_CONV_STATUS_SUCCESS ? Y26_CONV_STATUS_INVALID_ARGUMENT : status;
        }
        workspace->threaded_conv = y26_threaded_conv_create_spatial_rows(&cfg->model5_conv, thread_count);
        if (workspace->threaded_conv == nullptr) {
            y26_model5_island_release(workspace);
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        status = y26_build_silu_u8_to_s8_lut(cfg->model4_preact_scale,
                                             cfg->model4_preact_zero_point_u8,
                                             cfg->model4_postact_scale,
                                             cfg->model4_postact_zero_point_u8,
                                             workspace->model4_postact_lut_s8);
        if (status == Y26_CONV_STATUS_SUCCESS) {
            status = y26_build_silu_u8_to_s8_lut(cfg->model5_conv.output_scale,
                                                 cfg->model5_conv.output_zero_point_u8,
                                                 cfg->model5_postact_scale,
                                                 cfg->model5_postact_zero_point_u8,
                                                 workspace->model5_postact_lut_s8);
        }
        if (status == Y26_CONV_STATUS_SUCCESS) {
            const Y26ActivationRequantParams params = activation_params(*cfg, workspace->model5_element_count);
            status = y26_build_fixed_requant_params_per_channel(&params, workspace->model5_fixed_requant);
        }
        if (status != Y26_CONV_STATUS_SUCCESS) {
            y26_model5_island_release(workspace);
            return status;
        }
        workspace->model5_fixed_requant_count = static_cast<std::size_t>(cfg->model5_conv.params.output_c);
        workspace->workspace_bytes = workspace->model4_element_count +
                                     workspace->model5_element_count * sizeof(std::int32_t) +
                                     workspace->model5_fixed_requant_count * sizeof(Y26FixedRequantParams) +
                                     workspace->scalar_conv.workspace_bytes + workspace->scalar_conv.raw_i32_bytes +
                                     workspace->scalar_conv.prepacked_bytes + sizeof(workspace->model4_postact_lut_s8) +
                                     sizeof(workspace->model5_postact_lut_s8);
        Y26ThreadedConvPlan plan {};
        if (y26_threaded_conv_get_plan(workspace->threaded_conv, &plan) != Y26_CONV_STATUS_SUCCESS) {
            y26_model5_island_release(workspace);
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
        for (int worker = 0; worker < plan.thread_count; ++worker) {
            workspace->workspace_bytes += plan.workers[worker].workspace_bytes +
                                          plan.workers[worker].prepacked_bytes +
                                          static_cast<std::size_t>(plan.workers[worker].local_output_h) *
                                              static_cast<std::size_t>(plan.output_w) *
                                              static_cast<std::size_t>(plan.output_c) * sizeof(std::int32_t);
        }
        workspace->prepared = 1;
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        y26_model5_island_release(workspace);
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

extern "C" int y26_model5_island_apply_model4_postact(const Y26Model5IslandConfig* cfg,
                                                        const Y26Model5IslandWorkspace* workspace,
                                                        const std::uint8_t* model4_preact_nhwc_u8,
                                                        std::int8_t* model4_postact_nhwc_s8) {
    if (!config_valid(cfg) || !workspace_initialized(workspace) || workspace->prepared != 1) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    return apply_direct_lut(model4_preact_nhwc_u8,
                            model4_postact_nhwc_s8,
                            workspace->model4_element_count,
                            workspace->model4_postact_lut_s8);
}

extern "C" int y26_model5_island_run_scalar(const Y26Model5IslandConfig* cfg,
                                              Y26Model5IslandWorkspace* workspace,
                                              const std::uint8_t* model4_preact_nhwc_u8,
                                              std::int8_t* model5_postact_nhwc_s8,
                                              Y26Model5IslandTimingUs* timing) {
    if (!config_valid(cfg) || !workspace_initialized(workspace) || workspace->prepared != 1 ||
        model4_preact_nhwc_u8 == nullptr || model5_postact_nhwc_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    reset_timing(timing);
    const auto total_begin = Clock::now();
    const auto post4_begin = total_begin;
    int status = y26_model5_island_apply_model4_postact(
        cfg, workspace, model4_preact_nhwc_u8, workspace->model4_postact_nhwc_s8);
    const auto post4_end = Clock::now();
    const Y26Stage5Block0Config scalar_cfg = scalar_config(*cfg);
    const auto conv_begin = post4_end;
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = y26_stage5_block0_run_scalar(&scalar_cfg,
                                              &workspace->scalar_conv,
                                              workspace->model4_postact_nhwc_s8,
                                              workspace->model5_corrected_nhwc_i32);
    }
    const auto conv_end = Clock::now();
    const auto post5_begin = conv_end;
    if (status == Y26_CONV_STATUS_SUCCESS) {
        const Y26ActivationRequantParams params = activation_params(*cfg, workspace->model5_element_count);
        status = y26_activation_requant_silu_int8_lut_scalar_unrolled(
            &params,
            workspace->model5_corrected_nhwc_i32,
            workspace->model5_postact_lut_s8,
            model5_postact_nhwc_s8);
    }
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->model4_postact_us = elapsed_us(post4_begin, post4_end);
        timing->model5_conv_us = elapsed_us(conv_begin, conv_end);
        timing->model5_compute_us = timing->model5_conv_us;
        timing->model5_postact_us = elapsed_us(post5_begin, end);
        timing->total_us = elapsed_us(total_begin, end);
    }
    return status;
}

extern "C" int y26_model5_island_run_ime_cluster0(const Y26Model5IslandConfig* cfg,
                                                    Y26Model5IslandWorkspace* workspace,
                                                    const std::uint8_t* model4_preact_nhwc_u8,
                                                    std::int8_t* model5_postact_nhwc_s8,
                                                    Y26Model5IslandTimingUs* timing) {
    if (!config_valid(cfg) || !workspace_initialized(workspace) || workspace->prepared != 1 ||
        workspace->threaded_conv == nullptr || model4_preact_nhwc_u8 == nullptr ||
        model5_postact_nhwc_s8 == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    reset_timing(timing);
    const auto total_begin = Clock::now();
    int status = y26_model5_island_apply_model4_postact(
        cfg, workspace, model4_preact_nhwc_u8, workspace->model4_postact_nhwc_s8);
    const auto post4_end = Clock::now();
    Y26ThreadedConvTimingUs conv_timing {};
    if (status == Y26_CONV_STATUS_SUCCESS) {
        if (cfg->dataflow_mode == Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK) {
            status = y26_threaded_conv_run_ime_cluster0_stage39_fastpack(
                workspace->threaded_conv,
                workspace->model4_postact_nhwc_s8,
                workspace->model5_corrected_nhwc_i32,
                cfg->ime_accumulator_groups,
                &conv_timing);
        } else {
            status = y26_threaded_conv_run_ime_cluster0_stage37_pipelined(
                workspace->threaded_conv,
                workspace->model4_postact_nhwc_s8,
                workspace->model5_corrected_nhwc_i32,
                cfg->ime_accumulator_groups,
                &conv_timing);
        }
    }
    const auto activation_begin = Clock::now();
    if (status == Y26_CONV_STATUS_SUCCESS) {
        const Y26ActivationRequantParams params = activation_params(*cfg, workspace->model5_element_count);
        status = y26_activation_requant_silu_int8_lut_fixed_requant(&params,
                                                                    workspace->model5_fixed_requant,
                                                                    workspace->model5_corrected_nhwc_i32,
                                                                    workspace->model5_postact_lut_s8,
                                                                    model5_postact_nhwc_s8);
    }
    const auto end = Clock::now();
    if (timing != nullptr) {
        timing->model4_postact_us = elapsed_us(total_begin, post4_end);
        timing->model5_conv_us = conv_timing.total_us;
        timing->model5_im2col_pack_us = conv_timing.worker_im2col_pack_us;
        timing->model5_compute_us = conv_timing.worker_compute_us;
        timing->model5_correction_us = conv_timing.worker_correction_us;
        timing->model5_thread_overhead_us = std::max(0.0, conv_timing.total_us - conv_timing.worker_max_us);
        timing->model5_postact_us = elapsed_us(activation_begin, end);
        timing->total_us = elapsed_us(total_begin, end);
    }
    return status;
}

extern "C" int y26_model5_island_worker_affinity_ok(const Y26Model5IslandWorkspace* workspace) {
    return workspace_initialized(workspace) && workspace->threaded_conv != nullptr
               ? y26_threaded_conv_worker_affinity_ok(workspace->threaded_conv)
               : 0;
}

extern "C" int y26_model5_island_thread_count(const Y26Model5IslandWorkspace* workspace) {
    return workspace_initialized(workspace) && workspace->threaded_conv != nullptr
               ? y26_threaded_conv_thread_count(workspace->threaded_conv)
               : 0;
}
