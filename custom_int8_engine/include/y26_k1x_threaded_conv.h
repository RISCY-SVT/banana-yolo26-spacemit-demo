#pragma once

#include "y26_k1x_activation.h"
#include "y26_k1x_backbone_subset_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

enum Y26ThreadedConvPartition {
    Y26_THREADED_CONV_PARTITION_SPATIAL_ROWS = 0,
};

struct Y26ThreadedConvWorkerPlan {
    int cpu;
    int row_begin;
    int row_end;
    int input_row_begin;
    int input_row_end;
    int local_output_offset;
    int local_output_h;
    int output_rows_written;
    int overcomputed_rows;
    int discarded_rows;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
};

struct Y26ThreadedConvPlan {
    int thread_count;
    int output_h;
    int output_w;
    int output_c;
    int kernel_h;
    int kernel_w;
    int input_c;
    int total_overcomputed_rows;
    int total_discarded_rows;
    long long estimated_extra_macs;
    double estimated_extra_mac_pct;
    Y26ThreadedConvWorkerPlan workers[4];
};

struct Y26ThreadedConvTimingUs {
    double conv_us;
    double correction_us;
    double total_us;
    double worker_max_us;
    double worker_min_us;
    double worker_im2col_pack_us;
    double worker_compute_us;
    double worker_correction_us;
    double worker_copy_us;
    double worker_other_us;
};

struct Y26ThreadedActivationTimingUs {
    double total_us;
    double worker_max_us;
    double worker_min_us;
};

struct Y26ThreadedConvWorkspace;

Y26ThreadedConvWorkspace* y26_threaded_conv_create_spatial_rows(const Y26Stage7ConvNodeConfig* cfg,
                                                                int thread_count);

void y26_threaded_conv_destroy(Y26ThreadedConvWorkspace* workspace);

int y26_threaded_conv_run_ime_cluster0(const Y26ThreadedConvWorkspace* workspace,
                                       const std::int8_t* input_nhwc_s8,
                                       std::int32_t* corrected_output_nhwc,
                                       Y26ThreadedConvTimingUs* timing);

int y26_threaded_conv_run_ime_cluster0_stage36_pipelined_cv2(
    const Y26ThreadedConvWorkspace* workspace,
    const std::int8_t* input_nhwc_s8,
    std::int32_t* corrected_output_nhwc,
    int accumulator_groups,
    Y26ThreadedConvTimingUs* timing);

int y26_threaded_conv_run_ime_cluster0_stage37_pipelined(
    const Y26ThreadedConvWorkspace* workspace,
    const std::int8_t* input_nhwc_s8,
    std::int32_t* corrected_output_nhwc,
    int accumulator_groups,
    Y26ThreadedConvTimingUs* timing);

int y26_threaded_conv_run_ime_cluster0_u8s8_fused_correction(
    const Y26ThreadedConvWorkspace* workspace,
    const std::int8_t* input_nhwc_s8_storage,
    std::int32_t* corrected_output_nhwc,
    Y26ThreadedConvTimingUs* timing);

int y26_threaded_conv_run_activation_rvv_f32_rows(const Y26ThreadedConvWorkspace* workspace,
                                                  const Y26ActivationRequantParams* params,
                                                  const std::int32_t* producer_i32,
                                                  const std::int8_t* lut_256_s8,
                                                  std::int8_t* consumer_input_s8,
                                                  Y26ThreadedActivationTimingUs* timing);

int y26_threaded_conv_thread_count(const Y26ThreadedConvWorkspace* workspace);

int y26_threaded_conv_get_plan(const Y26ThreadedConvWorkspace* workspace,
                               Y26ThreadedConvPlan* plan);

int y26_threaded_conv_worker_affinity_ok(const Y26ThreadedConvWorkspace* workspace);

}
