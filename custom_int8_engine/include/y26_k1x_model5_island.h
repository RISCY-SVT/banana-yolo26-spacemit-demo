#pragma once

#include "y26_k1x_backbone_subset_runner.h"
#include "y26_k1x_block_runner.h"
#include "y26_k1x_threaded_conv.h"

#include <cstddef>
#include <cstdint>

extern "C" {

enum {
    Y26_MODEL5_DATAFLOW_STAGE43_R0 = 0,
    Y26_MODEL5_DATAFLOW_STAGE44_STRIDE2_FASTPACK = 1,
};

struct Y26Model5IslandConfig {
    Y26Stage7ConvNodeConfig model5_conv;
    float model4_preact_scale;
    int model4_preact_zero_point_u8;
    float model4_postact_scale;
    int model4_postact_zero_point_u8;
    float model5_postact_scale;
    int model5_postact_zero_point_u8;
    int ime_accumulator_groups;
    int dataflow_mode;
};

struct Y26Model5IslandTimingUs {
    double model4_postact_us;
    double model5_conv_us;
    double model5_im2col_pack_us;
    double model5_compute_us;
    double model5_correction_us;
    double model5_thread_overhead_us;
    double model5_postact_us;
    double total_us;
};

struct Y26Model5IslandWorkspace {
    std::uint32_t lifecycle_magic;
    std::uint32_t lifecycle_version;
    Y26Stage5Block0Workspace scalar_conv;
    Y26ThreadedConvWorkspace* threaded_conv;
    std::int8_t* model4_postact_nhwc_s8;
    std::int32_t* model5_corrected_nhwc_i32;
    Y26FixedRequantParams* model5_fixed_requant;
    std::int8_t model4_postact_lut_s8[256];
    std::int8_t model5_postact_lut_s8[256];
    std::size_t model4_element_count;
    std::size_t model5_element_count;
    std::size_t model5_fixed_requant_count;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_model5_island_workspace_init(Y26Model5IslandWorkspace* workspace);

int y26_model5_island_prepare(const Y26Model5IslandConfig* cfg,
                              int thread_count,
                              Y26Model5IslandWorkspace* workspace);

void y26_model5_island_release(Y26Model5IslandWorkspace* workspace);

int y26_model5_island_apply_model4_postact(const Y26Model5IslandConfig* cfg,
                                           const Y26Model5IslandWorkspace* workspace,
                                           const std::uint8_t* model4_preact_nhwc_u8,
                                           std::int8_t* model4_postact_nhwc_s8);

int y26_model5_island_run_scalar(const Y26Model5IslandConfig* cfg,
                                 Y26Model5IslandWorkspace* workspace,
                                 const std::uint8_t* model4_preact_nhwc_u8,
                                 std::int8_t* model5_postact_nhwc_s8,
                                 Y26Model5IslandTimingUs* timing);

int y26_model5_island_run_ime_cluster0(const Y26Model5IslandConfig* cfg,
                                       Y26Model5IslandWorkspace* workspace,
                                       const std::uint8_t* model4_preact_nhwc_u8,
                                       std::int8_t* model5_postact_nhwc_s8,
                                       Y26Model5IslandTimingUs* timing);

int y26_model5_island_worker_affinity_ok(const Y26Model5IslandWorkspace* workspace);
int y26_model5_island_thread_count(const Y26Model5IslandWorkspace* workspace);

}
