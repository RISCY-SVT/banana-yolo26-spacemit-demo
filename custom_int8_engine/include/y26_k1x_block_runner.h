#pragma once

#include "y26_k1x_conv_kernels.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Stage5Block0Config {
    const char* block_id;
    const char* node_name;
    Y26Conv2DParams conv0_params;
    int kernel_h;
    int kernel_w;
    int activation_zero_point_u8;
    int input_storage_zero_point_s8;
    const std::int8_t* weights_ohwi_s8;
    std::size_t weight_count;
    const std::int32_t* bias_i32;
    std::size_t bias_count;
};

struct Y26Stage5Block0Workspace {
    Y26PrepackedConvWeights* conv0_weights;
    Y26ConvWorkspace* conv0_workspace;
    std::int32_t* raw_i32;
    std::size_t raw_i32_count;
    std::size_t raw_i32_bytes;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage5_block0_prepare(const Y26Stage5Block0Config* cfg,
                              Y26Stage5Block0Workspace* ws);

void y26_stage5_block0_release(Y26Stage5Block0Workspace* ws);

int y26_stage5_block0_output_h(const Y26Stage5Block0Config* cfg);
int y26_stage5_block0_output_w(const Y26Stage5Block0Config* cfg);
std::size_t y26_stage5_block0_output_count(const Y26Stage5Block0Config* cfg);

int y26_stage5_block0_run_scalar(const Y26Stage5Block0Config* cfg,
                                 Y26Stage5Block0Workspace* ws,
                                 const std::int8_t* input_nhwc_s8,
                                 std::int32_t* output_i32_nhwc);

int y26_stage5_block0_run_ime_cluster0_hotpath(const Y26Stage5Block0Config* cfg,
                                               Y26Stage5Block0Workspace* ws,
                                               const std::int8_t* input_nhwc_s8,
                                               std::int32_t* output_i32_nhwc);

const std::int32_t* y26_stage5_block0_raw_scratch(const Y26Stage5Block0Workspace* ws);

}
