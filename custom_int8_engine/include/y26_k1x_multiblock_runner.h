#pragma once

#include "y26_k1x_conv_kernels.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Stage6ConvNodeConfig {
    const char* node_name;
    Y26Conv2DParams params;
    int kernel_h;
    int kernel_w;
    int activation_zero_point_u8;
    int input_storage_zero_point_s8;
    float input_scale;
    float output_scale;
    int output_zero_point_u8;
    const float* weight_scales;
    std::size_t weight_scale_count;
    const std::int8_t* weights_ohwi_s8;
    std::size_t weight_count;
    const std::int32_t* bias_i32;
    std::size_t bias_count;
};

struct Y26Stage6MultiblockConfig {
    const char* subset_id;
    Y26Stage6ConvNodeConfig conv0;
    Y26Stage6ConvNodeConfig conv1;
    float act0_output_scale;
    int act0_output_zero_point_u8;
};

struct Y26Stage6TimingUs {
    double conv0_us;
    double activation_us;
    double conv1_us;
    double total_us;
};

struct Y26Stage6MultiblockWorkspace {
    Y26PrepackedConvWeights* conv0_weights;
    Y26PrepackedConvWeights* conv1_weights;
    Y26ConvWorkspace* conv0_workspace;
    Y26ConvWorkspace* conv1_workspace;
    std::int32_t* conv0_raw_i32;
    std::int32_t* conv0_i32;
    std::int8_t* conv1_input_s8;
    std::int32_t* conv1_raw_i32;
    std::size_t conv0_output_count;
    std::size_t conv1_input_count;
    std::size_t conv1_output_count;
    std::size_t conv0_raw_bytes;
    std::size_t conv0_i32_bytes;
    std::size_t conv1_input_bytes;
    std::size_t conv1_raw_bytes;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage6_multiblock_prepare(const Y26Stage6MultiblockConfig* cfg,
                                  Y26Stage6MultiblockWorkspace* ws);

void y26_stage6_multiblock_release(Y26Stage6MultiblockWorkspace* ws);

int y26_stage6_multiblock_conv0_output_h(const Y26Stage6MultiblockConfig* cfg);
int y26_stage6_multiblock_conv0_output_w(const Y26Stage6MultiblockConfig* cfg);
int y26_stage6_multiblock_conv1_output_h(const Y26Stage6MultiblockConfig* cfg);
int y26_stage6_multiblock_conv1_output_w(const Y26Stage6MultiblockConfig* cfg);
std::size_t y26_stage6_multiblock_conv0_output_count(const Y26Stage6MultiblockConfig* cfg);
std::size_t y26_stage6_multiblock_conv1_output_count(const Y26Stage6MultiblockConfig* cfg);

int y26_stage6_multiblock_run_scalar(const Y26Stage6MultiblockConfig* cfg,
                                     Y26Stage6MultiblockWorkspace* ws,
                                     const std::int8_t* input_nhwc_s8,
                                     std::int32_t* output_i32_nhwc,
                                     Y26Stage6TimingUs* timing);

int y26_stage6_multiblock_run_ime_cluster0_hotpath(const Y26Stage6MultiblockConfig* cfg,
                                                   Y26Stage6MultiblockWorkspace* ws,
                                                   const std::int8_t* input_nhwc_s8,
                                                   std::int32_t* output_i32_nhwc,
                                                   Y26Stage6TimingUs* timing);

const std::int32_t* y26_stage6_multiblock_conv0_i32(const Y26Stage6MultiblockWorkspace* ws);
const std::int8_t* y26_stage6_multiblock_conv1_input_s8(const Y26Stage6MultiblockWorkspace* ws);

}

