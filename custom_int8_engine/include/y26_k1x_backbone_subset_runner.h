#pragma once

#include "y26_k1x_activation.h"
#include "y26_k1x_conv_kernels.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Stage7ConvNodeConfig {
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

struct Y26Stage7BackboneSubsetConfig {
    const char* subset_id;
    Y26Stage7ConvNodeConfig conv0;
    Y26Stage7ConvNodeConfig conv1;
    Y26Stage7ConvNodeConfig conv2;
    float act0_output_scale;
    int act0_output_zero_point_u8;
    float act1_output_scale;
    int act1_output_zero_point_u8;
    int activation_mode;
};

struct Y26Stage7TimingUs {
    double conv0_us;
    double act0_requant_us;
    double conv1_us;
    double act1_requant_us;
    double conv2_us;
    double total_us;
    Y26ActivationSubbucketTimingUs act0_subbucket_us;
    Y26ActivationSubbucketTimingUs act1_subbucket_us;
};

struct Y26Stage7BackboneSubsetWorkspace {
    Y26PrepackedConvWeights* conv0_weights;
    Y26PrepackedConvWeights* conv1_weights;
    Y26PrepackedConvWeights* conv2_weights;
    Y26ConvWorkspace* conv0_workspace;
    Y26ConvWorkspace* conv1_workspace;
    Y26ConvWorkspace* conv2_workspace;
    std::int32_t* conv0_raw_i32;
    std::int32_t* conv0_i32;
    std::int8_t* conv1_input_s8;
    std::int32_t* conv1_raw_i32;
    std::int32_t* conv1_i32;
    std::int8_t* conv2_input_s8;
    std::int32_t* conv2_raw_i32;
    Y26FixedRequantParams* conv0_fixed_requant;
    Y26FixedRequantParams* conv1_fixed_requant;
    std::int8_t act0_lut_s8[256];
    std::int8_t act1_lut_s8[256];
    std::size_t conv0_output_count;
    std::size_t conv1_input_count;
    std::size_t conv1_output_count;
    std::size_t conv2_input_count;
    std::size_t conv2_output_count;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage7_backbone_subset_prepare(const Y26Stage7BackboneSubsetConfig* cfg,
                                       Y26Stage7BackboneSubsetWorkspace* ws);

void y26_stage7_backbone_subset_release(Y26Stage7BackboneSubsetWorkspace* ws);

std::size_t y26_stage7_backbone_subset_conv0_output_count(const Y26Stage7BackboneSubsetConfig* cfg);
std::size_t y26_stage7_backbone_subset_conv1_output_count(const Y26Stage7BackboneSubsetConfig* cfg);
std::size_t y26_stage7_backbone_subset_conv2_output_count(const Y26Stage7BackboneSubsetConfig* cfg);

int y26_stage7_backbone_subset_run_scalar(const Y26Stage7BackboneSubsetConfig* cfg,
                                          Y26Stage7BackboneSubsetWorkspace* ws,
                                          const std::int8_t* input_nhwc_s8,
                                          std::int32_t* output_i32_nhwc,
                                          Y26Stage7TimingUs* timing);

int y26_stage7_backbone_subset_run_ime_cluster0_hotpath(const Y26Stage7BackboneSubsetConfig* cfg,
                                                        Y26Stage7BackboneSubsetWorkspace* ws,
                                                        const std::int8_t* input_nhwc_s8,
                                                        std::int32_t* output_i32_nhwc,
                                                        Y26Stage7TimingUs* timing);

const std::int32_t* y26_stage7_backbone_subset_conv0_i32(const Y26Stage7BackboneSubsetWorkspace* ws);
const std::int8_t* y26_stage7_backbone_subset_conv1_input_s8(const Y26Stage7BackboneSubsetWorkspace* ws);
const std::int32_t* y26_stage7_backbone_subset_conv1_i32(const Y26Stage7BackboneSubsetWorkspace* ws);
const std::int8_t* y26_stage7_backbone_subset_conv2_input_s8(const Y26Stage7BackboneSubsetWorkspace* ws);

}
