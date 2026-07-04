#pragma once

#include "y26_k1x_backbone_subset_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Stage10BackboneExpansionConfig {
    const char* subset_id;
    Y26Stage7BackboneSubsetConfig stage9;
    Y26Stage7ConvNodeConfig branch0;
    float conv2_act_output_scale;
    int conv2_act_output_zero_point_u8;
    int split_axis;
    int split_output1_channel_offset;
    int split_output1_channels;
    int activation_mode;
};

struct Y26Stage10TimingUs {
    double conv0_ime_us;
    double act0_requant_lut_us;
    double conv1_ime_us;
    double act1_requant_lut_us;
    double conv2_ime_us;
    double act2_requant_lut_us;
    double split_us;
    double branch_conv_us;
    double branch_correction_us;
    double pack_layout_us;
    double workspace_memcpy_us;
    double total_us;
    Y26Stage7TimingUs stage9_timing_us;
};

struct Y26Stage10BackboneExpansionWorkspace {
    Y26Stage7BackboneSubsetWorkspace stage9_ws;
    Y26PrepackedConvWeights* branch0_weights;
    Y26ConvWorkspace* branch0_workspace;
    std::int32_t* conv2_i32;
    std::int8_t* conv2_act_s8;
    std::int8_t* split_output1_s8;
    std::int32_t* branch0_raw_i32;
    Y26FixedRequantParams* conv2_fixed_requant;
    std::int8_t act2_lut_s8[256];
    std::size_t conv2_count;
    std::size_t conv2_act_count;
    std::size_t split_output1_count;
    std::size_t branch0_output_count;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage10_backbone_expansion_prepare(const Y26Stage10BackboneExpansionConfig* cfg,
                                           Y26Stage10BackboneExpansionWorkspace* ws);

void y26_stage10_backbone_expansion_release(Y26Stage10BackboneExpansionWorkspace* ws);

std::size_t y26_stage10_backbone_expansion_output_count(const Y26Stage10BackboneExpansionConfig* cfg);

int y26_stage10_backbone_expansion_run_scalar(const Y26Stage10BackboneExpansionConfig* cfg,
                                              Y26Stage10BackboneExpansionWorkspace* ws,
                                              const std::int8_t* input_nhwc_s8,
                                              std::int32_t* output_i32_nhwc,
                                              Y26Stage10TimingUs* timing);

int y26_stage10_backbone_expansion_run_ime_cluster0_hotpath(const Y26Stage10BackboneExpansionConfig* cfg,
                                                            Y26Stage10BackboneExpansionWorkspace* ws,
                                                            const std::int8_t* input_nhwc_s8,
                                                            std::int32_t* output_i32_nhwc,
                                                            Y26Stage10TimingUs* timing);

const std::int8_t* y26_stage10_backbone_expansion_conv2_activation_s8(
    const Y26Stage10BackboneExpansionWorkspace* ws);

const std::int8_t* y26_stage10_backbone_expansion_split_output1_s8(
    const Y26Stage10BackboneExpansionWorkspace* ws);

}
