#pragma once

#include "y26_k1x_backbone_stage10_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Stage11BranchBlockConfig {
    const char* subset_id;
    Y26Stage10BackboneExpansionConfig stage10;
    Y26Stage7ConvNodeConfig branch1;
    float branch0_act_output_scale;
    int branch0_act_output_zero_point_u8;
    int activation_mode;
};

struct Y26Stage11TimingUs {
    double conv0_ime_us;
    double act0_requant_lut_us;
    double conv1_ime_us;
    double act1_requant_lut_us;
    double conv2_ime_us;
    double act2_requant_lut_us;
    double split_us;
    double branch_cv1_conv_us;
    double branch_cv1_activation_us;
    double branch_cv2_conv_us;
    double branch_cv2_correction_us;
    double residual_add_us;
    double concat_copy_us;
    double layout_or_pack_us;
    double total_us;
    Y26Stage10TimingUs stage10_timing_us;
};

struct Y26Stage11BranchBlockWorkspace {
    Y26Stage10BackboneExpansionWorkspace stage10_ws;
    Y26PrepackedConvWeights* branch1_weights;
    Y26ConvWorkspace* branch1_workspace;
    std::int32_t* branch0_i32;
    std::int8_t* branch0_act_s8;
    std::int32_t* branch1_raw_i32;
    Y26FixedRequantParams* branch0_fixed_requant;
    std::int8_t branch0_act_lut_s8[256];
    std::size_t branch0_output_count;
    std::size_t branch0_act_count;
    std::size_t branch1_output_count;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage11_branch_block_prepare(const Y26Stage11BranchBlockConfig* cfg,
                                     Y26Stage11BranchBlockWorkspace* ws);

void y26_stage11_branch_block_release(Y26Stage11BranchBlockWorkspace* ws);

std::size_t y26_stage11_branch_block_output_count(const Y26Stage11BranchBlockConfig* cfg);

int y26_stage11_branch_block_run_scalar(const Y26Stage11BranchBlockConfig* cfg,
                                        Y26Stage11BranchBlockWorkspace* ws,
                                        const std::int8_t* input_nhwc_s8,
                                        std::int32_t* output_i32_nhwc,
                                        Y26Stage11TimingUs* timing);

int y26_stage11_branch_block_run_ime_cluster0_hotpath(const Y26Stage11BranchBlockConfig* cfg,
                                                      Y26Stage11BranchBlockWorkspace* ws,
                                                      const std::int8_t* input_nhwc_s8,
                                                      std::int32_t* output_i32_nhwc,
                                                      Y26Stage11TimingUs* timing);

const std::int8_t* y26_stage11_branch_block_branch0_activation_s8(
    const Y26Stage11BranchBlockWorkspace* ws);

}
