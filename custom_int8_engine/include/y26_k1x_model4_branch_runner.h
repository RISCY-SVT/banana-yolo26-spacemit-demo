#pragma once

#include "y26_k1x_next_c2f_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Stage15Model4BranchConfig {
    const char* subset_id;
    Y26Stage14NextC2fConfig stage14;
    Y26Stage7ConvNodeConfig branch0;
    float split1_output_scale;
    int split1_output_zero_point_u8;
    float branch0_act_output_scale;
    int branch0_act_output_zero_point_u8;
    int activation_mode;
};

struct Y26Stage15TimingUs {
    double conv_us;
    double activation_requant_us;
    double split_us;
    double merge_us;
    double add_us;
    double concat_us;
    double post_qdq_us;
    double pack_layout_us;
    double correction_us;
    double branch0_conv_us;
    double branch0_correction_us;
    double branch0_activation_us;
    double total_us;
    double conv_share_pct;
    double activation_share_pct;
    double merge_share_pct;
    double pack_layout_share_pct;
    double split_branch_share_pct;
    Y26Stage14TimingUs stage14_timing_us;
};

struct Y26Stage15Model4BranchWorkspace {
    Y26Stage14NextC2fWorkspace stage14_ws;
    Y26PrepackedConvWeights* branch0_weights;
    Y26ConvWorkspace* branch0_workspace;
    std::int32_t* model4_cv1_i32;
    std::int8_t* model4_cv1_act_s8;
    std::int8_t* split1_input_s8;
    std::int32_t* branch0_raw_i32;
    std::int32_t* branch0_i32;
    std::int8_t* branch0_act_s8;
    Y26FixedRequantParams* model4_cv1_fixed_requant;
    Y26FixedRequantParams* branch0_fixed_requant;
    std::int8_t model4_cv1_to_split1_lut_s8[256];
    std::int8_t branch0_act_lut_s8[256];
    std::size_t model4_cv1_output_count;
    std::size_t split1_count;
    std::size_t branch0_output_count;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage15_model4_branch_prepare(const Y26Stage15Model4BranchConfig* cfg,
                                      Y26Stage15Model4BranchWorkspace* ws);

void y26_stage15_model4_branch_release(Y26Stage15Model4BranchWorkspace* ws);

std::size_t y26_stage15_model4_branch_output_count(const Y26Stage15Model4BranchConfig* cfg);

int y26_stage15_model4_branch_run_scalar(const Y26Stage15Model4BranchConfig* cfg,
                                         Y26Stage15Model4BranchWorkspace* ws,
                                         const std::int8_t* input_nhwc_s8,
                                         std::int32_t* output_i32_nhwc,
                                         Y26Stage15TimingUs* timing);

int y26_stage15_model4_branch_run_ime_cluster0_hotpath(const Y26Stage15Model4BranchConfig* cfg,
                                                       Y26Stage15Model4BranchWorkspace* ws,
                                                       const std::int8_t* input_nhwc_s8,
                                                       std::int32_t* output_i32_nhwc,
                                                       Y26Stage15TimingUs* timing);

const std::int8_t* y26_stage15_model4_branch_split1_input_s8(const Y26Stage15Model4BranchWorkspace* ws);
const std::int8_t* y26_stage15_model4_branch_branch0_act_s8(const Y26Stage15Model4BranchWorkspace* ws);
const std::int32_t* y26_stage15_model4_branch_model4_cv1_i32(const Y26Stage15Model4BranchWorkspace* ws);

}
