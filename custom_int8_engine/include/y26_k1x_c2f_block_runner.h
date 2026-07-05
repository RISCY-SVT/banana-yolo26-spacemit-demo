#pragma once

#include "y26_k1x_branch_block_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

enum Y26Stage12MergeMode {
    Y26_STAGE12_MERGE_MODE_A0_MATERIALIZED_FLOAT = 0,
    Y26_STAGE13_MERGE_MODE_A1_FUSED_ADD_CONCAT = 1,
    Y26_STAGE13_MERGE_MODE_A2_FUSED_QDQ_NHWC = 2,
};

struct Y26Stage12C2fBlockConfig {
    const char* subset_id;
    Y26Stage11BranchBlockConfig stage11;
    Y26Stage7ConvNodeConfig model2_cv2;
    float split1_output_scale;
    int split1_output_zero_point_u8;
    float concat_output_scale;
    int concat_output_zero_point_u8;
    int activation_mode;
    int merge_mode;
};

struct Y26Stage12TimingUs {
    double conv_us;
    double activation_requant_us;
    double split_us;
    double add_us;
    double concat_us;
    double post_concat_qdq_us;
    double pack_layout_us;
    double correction_us;
    double model2_cv2_conv_us;
    double total_us;
    double activation_share_pct;
    double conv_share_pct;
    double add_concat_share_pct;
    double pack_layout_share_pct;
    double split_copy_us;
    double add_compute_us;
    double concat_materialize_us;
    double pack_for_model2_cv2_us;
    double layout_copy_us;
    double other_us;
    double merge_total_us;
    double merge_share_pct;
    int merge_mode;
    Y26Stage11TimingUs stage11_timing_us;
};

struct Y26Stage12C2fBlockWorkspace {
    Y26Stage11BranchBlockWorkspace stage11_ws;
    Y26PrepackedConvWeights* model2_cv2_weights;
    Y26ConvWorkspace* model2_cv2_workspace;
    std::int32_t* branch1_i32;
    float* split0_f32;
    float* split1_f32;
    float* branch1_act_f32;
    float* add_f32;
    float* concat_f32;
    std::int8_t* concat_s8;
    std::int32_t* model2_cv2_raw_i32;
    std::size_t split0_count;
    std::size_t split1_count;
    std::size_t add_count;
    std::size_t concat_count;
    std::size_t model2_cv2_output_count;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage12_c2f_block_prepare(const Y26Stage12C2fBlockConfig* cfg,
                                  Y26Stage12C2fBlockWorkspace* ws);

void y26_stage12_c2f_block_release(Y26Stage12C2fBlockWorkspace* ws);

std::size_t y26_stage12_c2f_block_output_count(const Y26Stage12C2fBlockConfig* cfg);

int y26_stage12_c2f_block_run_scalar(const Y26Stage12C2fBlockConfig* cfg,
                                     Y26Stage12C2fBlockWorkspace* ws,
                                     const std::int8_t* input_nhwc_s8,
                                     std::int32_t* output_i32_nhwc,
                                     Y26Stage12TimingUs* timing);

int y26_stage12_c2f_block_run_ime_cluster0_hotpath(const Y26Stage12C2fBlockConfig* cfg,
                                                   Y26Stage12C2fBlockWorkspace* ws,
                                                   const std::int8_t* input_nhwc_s8,
                                                   std::int32_t* output_i32_nhwc,
                                                   Y26Stage12TimingUs* timing);

const std::int8_t* y26_stage12_c2f_block_concat_s8(const Y26Stage12C2fBlockWorkspace* ws);

}
