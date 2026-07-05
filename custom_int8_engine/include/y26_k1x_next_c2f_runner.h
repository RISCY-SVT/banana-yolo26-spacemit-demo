#pragma once

#include "y26_k1x_c2f_block_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Stage14NextC2fConfig {
    const char* subset_id;
    Y26Stage12C2fBlockConfig stage13;
    Y26Stage7ConvNodeConfig model3;
    Y26Stage7ConvNodeConfig model4_cv1;
    float model2_cv2_act_output_scale;
    int model2_cv2_act_output_zero_point_u8;
    float model3_act_output_scale;
    int model3_act_output_zero_point_u8;
    int activation_mode;
};

struct Y26Stage14TimingUs {
    double conv_us;
    double activation_requant_us;
    double split_copy_us;
    double merge_us;
    double post_qdq_us;
    double pack_layout_us;
    double view_span_us;
    double add_us;
    double concat_us;
    double correction_us;
    double model3_conv_us;
    double model3_correction_us;
    double model4_cv1_conv_us;
    double model4_cv1_correction_us;
    double total_us;
    double conv_share_pct;
    double activation_share_pct;
    double merge_share_pct;
    double pack_layout_share_pct;
    double split_branch_share_pct;
    Y26Stage12TimingUs stage13_timing_us;
};

struct Y26Stage14NextC2fWorkspace {
    Y26Stage12C2fBlockWorkspace stage13_ws;
    Y26PrepackedConvWeights* model3_weights;
    Y26PrepackedConvWeights* model4_cv1_weights;
    Y26ConvWorkspace* model3_workspace;
    Y26ConvWorkspace* model4_cv1_workspace;
    std::int32_t* model2_cv2_i32;
    std::int8_t* model3_input_s8;
    std::int32_t* model3_raw_i32;
    std::int32_t* model3_i32;
    std::int8_t* model4_cv1_input_s8;
    std::int32_t* model4_cv1_raw_i32;
    Y26FixedRequantParams* model2_cv2_fixed_requant;
    Y26FixedRequantParams* model3_fixed_requant;
    std::int8_t model2_cv2_act_lut_s8[256];
    std::int8_t model3_act_lut_s8[256];
    std::size_t model2_cv2_output_count;
    std::size_t model3_input_count;
    std::size_t model3_output_count;
    std::size_t model4_cv1_input_count;
    std::size_t model4_cv1_output_count;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int prepared;
};

int y26_stage14_next_c2f_prepare(const Y26Stage14NextC2fConfig* cfg,
                                 Y26Stage14NextC2fWorkspace* ws);

void y26_stage14_next_c2f_release(Y26Stage14NextC2fWorkspace* ws);

std::size_t y26_stage14_next_c2f_output_count(const Y26Stage14NextC2fConfig* cfg);

int y26_stage14_next_c2f_run_scalar(const Y26Stage14NextC2fConfig* cfg,
                                    Y26Stage14NextC2fWorkspace* ws,
                                    const std::int8_t* input_nhwc_s8,
                                    std::int32_t* output_i32_nhwc,
                                    Y26Stage14TimingUs* timing);

int y26_stage14_next_c2f_run_ime_cluster0_hotpath(const Y26Stage14NextC2fConfig* cfg,
                                                  Y26Stage14NextC2fWorkspace* ws,
                                                  const std::int8_t* input_nhwc_s8,
                                                  std::int32_t* output_i32_nhwc,
                                                  Y26Stage14TimingUs* timing);

const std::int8_t* y26_stage14_next_c2f_model3_input_s8(const Y26Stage14NextC2fWorkspace* ws);
const std::int32_t* y26_stage14_next_c2f_model3_i32(const Y26Stage14NextC2fWorkspace* ws);
const std::int8_t* y26_stage14_next_c2f_model4_cv1_input_s8(const Y26Stage14NextC2fWorkspace* ws);

}
