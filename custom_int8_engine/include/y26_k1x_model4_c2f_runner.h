#pragma once

#include "y26_k1x_model4_branch_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

enum Y26Stage16MergeMode {
    Y26_STAGE16_MERGE_MODE_A0_MATERIALIZED_FLOAT = 0,
    Y26_STAGE16_MERGE_MODE_A2_FUSED_QDQ_NHWC = 2,
    Y26_STAGE16_MERGE_MODE_C2_SPLIT0_CONCAT_LUT = 20,
    Y26_STAGE16_MERGE_MODE_STAGE24_B3_SPLIT1_LUT = 24,
    Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT = 26,
    Y26_STAGE16_MERGE_MODE_STAGE33_MODEL4_CV2_MIXED_SIGNEDNESS = 33,
    Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED4 = 3604,
    Y26_STAGE16_MERGE_MODE_STAGE36_CV2_PIPELINED6 = 3606,
    Y26_STAGE16_MERGE_MODE_STAGE37_BRANCH3X3_PIPELINED4 = 3704,
};

enum Y26Stage16OutputQuantizeMode {
    Y26_STAGE16_OUTPUT_QUANTIZE_SCALAR = 0,
    Y26_STAGE16_OUTPUT_QUANTIZE_RVV_F32 = 1,
    Y26_STAGE16_OUTPUT_QUANTIZE_STAGE38_RVV_DIRECT_STORE = 38,
};

struct Y26Stage16Model4C2fConfig {
    const char* subset_id;
    Y26Stage15Model4BranchConfig stage15;
    Y26Stage7ConvNodeConfig branch1;
    Y26Stage7ConvNodeConfig model4_cv2;
    float concat_output_scale;
    int concat_output_zero_point_u8;
    int activation_mode;
    int merge_mode;
};

struct Y26Stage16TimingUs {
    double conv_us;
    double activation_requant_us;
    double split_us;
    double merge_us;
    double add_us;
    double concat_us;
    double post_qdq_us;
    double output_quantize_us;
    double pack_layout_us;
    double input_adapter_us;
    double copy_layout_us;
    double hash_checksum_compare_us;
    double other_us;
    double correction_us;
    double conv_im2col_pack_us;
    double conv_compute_us;
    double conv_copy_us;
    double conv_worker_other_us;
    double copy_us;
    double branch1_conv_us;
    double branch1_correction_us;
    double branch1_im2col_pack_us;
    double branch1_compute_us;
    double branch1_copy_us;
    double branch1_worker_other_us;
    double branch1_activation_us;
    double model4_cv2_conv_us;
    double model4_cv2_correction_us;
    double model4_cv2_im2col_pack_us;
    double model4_cv2_compute_us;
    double model4_cv2_copy_us;
    double model4_cv2_worker_other_us;
    double thread_overhead_us;
    double total_us;
    double activation_share_pct;
    double conv_share_pct;
    double merge_share_pct;
    double pack_layout_share_pct;
    Y26Stage15TimingUs stage15_timing_us;
};

struct Y26Stage16Model4C2fWorkspace {
    Y26Stage15Model4BranchWorkspace stage15_ws;
    Y26PrepackedConvWeights* branch1_weights;
    Y26PrepackedConvWeights* model4_cv2_weights;
    Y26ConvWorkspace* branch1_workspace;
    Y26ConvWorkspace* model4_cv2_workspace;
    Y26ThreadedConvWorkspace* branch1_threaded_workspace;
    Y26ThreadedConvWorkspace* model4_cv2_threaded_workspace;
    std::int32_t* stage15_output_i32;
    std::int32_t* branch1_raw_i32;
    std::int32_t* branch1_i32;
    std::uint8_t* branch1_conv_code_u8;
    float* branch1_act_f32;
    std::int8_t* branch1_add_concat_lut_s8;
    std::int8_t* split0_concat_s8;
    std::int8_t* concat_s8;
    std::int32_t* model4_cv2_raw_i32;
    std::int32_t* model4_cv2_i32;
    std::int8_t model4_cv1_to_concat_lut_s8[256];
    std::int8_t split1_to_concat_lut_s8[256];
    float split1_dequant_f32_lut[256];
    std::size_t stage15_output_count;
    std::size_t branch1_output_count;
    std::size_t concat_count;
    std::size_t model4_cv2_output_count;
    std::size_t prepacked_bytes;
    std::size_t workspace_bytes;
    int branch1_thread_count;
    int model4_cv2_thread_count;
    int prepared;
};

int y26_stage16_model4_c2f_prepare(const Y26Stage16Model4C2fConfig* cfg,
                                   Y26Stage16Model4C2fWorkspace* ws);

int y26_stage16_model4_c2f_prepare_cut(const Y26Stage16Model4C2fConfig* cfg,
                                       Y26Stage16Model4C2fWorkspace* ws);

int y26_stage16_model4_c2f_prepare_threaded_branch0(const Y26Stage16Model4C2fConfig* cfg,
                                                    Y26Stage16Model4C2fWorkspace* ws,
                                                    int thread_count);

int y26_stage16_model4_c2f_prepare_cut_threaded_branch0(const Y26Stage16Model4C2fConfig* cfg,
                                                        Y26Stage16Model4C2fWorkspace* ws,
                                                        int thread_count);

int y26_stage16_model4_c2f_prepare_cut_threaded_branch1(const Y26Stage16Model4C2fConfig* cfg,
                                                        Y26Stage16Model4C2fWorkspace* ws,
                                                        int thread_count);

int y26_stage16_model4_c2f_prepare_cut_threaded_model4_cv2(const Y26Stage16Model4C2fConfig* cfg,
                                                           Y26Stage16Model4C2fWorkspace* ws,
                                                           int thread_count);

void y26_stage16_model4_c2f_release(Y26Stage16Model4C2fWorkspace* ws);

std::size_t y26_stage16_model4_c2f_output_count(const Y26Stage16Model4C2fConfig* cfg);
std::size_t y26_stage16_model4_c2f_cut_input_count(const Y26Stage16Model4C2fConfig* cfg);

int y26_stage16_model4_c2f_run_scalar(const Y26Stage16Model4C2fConfig* cfg,
                                      Y26Stage16Model4C2fWorkspace* ws,
                                      const std::int8_t* input_nhwc_s8,
                                      std::int32_t* output_i32_nhwc,
                                      Y26Stage16TimingUs* timing);

int y26_stage16_model4_c2f_run_ime_cluster0_hotpath(const Y26Stage16Model4C2fConfig* cfg,
                                                    Y26Stage16Model4C2fWorkspace* ws,
                                                    const std::int8_t* input_nhwc_s8,
                                                    std::int32_t* output_i32_nhwc,
                                                    Y26Stage16TimingUs* timing);

int y26_stage16_model4_c2f_run_ime_threaded_branch0_cluster0_hotpath(const Y26Stage16Model4C2fConfig* cfg,
                                                                     Y26Stage16Model4C2fWorkspace* ws,
                                                                     const std::int8_t* input_nhwc_s8,
                                                                     std::int32_t* output_i32_nhwc,
                                                                     int thread_activation,
                                                                     Y26Stage16TimingUs* timing);

int y26_stage16_model4_c2f_run_cut_u8_output(const Y26Stage16Model4C2fConfig* cfg,
                                             Y26Stage16Model4C2fWorkspace* ws,
                                             const std::uint8_t* model4_cv1_q_u8_nhwc,
                                             std::uint8_t* output_q_u8_nhwc,
                                             int use_ime,
                                             int use_threaded_branch0,
                                             int output_quantize_mode,
                                             Y26Stage16TimingUs* timing);

int y26_stage16_model4_c2f_threaded_worker_affinity_ok(const Y26Stage16Model4C2fWorkspace* ws);
int y26_stage16_model4_c2f_threaded_thread_count(const Y26Stage16Model4C2fWorkspace* ws);

const std::int8_t* y26_stage16_model4_c2f_concat_s8(const Y26Stage16Model4C2fWorkspace* ws);
const std::int32_t* y26_stage16_model4_c2f_branch1_i32(const Y26Stage16Model4C2fWorkspace* ws);

}
