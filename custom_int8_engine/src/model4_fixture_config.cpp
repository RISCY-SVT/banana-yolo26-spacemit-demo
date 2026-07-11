#include "y26_k1x_model4_fixture_config.h"

#include "stage16_model4_c2f_fixture.h"

#include <iterator>

namespace {

Y26Stage7ConvNodeConfig conv0_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.conv0_node_name,
                                   fixture.conv0_params,
                                   fixture.conv0_kernel_h,
                                   fixture.conv0_kernel_w,
                                   fixture.conv0_activation_zero_point_u8,
                                   fixture.conv0_input_storage_zero_point_s8,
                                   fixture.images_scale,
                                   fixture.conv0_output_scale,
                                   fixture.conv0_output_zero_point_u8,
                                   fixture.conv0_weight_scales,
                                   fixture.conv0_weight_scale_count,
                                   fixture.conv0_weights_ohwi_s8,
                                   fixture.conv0_weight_count,
                                   fixture.conv0_bias_i32,
                                   fixture.conv0_bias_count};
}

Y26Stage7ConvNodeConfig conv1_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.conv1_node_name,
                                   fixture.conv1_params,
                                   fixture.conv1_kernel_h,
                                   fixture.conv1_kernel_w,
                                   fixture.act0_output_zero_point_u8,
                                   fixture.conv1_input_storage_zero_point_s8,
                                   fixture.act0_output_scale,
                                   fixture.conv1_output_scale,
                                   fixture.conv1_output_zero_point_u8,
                                   fixture.conv1_weight_scales,
                                   fixture.conv1_weight_scale_count,
                                   fixture.conv1_weights_ohwi_s8,
                                   fixture.conv1_weight_count,
                                   fixture.conv1_bias_i32,
                                   fixture.conv1_bias_count};
}

Y26Stage7ConvNodeConfig conv2_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.conv2_node_name,
                                   fixture.conv2_params,
                                   fixture.conv2_kernel_h,
                                   fixture.conv2_kernel_w,
                                   fixture.act1_output_zero_point_u8,
                                   fixture.conv2_input_storage_zero_point_s8,
                                   fixture.act1_output_scale,
                                   fixture.conv2_output_scale,
                                   fixture.conv2_output_zero_point_u8,
                                   fixture.conv2_weight_scales,
                                   fixture.conv2_weight_scale_count,
                                   fixture.conv2_weights_ohwi_s8,
                                   fixture.conv2_weight_count,
                                   fixture.conv2_bias_i32,
                                   fixture.conv2_bias_count};
}

Y26Stage7BackboneSubsetConfig stage9_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    int activation_mode) {
    return Y26Stage7BackboneSubsetConfig{fixture.subset_id,
                                         conv0_config(fixture),
                                         conv1_config(fixture),
                                         conv2_config(fixture),
                                         fixture.act0_output_scale,
                                         fixture.act0_output_zero_point_u8,
                                         fixture.act1_output_scale,
                                         fixture.act1_output_zero_point_u8,
                                         activation_mode};
}

Y26Stage7ConvNodeConfig stage10_branch0_config(
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch0_node_name,
                                   fixture.branch0_params,
                                   fixture.branch0_kernel_h,
                                   fixture.branch0_kernel_w,
                                   fixture.branch0_activation_zero_point_u8,
                                   fixture.branch0_input_storage_zero_point_s8,
                                   fixture.split_output1_scale,
                                   fixture.branch0_output_scale,
                                   fixture.branch0_output_zero_point_u8,
                                   fixture.branch0_weight_scales,
                                   fixture.branch0_weight_scale_count,
                                   fixture.branch0_weights_ohwi_s8,
                                   fixture.branch0_weight_count,
                                   fixture.branch0_bias_i32,
                                   fixture.branch0_bias_count};
}

Y26Stage10BackboneExpansionConfig stage10_config(
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture,
    int activation_mode) {
    return Y26Stage10BackboneExpansionConfig{fixture.subset_id,
                                             stage9_config(*fixture.stage9_fixture, activation_mode),
                                             stage10_branch0_config(fixture),
                                             fixture.split_output1_scale,
                                             fixture.branch0_activation_zero_point_u8,
                                             1,
                                             16,
                                             16,
                                             activation_mode};
}

Y26Stage7ConvNodeConfig stage11_branch1_config(
    const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch1_node_name,
                                   fixture.branch1_params,
                                   fixture.branch1_kernel_h,
                                   fixture.branch1_kernel_w,
                                   fixture.branch1_activation_zero_point_u8,
                                   fixture.branch1_input_storage_zero_point_s8,
                                   fixture.branch0_act_output_scale,
                                   fixture.branch1_output_scale,
                                   fixture.branch1_output_zero_point_u8,
                                   fixture.branch1_weight_scales,
                                   fixture.branch1_weight_scale_count,
                                   fixture.branch1_weights_ohwi_s8,
                                   fixture.branch1_weight_count,
                                   fixture.branch1_bias_i32,
                                   fixture.branch1_bias_count};
}

Y26Stage11BranchBlockConfig stage11_config(
    const y26_stage11_branch_block_fixture::BranchBlockFixture& fixture,
    int activation_mode) {
    return Y26Stage11BranchBlockConfig{fixture.subset_id,
                                       stage10_config(*fixture.stage10_fixture, activation_mode),
                                       stage11_branch1_config(fixture),
                                       fixture.branch0_act_output_scale,
                                       fixture.branch1_activation_zero_point_u8,
                                       activation_mode};
}

Y26Stage7ConvNodeConfig model2_cv2_config(
    const y26_stage12_c2f_block_fixture::C2fBlockFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.model2_cv2_node_name,
                                   fixture.model2_cv2_params,
                                   fixture.model2_cv2_kernel_h,
                                   fixture.model2_cv2_kernel_w,
                                   fixture.concat_output_zero_point_u8,
                                   fixture.concat_input_storage_zero_point_s8,
                                   fixture.concat_output_scale,
                                   fixture.model2_cv2_output_scale,
                                   fixture.model2_cv2_output_zero_point_u8,
                                   fixture.model2_cv2_weight_scales,
                                   fixture.model2_cv2_weight_scale_count,
                                   fixture.model2_cv2_weights_ohwi_s8,
                                   fixture.model2_cv2_weight_count,
                                   fixture.model2_cv2_bias_i32,
                                   fixture.model2_cv2_bias_count};
}

Y26Stage12C2fBlockConfig stage12_config(
    const y26_stage12_c2f_block_fixture::C2fBlockFixture& fixture,
    int activation_mode) {
    return Y26Stage12C2fBlockConfig{fixture.subset_id,
                                    stage11_config(*fixture.stage11_fixture, activation_mode),
                                    model2_cv2_config(fixture),
                                    fixture.stage11_fixture->stage10_fixture->split_output1_scale,
                                    fixture.stage11_fixture->stage10_fixture->branch0_activation_zero_point_u8,
                                    fixture.concat_output_scale,
                                    fixture.concat_output_zero_point_u8,
                                    activation_mode,
                                    Y26_STAGE13_MERGE_MODE_A2_FUSED_QDQ_NHWC};
}

Y26Stage7ConvNodeConfig model3_config(
    const y26_stage14_next_c2f_fixture::NextC2fFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.model3_node_name,
                                   fixture.model3_params,
                                   fixture.model3_kernel_h,
                                   fixture.model3_kernel_w,
                                   fixture.model3_activation_zero_point_u8,
                                   fixture.model3_input_storage_zero_point_s8,
                                   fixture.model3_input_scale,
                                   fixture.model3_output_scale,
                                   fixture.model3_output_zero_point_u8,
                                   fixture.model3_weight_scales,
                                   fixture.model3_weight_scale_count,
                                   fixture.model3_weights_ohwi_s8,
                                   fixture.model3_weight_count,
                                   fixture.model3_bias_i32,
                                   fixture.model3_bias_count};
}

Y26Stage7ConvNodeConfig model4_cv1_config(
    const y26_stage14_next_c2f_fixture::NextC2fFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.model4_cv1_node_name,
                                   fixture.model4_cv1_params,
                                   fixture.model4_cv1_kernel_h,
                                   fixture.model4_cv1_kernel_w,
                                   fixture.model4_cv1_activation_zero_point_u8,
                                   fixture.model4_cv1_input_storage_zero_point_s8,
                                   fixture.model4_cv1_input_scale,
                                   fixture.model4_cv1_output_scale,
                                   fixture.model4_cv1_output_zero_point_u8,
                                   fixture.model4_cv1_weight_scales,
                                   fixture.model4_cv1_weight_scale_count,
                                   fixture.model4_cv1_weights_ohwi_s8,
                                   fixture.model4_cv1_weight_count,
                                   fixture.model4_cv1_bias_i32,
                                   fixture.model4_cv1_bias_count};
}

Y26Stage14NextC2fConfig stage14_config(
    const y26_stage14_next_c2f_fixture::NextC2fFixture& fixture,
    int activation_mode) {
    return Y26Stage14NextC2fConfig{fixture.subset_id,
                                   stage12_config(*fixture.stage12_fixture, activation_mode),
                                   model3_config(fixture),
                                   model4_cv1_config(fixture),
                                   fixture.model3_input_scale,
                                   fixture.model3_activation_zero_point_u8,
                                   fixture.model3_act_output_scale,
                                   fixture.model3_act_output_zero_point_u8,
                                   activation_mode};
}

Y26Stage7ConvNodeConfig stage15_branch0_config(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture) {
    return Y26Stage7ConvNodeConfig{fixture.branch0_node_name,
                                   fixture.branch0_params,
                                   fixture.branch0_kernel_h,
                                   fixture.branch0_kernel_w,
                                   fixture.branch0_activation_zero_point_u8,
                                   fixture.branch0_input_storage_zero_point_s8,
                                   fixture.split1_output_scale,
                                   fixture.branch0_output_scale,
                                   fixture.branch0_output_zero_point_u8,
                                   fixture.branch0_weight_scales,
                                   fixture.branch0_weight_scale_count,
                                   fixture.branch0_weights_ohwi_s8,
                                   fixture.branch0_weight_count,
                                   fixture.branch0_bias_i32,
                                   fixture.branch0_bias_count};
}

Y26Stage15Model4BranchConfig stage15_config(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture,
    int activation_mode) {
    return Y26Stage15Model4BranchConfig{fixture.subset_id,
                                        stage14_config(*fixture.stage14_fixture, activation_mode),
                                        stage15_branch0_config(fixture),
                                        fixture.split1_output_scale,
                                        fixture.split1_output_zero_point_u8,
                                        fixture.branch0_act_output_scale,
                                        fixture.branch0_act_output_zero_point_u8,
                                        activation_mode};
}

Y26Stage7ConvNodeConfig stage16_branch1_config(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    return Y26Stage7ConvNodeConfig{"/model.4/m.0/cv2/conv/Conv",
                                   fixture.branch1_params,
                                   fixture.branch1_kernel_h,
                                   fixture.branch1_kernel_w,
                                   fixture.branch1_activation_zero_point_u8,
                                   fixture.branch1_input_storage_zero_point_s8,
                                   fixture.branch1_input_scale,
                                   fixture.branch1_output_scale,
                                   fixture.branch1_output_zero_point_u8,
                                   fixture.branch1_weight_scales,
                                   fixture.branch1_weight_scale_count,
                                   fixture.branch1_weights_ohwi_s8,
                                   fixture.branch1_weight_count,
                                   fixture.branch1_bias_i32,
                                   fixture.branch1_bias_count};
}

Y26Stage7ConvNodeConfig model4_cv2_config(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    return Y26Stage7ConvNodeConfig{"/model.4/cv2/conv/Conv",
                                   fixture.model4_cv2_params,
                                   fixture.model4_cv2_kernel_h,
                                   fixture.model4_cv2_kernel_w,
                                   fixture.model4_cv2_activation_zero_point_u8,
                                   fixture.model4_cv2_input_storage_zero_point_s8,
                                   fixture.concat_output_scale,
                                   fixture.model4_cv2_output_scale,
                                   fixture.model4_cv2_output_zero_point_u8,
                                   fixture.model4_cv2_weight_scales,
                                   fixture.model4_cv2_weight_scale_count,
                                   fixture.model4_cv2_weights_ohwi_s8,
                                   fixture.model4_cv2_weight_count,
                                   fixture.model4_cv2_bias_i32,
                                   fixture.model4_cv2_bias_count};
}

const y26_stage16_model4_c2f_fixture::Model4C2fFixture* fixture_at(int fixture_id) {
    if (fixture_id < 0 || fixture_id >= static_cast<int>(std::size(y26_stage16_model4_c2f_fixture::kFixtures))) {
        return nullptr;
    }
    return y26_stage16_model4_c2f_fixture::kFixtures[fixture_id];
}

}  // namespace

extern "C" int y26_model4_fixture_count() {
    return static_cast<int>(std::size(y26_stage16_model4_c2f_fixture::kFixtures));
}

extern "C" int y26_model4_fixture_make(int fixture_id,
                                         int activation_mode,
                                         int merge_mode,
                                         Y26Stage16Model4C2fConfig* cfg,
                                         Y26Model4FixtureView* view) {
    const auto* fixture = fixture_at(fixture_id);
    if (fixture == nullptr || cfg == nullptr || view == nullptr) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    *cfg = Y26Stage16Model4C2fConfig{fixture->subset_id,
                                     stage15_config(*fixture->stage15_fixture, activation_mode),
                                     stage16_branch1_config(*fixture),
                                     model4_cv2_config(*fixture),
                                     fixture->concat_output_scale,
                                     fixture->concat_output_zero_point_u8,
                                     activation_mode,
                                     merge_mode};
    *view = Y26Model4FixtureView{
        fixture->label,
        fixture->stage15_fixture->stage14_fixture->stage12_fixture->stage11_fixture->stage10_fixture
            ->stage9_fixture->input_nhwc_s8,
        fixture->expected_branch1_i32_nhwc,
        fixture->expected_branch1_count,
        fixture->expected_concat_s8_nhwc,
        fixture->expected_concat_count,
        fixture->expected_model4_cv2_i32_nhwc,
        fixture->expected_model4_cv2_count,
    };
    return Y26_CONV_STATUS_SUCCESS;
}
