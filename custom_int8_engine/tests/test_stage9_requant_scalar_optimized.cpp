#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_activation.h"
#include "y26_k1x_conv_kernels.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::size_t mismatches_i8(const std::vector<std::int8_t>& actual, const std::vector<std::int8_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            ++mismatches;
        }
    }
    return mismatches + (actual.size() > expected.size() ? actual.size() - expected.size()
                                                         : expected.size() - actual.size());
}

int verify_boundary(const char* label,
                    const Y26ActivationRequantParams& params,
                    const std::int32_t* producer_i32,
                    const std::int8_t* expected_s8) {
    std::int8_t lut[256] {};
    std::vector<Y26FixedRequantParams> fixed(static_cast<std::size_t>(params.channels));
    std::vector<std::int8_t> baseline(params.element_count, 0);
    std::vector<std::int8_t> optimized(params.element_count, 0);
    std::vector<std::int8_t> fixed_lut(params.element_count, 0);
    if (y26_build_silu_u8_to_s8_lut(params.conv_output_scale,
                                    params.conv_output_zero_point_u8,
                                    params.act_output_scale,
                                    params.act_output_zero_point_u8,
                                    lut) != Y26_CONV_STATUS_SUCCESS ||
        y26_build_fixed_requant_params_per_channel(&params, fixed.data()) != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << label << " setup failed\n";
        return 1;
    }

    const int baseline_status =
        y26_activation_requant_silu_int8_lut(&params, producer_i32, lut, baseline.data());
    const int optimized_status =
        y26_activation_requant_silu_int8_lut_scalar_unrolled(&params, producer_i32, lut, optimized.data());
    const int fixed_status = y26_activation_requant_silu_int8_lut_fixed_requant(
        &params, fixed.data(), producer_i32, lut, fixed_lut.data());

    const std::vector<std::int8_t> expected(expected_s8, expected_s8 + params.element_count);
    const std::size_t baseline_mismatches = mismatches_i8(baseline, expected);
    const std::size_t optimized_mismatches = mismatches_i8(optimized, baseline);
    const std::size_t fixed_mismatches = mismatches_i8(fixed_lut, baseline);
    std::cout << "stage9_requant_boundary label=" << label << " baseline_status=" << baseline_status
              << " optimized_status=" << optimized_status << " fixed_status=" << fixed_status
              << " baseline_mismatches=" << baseline_mismatches
              << " optimized_mismatches=" << optimized_mismatches << " fixed_mismatches=" << fixed_mismatches
              << "\n";
#if defined(__riscv_vector)
    std::vector<std::int8_t> rvv_lut(params.element_count, 0);
    const int rvv_status =
        y26_activation_requant_silu_int8_lut_rvv_f32(&params, producer_i32, lut, rvv_lut.data());
    const std::size_t rvv_mismatches = mismatches_i8(rvv_lut, baseline);
    std::cout << "stage9_requant_boundary_rvv label=" << label << " rvv_status=" << rvv_status
              << " rvv_mismatches=" << rvv_mismatches << "\n";
    if (rvv_status != Y26_CONV_STATUS_SUCCESS || rvv_mismatches != 0) {
        return 1;
    }
#endif
    return baseline_status == Y26_CONV_STATUS_SUCCESS && optimized_status == Y26_CONV_STATUS_SUCCESS &&
                   fixed_status == Y26_CONV_STATUS_SUCCESS && baseline_mismatches == 0 &&
                   optimized_mismatches == 0 && fixed_mismatches == 0
               ? 0
               : 1;
}

}  // namespace

int main() {
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    const Y26ActivationRequantParams act0_params{fixture.expected_conv0_count,
                                                 fixture.conv0_params.output_c,
                                                 fixture.images_scale,
                                                 fixture.conv0_weight_scales,
                                                 fixture.conv0_output_scale,
                                                 fixture.conv0_output_zero_point_u8,
                                                 fixture.act0_output_scale,
                                                 fixture.act0_output_zero_point_u8};
    const Y26ActivationRequantParams act1_params{fixture.expected_conv1_count,
                                                 fixture.conv1_params.output_c,
                                                 fixture.act0_output_scale,
                                                 fixture.conv1_weight_scales,
                                                 fixture.conv1_output_scale,
                                                 fixture.conv1_output_zero_point_u8,
                                                 fixture.act1_output_scale,
                                                 fixture.act1_output_zero_point_u8};
    int failures = 0;
    failures += verify_boundary("act0", act0_params, fixture.expected_conv0_i32_nhwc, fixture.expected_act0_s8_nhwc);
    failures += verify_boundary("act1", act1_params, fixture.expected_conv1_i32_nhwc, fixture.expected_act1_s8_nhwc);
    return failures == 0 ? 0 : 1;
}
