#include "stage10_backbone_expansion_fixture.h"
#include "y26_k1x_activation.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::size_t mismatches_i8(const std::vector<std::int8_t>& actual, const std::int8_t* expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    return mismatches;
}

Y26ActivationRequantParams conv2_to_split_params(
    const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture) {
    const auto& stage9 = *fixture.stage9_fixture;
    return Y26ActivationRequantParams{stage9.expected_conv2_count,
                                      stage9.conv2_params.output_c,
                                      stage9.act1_output_scale,
                                      stage9.conv2_weight_scales,
                                      stage9.conv2_output_scale,
                                      stage9.conv2_output_zero_point_u8,
                                      fixture.split_output1_scale,
                                      fixture.branch0_activation_zero_point_u8};
}

#if defined(__riscv_vector)
unsigned read_frm() {
    unsigned frm = 0;
    asm volatile("frrm %0" : "=r"(frm));
    return frm & 7U;
}

void set_frm(unsigned frm) {
    switch (frm) {
        case 0:
            asm volatile("fsrmi 0" ::: "memory");
            break;
        case 1:
            asm volatile("fsrmi 1" ::: "memory");
            break;
        case 2:
            asm volatile("fsrmi 2" ::: "memory");
            break;
        case 3:
            asm volatile("fsrmi 3" ::: "memory");
            break;
        case 4:
            asm volatile("fsrmi 4" ::: "memory");
            break;
        default:
            asm volatile("fsrmi 0" ::: "memory");
            break;
    }
}
#endif

int verify_fixture(const y26_stage10_backbone_expansion_fixture::BackboneExpansionFixture& fixture) {
    const auto& stage9 = *fixture.stage9_fixture;
    const Y26ActivationRequantParams params = conv2_to_split_params(fixture);
    std::int8_t lut[256] {};
    int status = y26_build_silu_u8_to_s8_lut(stage9.conv2_output_scale,
                                             stage9.conv2_output_zero_point_u8,
                                             fixture.split_output1_scale,
                                             fixture.branch0_activation_zero_point_u8,
                                             lut);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "lut build failed label=" << fixture.label << " status=" << status << "\n";
        return 1;
    }

    std::vector<std::int8_t> baseline(params.element_count, 0);
    status = y26_activation_requant_silu_int8_lut(&params, stage9.expected_conv2_i32_nhwc, lut, baseline.data());
    const std::size_t baseline_mismatches = mismatches_i8(baseline, fixture.expected_conv2_act_s8_nhwc);
    std::cout << "stage10_rvv_rounding_baseline label=" << fixture.label << " status=" << status
              << " mismatches=" << baseline_mismatches << "\n";
    if (status != Y26_CONV_STATUS_SUCCESS || baseline_mismatches != 0) {
        return 1;
    }

#if defined(__riscv_vector)
    const unsigned saved_frm = read_frm();
    int failures = 0;
    for (unsigned frm : {0U, 1U, 2U, 3U, 4U}) {
        set_frm(frm);
        std::vector<std::int8_t> actual(params.element_count, 0);
        status = y26_activation_requant_silu_int8_lut_rvv_f32(
            &params, stage9.expected_conv2_i32_nhwc, lut, actual.data());
        const std::size_t mismatches = mismatches_i8(actual, fixture.expected_conv2_act_s8_nhwc);
        const unsigned after_frm = read_frm();
        std::cout << "stage10_rvv_rounding_control label=" << fixture.label << " ambient_frm=" << frm
                  << " status=" << status << " mismatches=" << mismatches << " after_frm=" << after_frm
                  << "\n";
        failures +=
            (status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 && after_frm == frm) ? 0 : 1;
    }
    set_frm(saved_frm);
    return failures == 0 ? 0 : 1;
#else
    std::cout << "stage10_rvv_rounding_control skipped_no_rvv label=" << fixture.label << "\n";
    return 0;
#endif
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage10_backbone_expansion_fixture::kFixtures) {
        failures += verify_fixture(*fixture);
    }
    return failures == 0 ? 0 : 1;
}
