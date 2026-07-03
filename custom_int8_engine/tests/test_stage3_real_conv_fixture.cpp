#include "stage3_real_conv_fixture.h"
#include "y26_k1x_vmadot.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int output_h(const y26_stage3_real_fixture::RealConvFixture& fixture) {
    return fixture.kernel_h == 1 ? y26_conv1x1_output_h(&fixture.params) : y26_conv3x3_output_h(&fixture.params);
}

int output_w(const y26_stage3_real_fixture::RealConvFixture& fixture) {
    return fixture.kernel_w == 1 ? y26_conv1x1_output_w(&fixture.params) : y26_conv3x3_output_w(&fixture.params);
}

std::int8_t weight_at(const y26_stage3_real_fixture::RealConvFixture& fixture, int oc, int kh, int kw, int ic) {
    const int index = ((oc * fixture.kernel_h + kh) * fixture.kernel_w + kw) * fixture.params.input_c + ic;
    return fixture.weights_ohwi_s8[index];
}

void scalar_raw_dot(const y26_stage3_real_fixture::RealConvFixture& fixture, std::vector<std::int32_t>& raw) {
    const int oh_count = output_h(fixture);
    const int ow_count = output_w(fixture);
    raw.assign(static_cast<std::size_t>(oh_count * ow_count * fixture.params.output_c), 0);
    for (int oh = 0; oh < oh_count; ++oh) {
        for (int ow = 0; ow < ow_count; ++ow) {
            for (int oc = 0; oc < fixture.params.output_c; ++oc) {
                std::int32_t acc = 0;
                for (int kh = 0; kh < fixture.kernel_h; ++kh) {
                    for (int kw = 0; kw < fixture.kernel_w; ++kw) {
                        const int ih = oh * fixture.params.stride_h + kh - fixture.params.pad_h;
                        const int iw = ow * fixture.params.stride_w + kw - fixture.params.pad_w;
                        for (int ic = 0; ic < fixture.params.input_c; ++ic) {
                            std::int8_t a = static_cast<std::int8_t>(fixture.input_storage_zero_point_s8);
                            if (ih >= 0 && iw >= 0 && ih < fixture.params.input_h && iw < fixture.params.input_w) {
                                a = fixture.input_nhwc_s8[(ih * fixture.params.input_w + iw) * fixture.params.input_c + ic];
                            }
                            acc += static_cast<std::int32_t>(a) *
                                   static_cast<std::int32_t>(weight_at(fixture, oc, kh, kw, ic));
                        }
                    }
                }
                raw[(oh * ow_count + ow) * fixture.params.output_c + oc] = acc;
            }
        }
    }
}

int prepack(const y26_stage3_real_fixture::RealConvFixture& fixture,
            std::vector<std::int8_t>& packed_b,
            std::vector<std::int32_t>& weight_sums) {
    const int kernel_k = fixture.kernel_h * fixture.kernel_w * fixture.params.input_c;
    packed_b.assign(y26_mmt4d_packed_b_bytes(fixture.params.output_c, kernel_k), 0);
    weight_sums.assign(static_cast<std::size_t>(fixture.params.output_c), 0);
    if (fixture.kernel_h == 1) {
        return y26_conv1x1_prepack_weights_mmt4d_s8(
            fixture.weights_ohwi_s8, &fixture.params, packed_b.data(), packed_b.size(), weight_sums.data());
    }
    return y26_conv3x3_prepack_weights_mmt4d_s8(
        fixture.weights_ohwi_s8, &fixture.params, packed_b.data(), packed_b.size(), weight_sums.data());
}

int run_prepacked_ime(const y26_stage3_real_fixture::RealConvFixture& fixture,
                      const std::vector<std::int8_t>& packed_b,
                      std::vector<std::int32_t>& raw) {
    const int oh_count = output_h(fixture);
    const int ow_count = output_w(fixture);
    raw.assign(static_cast<std::size_t>(oh_count * ow_count * fixture.params.output_c), 0);
    std::vector<std::int8_t> workspace(
        y26_conv_mmt4d_a_workspace_bytes(&fixture.params, fixture.kernel_h, fixture.kernel_w), 0);
    if (fixture.kernel_h == 1) {
        return y26_conv1x1_i8s8s32_nhwc_ime_prepacked(fixture.input_nhwc_s8,
                                                       packed_b.data(),
                                                       raw.data(),
                                                       &fixture.params,
                                                       fixture.input_storage_zero_point_s8,
                                                       workspace.data(),
                                                       workspace.size());
    }
    return y26_conv3x3_i8s8s32_nhwc_ime_prepacked(fixture.input_nhwc_s8,
                                                   packed_b.data(),
                                                   raw.data(),
                                                   &fixture.params,
                                                   fixture.input_storage_zero_point_s8,
                                                   workspace.data(),
                                                   workspace.size());
}

int compare_expected(const y26_stage3_real_fixture::RealConvFixture& fixture,
                     const std::vector<std::int32_t>& corrected) {
    const int ow_count = output_w(fixture);
    int mismatches = 0;
    for (int oh = 0; oh < fixture.compare_h; ++oh) {
        for (int ow = 0; ow < fixture.compare_w; ++ow) {
            for (int oc = 0; oc < fixture.params.output_c; ++oc) {
                const int got = corrected[(oh * ow_count + ow) * fixture.params.output_c + oc];
                const int expected = fixture.expected_i32_nhwc[(oh * fixture.compare_w + ow) * fixture.params.output_c + oc];
                if (got != expected) {
                    ++mismatches;
                    if (mismatches <= 8) {
                        std::cerr << fixture.label << " mismatch oh=" << oh << " ow=" << ow << " oc=" << oc
                                  << " got=" << got << " expected=" << expected << "\n";
                    }
                }
            }
        }
    }
    return mismatches;
}

int verify_fixture(const y26_stage3_real_fixture::RealConvFixture& fixture) {
    std::vector<std::int8_t> packed_b;
    std::vector<std::int32_t> weight_sums;
    const int prepack_status = prepack(fixture, packed_b, weight_sums);
    if (prepack_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << fixture.label << " prepack_status=" << prepack_status << "\n";
        return 1;
    }

    std::vector<std::int32_t> raw_scalar;
    scalar_raw_dot(fixture, raw_scalar);
    std::vector<std::int32_t> corrected_scalar(raw_scalar.size(), 0);
    const int correction_status = y26_conv2d_apply_u8_as_s8_correction_nhwc(raw_scalar.data(),
                                                                             fixture.bias_i32,
                                                                             weight_sums.data(),
                                                                             corrected_scalar.data(),
                                                                             output_h(fixture) * output_w(fixture),
                                                                             fixture.params.output_c,
                                                                             fixture.activation_zero_point_u8);
    if (correction_status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << fixture.label << " correction_status=" << correction_status << "\n";
        return 1;
    }
    const int scalar_mismatches = compare_expected(fixture, corrected_scalar);

    int ime_status = Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    int ime_mismatches = 0;
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::vector<std::int32_t> raw_ime;
        ime_status = run_prepacked_ime(fixture, packed_b, raw_ime);
        if (ime_status == Y26_CONV_STATUS_SUCCESS) {
            std::vector<std::int32_t> corrected_ime(raw_ime.size(), 0);
            const int ime_correction_status = y26_conv2d_apply_u8_as_s8_correction_nhwc(raw_ime.data(),
                                                                                        fixture.bias_i32,
                                                                                        weight_sums.data(),
                                                                                        corrected_ime.data(),
                                                                                        output_h(fixture) * output_w(fixture),
                                                                                        fixture.params.output_c,
                                                                                        fixture.activation_zero_point_u8);
            if (ime_correction_status != Y26_CONV_STATUS_SUCCESS) {
                std::cerr << fixture.label << " ime_correction_status=" << ime_correction_status << "\n";
                return 1;
            }
            ime_mismatches = compare_expected(fixture, corrected_ime);
        }
    }

    std::cout << "stage3_real_conv label=" << fixture.label << " scalar_mismatches=" << scalar_mismatches
              << " ime_status=" << ime_status << " ime_mismatches=" << ime_mismatches << "\n";
    if (scalar_mismatches != 0 || ime_mismatches != 0) {
        return 1;
    }
    if (y26_vmadot_4x4x8_ime_available_buildtime() && ime_status != Y26_CONV_STATUS_SUCCESS) {
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage3_real_fixture::kFixtures) {
        failures += verify_fixture(*fixture);
    }
    return failures == 0 ? 0 : 1;
}
