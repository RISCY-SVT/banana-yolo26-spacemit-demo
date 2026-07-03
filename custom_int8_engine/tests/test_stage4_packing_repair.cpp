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
                                a = fixture.input_nhwc_s8[(ih * fixture.params.input_w + iw) *
                                                              fixture.params.input_c +
                                                          ic];
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

int compare_expected(const y26_stage3_real_fixture::RealConvFixture& fixture,
                     const std::vector<std::int32_t>& corrected,
                     const char* path_name) {
    const int ow_count = output_w(fixture);
    int mismatches = 0;
    for (int oh = 0; oh < fixture.compare_h; ++oh) {
        for (int ow = 0; ow < fixture.compare_w; ++ow) {
            for (int oc = 0; oc < fixture.params.output_c; ++oc) {
                const int got = corrected[(oh * ow_count + ow) * fixture.params.output_c + oc];
                const int expected =
                    fixture.expected_i32_nhwc[(oh * fixture.compare_w + ow) * fixture.params.output_c + oc];
                if (got != expected) {
                    ++mismatches;
                    if (mismatches <= 8) {
                        std::cerr << fixture.label << " " << path_name << " mismatch oh=" << oh << " ow=" << ow
                                  << " oc=" << oc << " got=" << got << " expected=" << expected << "\n";
                    }
                }
            }
        }
    }
    return mismatches;
}

int apply_correction(const y26_stage3_real_fixture::RealConvFixture& fixture,
                     const Y26PrepackedConvWeights* weights,
                     const std::vector<std::int32_t>& raw,
                     std::vector<std::int32_t>& corrected) {
    corrected.assign(raw.size(), 0);
    return y26_conv2d_apply_u8_as_s8_correction_nhwc(raw.data(),
                                                     fixture.bias_i32,
                                                     y26_prepacked_conv_weights_sums(weights),
                                                     corrected.data(),
                                                     output_h(fixture) * output_w(fixture),
                                                     fixture.params.output_c,
                                                     fixture.activation_zero_point_u8);
}

int verify_fixture(const y26_stage3_real_fixture::RealConvFixture& fixture) {
    Y26PrepackedConvWeights* weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        fixture.weights_ohwi_s8, &fixture.params, fixture.kernel_h, fixture.kernel_w, fixture.label, nullptr);
    if (weights == nullptr) {
        std::cerr << fixture.label << " persistent prepack create failed\n";
        return 1;
    }
    Y26ConvWorkspace* workspace = y26_conv_workspace_create(&fixture.params, fixture.kernel_h, fixture.kernel_w);
    if (workspace == nullptr) {
        std::cerr << fixture.label << " workspace create failed\n";
        y26_prepacked_conv_weights_destroy(weights);
        return 1;
    }

    int failures = 0;
    const std::size_t expected_b_bytes = y26_mmt4d_packed_b_bytes(
        fixture.params.output_c, fixture.kernel_h * fixture.kernel_w * fixture.params.input_c);
    if (y26_prepacked_conv_weights_packed_b_bytes(weights) != expected_b_bytes ||
        y26_conv_workspace_bytes(workspace) !=
            y26_conv_mmt4d_a_workspace_bytes(&fixture.params, fixture.kernel_h, fixture.kernel_w)) {
        std::cerr << fixture.label << " size metadata mismatch\n";
        ++failures;
    }

    std::vector<std::int32_t> raw_scalar;
    scalar_raw_dot(fixture, raw_scalar);
    std::vector<std::int32_t> corrected_scalar;
    const int scalar_correction = apply_correction(fixture, weights, raw_scalar, corrected_scalar);
    const int scalar_mismatches =
        scalar_correction == Y26_CONV_STATUS_SUCCESS ? compare_expected(fixture, corrected_scalar, "scalar") : 1;
    failures += scalar_mismatches == 0 ? 0 : 1;

    int m_major_status = Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    int n_major_status = Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
    int m_major_mismatches = 0;
    int n_major_mismatches = 0;
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::vector<std::int32_t> raw_ime(static_cast<std::size_t>(
            output_h(fixture) * output_w(fixture) * fixture.params.output_c));
        std::vector<std::int32_t> corrected_ime;
        m_major_status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(fixture.input_nhwc_s8,
                                                                   weights,
                                                                   raw_ime.data(),
                                                                   fixture.input_storage_zero_point_s8,
                                                                   workspace,
                                                                   Y26_CONV_LOOP_ORDER_M_MAJOR);
        if (m_major_status == Y26_CONV_STATUS_SUCCESS &&
            apply_correction(fixture, weights, raw_ime, corrected_ime) == Y26_CONV_STATUS_SUCCESS) {
            m_major_mismatches = compare_expected(fixture, corrected_ime, "m_major");
        } else {
            m_major_mismatches = 1;
        }

        n_major_status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(fixture.input_nhwc_s8,
                                                                   weights,
                                                                   raw_ime.data(),
                                                                   fixture.input_storage_zero_point_s8,
                                                                   workspace,
                                                                   Y26_CONV_LOOP_ORDER_N_MAJOR);
        if (n_major_status == Y26_CONV_STATUS_SUCCESS &&
            apply_correction(fixture, weights, raw_ime, corrected_ime) == Y26_CONV_STATUS_SUCCESS) {
            n_major_mismatches = compare_expected(fixture, corrected_ime, "n_major");
        } else {
            n_major_mismatches = 1;
        }
        failures += (m_major_status == Y26_CONV_STATUS_SUCCESS && m_major_mismatches == 0) ? 0 : 1;
        failures += (n_major_status == Y26_CONV_STATUS_SUCCESS && n_major_mismatches == 0) ? 0 : 1;
    }

    std::cout << "stage4_packing_repair label=" << fixture.label
              << " prepack_bytes=" << y26_prepacked_conv_weights_total_bytes(weights)
              << " workspace_bytes=" << y26_conv_workspace_bytes(workspace)
              << " scalar_mismatches=" << scalar_mismatches << " m_major_status=" << m_major_status
              << " m_major_mismatches=" << m_major_mismatches << " n_major_status=" << n_major_status
              << " n_major_mismatches=" << n_major_mismatches << "\n";

    y26_conv_workspace_destroy(workspace);
    y26_prepacked_conv_weights_destroy(weights);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto* fixture : y26_stage3_real_fixture::kFixtures) {
        failures += verify_fixture(*fixture);
    }
    return failures == 0 ? 0 : 1;
}
