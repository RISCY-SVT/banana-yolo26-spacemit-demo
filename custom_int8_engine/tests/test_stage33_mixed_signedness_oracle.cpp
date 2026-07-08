#define Y26_STAGE16_NO_TEST_MAIN 1
#include "test_stage16_model4_c2f_runner.cpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::uint8_t u8_code_from_s8_storage_local(std::int8_t value) {
    return static_cast<std::uint8_t>(static_cast<int>(value) + 128);
}

std::int8_t weight_at_local(const Y26Stage7ConvNodeConfig& cfg, int oc, int kh, int kw, int ic) {
    const int index = ((oc * cfg.kernel_h + kh) * cfg.kernel_w + kw) * cfg.params.input_c + ic;
    return cfg.weights_ohwi_s8[index];
}

std::int32_t weight_sum_local(const Y26Stage7ConvNodeConfig& cfg, int oc) {
    std::int32_t sum = 0;
    for (int kh = 0; kh < cfg.kernel_h; ++kh) {
        for (int kw = 0; kw < cfg.kernel_w; ++kw) {
            for (int ic = 0; ic < cfg.params.input_c; ++ic) {
                sum += weight_at_local(cfg, oc, kh, kw, ic);
            }
        }
    }
    return sum;
}

int verify_model4_cv2_mixed_oracle(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage16Model4C2fConfig cfg =
        stage16_config_from_fixture(fixture,
                                    Y26_ACTIVATION_MODE_INT8_LUT,
                                    Y26_STAGE16_MERGE_MODE_STAGE26_BRANCH1_ADD_LUT);
    Y26Stage16Model4C2fWorkspace ws {};
    int status = y26_stage16_model4_c2f_prepare(&cfg, &ws);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage33 prepare failed status=" << status << "\n";
        return 1;
    }

    const std::int8_t* upstream_input = fixture.stage15_fixture->stage14_fixture->stage12_fixture->stage11_fixture
                                            ->stage10_fixture->stage9_fixture->input_nhwc_s8;
    std::vector<std::int32_t> runner_output(y26_stage16_model4_c2f_output_count(&cfg), 0);
    Y26Stage16TimingUs timing {};
    status = y26_stage16_model4_c2f_run_scalar(&cfg, &ws, upstream_input, runner_output.data(), &timing);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "stage33 scalar runner failed status=" << status << "\n";
        y26_stage16_model4_c2f_release(&ws);
        return 1;
    }

    const std::int8_t* concat = y26_stage16_model4_c2f_concat_s8(&ws);
    const Y26Stage7ConvNodeConfig& conv = cfg.model4_cv2;
    const int output_h = y26_conv1x1_output_h(&conv.params);
    const int output_w = y26_conv1x1_output_w(&conv.params);
    const int output_m = output_h * output_w;
    std::vector<std::int32_t> mixed_output(runner_output.size(), 0);

    std::size_t mismatches = 0;
    int max_abs_diff = 0;
    for (int m = 0; m < output_m; ++m) {
        const std::int8_t* src = concat + static_cast<std::size_t>(m) * conv.params.input_c;
        for (int oc = 0; oc < conv.params.output_c; ++oc) {
            std::int64_t mixed_acc = static_cast<std::int64_t>(conv.bias_i32[oc]) -
                                     static_cast<std::int64_t>(conv.activation_zero_point_u8) *
                                         static_cast<std::int64_t>(weight_sum_local(conv, oc));
            for (int ic = 0; ic < conv.params.input_c; ++ic) {
                mixed_acc += static_cast<std::int64_t>(u8_code_from_s8_storage_local(src[ic])) *
                             static_cast<std::int64_t>(weight_at_local(conv, oc, 0, 0, ic));
            }
            const std::int32_t value = static_cast<std::int32_t>(mixed_acc);
            mixed_output[static_cast<std::size_t>(m) * conv.params.output_c + static_cast<std::size_t>(oc)] = value;
            const int diff = value - runner_output[static_cast<std::size_t>(m) * conv.params.output_c +
                                                   static_cast<std::size_t>(oc)];
            if (diff != 0) {
                ++mismatches;
                const int abs_diff = diff < 0 ? -diff : diff;
                if (abs_diff > max_abs_diff) {
                    max_abs_diff = abs_diff;
                }
            }
        }
    }

    long long checksum = 0;
    for (std::int32_t value : mixed_output) {
        checksum += value;
    }
    std::cout << "stage33_mixed_signedness_oracle"
              << " fixture=" << fixture.label
              << " status=" << status
              << " activation_zero_point_u8=" << conv.activation_zero_point_u8
              << " input_storage_zero_point_s8=" << conv.input_storage_zero_point_s8
              << " output_elements=" << mixed_output.size()
              << " mismatches=" << mismatches
              << " max_abs_diff=" << max_abs_diff
              << " checksum=" << checksum
              << "\n";
    y26_stage16_model4_c2f_release(&ws);
    return mismatches == 0 && max_abs_diff == 0 ? 0 : 1;
}

}  // namespace

int main() {
    return verify_model4_cv2_mixed_oracle(y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture);
}
