#include "y26_k1x_model5_island.h"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

class ScopedRoundToNearest {
public:
    ScopedRoundToNearest() : saved_(std::fegetround()), active_(std::fesetround(FE_TONEAREST) == 0) {}
    ~ScopedRoundToNearest() {
        if (active_ && saved_ != -1) {
            (void)std::fesetround(saved_);
        }
    }
    bool active() const { return active_; }

private:
    int saved_ = -1;
    bool active_ = false;
};

std::uint8_t quantize(float value, float scale, int zero_point) {
    const long rounded = std::lrint(static_cast<double>(value) / static_cast<double>(scale));
    return static_cast<std::uint8_t>(std::clamp<long>(rounded + zero_point, 0, 255));
}

float silu(float value) {
    return value / (1.0F + std::exp(-value));
}

int run_test() {
    ScopedRoundToNearest rounding;
    if (!rounding.active()) {
        std::cerr << "failed to set FE_TONEAREST\n";
        return 1;
    }
    constexpr int input_h = 5;
    constexpr int input_w = 5;
    constexpr int input_c = 8;
    constexpr int output_c = 4;
    constexpr int output_h = 3;
    constexpr int output_w = 3;
    constexpr float preact_scale = 0.125F;
    constexpr int preact_zp = 121;
    constexpr float post4_scale = 0.0625F;
    constexpr int post4_zp = 13;
    constexpr float conv_scale = 0.25F;
    constexpr int conv_zp = 127;
    constexpr float post5_scale = 0.125F;
    constexpr int post5_zp = 9;

    std::array<float, output_c> weight_scales {0.03125F, 0.046875F, 0.0625F, 0.078125F};
    std::array<std::int32_t, output_c> bias {3, -7, 11, -13};
    std::vector<std::int8_t> weights(output_c * 3 * 3 * input_c);
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = static_cast<std::int8_t>(static_cast<int>(index % 11) - 5);
    }
    std::vector<std::uint8_t> preact_u8(input_h * input_w * input_c);
    for (std::size_t index = 0; index < preact_u8.size(); ++index) {
        preact_u8[index] = static_cast<std::uint8_t>((index * 37 + 19) & 255U);
    }

    Y26Model5IslandConfig config {};
    config.model5_conv.node_name = "/model.5/conv/Conv";
    config.model5_conv.params = Y26Conv2DParams{input_h, input_w, input_c, output_c, 2, 2, 1, 1};
    config.model5_conv.kernel_h = 3;
    config.model5_conv.kernel_w = 3;
    config.model5_conv.activation_zero_point_u8 = post4_zp;
    config.model5_conv.input_storage_zero_point_s8 = post4_zp - 128;
    config.model5_conv.input_scale = post4_scale;
    config.model5_conv.output_scale = conv_scale;
    config.model5_conv.output_zero_point_u8 = conv_zp;
    config.model5_conv.weight_scales = weight_scales.data();
    config.model5_conv.weight_scale_count = weight_scales.size();
    config.model5_conv.weights_ohwi_s8 = weights.data();
    config.model5_conv.weight_count = weights.size();
    config.model5_conv.bias_i32 = bias.data();
    config.model5_conv.bias_count = bias.size();
    config.model4_preact_scale = preact_scale;
    config.model4_preact_zero_point_u8 = preact_zp;
    config.model4_postact_scale = post4_scale;
    config.model4_postact_zero_point_u8 = post4_zp;
    config.model5_postact_scale = post5_scale;
    config.model5_postact_zero_point_u8 = post5_zp;
    config.ime_accumulator_groups = 4;

    Y26Model5IslandWorkspace invalid_workspace;
    std::memset(&invalid_workspace, 0xa5, sizeof(invalid_workspace));
    if (y26_model5_island_prepare(&config, 2, &invalid_workspace) != Y26_CONV_STATUS_INVALID_ARGUMENT) {
        std::cerr << "prepare accepted a workspace without explicit init\n";
        return 1;
    }
    y26_model5_island_release(&invalid_workspace);

    Y26Model5IslandWorkspace workspace {};
    if (y26_model5_island_workspace_init(&workspace) != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "workspace init failed\n";
        return 1;
    }
    int status = y26_model5_island_prepare(&config, 2, &workspace);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        std::cerr << "prepare status=" << status << "\n";
        return 1;
    }
    Y26ThreadedConvPlan plan {};
    if (y26_threaded_conv_get_plan(workspace.threaded_conv, &plan) != Y26_CONV_STATUS_SUCCESS ||
        plan.output_h != output_h || plan.output_w != output_w || plan.thread_count != 2) {
        std::cerr << "unexpected stride2 worker plan\n";
        y26_model5_island_release(&workspace);
        return 1;
    }

    std::vector<std::int8_t> actual(output_h * output_w * output_c);
    Y26Model5IslandTimingUs timing {};
    status = y26_model5_island_run_scalar(&config, &workspace, preact_u8.data(), actual.data(), &timing);

    std::vector<std::uint8_t> post4_u8(preact_u8.size());
    for (std::size_t index = 0; index < preact_u8.size(); ++index) {
        const float value = (static_cast<int>(preact_u8[index]) - preact_zp) * preact_scale;
        post4_u8[index] = quantize(silu(value), post4_scale, post4_zp);
    }
    std::size_t mismatches = 0;
    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < output_c; ++oc) {
                std::int32_t accumulator = bias[oc];
                for (int kh = 0; kh < 3; ++kh) {
                    const int ih = oh * 2 + kh - 1;
                    for (int kw = 0; kw < 3; ++kw) {
                        const int iw = ow * 2 + kw - 1;
                        for (int ic = 0; ic < input_c; ++ic) {
                            const int input_code = ih >= 0 && ih < input_h && iw >= 0 && iw < input_w
                                                       ? post4_u8[(ih * input_w + iw) * input_c + ic]
                                                       : post4_zp;
                            const std::size_t weight_index =
                                static_cast<std::size_t>(((oc * 3 + kh) * 3 + kw) * input_c + ic);
                            accumulator += (input_code - post4_zp) * static_cast<int>(weights[weight_index]);
                        }
                    }
                }
                const float conv_real = static_cast<float>(accumulator) * post4_scale * weight_scales[oc];
                const std::uint8_t conv_code = quantize(conv_real, conv_scale, conv_zp);
                const float conv_dq = (static_cast<int>(conv_code) - conv_zp) * conv_scale;
                const std::uint8_t expected_code = quantize(silu(conv_dq), post5_scale, post5_zp);
                const std::int8_t expected_s8 = static_cast<std::int8_t>(static_cast<int>(expected_code) - 128);
                const std::size_t output_index =
                    static_cast<std::size_t>((oh * output_w + ow) * output_c + oc);
                mismatches += actual[output_index] != expected_s8 ? 1U : 0U;
            }
        }
    }
    std::cout << "stage43_model5_scalar status=" << status << " mismatches=" << mismatches
              << " output_count=" << actual.size() << "\n";
    y26_model5_island_release(&workspace);
    return status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 ? 0 : 1;
}

}  // namespace

int main() {
    return run_test();
}
