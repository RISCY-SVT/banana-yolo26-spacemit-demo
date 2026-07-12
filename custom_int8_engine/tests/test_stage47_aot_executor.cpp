#include "y26_k1x_stage47_aot.h"

#include "y26_k1x_conv_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::int8_t storage_from_code(int code) {
    return static_cast<std::int8_t>(code - 128);
}

int test_layout_roundtrip() {
    constexpr int h = 3;
    constexpr int w = 5;
    constexpr int c = 8;
    std::vector<std::uint8_t> input(static_cast<std::size_t>(h) * w * c);
    std::vector<std::int8_t> resident(input.size());
    std::vector<std::uint8_t> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) input[index] = static_cast<std::uint8_t>((index * 17U + 3U) & 255U);
    y26::stage47::nchw_u8_to_nhwc_s8(input.data(), resident.data(), h, w, c);
    y26::stage47::nhwc_s8_to_nchw_u8(resident.data(), output.data(), h, w, c);
    return input == output ? 0 : 1;
}

int test_integrated_scalar_tail() {
    constexpr int h = 3;
    constexpr int w = 5;
    constexpr int input_c = 8;
    constexpr int output_c = 20;
    constexpr int input_zero_point = 121;
    constexpr int output_zero_point = 127;
    constexpr float input_scale = 0.125f;
    constexpr float weight_scale = 0.125f;
    constexpr float output_scale = input_scale * weight_scale;
    std::vector<std::int8_t> input(static_cast<std::size_t>(h) * w * input_c);
    std::vector<std::int8_t> weights(static_cast<std::size_t>(output_c) * input_c);
    std::vector<float> scales(output_c, weight_scale);
    std::vector<std::int32_t> bias(output_c);
    std::vector<std::int8_t> output(static_cast<std::size_t>(h) * w * output_c);
    std::vector<std::int8_t> expected(output.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = storage_from_code(input_zero_point + static_cast<int>(index % 9U) - 4);
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = static_cast<std::int8_t>(static_cast<int>(index % 7U) - 3);
    }
    for (int channel = 0; channel < output_c; ++channel) bias[channel] = channel - 10;

    y26::stage47::ConvSpec spec;
    spec.input = {h, w, input_c, input_scale, input_zero_point};
    spec.output_h = h;
    spec.output_w = w;
    spec.output_c = output_c;
    spec.kernel_h = 1;
    spec.kernel_w = 1;
    spec.stride_h = 1;
    spec.stride_w = 1;
    spec.conv_output_scale = output_scale;
    spec.conv_output_zero_point_u8 = output_zero_point;
    spec.weights_ohwi_s8 = weights.data();
    spec.weight_count = weights.size();
    spec.weight_scales = scales.data();
    spec.weight_scale_count = scales.size();
    spec.bias_i32 = bias.data();
    spec.bias_count = bias.size();
    const y26::stage47::TensorSpec output_spec{h, w, output_c, output_scale, output_zero_point};
    spec.segments.push_back({0, output_c, output_spec, false});

    y26::stage47::IntegratedConv conv;
    if (conv.prepare(spec) != Y26_CONV_STATUS_SUCCESS) return 2;
    y26::stage47::WorkerPool pool(1);
    y26::stage47::RunOptions options;
    options.kernel = y26::stage47::KernelShape::scalar;
    options.workers = 1;
    const std::array<std::int8_t*, 2> outputs{output.data(), nullptr};
    if (conv.run(pool, input.data(), outputs, 1, options, nullptr) != Y26_CONV_STATUS_SUCCESS) return 3;

    for (int pixel = 0; pixel < h * w; ++pixel) {
        for (int oc = 0; oc < output_c; ++oc) {
            std::int32_t accumulator = bias[oc];
            for (int ic = 0; ic < input_c; ++ic) {
                const int code = static_cast<int>(input[static_cast<std::size_t>(pixel) * input_c + ic]) + 128;
                accumulator += (code - input_zero_point) * weights[static_cast<std::size_t>(oc) * input_c + ic];
            }
            const int code = std::clamp(accumulator + output_zero_point, 0, 255);
            expected[static_cast<std::size_t>(pixel) * output_c + oc] = storage_from_code(code);
        }
    }
    return output == expected ? 0 : 4;
}

int test_reject_grouped() {
    std::array<std::int8_t, 8> weights{};
    std::array<float, 1> scales{1.0f};
    std::array<std::int32_t, 1> bias{};
    y26::stage47::ConvSpec spec;
    spec.input = {1, 1, 8, 1.0f, 0};
    spec.output_h = 1;
    spec.output_w = 1;
    spec.output_c = 1;
    spec.kernel_h = 1;
    spec.kernel_w = 1;
    spec.stride_h = 1;
    spec.stride_w = 1;
    spec.group = 8;
    spec.conv_output_scale = 1.0f;
    spec.weights_ohwi_s8 = weights.data();
    spec.weight_count = weights.size();
    spec.weight_scales = scales.data();
    spec.weight_scale_count = scales.size();
    spec.bias_i32 = bias.data();
    spec.bias_count = bias.size();
    spec.segments.push_back({0, 1, {1, 1, 1, 1.0f, 0}, false});
    y26::stage47::IntegratedConv conv;
    return conv.prepare(spec) == Y26_CONV_STATUS_INVALID_ARGUMENT ? 0 : 1;
}

}  // namespace

int main() {
    const int layout = test_layout_roundtrip();
    const int scalar = test_integrated_scalar_tail();
    const int grouped = test_reject_grouped();
    std::cout << "layout_roundtrip=" << layout << '\n';
    std::cout << "integrated_scalar_tail=" << scalar << '\n';
    std::cout << "grouped_rejection=" << grouped << '\n';
    return layout == 0 && scalar == 0 && grouped == 0 ? 0 : 1;
}
