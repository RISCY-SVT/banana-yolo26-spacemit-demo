#include "stage3_real_conv_fixture.h"
#include "y26_k1x_vmadot.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchResult {
    double mean_us;
    std::int64_t checksum;
    int status;
};

Y26Conv2DParams full_shape_params(const y26_stage3_real_fixture::RealConvFixture& fixture) {
    Y26Conv2DParams params = fixture.params;
    params.input_h = 160;
    params.input_w = 160;
    return params;
}

int output_h(const Y26Conv2DParams& params, int kernel_h) {
    return kernel_h == 1 ? y26_conv1x1_output_h(&params) : y26_conv3x3_output_h(&params);
}

int output_w(const Y26Conv2DParams& params, int kernel_w) {
    return kernel_w == 1 ? y26_conv1x1_output_w(&params) : y26_conv3x3_output_w(&params);
}

std::vector<std::int8_t> make_input(const Y26Conv2DParams& params, int seed) {
    std::vector<std::int8_t> input(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int q = static_cast<int>((i * 37 + seed * 17) & 255);
        input[i] = static_cast<std::int8_t>(q - 128);
    }
    return input;
}

std::vector<std::int8_t> tiled_weights(const y26_stage3_real_fixture::RealConvFixture& fixture,
                                       const Y26Conv2DParams& params) {
    const int kernel_h = fixture.kernel_h;
    const int kernel_w = fixture.kernel_w;
    std::vector<std::int8_t> weights(
        static_cast<std::size_t>(params.output_c * kernel_h * kernel_w * params.input_c), 0);
    for (int oc = 0; oc < params.output_c; ++oc) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                for (int ic = 0; ic < params.input_c; ++ic) {
                    const int src = ((oc % fixture.params.output_c * kernel_h + kh) * kernel_w + kw) *
                                        fixture.params.input_c +
                                    (ic % fixture.params.input_c);
                    const int dst = ((oc * kernel_h + kh) * kernel_w + kw) * params.input_c + ic;
                    weights[dst] = fixture.weights_ohwi_s8[src];
                }
            }
        }
    }
    return weights;
}

std::int64_t checksum_i32(const std::vector<std::int32_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::int64_t{0});
}

template <typename Fn>
BenchResult measure(int iterations, Fn&& fn) {
    int status = 0;
    std::int64_t checksum = 0;
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = fn();
        status = result.first;
        checksum += result.second;
    }
    const auto end = Clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return {static_cast<double>(ns) / 1000.0 / static_cast<double>(iterations), checksum, status};
}

std::int64_t pack_a_panels_checksum(const std::vector<std::int8_t>& input,
                                    const Y26Conv2DParams& params,
                                    int kernel_h,
                                    int kernel_w,
                                    std::int8_t storage_zero_point,
                                    std::vector<std::int8_t>& workspace) {
    const int oh_count = output_h(params, kernel_h);
    const int ow_count = output_w(params, kernel_w);
    const int output_m = oh_count * ow_count;
    const int kernel_k = kernel_h * kernel_w * params.input_c;
    const int k_padded = ((kernel_k + 7) / 8) * 8;
    std::int64_t checksum = 0;
    for (int m0 = 0; m0 < output_m; m0 += 4) {
        for (int m = 0; m < 4; ++m) {
            const int flat_m = m0 + m;
            for (int k = 0; k < k_padded; ++k) {
                std::int8_t value = 0;
                if (flat_m < output_m && k < kernel_k) {
                    const int oh = flat_m / ow_count;
                    const int ow = flat_m % ow_count;
                    const int kh = k / (kernel_w * params.input_c);
                    const int rem = k % (kernel_w * params.input_c);
                    const int kw = rem / params.input_c;
                    const int ic = rem % params.input_c;
                    const int ih = oh * params.stride_h + kh - params.pad_h;
                    const int iw = ow * params.stride_w + kw - params.pad_w;
                    if (ih < 0 || iw < 0 || ih >= params.input_h || iw >= params.input_w) {
                        value = storage_zero_point;
                    } else {
                        value = input[(ih * params.input_w + iw) * params.input_c + ic];
                    }
                }
                workspace[m * k_padded + k] = value;
                checksum += value;
            }
        }
    }
    return checksum;
}

void run_fixture(const y26_stage3_real_fixture::RealConvFixture& fixture, int iterations) {
    const Y26Conv2DParams params = full_shape_params(fixture);
    const int oh_count = output_h(params, fixture.kernel_h);
    const int ow_count = output_w(params, fixture.kernel_w);
    const int output_count = oh_count * ow_count * params.output_c;
    const int kernel_k = fixture.kernel_h * fixture.kernel_w * params.input_c;
    std::vector<std::int8_t> input = make_input(params, fixture.kernel_h == 1 ? 11 : 23);
    std::vector<std::int8_t> weights = tiled_weights(fixture, params);
    std::vector<std::int32_t> output(static_cast<std::size_t>(output_count), 0);
    std::vector<std::int32_t> corrected(static_cast<std::size_t>(output_count), 0);
    std::vector<std::int8_t> packed_b(y26_mmt4d_packed_b_bytes(params.output_c, kernel_k), 0);
    std::vector<std::int32_t> weight_sums(static_cast<std::size_t>(params.output_c), 0);
    std::vector<std::int32_t> bias(static_cast<std::size_t>(params.output_c), 0);
    for (int oc = 0; oc < params.output_c; ++oc) {
        bias[oc] = fixture.bias_i32[oc % fixture.params.output_c];
    }
    std::vector<std::int8_t> workspace(
        y26_conv_mmt4d_a_workspace_bytes(&params, fixture.kernel_h, fixture.kernel_w), 0);

    const int prepack_status = fixture.kernel_h == 1
                                   ? y26_conv1x1_prepack_weights_mmt4d_s8(
                                         weights.data(), &params, packed_b.data(), packed_b.size(), weight_sums.data())
                                   : y26_conv3x3_prepack_weights_mmt4d_s8(
                                         weights.data(), &params, packed_b.data(), packed_b.size(), weight_sums.data());

    const auto scalar = measure(iterations, [&]() {
        int status = fixture.kernel_h == 1
                         ? y26_conv1x1_i8s8s32_nhwc_scalar(input.data(), weights.data(), nullptr, output.data(), &params)
                         : y26_conv3x3_i8s8s32_nhwc_scalar(input.data(), weights.data(), nullptr, output.data(), &params);
        return std::make_pair(status, checksum_i32(output));
    });

    const auto weight_prepack = measure(iterations, [&]() {
        int status = fixture.kernel_h == 1
                         ? y26_conv1x1_prepack_weights_mmt4d_s8(
                               weights.data(), &params, packed_b.data(), packed_b.size(), weight_sums.data())
                         : y26_conv3x3_prepack_weights_mmt4d_s8(
                               weights.data(), &params, packed_b.data(), packed_b.size(), weight_sums.data());
        return std::make_pair(status, static_cast<std::int64_t>(packed_b[0]) + weight_sums[0]);
    });

    const auto activation_pack = measure(iterations, [&]() {
        const auto checksum = pack_a_panels_checksum(input,
                                                     params,
                                                     fixture.kernel_h,
                                                     fixture.kernel_w,
                                                     static_cast<std::int8_t>(fixture.input_storage_zero_point_s8),
                                                     workspace);
        return std::make_pair(0, checksum);
    });

    const auto correction = measure(iterations, [&]() {
        const int status = y26_conv2d_apply_u8_as_s8_correction_nhwc(output.data(),
                                                                      bias.data(),
                                                                      weight_sums.data(),
                                                                      corrected.data(),
                                                                      oh_count * ow_count,
                                                                      params.output_c,
                                                                      fixture.activation_zero_point_u8);
        return std::make_pair(status, checksum_i32(corrected));
    });

    BenchResult old_wrapper {0.0, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME};
    BenchResult prepacked {0.0, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        old_wrapper = measure(iterations, [&]() {
            int status = fixture.kernel_h == 1
                             ? y26_conv1x1_i8s8s32_nhwc_ime(input.data(), weights.data(), nullptr, output.data(), &params)
                             : y26_conv3x3_i8s8s32_nhwc_ime(input.data(), weights.data(), nullptr, output.data(), &params);
            return std::make_pair(status, checksum_i32(output));
        });
        prepacked = measure(iterations, [&]() {
            int status = fixture.kernel_h == 1
                             ? y26_conv1x1_i8s8s32_nhwc_ime_prepacked(input.data(),
                                                                       packed_b.data(),
                                                                       output.data(),
                                                                       &params,
                                                                       fixture.input_storage_zero_point_s8,
                                                                       workspace.data(),
                                                                       workspace.size())
                             : y26_conv3x3_i8s8s32_nhwc_ime_prepacked(input.data(),
                                                                       packed_b.data(),
                                                                       output.data(),
                                                                       &params,
                                                                       fixture.input_storage_zero_point_s8,
                                                                       workspace.data(),
                                                                       workspace.size());
            return std::make_pair(status, checksum_i32(output));
        });
    }

    std::cout << "case=" << fixture.label << " shape=" << params.input_h << "x" << params.input_w << "x"
              << params.input_c << "->" << params.output_c << " kernel=" << fixture.kernel_h << "x"
              << fixture.kernel_w << " iterations=" << iterations << " prepack_status=" << prepack_status << "\n";
    std::cout << "  scalar_us=" << scalar.mean_us << " status=" << scalar.status << " checksum=" << scalar.checksum
              << "\n";
    std::cout << "  old_on_the_fly_ime_us=" << old_wrapper.mean_us << " status=" << old_wrapper.status
              << " checksum=" << old_wrapper.checksum << "\n";
    std::cout << "  prepacked_ime_us=" << prepacked.mean_us << " status=" << prepacked.status
              << " checksum=" << prepacked.checksum << "\n";
    std::cout << "  weight_prepack_us=" << weight_prepack.mean_us << " status=" << weight_prepack.status
              << " checksum=" << weight_prepack.checksum << "\n";
    std::cout << "  activation_pack_or_im2col_us=" << activation_pack.mean_us << " status=" << activation_pack.status
              << " checksum=" << activation_pack.checksum << "\n";
    std::cout << "  correction_us=" << correction.mean_us << " status=" << correction.status
              << " checksum=" << correction.checksum << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 3;
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
    }
    std::cout << "STAGE3_PACKING_BENCH_BEGIN\n";
    std::cout << "note=kernel/block microbenchmark only, not YOLO26 inference FPS\n";
    for (const auto* fixture : y26_stage3_real_fixture::kFixtures) {
        run_fixture(*fixture, iterations);
    }
    std::cout << "STAGE3_PACKING_BENCH_END\n";
    return 0;
}
