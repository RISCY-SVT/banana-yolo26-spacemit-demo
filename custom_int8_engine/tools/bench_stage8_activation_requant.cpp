#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_activation.h"
#include "y26_k1x_backbone_subset_runner.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
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
    Y26Stage7TimingUs timing;
    std::int64_t checksum;
    int status;
    std::size_t mismatches;
};

struct LookupBenchResult {
    double mean_us;
    std::int64_t checksum;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

const char* activation_mode_name(int mode) {
    switch (mode) {
        case Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE:
            return "scalar_float_reference";
        case Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY:
            return "fixed_requant_only";
        case Y26_ACTIVATION_MODE_INT8_LUT:
            return "int8_lut";
        case Y26_ACTIVATION_MODE_FUSED_LUT_PACK:
            return "fused_lut_pack";
        default:
            return "unknown";
    }
}

std::int64_t checksum_i32(const std::vector<std::int32_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::int64_t{0});
}

std::int64_t checksum_i8(const std::vector<std::int8_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::int64_t{0});
}

std::size_t mismatches_i32(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
    mismatches += actual.size() > expected.size() ? actual.size() - expected.size() : expected.size() - actual.size();
    return mismatches;
}

std::vector<std::int8_t> make_input(const Y26Conv2DParams& params, int seed) {
    std::vector<std::int8_t> input(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const int q = static_cast<int>((i * 37 + seed * 19) & 255);
        input[i] = static_cast<std::int8_t>(q - 128);
    }
    return input;
}

Y26Stage7ConvNodeConfig conv0_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    const Y26Conv2DParams& params) {
    return Y26Stage7ConvNodeConfig{fixture.conv0_node_name,
                                   params,
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
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    const Y26Conv2DParams& params) {
    return Y26Stage7ConvNodeConfig{fixture.conv1_node_name,
                                   params,
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
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    const Y26Conv2DParams& params) {
    return Y26Stage7ConvNodeConfig{fixture.conv2_node_name,
                                   params,
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

Y26Stage7BackboneSubsetConfig full_shape_config(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture,
    int activation_mode) {
    return Y26Stage7BackboneSubsetConfig{
        fixture.subset_id,
        conv0_config(fixture, Y26Conv2DParams{640, 640, 3, 16, 2, 2, 1, 1}),
        conv1_config(fixture, Y26Conv2DParams{320, 320, 16, 32, 2, 2, 1, 1}),
        conv2_config(fixture, Y26Conv2DParams{160, 160, 32, 32, 1, 1, 0, 0}),
        fixture.act0_output_scale,
        fixture.act0_output_zero_point_u8,
        fixture.act1_output_scale,
        fixture.act1_output_zero_point_u8,
        activation_mode,
    };
}

Y26ActivationRequantParams act0_params_for(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26ActivationRequantParams{static_cast<std::size_t>(320 * 320 * 16),
                                      16,
                                      fixture.images_scale,
                                      fixture.conv0_weight_scales,
                                      fixture.conv0_output_scale,
                                      fixture.conv0_output_zero_point_u8,
                                      fixture.act0_output_scale,
                                      fixture.act0_output_zero_point_u8};
}

Y26ActivationRequantParams act1_params_for(
    const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    return Y26ActivationRequantParams{static_cast<std::size_t>(160 * 160 * 32),
                                      32,
                                      fixture.act0_output_scale,
                                      fixture.conv1_weight_scales,
                                      fixture.conv1_output_scale,
                                      fixture.conv1_output_zero_point_u8,
                                      fixture.act1_output_scale,
                                      fixture.act1_output_zero_point_u8};
}

BenchResult run_mode(int mode,
                     int iterations,
                     bool use_ime,
                     const std::vector<std::int8_t>& input,
                     const std::vector<std::int32_t>& expected_output) {
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    Y26Stage7BackboneSubsetConfig cfg = full_shape_config(fixture, mode);
    Y26Stage7BackboneSubsetWorkspace ws {};
    const int prepare_status = y26_stage7_backbone_subset_prepare(&cfg, &ws);
    if (prepare_status != Y26_CONV_STATUS_SUCCESS) {
        return {0.0, {}, 0, prepare_status, expected_output.size()};
    }
    std::vector<std::int32_t> output(y26_stage7_backbone_subset_conv2_output_count(&cfg), 0);
    Y26Stage7TimingUs timing_sum {};
    int status = Y26_CONV_STATUS_SUCCESS;
    std::int64_t checksum = 0;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        Y26Stage7TimingUs timing {};
        if (use_ime) {
            status = y26_stage7_backbone_subset_run_ime_cluster0_hotpath(
                &cfg, &ws, input.data(), output.data(), &timing);
        } else {
            status = y26_stage7_backbone_subset_run_scalar(&cfg, &ws, input.data(), output.data(), &timing);
        }
        checksum += checksum_i32(output);
        timing_sum.conv0_us += timing.conv0_us;
        timing_sum.act0_requant_us += timing.act0_requant_us;
        timing_sum.conv1_us += timing.conv1_us;
        timing_sum.act1_requant_us += timing.act1_requant_us;
        timing_sum.conv2_us += timing.conv2_us;
        timing_sum.total_us += timing.total_us;
        if (status != Y26_CONV_STATUS_SUCCESS) {
            break;
        }
    }
    const auto end = Clock::now();
    const double denom = static_cast<double>(std::max(1, iterations));
    timing_sum.conv0_us /= denom;
    timing_sum.act0_requant_us /= denom;
    timing_sum.conv1_us /= denom;
    timing_sum.act1_requant_us /= denom;
    timing_sum.conv2_us /= denom;
    timing_sum.total_us /= denom;
    const std::size_t mismatches = expected_output.empty() ? 0 : mismatches_i32(output, expected_output);
    y26_stage7_backbone_subset_release(&ws);
    return {elapsed_us(begin, end) / denom, timing_sum, checksum, status, mismatches};
}

LookupBenchResult bench_lookup_scalar(const std::vector<std::uint8_t>& codes,
                                      const std::int8_t* lut,
                                      int iterations) {
    std::vector<std::int8_t> output(codes.size(), 0);
    std::int64_t checksum = 0;
    const auto begin = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (std::size_t i = 0; i < codes.size(); ++i) {
            output[i] = lut[codes[i]];
        }
        checksum += checksum_i8(output);
    }
    return {elapsed_us(begin, Clock::now()) / static_cast<double>(iterations), checksum};
}

LookupBenchResult bench_lookup_unrolled4(const std::vector<std::uint8_t>& codes,
                                         const std::int8_t* lut,
                                         int iterations) {
    std::vector<std::int8_t> output(codes.size(), 0);
    std::int64_t checksum = 0;
    const auto begin = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        std::size_t i = 0;
        for (; i + 4 <= codes.size(); i += 4) {
            output[i + 0] = lut[codes[i + 0]];
            output[i + 1] = lut[codes[i + 1]];
            output[i + 2] = lut[codes[i + 2]];
            output[i + 3] = lut[codes[i + 3]];
        }
        for (; i < codes.size(); ++i) {
            output[i] = lut[codes[i]];
        }
        checksum += checksum_i8(output);
    }
    return {elapsed_us(begin, Clock::now()) / static_cast<double>(iterations), checksum};
}

void print_mode_result(const char* label, const BenchResult& result) {
    std::cout << label << " mode=" << label << " total_us=" << result.mean_us << " status=" << result.status
              << " checksum=" << result.checksum << " mismatches=" << result.mismatches
              << " conv0_us=" << result.timing.conv0_us << " act0_requant_us=" << result.timing.act0_requant_us
              << " conv1_us=" << result.timing.conv1_us
              << " act1_requant_us=" << result.timing.act1_requant_us << " conv2_us=" << result.timing.conv2_us
              << " activation_total_us=" << (result.timing.act0_requant_us + result.timing.act1_requant_us)
              << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    const int lookup_iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3;
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    const Y26Stage7BackboneSubsetConfig baseline_cfg =
        full_shape_config(fixture, Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE);
    std::vector<std::int8_t> input = make_input(baseline_cfg.conv0.params, 31);
    std::vector<std::int32_t> empty_expected;

    const BenchResult scalar_ref =
        run_mode(Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE, iterations, false, input, empty_expected);

    std::vector<std::int32_t> baseline_output(
        static_cast<std::size_t>(baseline_cfg.conv2.params.input_h * baseline_cfg.conv2.params.input_w *
                                 baseline_cfg.conv2.params.output_c),
        0);
    {
        Y26Stage7BackboneSubsetConfig cfg = baseline_cfg;
        Y26Stage7BackboneSubsetWorkspace ws {};
        if (y26_stage7_backbone_subset_prepare(&cfg, &ws) == Y26_CONV_STATUS_SUCCESS) {
            (void)y26_stage7_backbone_subset_run_scalar(&cfg, &ws, input.data(), baseline_output.data(), nullptr);
            y26_stage7_backbone_subset_release(&ws);
        }
    }

    BenchResult ime_ref {0.0, {}, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME, 0};
    BenchResult ime_fixed {0.0, {}, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME, 0};
    BenchResult ime_lut {0.0, {}, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME, 0};
    BenchResult ime_fused_lut {0.0, {}, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME, 0};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        ime_ref = run_mode(
            Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE, iterations, true, input, baseline_output);
        ime_fixed =
            run_mode(Y26_ACTIVATION_MODE_FIXED_REQUANT_ONLY, iterations, true, input, baseline_output);
        ime_lut = run_mode(Y26_ACTIVATION_MODE_INT8_LUT, iterations, true, input, baseline_output);
        ime_fused_lut = run_mode(Y26_ACTIVATION_MODE_FUSED_LUT_PACK, iterations, true, input, baseline_output);
    }

    Y26Stage7BackboneSubsetConfig profile_cfg =
        full_shape_config(fixture, Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE);
    Y26Stage7BackboneSubsetWorkspace profile_ws {};
    Y26ActivationSubbucketTimingUs act0_profile {};
    Y26ActivationSubbucketTimingUs act1_profile {};
    if (y26_stage7_backbone_subset_prepare(&profile_cfg, &profile_ws) == Y26_CONV_STATUS_SUCCESS) {
        std::vector<std::int32_t> profile_output(y26_stage7_backbone_subset_conv2_output_count(&profile_cfg), 0);
        if (y26_vmadot_4x4x8_ime_available_buildtime()) {
            (void)y26_stage7_backbone_subset_run_ime_cluster0_hotpath(
                &profile_cfg, &profile_ws, input.data(), profile_output.data(), nullptr);
        } else {
            (void)y26_stage7_backbone_subset_run_scalar(
                &profile_cfg, &profile_ws, input.data(), profile_output.data(), nullptr);
        }

        std::vector<std::uint8_t> act0_conv_code(act0_params_for(fixture).element_count, 0);
        std::vector<float> act0_dq(act0_conv_code.size(), 0.0f);
        std::vector<float> act0_silu(act0_conv_code.size(), 0.0f);
        std::vector<std::uint8_t> act0_q(act0_conv_code.size(), 0);
        std::vector<std::int8_t> act0_out(act0_conv_code.size(), 0);
        const Y26ActivationRequantParams act0_params = act0_params_for(fixture);
        (void)y26_activation_requant_silu_profile_scalar_float(&act0_params,
                                                               y26_stage7_backbone_subset_conv0_i32(&profile_ws),
                                                               act0_conv_code.data(),
                                                               act0_dq.data(),
                                                               act0_silu.data(),
                                                               act0_q.data(),
                                                               act0_out.data(),
                                                               &act0_profile);

        std::vector<std::uint8_t> act1_conv_code(act1_params_for(fixture).element_count, 0);
        std::vector<float> act1_dq(act1_conv_code.size(), 0.0f);
        std::vector<float> act1_silu(act1_conv_code.size(), 0.0f);
        std::vector<std::uint8_t> act1_q(act1_conv_code.size(), 0);
        std::vector<std::int8_t> act1_out(act1_conv_code.size(), 0);
        const Y26ActivationRequantParams act1_params = act1_params_for(fixture);
        (void)y26_activation_requant_silu_profile_scalar_float(&act1_params,
                                                               y26_stage7_backbone_subset_conv1_i32(&profile_ws),
                                                               act1_conv_code.data(),
                                                               act1_dq.data(),
                                                               act1_silu.data(),
                                                               act1_q.data(),
                                                               act1_out.data(),
                                                               &act1_profile);

        std::int8_t act0_lut[256] {};
        (void)y26_build_silu_u8_to_s8_lut(fixture.conv0_output_scale,
                                          fixture.conv0_output_zero_point_u8,
                                          fixture.act0_output_scale,
                                          fixture.act0_output_zero_point_u8,
                                          act0_lut);
        const auto lookup_scalar = bench_lookup_scalar(act0_conv_code, act0_lut, lookup_iterations);
        const auto lookup_unrolled = bench_lookup_unrolled4(act0_conv_code, act0_lut, lookup_iterations);

        std::cout << "STAGE8_ACTIVATION_REQUANT_BENCH_BEGIN\n";
        std::cout << "note=selected-subset activation/requant microbenchmark only, not YOLO26 inference FPS\n";
        std::cout << "subset=" << fixture.subset_id << " iterations=" << iterations
                  << " lookup_iterations=" << lookup_iterations << "\n";
        print_mode_result(activation_mode_name(Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE), scalar_ref);
        print_mode_result("ime_scalar_float_reference", ime_ref);
        print_mode_result("ime_fixed_requant_only", ime_fixed);
        print_mode_result("ime_int8_lut", ime_lut);
        print_mode_result("ime_fused_lut_pack", ime_fused_lut);
        std::cout << "act0_subbucket corr_i32_to_conv_out_quant_code_us="
                  << act0_profile.corr_i32_to_conv_out_quant_code_us
                  << " conv_out_code_to_float_dequant_us=" << act0_profile.conv_out_code_to_float_dequant_us
                  << " float_silu_sigmoid_mul_us=" << act0_profile.float_silu_sigmoid_mul_us
                  << " act_quant_float_to_uint8_us=" << act0_profile.act_quant_float_to_uint8_us
                  << " signed_storage_shift_us=" << act0_profile.signed_storage_shift_us
                  << " layout_or_pack_handoff_us=" << act0_profile.layout_or_pack_handoff_us
                  << " combined_current_fallback_us=" << act0_profile.combined_current_fallback_us << "\n";
        std::cout << "act1_subbucket corr_i32_to_conv_out_quant_code_us="
                  << act1_profile.corr_i32_to_conv_out_quant_code_us
                  << " conv_out_code_to_float_dequant_us=" << act1_profile.conv_out_code_to_float_dequant_us
                  << " float_silu_sigmoid_mul_us=" << act1_profile.float_silu_sigmoid_mul_us
                  << " act_quant_float_to_uint8_us=" << act1_profile.act_quant_float_to_uint8_us
                  << " signed_storage_shift_us=" << act1_profile.signed_storage_shift_us
                  << " layout_or_pack_handoff_us=" << act1_profile.layout_or_pack_handoff_us
                  << " combined_current_fallback_us=" << act1_profile.combined_current_fallback_us << "\n";
        std::cout << "lookup_mechanism scalar_table_us=" << lookup_scalar.mean_us
                  << " scalar_checksum=" << lookup_scalar.checksum
                  << " unrolled4_us=" << lookup_unrolled.mean_us
                  << " unrolled4_checksum=" << lookup_unrolled.checksum
                  << " rvv_vluxei_status=not-implemented rvv_vrgather_status=not-implemented\n";
        if (ime_ref.status == Y26_CONV_STATUS_SUCCESS && ime_lut.status == Y26_CONV_STATUS_SUCCESS) {
            const double before_activation = ime_ref.timing.act0_requant_us + ime_ref.timing.act1_requant_us;
            const double after_activation = ime_lut.timing.act0_requant_us + ime_lut.timing.act1_requant_us;
            std::cout << "stage8_delta activation_before_us=" << before_activation
                      << " activation_after_us=" << after_activation
                      << " total_before_us=" << ime_ref.mean_us << " total_after_us=" << ime_lut.mean_us
                      << " activation_share_after_pct=" << (100.0 * after_activation / ime_lut.mean_us)
                      << " speedup_vs_stage7_ime=" << (ime_ref.mean_us / ime_lut.mean_us)
                      << " speedup_vs_scalar_total=" << (scalar_ref.mean_us / ime_lut.mean_us) << "\n";
        }
        std::cout << "STAGE8_ACTIVATION_REQUANT_BENCH_END\n";
        y26_stage7_backbone_subset_release(&profile_ws);
    }

    return scalar_ref.status == Y26_CONV_STATUS_SUCCESS && ime_lut.status != Y26_CONV_STATUS_INVALID_ARGUMENT ? 0 : 1;
}
