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

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

const char* activation_mode_name(int mode) {
    switch (mode) {
        case Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE:
            return "scalar_float_reference";
        case Y26_ACTIVATION_MODE_INT8_LUT:
            return "A0_int8_lut";
        case Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT:
            return "A1_scalar_unrolled_lut";
        case Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT:
            return "A2_rvv_f32_lut";
        case Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT:
            return "A3_fixed_requant_lut";
        case Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT:
            return "A4_fused_current_layout";
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
    return mismatches + (actual.size() > expected.size() ? actual.size() - expected.size()
                                                         : expected.size() - actual.size());
}

std::size_t mismatches_i8(const std::vector<std::int8_t>& actual, const std::int8_t* expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        mismatches += actual[i] != expected[i] ? 1U : 0U;
    }
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

void print_mode_result(const char* label, const BenchResult& result) {
    const double activation_total = result.timing.act0_requant_us + result.timing.act1_requant_us;
    std::cout << label << " mode=" << label << " total_us=" << result.mean_us << " status=" << result.status
              << " checksum=" << result.checksum << " mismatches=" << result.mismatches
              << " conv0_us=" << result.timing.conv0_us << " act0_requant_us=" << result.timing.act0_requant_us
              << " conv1_us=" << result.timing.conv1_us
              << " act1_requant_us=" << result.timing.act1_requant_us << " conv2_us=" << result.timing.conv2_us
              << " activation_total_us=" << activation_total
              << " activation_share_pct=" << (result.mean_us > 0.0 ? 100.0 * activation_total / result.mean_us : 0.0)
              << "\n";
}

void print_stage9_profile(const char* label,
                          const Y26Stage9ActivationTimingUs& act0,
                          const Y26Stage9ActivationTimingUs& act1,
                          std::size_t act0_mismatches,
                          std::size_t act1_mismatches) {
    std::cout << label << " act0_requant_arithmetic_us=" << act0.requant_arithmetic_us
              << " act0_lut_lookup_store_us=" << act0.lut_lookup_us
              << " act0_packa_handoff_us=" << act0.packa_handoff_us << " act0_total_us=" << act0.total_us
              << " act0_mismatches=" << act0_mismatches
              << " act1_requant_arithmetic_us=" << act1.requant_arithmetic_us
              << " act1_lut_lookup_store_us=" << act1.lut_lookup_us
              << " act1_packa_handoff_us=" << act1.packa_handoff_us << " act1_total_us=" << act1.total_us
              << " act1_mismatches=" << act1_mismatches
              << " activation_total_us=" << (act0.total_us + act1.total_us) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1;
    const int profile_iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3;
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    const Y26Stage7BackboneSubsetConfig baseline_cfg =
        full_shape_config(fixture, Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE);
    const std::vector<std::int8_t> input = make_input(baseline_cfg.conv0.params, 31);
    const std::vector<std::int32_t> empty_expected;
    const BenchResult scalar_ref =
        run_mode(Y26_ACTIVATION_MODE_SCALAR_FLOAT_REFERENCE, iterations, false, input, empty_expected);

    std::vector<std::int32_t> baseline_output(
        static_cast<std::size_t>(baseline_cfg.conv2.params.input_h * baseline_cfg.conv2.params.input_w *
                                 baseline_cfg.conv2.params.output_c),
        0);
    {
        Y26Stage7BackboneSubsetWorkspace ws {};
        if (y26_stage7_backbone_subset_prepare(&baseline_cfg, &ws) == Y26_CONV_STATUS_SUCCESS) {
            (void)y26_stage7_backbone_subset_run_scalar(
                &baseline_cfg, &ws, input.data(), baseline_output.data(), nullptr);
            y26_stage7_backbone_subset_release(&ws);
        }
    }

    BenchResult modes[5] {};
    const int mode_ids[5] = {Y26_ACTIVATION_MODE_INT8_LUT,
                             Y26_ACTIVATION_MODE_STAGE9_SCALAR_UNROLLED_LUT,
                             Y26_ACTIVATION_MODE_STAGE9_RVV_F32_LUT,
                             Y26_ACTIVATION_MODE_STAGE9_FIXED_REQUANT_LUT,
                             Y26_ACTIVATION_MODE_STAGE9_FUSED_CURRENT_LAYOUT};
    if (y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)y26_k1x_ime_probe_once();
        for (int i = 0; i < 5; ++i) {
            modes[i] = run_mode(mode_ids[i], iterations, true, input, baseline_output);
        }
    } else {
        for (auto& item : modes) {
            item = {0.0, {}, 0, Y26_CONV_STATUS_NOT_BUILT_WITH_IME, 0};
        }
    }

    Y26Stage7BackboneSubsetWorkspace profile_ws {};
    Y26Stage7BackboneSubsetConfig profile_cfg =
        full_shape_config(fixture, Y26_ACTIVATION_MODE_INT8_LUT);
    std::vector<std::uint8_t> act0_code(act0_params_for(fixture).element_count, 0);
    std::vector<std::uint8_t> act1_code(act1_params_for(fixture).element_count, 0);
    std::vector<std::int8_t> act0_out(act0_code.size(), 0);
    std::vector<std::int8_t> act1_out(act1_code.size(), 0);
    std::vector<std::int8_t> act0_out_fixed(act0_code.size(), 0);
    std::vector<std::int8_t> act1_out_fixed(act1_code.size(), 0);
    std::vector<std::int8_t> act0_ref(act0_code.size(), 0);
    std::vector<std::int8_t> act1_ref(act1_code.size(), 0);
    Y26Stage9ActivationTimingUs act0_scalar_profile {};
    Y26Stage9ActivationTimingUs act1_scalar_profile {};
    Y26Stage9ActivationTimingUs act0_fixed_profile {};
    Y26Stage9ActivationTimingUs act1_fixed_profile {};
    double packa_us = 0.0;
    std::int64_t packa_checksum = 0;
    std::size_t packa_mismatches = 0;

    if (y26_stage7_backbone_subset_prepare(&profile_cfg, &profile_ws) == Y26_CONV_STATUS_SUCCESS) {
        std::vector<std::int32_t> profile_output(y26_stage7_backbone_subset_conv2_output_count(&profile_cfg), 0);
        if (y26_vmadot_4x4x8_ime_available_buildtime()) {
            (void)y26_stage7_backbone_subset_run_ime_cluster0_hotpath(
                &profile_cfg, &profile_ws, input.data(), profile_output.data(), nullptr);
        } else {
            (void)y26_stage7_backbone_subset_run_scalar(
                &profile_cfg, &profile_ws, input.data(), profile_output.data(), nullptr);
        }
        const Y26ActivationRequantParams act0_params = act0_params_for(fixture);
        const Y26ActivationRequantParams act1_params = act1_params_for(fixture);
        (void)y26_activation_requant_silu_int8_lut(
            &act0_params, profile_ws.conv0_i32, profile_ws.act0_lut_s8, act0_ref.data());
        (void)y26_activation_requant_silu_int8_lut(
            &act1_params, profile_ws.conv1_i32, profile_ws.act1_lut_s8, act1_ref.data());
        for (int iter = 0; iter < profile_iterations; ++iter) {
            Y26Stage9ActivationTimingUs t0 {};
            Y26Stage9ActivationTimingUs t1 {};
            (void)y26_activation_requant_silu_int8_lut_scalar_unrolled_profile(&act0_params,
                                                                               profile_ws.conv0_i32,
                                                                               profile_ws.act0_lut_s8,
                                                                               act0_code.data(),
                                                                               act0_out.data(),
                                                                               &t0);
            (void)y26_activation_requant_silu_int8_lut_scalar_unrolled_profile(&act1_params,
                                                                               profile_ws.conv1_i32,
                                                                               profile_ws.act1_lut_s8,
                                                                               act1_code.data(),
                                                                               act1_out.data(),
                                                                               &t1);
            act0_scalar_profile.requant_arithmetic_us += t0.requant_arithmetic_us;
            act0_scalar_profile.lut_lookup_us += t0.lut_lookup_us;
            act0_scalar_profile.store_write_us += t0.store_write_us;
            act0_scalar_profile.total_us += t0.total_us;
            act1_scalar_profile.requant_arithmetic_us += t1.requant_arithmetic_us;
            act1_scalar_profile.lut_lookup_us += t1.lut_lookup_us;
            act1_scalar_profile.store_write_us += t1.store_write_us;
            act1_scalar_profile.total_us += t1.total_us;

            Y26Stage9ActivationTimingUs f0 {};
            Y26Stage9ActivationTimingUs f1 {};
            (void)y26_activation_requant_silu_int8_lut_fixed_requant_profile(&act0_params,
                                                                             profile_ws.conv0_fixed_requant,
                                                                             profile_ws.conv0_i32,
                                                                             profile_ws.act0_lut_s8,
                                                                             act0_code.data(),
                                                                             act0_out_fixed.data(),
                                                                             &f0);
            (void)y26_activation_requant_silu_int8_lut_fixed_requant_profile(&act1_params,
                                                                             profile_ws.conv1_fixed_requant,
                                                                             profile_ws.conv1_i32,
                                                                             profile_ws.act1_lut_s8,
                                                                             act1_code.data(),
                                                                             act1_out_fixed.data(),
                                                                             &f1);
            act0_fixed_profile.requant_arithmetic_us += f0.requant_arithmetic_us;
            act0_fixed_profile.lut_lookup_us += f0.lut_lookup_us;
            act0_fixed_profile.store_write_us += f0.store_write_us;
            act0_fixed_profile.total_us += f0.total_us;
            act1_fixed_profile.requant_arithmetic_us += f1.requant_arithmetic_us;
            act1_fixed_profile.lut_lookup_us += f1.lut_lookup_us;
            act1_fixed_profile.store_write_us += f1.store_write_us;
            act1_fixed_profile.total_us += f1.total_us;
        }
        const double denom = static_cast<double>(profile_iterations);
        act0_scalar_profile.requant_arithmetic_us /= denom;
        act0_scalar_profile.lut_lookup_us /= denom;
        act0_scalar_profile.store_write_us /= denom;
        act0_scalar_profile.total_us /= denom;
        act1_scalar_profile.requant_arithmetic_us /= denom;
        act1_scalar_profile.lut_lookup_us /= denom;
        act1_scalar_profile.store_write_us /= denom;
        act1_scalar_profile.total_us /= denom;
        act0_fixed_profile.requant_arithmetic_us /= denom;
        act0_fixed_profile.lut_lookup_us /= denom;
        act0_fixed_profile.store_write_us /= denom;
        act0_fixed_profile.total_us /= denom;
        act1_fixed_profile.requant_arithmetic_us /= denom;
        act1_fixed_profile.lut_lookup_us /= denom;
        act1_fixed_profile.store_write_us /= denom;
        act1_fixed_profile.total_us /= denom;

        const int h = profile_cfg.conv2.params.input_h;
        const int w = profile_cfg.conv2.params.input_w;
        const int c = profile_cfg.conv2.params.input_c;
        const int output_m = h * w;
        const int k_padded = ((c + 7) / 8) * 8;
        const std::size_t packed_bytes =
            static_cast<std::size_t>((output_m + 3) / 4) * static_cast<std::size_t>(k_padded / 8) * 32U;
        std::vector<std::int8_t> packed(packed_bytes, 0);
        std::vector<std::int8_t> unpacked(profile_ws.conv2_input_count, 0);
        const auto pack_begin = Clock::now();
        (void)y26_activation_packa_1x1_mmt4d_4x8_from_nhwc(
            profile_ws.conv2_input_s8, h, w, c, packed.data(), packed.size());
        const auto pack_end = Clock::now();
        (void)y26_activation_unpacka_1x1_mmt4d_4x8_to_nhwc(packed.data(), h, w, c, unpacked.data());
        packa_us = elapsed_us(pack_begin, pack_end);
        packa_checksum = checksum_i8(packed);
        packa_mismatches = mismatches_i8(unpacked, profile_ws.conv2_input_s8);
        y26_stage7_backbone_subset_release(&profile_ws);
    }

    std::cout << "STAGE9_ACTIVATION_FUSION_BENCH_BEGIN\n";
    std::cout << "note=selected-subset activation/requant microbenchmark only, not YOLO26 inference FPS\n";
    std::cout << "subset=" << fixture.subset_id << " iterations=" << iterations
              << " profile_iterations=" << profile_iterations << "\n";
    print_mode_result("scalar_float_reference", scalar_ref);
    for (int i = 0; i < 5; ++i) {
        print_mode_result(activation_mode_name(mode_ids[i]), modes[i]);
    }
    print_stage9_profile("A1_scalar_unrolled_profile",
                         act0_scalar_profile,
                         act1_scalar_profile,
                         mismatches_i8(act0_out, act0_ref.data()),
                         mismatches_i8(act1_out, act1_ref.data()));
    print_stage9_profile("A3_fixed_requant_profile",
                         act0_fixed_profile,
                         act1_fixed_profile,
                         mismatches_i8(act0_out_fixed, act0_ref.data()),
                         mismatches_i8(act1_out_fixed, act1_ref.data()));
    std::cout << "A5_packa_handoff conv2_1x1_packa_us=" << packa_us
              << " packa_checksum=" << packa_checksum << " packa_unpack_mismatches=" << packa_mismatches
              << " status=sidecar-not-integrated\n";
    std::cout << "STAGE9_ACTIVATION_FUSION_BENCH_END\n";
    return scalar_ref.status == Y26_CONV_STATUS_SUCCESS && modes[0].status != Y26_CONV_STATUS_INVALID_ARGUMENT ? 0 : 1;
}
