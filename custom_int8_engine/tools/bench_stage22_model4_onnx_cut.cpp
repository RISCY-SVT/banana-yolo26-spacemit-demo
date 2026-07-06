#define Y26_STAGE16_NO_TEST_MAIN 1
#include "../tests/test_stage16_model4_c2f_runner.cpp"

#include "y26_k1x_threaded_conv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kFullH = 80;
constexpr int kFullW = 80;
constexpr int kModel4Cv1C = 64;
constexpr int kSplitC = 32;

struct Protocol {
    int warmup = 10;
    int runs = 100;
    int repeats = 5;
};

struct Timing {
    double conv_us = 0.0;
    double activation_requant_us = 0.0;
    double split_us = 0.0;
    double merge_us = 0.0;
    double post_qdq_us = 0.0;
    double thread_overhead_us = 0.0;
    double correction_us = 0.0;
    double total_us = 0.0;
    double branch0_conv_us = 0.0;
    double branch1_conv_us = 0.0;
    double model4_cv2_conv_us = 0.0;
};

struct Summary {
    double mean_total_us = 0.0;
    double stddev_total_us = 0.0;
    double min_total_us = 0.0;
    double max_total_us = 0.0;
    double cv_total_pct = 0.0;
    double mean_conv_us = 0.0;
    double mean_activation_requant_us = 0.0;
    double mean_merge_us = 0.0;
    double mean_thread_overhead_us = 0.0;
    double mean_correction_us = 0.0;
    double mean_branch0_conv_us = 0.0;
    double mean_branch1_conv_us = 0.0;
    double mean_model4_cv2_conv_us = 0.0;
    std::size_t mismatches = 0;
    int max_abs_diff = 0;
    unsigned long long checksum = 0;
    int status = Y26_CONV_STATUS_SUCCESS;
    int affinity_ok = 1;
};

struct Options {
    std::string fixture_dir;
    std::string mode = "scalar";
    Protocol protocol {};
    bool frm_sweep = false;
    std::string dump_actual;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) / 1000.0;
}

#if defined(__riscv)
unsigned stage22_read_frm() {
    unsigned frm = 0;
    asm volatile("frrm %0" : "=r"(frm));
    return frm & 7U;
}

void stage22_set_frm(unsigned frm) {
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

class ScopedRiscvRne {
public:
    ScopedRiscvRne() : saved_(stage22_read_frm()) {
        stage22_set_frm(0);
    }
    ~ScopedRiscvRne() {
        stage22_set_frm(saved_);
    }

private:
    unsigned saved_;
};
#else
class ScopedRiscvRne {
public:
    ScopedRiscvRne() = default;
};
#endif

int output_h_for_kernel(const Y26Conv2DParams& params, int kernel_h) {
    return kernel_h == 1 ? y26_conv1x1_output_h(&params) : y26_conv3x3_output_h(&params);
}

int output_w_for_kernel(const Y26Conv2DParams& params, int kernel_w) {
    return kernel_w == 1 ? y26_conv1x1_output_w(&params) : y26_conv3x3_output_w(&params);
}

std::size_t output_count_for_kernel(const Y26Stage7ConvNodeConfig& cfg) {
    return static_cast<std::size_t>(output_h_for_kernel(cfg.params, cfg.kernel_h)) *
           static_cast<std::size_t>(output_w_for_kernel(cfg.params, cfg.kernel_w)) *
           static_cast<std::size_t>(cfg.params.output_c);
}

std::int8_t signed_storage_from_u8(std::uint8_t value) {
    return static_cast<std::int8_t>(static_cast<int>(value) - 128);
}

std::int8_t weight_at(const Y26Stage7ConvNodeConfig& cfg, int oc, int kh, int kw, int ic) {
    const int index = ((oc * cfg.kernel_h + kh) * cfg.kernel_w + kw) * cfg.params.input_c + ic;
    return cfg.weights_ohwi_s8[index];
}

int scalar_raw_dot(const Y26Stage7ConvNodeConfig& cfg,
                   const std::int8_t* input_nhwc_s8,
                   std::int32_t* raw_i32_nhwc) {
    const int output_h = output_h_for_kernel(cfg.params, cfg.kernel_h);
    const int output_w = output_w_for_kernel(cfg.params, cfg.kernel_w);
    const std::int8_t pad = static_cast<std::int8_t>(cfg.input_storage_zero_point_s8);
    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < cfg.params.output_c; ++oc) {
                std::int32_t acc = 0;
                for (int kh = 0; kh < cfg.kernel_h; ++kh) {
                    const int ih = oh * cfg.params.stride_h + kh - cfg.params.pad_h;
                    const bool valid_h = ih >= 0 && ih < cfg.params.input_h;
                    for (int kw = 0; kw < cfg.kernel_w; ++kw) {
                        const int iw = ow * cfg.params.stride_w + kw - cfg.params.pad_w;
                        const bool inside = valid_h && iw >= 0 && iw < cfg.params.input_w;
                        const std::int8_t* src =
                            inside ? input_nhwc_s8 + (ih * cfg.params.input_w + iw) * cfg.params.input_c : nullptr;
                        for (int ic = 0; ic < cfg.params.input_c; ++ic) {
                            const std::int8_t a = inside ? src[ic] : pad;
                            acc += static_cast<std::int32_t>(a) *
                                   static_cast<std::int32_t>(weight_at(cfg, oc, kh, kw, ic));
                        }
                    }
                }
                raw_i32_nhwc[(oh * output_w + ow) * cfg.params.output_c + oc] = acc;
            }
        }
    }
    return Y26_CONV_STATUS_SUCCESS;
}

int apply_correction(const Y26Stage7ConvNodeConfig& cfg,
                     const Y26PrepackedConvWeights* weights,
                     const std::int32_t* raw_i32,
                     std::int32_t* corrected_i32) {
    return y26_conv2d_apply_u8_as_s8_correction_nhwc(raw_i32,
                                                     cfg.bias_i32,
                                                     y26_prepacked_conv_weights_sums(weights),
                                                     corrected_i32,
                                                     output_h_for_kernel(cfg.params, cfg.kernel_h) *
                                                         output_w_for_kernel(cfg.params, cfg.kernel_w),
                                                     cfg.params.output_c,
                                                     cfg.activation_zero_point_u8);
}

int run_conv_scalar(const Y26Stage7ConvNodeConfig& cfg,
                    const Y26PrepackedConvWeights* weights,
                    const std::int8_t* input_s8,
                    std::vector<std::int32_t>& raw,
                    std::vector<std::int32_t>& output,
                    double& conv_us,
                    double& correction_us) {
    const auto begin = Clock::now();
    int status = scalar_raw_dot(cfg, input_s8, raw.data());
    const auto correction_begin = Clock::now();
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg, weights, raw.data(), output.data());
    }
    const auto end = Clock::now();
    conv_us = elapsed_us(begin, end);
    correction_us = elapsed_us(correction_begin, end);
    return status;
}

int run_conv_ime(const Y26Stage7ConvNodeConfig& cfg,
                 const Y26PrepackedConvWeights* weights,
                 Y26ConvWorkspace* workspace,
                 const std::int8_t* input_s8,
                 std::vector<std::int32_t>& raw,
                 std::vector<std::int32_t>& output,
                 double& conv_us,
                 double& correction_us) {
    const auto begin = Clock::now();
    int status = y26_conv2d_i8s8s32_nhwc_ime_prepacked_v1(input_s8,
                                                          weights,
                                                          raw.data(),
                                                          cfg.input_storage_zero_point_s8,
                                                          workspace,
                                                          Y26_CONV_LOOP_ORDER_M_MAJOR);
    const auto correction_begin = Clock::now();
    if (status == Y26_CONV_STATUS_SUCCESS) {
        status = apply_correction(cfg, weights, raw.data(), output.data());
    }
    const auto end = Clock::now();
    conv_us = elapsed_us(begin, end);
    correction_us = elapsed_us(correction_begin, end);
    return status;
}

Y26ActivationRequantParams activation_params(const Y26Stage7ConvNodeConfig& producer,
                                             std::size_t output_count,
                                             float act_output_scale,
                                             int act_output_zero_point_u8) {
    return Y26ActivationRequantParams{output_count,
                                      producer.params.output_c,
                                      producer.input_scale,
                                      producer.weight_scales,
                                      producer.output_scale,
                                      producer.output_zero_point_u8,
                                      act_output_scale,
                                      act_output_zero_point_u8};
}

float silu_f32(float value) {
    return value / (1.0f + std::exp(-value));
}

float dequant_signed_storage(std::int8_t value, float scale, int zero_point_u8) {
    const int code = static_cast<int>(value) + 128;
    return static_cast<float>(code - zero_point_u8) * scale;
}

std::int8_t quantize_s8(float value, float scale, int zero_point_u8) {
    const std::uint8_t code = y26_quantize_u8_nearest_even_f32(value, scale, zero_point_u8);
    return signed_storage_from_u8(code);
}

std::uint8_t accumulator_to_conv_code(const Y26Stage7ConvNodeConfig& cfg, std::int32_t acc, int channel) {
    const float acc_scale = cfg.input_scale * cfg.weight_scales[channel];
    const float conv_float = static_cast<float>(acc) * acc_scale;
    return y26_quantize_u8_nearest_even_f32(conv_float, cfg.output_scale, cfg.output_zero_point_u8);
}

float accumulator_to_silu_float(const Y26Stage7ConvNodeConfig& cfg, std::int32_t acc, int channel) {
    const std::uint8_t code = accumulator_to_conv_code(cfg, acc, channel);
    const float x = (static_cast<int>(code) - cfg.output_zero_point_u8) * cfg.output_scale;
    return silu_f32(x);
}

Y26Stage7ConvNodeConfig fullshape_branch0(
    const y26_stage15_model4_branch_fixture::Model4BranchFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg = stage15_branch0_config_from_fixture(fixture);
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

Y26Stage7ConvNodeConfig fullshape_branch1(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg = branch1_config_from_fixture(fixture);
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

Y26Stage7ConvNodeConfig fullshape_model4_cv2(
    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture) {
    Y26Stage7ConvNodeConfig cfg = model4_cv2_config_from_fixture(fixture);
    cfg.params.input_h = kFullH;
    cfg.params.input_w = kFullW;
    return cfg;
}

std::vector<std::uint8_t> read_u8_file(const std::string& path, std::size_t expected_count) {
    std::vector<std::uint8_t> values(expected_count);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size()));
    if (in.gcount() != static_cast<std::streamsize>(values.size())) {
        return {};
    }
    char extra = 0;
    if (in.read(&extra, 1)) {
        return {};
    }
    return values;
}

bool write_u8_file(const std::string& path, const std::vector<std::uint8_t>& values) {
    if (path.empty()) {
        return true;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()));
    return static_cast<bool>(out);
}

unsigned long long checksum_u8(const std::vector<std::uint8_t>& values) {
    unsigned long long sum = 0;
    for (std::uint8_t value : values) {
        sum += value;
    }
    return sum;
}

void compare_u8(const std::vector<std::uint8_t>& actual,
                const std::vector<std::uint8_t>& expected,
                std::size_t& mismatches,
                int& max_abs_diff) {
    mismatches = 0;
    max_abs_diff = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const int diff = std::abs(static_cast<int>(actual[i]) - static_cast<int>(expected[i]));
        if (diff != 0) {
            ++mismatches;
            max_abs_diff = std::max(max_abs_diff, diff);
        }
    }
}

struct Context {
    explicit Context(const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture_in)
        : fixture(fixture_in),
          branch0(fullshape_branch0(*fixture_in.stage15_fixture)),
          branch1(fullshape_branch1(fixture_in)),
          model4_cv2(fullshape_model4_cv2(fixture_in)),
          model4_cv1_q_u8(static_cast<std::size_t>(kFullH) * kFullW * kModel4Cv1C),
          split1_s8(static_cast<std::size_t>(kFullH) * kFullW * kSplitC),
          split0_concat_s8(static_cast<std::size_t>(kFullH) * kFullW * kSplitC),
          branch0_raw(output_count_for_kernel(branch0)),
          branch0_i32(output_count_for_kernel(branch0)),
          branch0_act_s8(output_count_for_kernel(branch0)),
          branch1_raw(output_count_for_kernel(branch1)),
          branch1_i32(output_count_for_kernel(branch1)),
          branch1_act_f32(output_count_for_kernel(branch1)),
          concat_s8(static_cast<std::size_t>(kFullH) * kFullW * kSplitC * 3),
          model4_cv2_raw(output_count_for_kernel(model4_cv2)),
          output_i32(output_count_for_kernel(model4_cv2)),
          output_q_u8(output_count_for_kernel(model4_cv2)),
          expected_q_u8(output_count_for_kernel(model4_cv2)) {}

    const y26_stage16_model4_c2f_fixture::Model4C2fFixture& fixture;
    Y26Stage7ConvNodeConfig branch0;
    Y26Stage7ConvNodeConfig branch1;
    Y26Stage7ConvNodeConfig model4_cv2;
    Y26PrepackedConvWeights* branch0_weights = nullptr;
    Y26PrepackedConvWeights* branch1_weights = nullptr;
    Y26PrepackedConvWeights* model4_cv2_weights = nullptr;
    Y26ConvWorkspace* branch0_workspace = nullptr;
    Y26ConvWorkspace* branch1_workspace = nullptr;
    Y26ConvWorkspace* model4_cv2_workspace = nullptr;
    Y26ThreadedConvWorkspace* branch0_threaded = nullptr;
    std::vector<std::uint8_t> model4_cv1_q_u8;
    std::vector<std::int8_t> split1_s8;
    std::vector<std::int8_t> split0_concat_s8;
    std::vector<std::int32_t> branch0_raw;
    std::vector<std::int32_t> branch0_i32;
    std::vector<std::int8_t> branch0_act_s8;
    std::vector<std::int32_t> branch1_raw;
    std::vector<std::int32_t> branch1_i32;
    std::vector<float> branch1_act_f32;
    std::vector<std::int8_t> concat_s8;
    std::vector<std::int32_t> model4_cv2_raw;
    std::vector<std::int32_t> output_i32;
    std::vector<std::uint8_t> output_q_u8;
    std::vector<std::uint8_t> expected_q_u8;
    std::int8_t split_lut[256] {};
    std::int8_t split0_concat_lut[256] {};
    std::int8_t branch0_lut[256] {};
};

bool prepare(Context& ctx, int thread_count) {
    ctx.branch0_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        ctx.branch0.weights_ohwi_s8, &ctx.branch0.params, ctx.branch0.kernel_h, ctx.branch0.kernel_w,
        ctx.branch0.node_name, ctx.branch0.weight_scales);
    ctx.branch1_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        ctx.branch1.weights_ohwi_s8, &ctx.branch1.params, ctx.branch1.kernel_h, ctx.branch1.kernel_w,
        ctx.branch1.node_name, ctx.branch1.weight_scales);
    ctx.model4_cv2_weights = y26_prepacked_conv_weights_create_mmt4d_s8(
        ctx.model4_cv2.weights_ohwi_s8, &ctx.model4_cv2.params, ctx.model4_cv2.kernel_h, ctx.model4_cv2.kernel_w,
        ctx.model4_cv2.node_name, ctx.model4_cv2.weight_scales);
    ctx.branch0_workspace = y26_conv_workspace_create(&ctx.branch0.params, ctx.branch0.kernel_h, ctx.branch0.kernel_w);
    ctx.branch1_workspace = y26_conv_workspace_create(&ctx.branch1.params, ctx.branch1.kernel_h, ctx.branch1.kernel_w);
    ctx.model4_cv2_workspace =
        y26_conv_workspace_create(&ctx.model4_cv2.params, ctx.model4_cv2.kernel_h, ctx.model4_cv2.kernel_w);
    if (thread_count > 0) {
        ctx.branch0_threaded = y26_threaded_conv_create_spatial_rows(&ctx.branch0, thread_count);
    }
    y26_build_silu_u8_to_s8_lut(ctx.branch0.output_scale,
                                ctx.branch0.output_zero_point_u8,
                                ctx.fixture.stage15_fixture->branch0_act_output_scale,
                                ctx.fixture.stage15_fixture->branch0_act_output_zero_point_u8,
                                ctx.branch0_lut);
    y26_build_silu_u8_to_s8_lut(ctx.fixture.stage15_fixture->stage14_fixture->model4_cv1_output_scale,
                                ctx.fixture.stage15_fixture->stage14_fixture->model4_cv1_output_zero_point_u8,
                                ctx.fixture.stage15_fixture->split1_output_scale,
                                ctx.fixture.stage15_fixture->split1_output_zero_point_u8,
                                ctx.split_lut);
    y26_build_silu_u8_to_s8_lut(ctx.fixture.stage15_fixture->stage14_fixture->model4_cv1_output_scale,
                                ctx.fixture.stage15_fixture->stage14_fixture->model4_cv1_output_zero_point_u8,
                                ctx.fixture.concat_output_scale,
                                ctx.fixture.concat_output_zero_point_u8,
                                ctx.split0_concat_lut);
    return ctx.branch0_weights != nullptr && ctx.branch1_weights != nullptr && ctx.model4_cv2_weights != nullptr &&
           ctx.branch0_workspace != nullptr && ctx.branch1_workspace != nullptr && ctx.model4_cv2_workspace != nullptr &&
           (thread_count == 0 || ctx.branch0_threaded != nullptr);
}

void destroy(Context& ctx) {
    y26_threaded_conv_destroy(ctx.branch0_threaded);
    y26_conv_workspace_destroy(ctx.model4_cv2_workspace);
    y26_conv_workspace_destroy(ctx.branch1_workspace);
    y26_conv_workspace_destroy(ctx.branch0_workspace);
    y26_prepacked_conv_weights_destroy(ctx.model4_cv2_weights);
    y26_prepacked_conv_weights_destroy(ctx.branch1_weights);
    y26_prepacked_conv_weights_destroy(ctx.branch0_weights);
}

void build_split_inputs(Context& ctx) {
    for (int m = 0; m < kFullH * kFullW; ++m) {
        for (int c = 0; c < kSplitC; ++c) {
            const std::uint8_t split0_code = ctx.model4_cv1_q_u8[static_cast<std::size_t>(m) * kModel4Cv1C + c];
            const std::uint8_t split1_code =
                ctx.model4_cv1_q_u8[static_cast<std::size_t>(m) * kModel4Cv1C + kSplitC + c];
            ctx.split0_concat_s8[static_cast<std::size_t>(m) * kSplitC + c] = ctx.split0_concat_lut[split0_code];
            ctx.split1_s8[static_cast<std::size_t>(m) * kSplitC + c] = ctx.split_lut[split1_code];
        }
    }
}

void build_branch1_activation_float(Context& ctx) {
    const int channels = ctx.branch1.params.output_c;
    for (std::size_t i = 0; i < ctx.branch1_i32.size(); ++i) {
        const int c = static_cast<int>(i % static_cast<std::size_t>(channels));
        ctx.branch1_act_f32[i] = accumulator_to_silu_float(ctx.branch1, ctx.branch1_i32[i], c);
    }
}

void build_concat_qdq(Context& ctx) {
    for (int m = 0; m < kFullH * kFullW; ++m) {
        std::int8_t* dst = ctx.concat_s8.data() + static_cast<std::size_t>(m) * kSplitC * 3;
        for (int c = 0; c < kSplitC; ++c) {
            dst[c] = ctx.split0_concat_s8[static_cast<std::size_t>(m) * kSplitC + c];
            const std::int8_t split1_s8 = ctx.split1_s8[static_cast<std::size_t>(m) * kSplitC + c];
            const float split1_value = dequant_signed_storage(split1_s8,
                                                             ctx.fixture.stage15_fixture->split1_output_scale,
                                                             ctx.fixture.stage15_fixture->split1_output_zero_point_u8);
            dst[kSplitC + c] =
                quantize_s8(split1_value, ctx.fixture.concat_output_scale, ctx.fixture.concat_output_zero_point_u8);
            const float add_value = split1_value + ctx.branch1_act_f32[static_cast<std::size_t>(m) * kSplitC + c];
            dst[kSplitC * 2 + c] =
                quantize_s8(add_value, ctx.fixture.concat_output_scale, ctx.fixture.concat_output_zero_point_u8);
        }
    }
}

void quantize_model4_cv2_output(Context& ctx) {
    const int channels = ctx.model4_cv2.params.output_c;
    for (std::size_t i = 0; i < ctx.output_i32.size(); ++i) {
        const int c = static_cast<int>(i % static_cast<std::size_t>(channels));
        ctx.output_q_u8[i] = accumulator_to_conv_code(ctx.model4_cv2, ctx.output_i32[i], c);
    }
}

int run_once(Context& ctx, bool use_ime, bool use_threaded_branch0, Timing& timing) {
    [[maybe_unused]] ScopedRiscvRne rne_guard;
    timing = Timing {};
    const auto total_begin = Clock::now();
    const auto split_begin = Clock::now();
    build_split_inputs(ctx);
    const auto split_end = Clock::now();

    double branch0_conv_us = 0.0;
    double branch0_correction_us = 0.0;
    int status = Y26_CONV_STATUS_SUCCESS;
    if (use_threaded_branch0) {
        Y26ThreadedConvTimingUs threaded {};
        status =
            y26_threaded_conv_run_ime_cluster0(ctx.branch0_threaded, ctx.split1_s8.data(), ctx.branch0_i32.data(), &threaded);
        branch0_conv_us = threaded.total_us;
        branch0_correction_us = threaded.correction_us;
        timing.thread_overhead_us += std::max(0.0, threaded.total_us - threaded.worker_max_us);
    } else if (use_ime) {
        status = run_conv_ime(ctx.branch0,
                              ctx.branch0_weights,
                              ctx.branch0_workspace,
                              ctx.split1_s8.data(),
                              ctx.branch0_raw,
                              ctx.branch0_i32,
                              branch0_conv_us,
                              branch0_correction_us);
    } else {
        status = run_conv_scalar(ctx.branch0,
                                 ctx.branch0_weights,
                                 ctx.split1_s8.data(),
                                 ctx.branch0_raw,
                                 ctx.branch0_i32,
                                 branch0_conv_us,
                                 branch0_correction_us);
    }
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto act_begin = Clock::now();
    const Y26ActivationRequantParams branch0_params =
        activation_params(ctx.branch0,
                          ctx.branch0_i32.size(),
                          ctx.fixture.stage15_fixture->branch0_act_output_scale,
                          ctx.fixture.stage15_fixture->branch0_act_output_zero_point_u8);
    status = use_ime ? y26_activation_requant_silu_int8_lut_rvv_f32(
                           &branch0_params, ctx.branch0_i32.data(), ctx.branch0_lut, ctx.branch0_act_s8.data())
                     : y26_activation_requant_silu_int8_lut(
                           &branch0_params, ctx.branch0_i32.data(), ctx.branch0_lut, ctx.branch0_act_s8.data());
    const auto act_end = Clock::now();
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }

    double branch1_conv_us = 0.0;
    double branch1_correction_us = 0.0;
    status = use_ime ? run_conv_ime(ctx.branch1,
                                    ctx.branch1_weights,
                                    ctx.branch1_workspace,
                                    ctx.branch0_act_s8.data(),
                                    ctx.branch1_raw,
                                    ctx.branch1_i32,
                                    branch1_conv_us,
                                    branch1_correction_us)
                     : run_conv_scalar(ctx.branch1,
                                       ctx.branch1_weights,
                                       ctx.branch0_act_s8.data(),
                                       ctx.branch1_raw,
                                       ctx.branch1_i32,
                                       branch1_conv_us,
                                       branch1_correction_us);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    const auto branch1_act_begin = Clock::now();
    build_branch1_activation_float(ctx);
    const auto branch1_act_end = Clock::now();

    const auto merge_begin = Clock::now();
    build_concat_qdq(ctx);
    const auto merge_end = Clock::now();

    double model4_cv2_conv_us = 0.0;
    double model4_cv2_correction_us = 0.0;
    status = use_ime ? run_conv_ime(ctx.model4_cv2,
                                    ctx.model4_cv2_weights,
                                    ctx.model4_cv2_workspace,
                                    ctx.concat_s8.data(),
                                    ctx.model4_cv2_raw,
                                    ctx.output_i32,
                                    model4_cv2_conv_us,
                                    model4_cv2_correction_us)
                     : run_conv_scalar(ctx.model4_cv2,
                                       ctx.model4_cv2_weights,
                                       ctx.concat_s8.data(),
                                       ctx.model4_cv2_raw,
                                       ctx.output_i32,
                                       model4_cv2_conv_us,
                                       model4_cv2_correction_us);
    if (status != Y26_CONV_STATUS_SUCCESS) {
        return status;
    }
    quantize_model4_cv2_output(ctx);
    const auto total_end = Clock::now();

    timing.split_us = elapsed_us(split_begin, split_end);
    timing.conv_us = branch0_conv_us + branch1_conv_us + model4_cv2_conv_us;
    timing.branch0_conv_us = branch0_conv_us;
    timing.branch1_conv_us = branch1_conv_us;
    timing.model4_cv2_conv_us = model4_cv2_conv_us;
    timing.correction_us = branch0_correction_us + branch1_correction_us + model4_cv2_correction_us;
    timing.activation_requant_us = elapsed_us(act_begin, act_end) + elapsed_us(branch1_act_begin, branch1_act_end);
    timing.merge_us = elapsed_us(merge_begin, merge_end);
    timing.post_qdq_us = timing.merge_us;
    timing.total_us = elapsed_us(total_begin, total_end);
    return status;
}

Summary summarize(const std::vector<Timing>& repeats,
                  std::size_t mismatches,
                  int max_abs_diff,
                  unsigned long long checksum,
                  int status,
                  int affinity_ok) {
    Summary out {};
    out.mismatches = mismatches;
    out.max_abs_diff = max_abs_diff;
    out.checksum = checksum;
    out.status = status;
    out.affinity_ok = affinity_ok;
    if (repeats.empty()) {
        return out;
    }
    std::vector<double> totals;
    totals.reserve(repeats.size());
    for (const Timing& timing : repeats) {
        totals.push_back(timing.total_us);
        out.mean_conv_us += timing.conv_us;
        out.mean_activation_requant_us += timing.activation_requant_us;
        out.mean_merge_us += timing.merge_us;
        out.mean_thread_overhead_us += timing.thread_overhead_us;
        out.mean_correction_us += timing.correction_us;
        out.mean_branch0_conv_us += timing.branch0_conv_us;
        out.mean_branch1_conv_us += timing.branch1_conv_us;
        out.mean_model4_cv2_conv_us += timing.model4_cv2_conv_us;
    }
    const double denom = static_cast<double>(repeats.size());
    out.mean_total_us = std::accumulate(totals.begin(), totals.end(), 0.0) / denom;
    out.mean_conv_us /= denom;
    out.mean_activation_requant_us /= denom;
    out.mean_merge_us /= denom;
    out.mean_thread_overhead_us /= denom;
    out.mean_correction_us /= denom;
    out.mean_branch0_conv_us /= denom;
    out.mean_branch1_conv_us /= denom;
    out.mean_model4_cv2_conv_us /= denom;
    out.min_total_us = *std::min_element(totals.begin(), totals.end());
    out.max_total_us = *std::max_element(totals.begin(), totals.end());
    double variance = 0.0;
    for (double value : totals) {
        const double diff = value - out.mean_total_us;
        variance += diff * diff;
    }
    out.stddev_total_us = repeats.size() > 1 ? std::sqrt(variance / static_cast<double>(repeats.size() - 1)) : 0.0;
    out.cv_total_pct = out.mean_total_us > 0.0 ? 100.0 * out.stddev_total_us / out.mean_total_us : 0.0;
    return out;
}

int run_candidate(Context& ctx, const Options& options, Summary& summary) {
    const bool use_ime = options.mode == "ime" || options.mode == "ime_threaded";
    const bool use_threaded = options.mode == "ime_threaded";
    if (use_ime) {
        (void)y26_k1x_ime_probe_once();
    }
    Timing ignored {};
    for (int i = 0; i < options.protocol.warmup; ++i) {
        const int status = run_once(ctx, use_ime, use_threaded, ignored);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            summary.status = status;
            return status;
        }
    }
    std::vector<Timing> repeats;
    repeats.reserve(static_cast<std::size_t>(options.protocol.repeats));
    int status = Y26_CONV_STATUS_SUCCESS;
    for (int repeat = 0; repeat < options.protocol.repeats; ++repeat) {
        Timing sum {};
        for (int run = 0; run < options.protocol.runs; ++run) {
            Timing timing {};
            status = run_once(ctx, use_ime, use_threaded, timing);
            if (status != Y26_CONV_STATUS_SUCCESS) {
                break;
            }
            sum.conv_us += timing.conv_us;
            sum.activation_requant_us += timing.activation_requant_us;
            sum.split_us += timing.split_us;
            sum.merge_us += timing.merge_us;
            sum.post_qdq_us += timing.post_qdq_us;
            sum.thread_overhead_us += timing.thread_overhead_us;
            sum.correction_us += timing.correction_us;
            sum.total_us += timing.total_us;
            sum.branch0_conv_us += timing.branch0_conv_us;
            sum.branch1_conv_us += timing.branch1_conv_us;
            sum.model4_cv2_conv_us += timing.model4_cv2_conv_us;
        }
        const double denom = static_cast<double>(options.protocol.runs);
        sum.conv_us /= denom;
        sum.activation_requant_us /= denom;
        sum.split_us /= denom;
        sum.merge_us /= denom;
        sum.post_qdq_us /= denom;
        sum.thread_overhead_us /= denom;
        sum.correction_us /= denom;
        sum.total_us /= denom;
        sum.branch0_conv_us /= denom;
        sum.branch1_conv_us /= denom;
        sum.model4_cv2_conv_us /= denom;
        repeats.push_back(sum);
        if (status != Y26_CONV_STATUS_SUCCESS) {
            break;
        }
    }
    std::size_t mismatches = 0;
    int max_abs_diff = 0;
    compare_u8(ctx.output_q_u8, ctx.expected_q_u8, mismatches, max_abs_diff);
    const int affinity_ok = !use_threaded || ctx.branch0_threaded == nullptr
                                ? 1
                                : y26_threaded_conv_worker_affinity_ok(ctx.branch0_threaded);
    summary = summarize(repeats, mismatches, max_abs_diff, checksum_u8(ctx.output_q_u8), status, affinity_ok);
    return status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 && affinity_ok == 1 ? 0 : 1;
}

#if defined(__riscv)
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

int run_frm_sweep(Context& ctx, const Options& options) {
#if defined(__riscv)
    const unsigned saved = read_frm();
    int failures = 0;
    for (unsigned frm : {0U, 1U, 2U, 3U, 4U}) {
        set_frm(frm);
        Summary summary {};
        Options one = options;
        one.protocol = Protocol{0, 1, 1};
        const int status = run_candidate(ctx, one, summary);
        const unsigned after = read_frm();
        std::cout << "stage22_frm ambient_frm=" << frm
                  << " status=" << status
                  << " mismatches=" << summary.mismatches
                  << " max_abs_diff=" << summary.max_abs_diff
                  << " after_frm=" << after
                  << " checksum=" << summary.checksum
                  << "\n";
        failures += status == 0 && after == frm ? 0 : 1;
    }
    set_frm(saved);
    return failures == 0 ? 0 : 1;
#else
    (void)ctx;
    (void)options;
    std::cout << "stage22_frm skipped_non_riscv\n";
    return 0;
#endif
}

Options parse_options(int argc, char** argv) {
    Options options {};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--fixture-dir") {
            options.fixture_dir = require_value("--fixture-dir");
        } else if (arg == "--mode") {
            options.mode = require_value("--mode");
        } else if (arg == "--warmup") {
            options.protocol.warmup = std::max(0, std::atoi(require_value("--warmup").c_str()));
        } else if (arg == "--runs") {
            options.protocol.runs = std::max(1, std::atoi(require_value("--runs").c_str()));
        } else if (arg == "--repeats") {
            options.protocol.repeats = std::max(1, std::atoi(require_value("--repeats").c_str()));
        } else if (arg == "--dump-actual") {
            options.dump_actual = require_value("--dump-actual");
        } else if (arg == "--frm-sweep") {
            options.frm_sweep = true;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    if (options.fixture_dir.empty()) {
        std::cerr << "usage: bench_stage22_model4_onnx_cut --fixture-dir <dir> [--mode scalar|ime|ime_threaded]\n";
        return 2;
    }
    const bool use_threaded = options.mode == "ime_threaded";
    const int thread_count = use_threaded ? 4 : 0;
    Context ctx(y26_stage16_model4_c2f_fixture::kSyntheticSeededFixture);
    if (!prepare(ctx, thread_count)) {
        std::cerr << "prepare failed\n";
        return 1;
    }
    ctx.model4_cv1_q_u8 = read_u8_file(options.fixture_dir + "/model4_cv1_conv_q_u8_nhwc.bin",
                                       static_cast<std::size_t>(kFullH) * kFullW * kModel4Cv1C);
    ctx.expected_q_u8 = read_u8_file(options.fixture_dir + "/model4_cv2_conv_q_u8_expected_nhwc.bin",
                                    output_count_for_kernel(ctx.model4_cv2));
    if (ctx.model4_cv1_q_u8.empty() || ctx.expected_q_u8.empty()) {
        std::cerr << "failed to read Stage22 cut fixture files from " << options.fixture_dir << "\n";
        destroy(ctx);
        return 1;
    }

    Summary summary {};
    const int status = run_candidate(ctx, options, summary);
    const bool dump_ok = write_u8_file(options.dump_actual, ctx.output_q_u8);
    std::cout << "stage22_cut_compare"
              << " mode=" << options.mode
              << " warmup=" << options.protocol.warmup
              << " runs=" << options.protocol.runs
              << " repeats=" << options.protocol.repeats
              << " status=" << summary.status
              << " mismatches=" << summary.mismatches
              << " max_abs_diff=" << summary.max_abs_diff
              << " checksum=" << summary.checksum
              << " expected_checksum=" << checksum_u8(ctx.expected_q_u8)
              << " affinity_ok=" << summary.affinity_ok
              << " mean_total_us=" << summary.mean_total_us
              << " stddev_total_us=" << summary.stddev_total_us
              << " min_total_us=" << summary.min_total_us
              << " max_total_us=" << summary.max_total_us
              << " cv_total_pct=" << summary.cv_total_pct
              << " mean_conv_us=" << summary.mean_conv_us
              << " mean_activation_requant_us=" << summary.mean_activation_requant_us
              << " mean_merge_us=" << summary.mean_merge_us
              << " mean_thread_overhead_us=" << summary.mean_thread_overhead_us
              << " mean_correction_us=" << summary.mean_correction_us
              << " mean_branch0_conv_us=" << summary.mean_branch0_conv_us
              << " mean_branch1_conv_us=" << summary.mean_branch1_conv_us
              << " mean_model4_cv2_conv_us=" << summary.mean_model4_cv2_conv_us
              << " conv_share_pct="
              << (summary.mean_total_us > 0.0 ? 100.0 * summary.mean_conv_us / summary.mean_total_us : 0.0)
              << " activation_share_pct="
              << (summary.mean_total_us > 0.0 ? 100.0 * summary.mean_activation_requant_us / summary.mean_total_us : 0.0)
              << " merge_share_pct="
              << (summary.mean_total_us > 0.0 ? 100.0 * summary.mean_merge_us / summary.mean_total_us : 0.0)
              << " dump_actual_ok=" << (dump_ok ? 1 : 0)
              << " note=selected-subset-not-model-fps"
              << "\n";
    int failures = status == 0 && dump_ok ? 0 : 1;
    if (options.frm_sweep) {
        failures += run_frm_sweep(ctx, options);
    }
    destroy(ctx);
    return failures == 0 ? 0 : 1;
}
