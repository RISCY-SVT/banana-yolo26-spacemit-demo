#include "y26_k1x_int8_v1.h"
#include "y26_k1x_stage51_q62.h"
#include "y26_k1x_stage61_attention_ntail.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<int, 52> kWidths {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    63, 64, 65, 99, 100, 101, 120, 121, 122, 143, 144, 145,
    168, 169, 170, 195, 196, 197, 255, 256, 257,
};

constexpr std::array<int, 13> kInnerSizes {
    1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
};

std::uint64_t mix(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint8_t code_for(std::uint64_t tag, int first, int second, int mode) noexcept {
    if (mode == 1) return ((first + second) & 1) == 0 ? 0U : 255U;
    if (mode == 2) return (first % 3) == 0 ? 0U : ((second % 3) == 0 ? 255U : 128U);
    return static_cast<std::uint8_t>(mix(tag ^ (static_cast<std::uint64_t>(first) << 32U) ^
                                             static_cast<std::uint32_t>(second)) & 0xffU);
}

std::uint64_t fnv1a(const std::vector<std::uint8_t>& values) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint8_t value : values) {
        hash ^= value;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::size_t raw_index(int row, int column) noexcept {
    return static_cast<std::size_t>((column / 4) * 3 + row / 4) * 16U +
           static_cast<std::size_t>(row % 4) * 4U + column % 4;
}

const char* strategy_name(y26::stage61::Ntail13Strategy strategy) noexcept {
    return strategy == y26::stage61::Ntail13Strategy::n8_n8 ? "n8n8" : "n16";
}

struct CaseResult {
    std::uint64_t hash = 0;
    int kernel_calls = 0;
    int padded_k_lanes = 0;
    int padded_n_columns = 0;
};

CaseResult run_case(int n, int k, int live_rows, int mode,
                    y26::stage61::Ntail13Strategy strategy) {
    constexpr int kRows = 12;
    constexpr int kColumns = 16;
    constexpr int kLanes = 8;
    constexpr std::int32_t kCanary = 0x13579bdf;
    constexpr std::size_t kGuard = 16;
    const int left_zero_point = (n * 37 + k * 11 + mode * 19) & 255;
    const int right_zero_point = (n * 13 + k * 29 + mode * 7) & 255;
    const int output_zero_point = (n * 17 + k * 5 + mode * 31) & 255;
    const std::int64_t multiplier = (INT64_C(1) << 54) +
        static_cast<std::int64_t>((n * 131 + k * 17 + mode) & 0xffff);
    const int k_tiles = (k + kLanes - 1) / kLanes;
    const int padded_k = k_tiles * kLanes;
    const int n_blocks = (n + kColumns - 1) / kColumns;
    const std::int8_t neutral_left = y26::int8_v1::signed_storage(
        static_cast<std::uint8_t>(left_zero_point));
    const std::int8_t neutral_right = y26::int8_v1::signed_storage(
        static_cast<std::uint8_t>(right_zero_point));

    std::vector<std::int8_t> packed_a(
        static_cast<std::size_t>(k_tiles) * kRows * kLanes, neutral_left);
    std::vector<std::int8_t> packed_b(
        static_cast<std::size_t>(n_blocks) * k_tiles * kColumns * kLanes,
        neutral_right);
    for (int inner = 0; inner < k; ++inner) {
        const int tile = inner / kLanes;
        const int lane = inner % kLanes;
        for (int row = 0; row < live_rows; ++row) {
            const auto semantic = code_for(0x413131ULL + n, row, inner, mode);
            packed_a[(static_cast<std::size_t>(tile) * kRows + row) * kLanes + lane] =
                y26::int8_v1::signed_storage(semantic);
        }
        for (int column = 0; column < n; ++column) {
            const int block = column / kColumns;
            const int block_column = column % kColumns;
            const auto semantic = code_for(0x423232ULL + n, inner, column, mode);
            packed_b[(((static_cast<std::size_t>(block) * k_tiles + tile) * kColumns +
                        block_column) * kLanes) + lane] =
                y26::int8_v1::signed_storage(semantic);
        }
    }

    std::vector<std::uint8_t> result(static_cast<std::size_t>(live_rows) * n);
    std::array<std::int32_t, kGuard + kRows * kColumns + kGuard> guarded {};
    std::fill(guarded.begin(), guarded.end(), kCanary);
    std::int32_t* raw = guarded.data() + kGuard;
    y26::stage51::VectorFixedPointState vector_state;
    if (!y26::stage51::begin_q62_vector_rne(&vector_state)) {
        throw std::runtime_error("cannot establish Q62 vector state");
    }

    CaseResult case_result;
    case_result.padded_k_lanes = padded_k - k;
    for (int block = 0; block < n_blocks; ++block) {
        std::fill(guarded.begin(), guarded.end(), kCanary);
        const int column_begin = block * kColumns;
        const int live_columns = std::min(kColumns, n - column_begin);
        const auto route = y26::stage61::ntail_route_count(live_columns, strategy);
        case_result.kernel_calls += route.n4 + route.n8 + route.n16;
        case_result.padded_n_columns += route.padded_dead_columns;
        const auto* panel = packed_b.data() +
            static_cast<std::size_t>(block) * k_tiles * kColumns * kLanes;
        if (!y26::stage61::run_m12n_tail(
                packed_a.data(), panel, k_tiles, live_columns, strategy,
                raw, kRows * kColumns)) {
            throw std::runtime_error("valid N-tail call rejected");
        }
        const std::size_t written =
            static_cast<std::size_t>((live_columns + 3) / 4) * 3U * 16U;
        if (!std::all_of(guarded.begin(), guarded.begin() + kGuard,
                         [](std::int32_t value) { return value == kCanary; }) ||
            !std::all_of(guarded.begin() + kGuard + written, guarded.end(),
                         [](std::int32_t value) { return value == kCanary; })) {
            throw std::runtime_error("N-tail kernel changed a red zone");
        }

        for (int row = 0; row < live_rows; ++row) {
            std::int64_t row_sum = 0;
            for (int inner = 0; inner < padded_k; ++inner) {
                const int tile = inner / kLanes;
                const int lane = inner % kLanes;
                row_sum += packed_a[(static_cast<std::size_t>(tile) * kRows + row) *
                                    kLanes + lane];
            }
            for (int group = 0; group < (live_columns + 3) / 4; ++group) {
                alignas(32) std::array<std::int64_t, 4> corrected {};
                alignas(32) std::array<std::int64_t, 4> multipliers {
                    multiplier * 2, multiplier * 2, multiplier * 2, multiplier * 2,
                };
                alignas(32) std::array<std::int64_t, 4> rounded {};
                const int group_live = std::min(4, live_columns - group * 4);
                for (int lane = 0; lane < group_live; ++lane) {
                    const int local_column = group * 4 + lane;
                    std::int64_t column_sum = 0;
                    std::int64_t semantic_accumulator = 0;
                    for (int inner = 0; inner < padded_k; ++inner) {
                        const int tile = inner / kLanes;
                        const int inner_lane = inner % kLanes;
                        const auto a = packed_a[
                            (static_cast<std::size_t>(tile) * kRows + row) * kLanes +
                            inner_lane];
                        const auto b = panel[
                            (static_cast<std::size_t>(tile) * kColumns + local_column) *
                            kLanes + inner_lane];
                        column_sum += b;
                        if (inner < k) {
                            semantic_accumulator +=
                                (static_cast<int>(a) + 128 - left_zero_point) *
                                (static_cast<int>(b) + 128 - right_zero_point);
                        }
                    }
                    corrected[static_cast<std::size_t>(lane)] =
                        static_cast<std::int64_t>(raw[raw_index(row, local_column)]) +
                        static_cast<std::int64_t>(128 - right_zero_point) * row_sum +
                        static_cast<std::int64_t>(128 - left_zero_point) * column_sum +
                        static_cast<std::int64_t>(padded_k) *
                            (128 - left_zero_point) * (128 - right_zero_point);
                    if (corrected[static_cast<std::size_t>(lane)] != semantic_accumulator) {
                        throw std::runtime_error("zero-point corrected accumulator mismatch");
                    }
                }
                y26::stage51::q62_vsmul_m63_i64x4(
                    corrected.data(), multipliers.data(), rounded.data());
                for (int lane = 0; lane < group_live; ++lane) {
                    const int column = column_begin + group * 4 + lane;
                    std::uint8_t scalar = 0;
                    const y26::int8_v1::RequantAsset asset {
                        multiplier, 62, output_zero_point, 0, 255,
                    };
                    if (!y26::int8_v1::requantize_u8(
                            corrected[static_cast<std::size_t>(lane)], asset, &scalar)) {
                        throw std::runtime_error("scalar requantization failed");
                    }
                    const auto vectorized = static_cast<std::uint8_t>(
                        std::clamp<std::int64_t>(rounded[static_cast<std::size_t>(lane)] +
                            output_zero_point, 0, 255));
                    if (scalar != vectorized) {
                        throw std::runtime_error("Q62 vector epilogue mismatch");
                    }
                    result[static_cast<std::size_t>(row) * n + column] = vectorized;
                }
            }
        }
    }
    const auto vector_result = y26::stage51::end_q62_vector_rne(&vector_state);
    if (!vector_result.restored || vector_result.saturated) {
        throw std::runtime_error("Q62 vector state was not cleanly restored");
    }
    case_result.hash = fnv1a(result);
    return case_result;
}

void invalid_argument_tests() {
    alignas(32) std::array<std::int8_t, 12 * 8> a {};
    alignas(32) std::array<std::int8_t, 16 * 8> b {};
    alignas(32) std::array<std::int32_t, 12 * 16> c {};
    using y26::stage61::Ntail13Strategy;
    const auto strategy = Ntail13Strategy::n8_n8;
    if (y26::stage61::run_m12n_tail(a.data(), b.data(), 1, 0, strategy,
                                     c.data(), c.size()) ||
        y26::stage61::run_m12n_tail(a.data(), b.data(), 1, 17, strategy,
                                     c.data(), c.size()) ||
        y26::stage61::run_m12n_tail(a.data(), b.data(), 0, 4, strategy,
                                     c.data(), c.size()) ||
        y26::stage61::run_m12n_tail(a.data(), b.data(), 1, 8, strategy,
                                     c.data(), 95) ||
        y26::stage61::run_m12n_tail(a.data(), b.data(), 1, 4, strategy,
                                     reinterpret_cast<std::int32_t*>(a.data()), c.size())) {
        throw std::runtime_error("invalid or aliased N-tail call was accepted");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool emit_tsv = argc == 2 && std::string_view(argv[1]) == "--tsv";
        if (argc > 2 || (argc == 2 && !emit_tsv)) return 2;
        invalid_argument_tests();
        if (emit_tsv) {
            std::cout << "n\tk\tm\tmode\tstrategy\toutput_hash\tkernel_calls\t"
                         "padded_k_lanes\tpadded_n_columns\tstatus\n";
        }
        int cases = 0;
        for (std::size_t index = 0; index < kWidths.size(); ++index) {
            const int n = kWidths[index];
            const int k = kInnerSizes[index % kInnerSizes.size()];
            const int m = 1 + static_cast<int>(index % 12U);
            for (int mode = 0; mode < 3; ++mode) {
                for (const auto strategy : {
                         y26::stage61::Ntail13Strategy::n8_n8,
                         y26::stage61::Ntail13Strategy::padded_n16}) {
                    const auto result = run_case(n, k, m, mode, strategy);
                    ++cases;
                    if (emit_tsv) {
                        std::cout << n << '\t' << k << '\t' << m << '\t' << mode << '\t'
                                  << strategy_name(strategy) << "\t0x" << std::hex
                                  << std::setw(16) << std::setfill('0') << result.hash << std::dec
                                  << '\t' << result.kernel_calls << '\t'
                                  << result.padded_k_lanes << '\t'
                                  << result.padded_n_columns << "\tpass\n";
                    }
                }
            }
        }
        std::cout << "stage61_attention_ntail_cases=" << cases << "\n";
        std::cout << "stage61_attention_ntail_status=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
