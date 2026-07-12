#include "y26_k1x_stage48_nchwc8.h"

#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_int8_v1.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

template <typename T>
void write_binary(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) throw std::runtime_error("test package write failed");
}

std::filesystem::path make_zero_package() {
#if defined(__linux__)
    const long process = static_cast<long>(getpid());
#else
    const long process = 0;
#endif
    const auto root = std::filesystem::temp_directory_path() /
        ("y26_stage48_test_" + std::to_string(process));
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    std::ofstream meta(root / "model5_meta.tsv");
    meta << "contract_id\tprofile_id\tlayout_id\tinput_h\tinput_w\tinput_c\toutput_h\toutput_w\toutput_c"
            "\tkernel_h\tkernel_w\tstride_h\tstride_w\tpad_h\tpad_w\tinput_zero_point"
            "\tconv_output_zero_point\toutput_zero_point\tk\tk_tiles\tn_blocks"
            "\taccumulator_absolute_bound\tint32_safe\n";
    meta << y26::int8_v1::kContractId << '\t' << y26::int8_v1::kGeneralProfile << '\t'
         << y26::int8_v1::kNchwc8LayoutId
         << "\t80\t80\t128\t40\t40\t128\t3\t3\t2\t2\t1\t1\t9\t136\t10\t1152\t144\t8\t1000000\t1\n";
    meta.close();
    write_binary(root / "weights_packed_n16k8_s8.bin", std::vector<std::int8_t>(8U * 144U * 16U * 8U));
    write_binary(root / "weight_sums_i32.bin", std::vector<std::int32_t>(128));
    write_binary(root / "bias_i32.bin", std::vector<std::int32_t>(128));
    write_binary(root / "requant_multiplier_i64.bin", std::vector<std::int64_t>(128, 1));
    write_binary(root / "requant_right_shift_i32.bin", std::vector<std::int32_t>(128, 0));
    std::vector<std::int8_t> lut(256);
    for (int code = 0; code < 256; ++code) lut[static_cast<std::size_t>(code)] = y26::int8_v1::signed_storage(code);
    write_binary(root / "silu_lut_s8.bin", lut);
    return root;
}

int test_layout_roundtrip() {
    constexpr int n = 1;
    constexpr int c = 16;
    constexpr int h = 3;
    constexpr int w = 5;
    std::vector<std::uint8_t> input(n * c * h * w);
    std::vector<std::int8_t> blocked(input.size());
    std::vector<std::uint8_t> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>((index * 17U + 9U) & 255U);
    }
    y26::stage48::nchw_u8_to_nchwc8_s8(input.data(), blocked.data(), n, c, h, w);
    y26::stage48::nchwc8_s8_to_nchw_u8(blocked.data(), output.data(), n, c, h, w);
    return input == output ? 0 : 1;
}

int test_direct_pack(const std::filesystem::path& package) {
    y26::stage48::Model5DirectConv conv;
    if (conv.prepare(package, 1) != Y26_CONV_STATUS_SUCCESS) return 1;
    std::vector<std::uint8_t> input_nchw(conv.input_bytes());
    for (std::size_t index = 0; index < input_nchw.size(); ++index) {
        input_nchw[index] = static_cast<std::uint8_t>((index * 29U + 7U) & 255U);
    }
    std::vector<std::int8_t> blocked(input_nchw.size());
    y26::stage48::nchw_u8_to_nchwc8_s8(input_nchw.data(), blocked.data(), 1, 128, 80, 80);
    for (y26::stage48::MBlock block : {y26::stage48::MBlock::m4, y26::stage48::MBlock::m8, y26::stage48::MBlock::m12}) {
        const int rows = static_cast<int>(block);
        std::vector<std::int8_t> reference(static_cast<std::size_t>(rows) * 1152);
        std::vector<std::int8_t> actual(reference.size());
        if (conv.debug_pack_a(blocked.data(), 4, block, y26::stage48::LoadStrategy::c8_u64,
                              reference.data(), reference.size()) != Y26_CONV_STATUS_SUCCESS) return 2;
        for (y26::stage48::LoadStrategy strategy : {
                 y26::stage48::LoadStrategy::c8_u64,
                 y26::stage48::LoadStrategy::rvv_vlse64,
                 y26::stage48::LoadStrategy::rvv_vlseg2e64}) {
            if (conv.debug_pack_a(blocked.data(), 4, block, strategy, actual.data(), actual.size()) !=
                    Y26_CONV_STATUS_SUCCESS ||
                actual != reference) return 3;
        }
    }
    return 0;
}

int test_scalar_workers(const std::filesystem::path& package) {
    y26::stage48::Model5DirectConv conv;
    if (conv.prepare(package, 4) != Y26_CONV_STATUS_SUCCESS) return 1;
    std::vector<std::int8_t> input(conv.input_bytes(), y26::int8_v1::signed_storage(9));
    std::vector<std::int8_t> output(conv.output_bytes());
    const std::int8_t expected = y26::int8_v1::signed_storage(136);
    for (int workers = 1; workers <= 4; ++workers) {
        for (y26::stage48::MBlock block : {
                 y26::stage48::MBlock::m4, y26::stage48::MBlock::m8, y26::stage48::MBlock::m12}) {
            for (y26::stage48::PartitionPolicy partition : {
                     y26::stage48::PartitionPolicy::spatial,
                     y26::stage48::PartitionPolicy::output_channel}) {
                std::fill(output.begin(), output.end(), std::int8_t{-1});
                y26::stage48::RunOptions options;
                options.route = y26::stage48::ComputeRoute::scalar;
                options.m_block = block;
                options.load_strategy = y26::stage48::LoadStrategy::c8_u64;
                options.partition = partition;
                options.workers = workers;
                if (conv.run(input.data(), output.data(), options, nullptr) != Y26_CONV_STATUS_SUCCESS) return 2;
                for (std::int8_t value : output) {
                    if (value != expected) return 3;
                }
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    const auto package = make_zero_package();
    const int layout = test_layout_roundtrip();
    const int pack = test_direct_pack(package);
    const int workers = test_scalar_workers(package);
    std::filesystem::remove_all(package);
    std::cout << "layout=" << layout << '\n';
    std::cout << "pack=" << pack << '\n';
    std::cout << "workers=" << workers << '\n';
    return layout == 0 && pack == 0 && workers == 0 ? 0 : 1;
}
