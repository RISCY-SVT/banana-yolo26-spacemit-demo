#include "y26_k1x_int8_v1.h"
#include "y26_k1x_package.h"

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

using y26::int8_v1::sha256_file;

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary);
    stream << text;
    if (!stream) throw std::runtime_error("test text write failed");
}

template <typename T>
void write_values(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) throw std::runtime_error("test binary write failed");
}

std::filesystem::path make_package(const std::filesystem::path& root) {
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    write_text(root / "package.json",
               "{\n"
               "  \"byte_order\": \"little-endian\",\n"
               "  \"contract_id\": \"K1X_INT8_V1\",\n"
               "  \"layout_id\": \"NCHWc8_SPATIAL_INNER_V1\",\n"
               "  \"model_sha256\": \"30a94e4738606673b5e0a73499cbc977167f046f8fa8637d6040ce744f429c0c\",\n"
               "  \"profile_id\": \"K1X_INT8_V1_GENERAL\",\n"
               "  \"schema_version\": 2,\n"
               "  \"source_lineage_id\": \"stage49-test\"\n"
               "}\n");
    write_text(root / "model5_meta.tsv", "field\tvalue\ninput_zero_point\t9\n");
    write_values(root / "weights_packed_n16k8_s8.bin", std::vector<std::int8_t>(128, 1));
    write_values(root / "weight_sums_i32.bin", std::vector<std::int32_t>(16, 8));
    write_values(root / "bias_i32.bin", std::vector<std::int32_t>(16, 3));
    write_values(root / "requant_multiplier_i64.bin", std::vector<std::int64_t>(16, 1LL << 61));
    write_values(root / "requant_right_shift_i32.bin", std::vector<std::int32_t>(16, 62));
    write_values(root / "silu_lut_s8.bin", std::vector<std::int8_t>(256));

    constexpr std::array<const char*, 8> files {{
        "bias_i32.bin", "model5_meta.tsv", "package.json", "requant_multiplier_i64.bin",
        "requant_right_shift_i32.bin", "silu_lut_s8.bin", "weight_sums_i32.bin",
        "weights_packed_n16k8_s8.bin",
    }};
    std::ofstream manifest(root / "asset_hashes.tsv", std::ios::binary);
    manifest << "path\tbytes\tsha256\n";
    for (const char* name : files) {
        const auto path = root / name;
        manifest << name << '\t' << std::filesystem::file_size(path) << '\t' << sha256_file(path) << '\n';
    }
    if (!manifest) throw std::runtime_error("test manifest write failed");
    return root;
}

void flip_first_byte(const std::filesystem::path& path) {
    std::fstream stream(path, std::ios::in | std::ios::out | std::ios::binary);
    char value = 0;
    stream.read(&value, 1);
    value ^= 0x01;
    stream.seekp(0);
    stream.write(&value, 1);
    if (!stream) throw std::runtime_error("test corruption failed");
}

bool verify(const std::filesystem::path& root, const std::string& trusted) {
    return y26::int8_v1::verify_package(root, trusted,
                                        y26::int8_v1::kContractId,
                                        y26::int8_v1::kGeneralProfile,
                                        y26::int8_v1::kNchwc8LayoutId).ok;
}

int run() {
#if defined(__linux__)
    const long process = static_cast<long>(getpid());
#else
    const long process = 0;
#endif
    const auto root = std::filesystem::temp_directory_path() /
        ("y26_stage49_integrity_" + std::to_string(process));
    const auto base = make_package(root / "base");
    const std::string trusted = sha256_file(base / "asset_hashes.tsv");
    if (!verify(base, trusted)) return 1;

    constexpr std::array<const char*, 8> corrupt_assets {{
        "weights_packed_n16k8_s8.bin", "bias_i32.bin", "requant_multiplier_i64.bin",
        "requant_right_shift_i32.bin", "silu_lut_s8.bin", "model5_meta.tsv",
        "package.json", "weight_sums_i32.bin",
    }};
    for (std::size_t index = 0; index < corrupt_assets.size(); ++index) {
        const auto copy = root / ("corrupt_" + std::to_string(index));
        std::filesystem::copy(base, copy, std::filesystem::copy_options::recursive);
        flip_first_byte(copy / corrupt_assets[index]);
        if (verify(copy, trusted)) return static_cast<int>(10 + index);
    }

    const auto manifest_copy = root / "corrupt_manifest";
    std::filesystem::copy(base, manifest_copy, std::filesystem::copy_options::recursive);
    flip_first_byte(manifest_copy / "asset_hashes.tsv");
    if (verify(manifest_copy, trusted)) return 30;

    const auto unexpected = root / "unexpected";
    std::filesystem::copy(base, unexpected, std::filesystem::copy_options::recursive);
    write_text(unexpected / "extra.bin", "x");
    if (verify(unexpected, trusted)) return 31;

    const auto executable = root / "executable";
    std::filesystem::copy(base, executable, std::filesystem::copy_options::recursive);
    std::filesystem::permissions(executable / "bias_i32.bin", std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::add);
    if (verify(executable, trusted)) return 32;

    const auto alias_buffer = std::array<std::uint8_t, 64> {};
    if (!y26::int8_v1::ranges_overlap(alias_buffer.data(), 32, alias_buffer.data() + 16, 16) ||
        y26::int8_v1::ranges_overlap(alias_buffer.data(), 16, alias_buffer.data() + 16, 16)) return 33;

    std::filesystem::remove_all(root);
    return 0;
}

}  // namespace

int main() {
    const int status = run();
    std::cout << "package_integrity=" << status << '\n';
    return status == 0 ? 0 : 1;
}
