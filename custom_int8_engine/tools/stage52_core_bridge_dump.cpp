#include "y26_k1x_package.h"
#include "y26_k1x_stage48_nchwc8.h"
#include "y26_k1x_stage49_slice.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path, std::size_t expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamsize>(expected)) {
        throw std::runtime_error("input size mismatch");
    }
    stream.seekg(0);
    std::vector<std::uint8_t> bytes(expected);
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("input read failed");
    }
    return bytes;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            throw std::runtime_error(
                "usage: stage52_core_bridge_dump CORE INPUT_NCHW_U8 OUTPUT_DIR [scalar|e2c]");
        }
        const std::filesystem::path package = std::filesystem::canonical(argv[1]);
        const std::filesystem::path output = argv[3];
        std::filesystem::create_directories(output);
        y26::stage49::PersistentSlice executor;
        const std::string manifest = y26::int8_v1::sha256_file(package / "asset_hashes.tsv");
        if (executor.prepare(package, manifest, 4) != 0) throw std::runtime_error(executor.last_error());
        const auto logical = read_bytes(argv[2], 128U * 80U * 80U);
        std::vector<std::int8_t> physical(logical.size());
        y26::stage48::nchw_u8_to_nchwc8_s8(logical.data(), physical.data(), 1, 128, 80, 80);
        std::vector<std::int8_t> final(executor.tensor_bytes(executor.output_tensor_id()));
        y26::stage49::RunOptions options;
        const bool e2c = argc == 5 && std::string(argv[4]) == "e2c";
        if (argc == 5 && !e2c && std::string(argv[4]) != "scalar") {
            throw std::runtime_error("unknown route");
        }
        options.route = e2c ? y26::stage49::ComputeRoute::ime : y26::stage49::ComputeRoute::scalar;
        options.kernel = y26::stage49::KernelShape::m12n16;
        options.load = y26::stage49::LoadStrategy::vlseg2_pair_vlse;
        options.epilogue = e2c ? y26::stage49::EpilogueStrategy::rvv_q62
                               : y26::stage49::EpilogueStrategy::inline_scalar;
        options.partition = y26::stage49::PartitionPolicy::spatial;
        options.nonconv = e2c ? y26::stage49::NonConvStrategy::explicit_rvv_lut
                              : y26::stage49::NonConvStrategy::serial_scalar;
        options.scheduler = y26::stage49::SchedulerStrategy::active_workers_complete;
        options.workers = 4;
        options.capture_intermediates = true;
        if (executor.run_slice(physical.data(), final.data(), options, nullptr) != 0) {
            throw std::runtime_error(executor.last_error());
        }
        for (int tensor = 0; tensor < executor.tensor_count(); ++tensor) {
            std::vector<std::int8_t> values(executor.tensor_bytes(tensor));
            if (executor.copy_captured_tensor(tensor, values.data(), values.size()) != 0) continue;
            std::ofstream stream(output / ("tensor_" + std::to_string(tensor) + ".nchwc8_s8"), std::ios::binary);
            stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()));
        }
        std::cout << "captured=" << executor.tensor_count() << " output_bytes=" << final.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
