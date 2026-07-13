#include "y26_k1x_full_executor.h"
#include "y26_k1x_package.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_input(const std::filesystem::path& path) {
    constexpr std::size_t kElements = 3U * 640U * 640U;
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamsize>(kElements * sizeof(float))) {
        throw std::runtime_error("invalid preprocessed input");
    }
    stream.seekg(0);
    std::vector<float> input(kElements);
    if (!stream.read(reinterpret_cast<char*>(input.data()),
                     static_cast<std::streamsize>(input.size() * sizeof(float)))) {
        throw std::runtime_error("cannot read preprocessed input");
    }
    return input;
}

std::size_t tensor_elements(const std::filesystem::path& path, int wanted) {
    std::ifstream stream(path);
    std::string line;
    if (!stream || !std::getline(stream, line)) throw std::runtime_error("cannot read tensors.tsv");
    while (std::getline(stream, line)) {
        std::vector<std::string> fields;
        std::size_t begin = 0;
        for (;;) {
            const std::size_t end = line.find('\t', begin);
            fields.emplace_back(line.substr(begin, end == std::string::npos ? end : end - begin));
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        if (fields.size() > 8 && std::stoi(fields[0]) == wanted) {
            return static_cast<std::size_t>(std::stoull(fields[8]));
        }
    }
    throw std::runtime_error("tensor descriptor missing");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6) {
            throw std::runtime_error(
                "usage: stage52_boundary_dump PACKAGE INPUT_F32 OUTPUT_DIR [--scalar] [TENSOR_ID]");
        }
        const std::filesystem::path package = std::filesystem::canonical(argv[1]);
        const std::filesystem::path output_dir = argv[3];
        std::filesystem::create_directories(output_dir);
        y26::stage52::RunConfig config;
        int selected_tensor = -1;
        for (int index = 4; index < argc; ++index) {
            if (std::string(argv[index]) == "--scalar") {
                config.compute = y26::stage52::ComputeMode::scalar;
            } else if (selected_tensor < 0) {
                selected_tensor = std::stoi(argv[index]);
            } else {
                throw std::runtime_error("duplicate tensor selection");
            }
        }
        config.capture_boundaries = selected_tensor < 0;
        y26::stage52::FullExecutor executor;
        const std::string manifest = y26::int8_v1::sha256_file(package / "asset_hashes.tsv");
        if (executor.prepare(package, manifest, config) != 0) {
            throw std::runtime_error(executor.last_error());
        }
        const std::vector<float> input = read_input(argv[2]);
        std::vector<float> output(300U * 6U);
        y26::stage52::RunTiming timing;
        if (executor.run_preprocessed(input.data(), input.size(), output.data(), output.size(), &timing) != 0) {
            throw std::runtime_error(executor.last_error());
        }
        const int first_tensor = selected_tensor < 0 ? 0 : selected_tensor;
        const int last_tensor = selected_tensor < 0 ? executor.tensor_count() : first_tensor + 1;
        for (int tensor = first_tensor; tensor < last_tensor; ++tensor) {
            const std::size_t count = tensor_elements(package / "tensors.tsv", tensor);
            std::vector<std::uint8_t> values(count);
            if (executor.copy_boundary(tensor, values.data(), values.size()) != 0) {
                throw std::runtime_error("cannot copy captured tensor " + std::to_string(tensor));
            }
            std::ofstream stream(output_dir / ("tensor_" + std::to_string(tensor) + ".u8"), std::ios::binary);
            stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size()));
        }
        std::ofstream final_stream(output_dir / "output_1x300x6.f32", std::ios::binary);
        final_stream.write(reinterpret_cast<const char*>(output.data()),
                           static_cast<std::streamsize>(output.size() * sizeof(float)));
        std::cout << "mode=" << y26::stage52::compute_mode_name(config.compute)
                  << " tensors=" << executor.tensor_count() << " operations=" << executor.operation_count()
                  << " wall_us=" << timing.total_us << " output_hash=0x" << std::hex << timing.output_hash
                  << std::dec << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
