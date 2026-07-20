#include "y26_k1x_full_executor.h"
#include "y26_k1x_int8_v1.h"
#include "y26_k1x_package.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_input(const std::filesystem::path& path, std::size_t elements) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open input: " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes != static_cast<std::streamsize>(elements * sizeof(float))) {
        throw std::runtime_error("input size does not match the prepared static profile");
    }
    stream.seekg(0);
    std::vector<float> result(elements);
    if (!stream.read(reinterpret_cast<char*>(result.data()), bytes)) {
        throw std::runtime_error("failed to read input");
    }
    return result;
}

unsigned read_frm() noexcept {
#if defined(__riscv)
    unsigned value = 0;
    asm volatile("frrm %0" : "=r"(value));
    return value & 7U;
#else
    return 0;
#endif
}

void write_frm(unsigned value) noexcept {
#if defined(__riscv)
    asm volatile("fsrm %0" : : "r"(value & 7U) : "memory");
#else
    (void)value;
#endif
}

unsigned read_vcsr() noexcept {
#if defined(__riscv)
    unsigned value = 0;
    asm volatile("csrr %0, vcsr" : "=r"(value));
    return value & 7U;
#else
    return 0;
#endif
}

void write_vcsr(unsigned value) noexcept {
#if defined(__riscv)
    asm volatile("csrw vcsr, %0" : : "r"(value & 7U) : "memory");
#else
    (void)value;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: stage52_state_probe PACKAGE INPUT_F32\n";
        return 2;
    }
    try {
        const std::filesystem::path package = argv[1];
        y26::stage52::RunConfig config;
        config.workers = 4;
        config.worker_cpu_begin = 0;
        config.controller_cpu = 4;
        config.scheduler = y26::stage52::SchedulerMode::safe;
        config.compute = y26::stage52::ComputeMode::optimized;
        config.allow_stage60_static_profiles = true;
        y26::stage52::FullExecutor executor;
        const std::string manifest = y26::int8_v1::sha256_file(package / "asset_hashes.tsv");
        if (executor.prepare(package, manifest, config) != 0) {
            throw std::runtime_error("prepare failed: " + executor.last_error());
        }
        const std::vector<float> input = read_input(argv[2], executor.input_elements());

        const unsigned saved_frm = read_frm();
        const unsigned saved_vcsr = read_vcsr();
        constexpr std::array<unsigned, 5> ambient_vcsr {0U, 3U, 4U, 7U, 2U};
        std::uint64_t expected_hash = 0;
        bool pass = true;
        std::array<float, 1800> output {};
        std::cout << "frm\tambient_vcsr\tafter_frm\tafter_vcsr\toutput_hash\taffinity_ok"
                     "\tcpu4_7_ime_count\tstatus\n";
        for (unsigned frm = 0; frm < 5U; ++frm) {
            write_frm(frm);
            write_vcsr(ambient_vcsr[frm]);
            y26::stage52::RunTiming timing;
            const int status = executor.run_preprocessed(input.data(), input.size(), output.data(),
                                                         output.size(), &timing);
            const unsigned after_frm = read_frm();
            const unsigned after_vcsr = read_vcsr();
            if (expected_hash == 0) expected_hash = timing.output_hash;
            const bool row_pass = status == 0 && after_frm == frm &&
                                  after_vcsr == ambient_vcsr[frm] &&
                                  timing.output_hash == expected_hash &&
                                  timing.affinity_ok == 1 && timing.cpu4_7_ime_count == 0;
            pass = pass && row_pass;
            std::cout << frm << '\t' << ambient_vcsr[frm] << '\t' << after_frm << '\t'
                      << after_vcsr << "\t0x" << std::hex << timing.output_hash << std::dec
                      << '\t' << timing.affinity_ok << '\t' << timing.cpu4_7_ime_count << '\t'
                      << (row_pass ? "pass" : "fail") << '\n';
        }
        write_frm(saved_frm);
        write_vcsr(saved_vcsr);
        std::cout << "restored_process_state\tfrm=" << read_frm() << "\tvcsr=" << read_vcsr()
                  << "\texpected_frm=" << saved_frm << "\texpected_vcsr=" << saved_vcsr << '\n';
        return pass && read_frm() == saved_frm && read_vcsr() == saved_vcsr ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
