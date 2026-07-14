#include "y26_k1x_stage48_nchwc8.h"

#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_int8_v1.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace y26::stage48 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int kMaximumWorkers = 4;
constexpr int kNBlock = 16;
constexpr int kInputH = 80;
constexpr int kInputW = 80;
constexpr int kInputC = 128;
constexpr int kOutputH = 40;
constexpr int kOutputW = 40;
constexpr int kOutputC = 128;
constexpr int kKernelH = 3;
constexpr int kKernelW = 3;
constexpr int kStride = 2;
constexpr int kPad = 1;
constexpr int kKernelK = kKernelH * kKernelW * kInputC;
constexpr int kKTiles = kKernelK / 8;
constexpr int kNBlocks = kOutputC / kNBlock;

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

int rows_for(MBlock block) {
    return static_cast<int>(block);
}

bool pin_current_thread(int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

int current_cpu() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> values;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        values.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return values;
}

std::unordered_map<std::string, std::string> read_single_tsv_row(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::string header_line;
    std::string value_line;
    if (!stream || !std::getline(stream, header_line) || !std::getline(stream, value_line)) {
        throw std::runtime_error("cannot read single-row TSV: " + path.string());
    }
    const auto header = split_tsv(header_line);
    const auto values = split_tsv(value_line);
    if (header.size() != values.size() || header.empty()) {
        throw std::runtime_error("malformed single-row TSV: " + path.string());
    }
    std::unordered_map<std::string, std::string> result;
    for (std::size_t index = 0; index < header.size(); ++index) {
        result.emplace(header[index], values[index]);
    }
    std::string unexpected;
    if (std::getline(stream, unexpected) && !unexpected.empty()) {
        throw std::runtime_error("multiple data rows in TSV: " + path.string());
    }
    return result;
}

const std::string& text_field(const std::unordered_map<std::string, std::string>& row, const char* name) {
    const auto found = row.find(name);
    if (found == row.end()) {
        throw std::runtime_error(std::string("missing model5 metadata field: ") + name);
    }
    return found->second;
}

std::int64_t integer_field(const std::unordered_map<std::string, std::string>& row, const char* name) {
    const std::string& text = text_field(row, name);
    std::int64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) {
        throw std::runtime_error(std::string("invalid integer model5 metadata field: ") + name);
    }
    return result;
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t expected_count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open binary asset: " + path.string());
    }
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) != expected_count * sizeof(T)) {
        throw std::runtime_error("binary asset size mismatch: " + path.string());
    }
    stream.seekg(0);
    std::vector<T> result(expected_count);
    if (bytes != 0 && !stream.read(reinterpret_cast<char*>(result.data()), bytes)) {
        throw std::runtime_error("binary asset read failed: " + path.string());
    }
    return result;
}

void scalar_block(const std::int8_t* a,
                  const std::int8_t* b,
                  int k_tiles,
                  int rows,
                  std::int32_t* c) {
    std::fill(c, c + static_cast<std::size_t>(rows) * kNBlock, 0);
    const int row_groups = rows / 4;
    for (int tile = 0; tile < k_tiles; ++tile) {
        const std::int8_t* at = a + static_cast<std::size_t>(tile) * rows * 8;
        const std::int8_t* bt = b + static_cast<std::size_t>(tile) * kNBlock * 8;
        for (int output_group = 0; output_group < 4; ++output_group) {
            for (int row_group = 0; row_group < row_groups; ++row_group) {
                std::int32_t* result = c + static_cast<std::size_t>(output_group * row_groups + row_group) * 16;
                for (int row = 0; row < 4; ++row) {
                    for (int output = 0; output < 4; ++output) {
                        std::int32_t sum = result[row * 4 + output];
                        for (int lane = 0; lane < 8; ++lane) {
                            sum += static_cast<std::int32_t>(at[(row_group * 4 + row) * 8 + lane]) *
                                   static_cast<std::int32_t>(bt[(output_group * 4 + output) * 8 + lane]);
                        }
                        result[row * 4 + output] = sum;
                    }
                }
            }
        }
    }
}

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)

#define Y26_STAGE48_INIT_ACC(reg) "vxor.vv " #reg ", " #reg ", " #reg "\n\t"
#define Y26_STAGE48_DOT(acc, a, b) "smt.vmadot " #acc ", " #a ", " #b "\n\t"
#define Y26_STAGE48_STORE(reg) "vse32.v " #reg ", (t2)\n\t" "addi t2, t2, 64\n\t"

extern "C" __attribute__((noinline)) void y26_stage48_kernel_m4n16(const std::int8_t* a,
                                                                    const std::int8_t* b,
                                                                    int k_tiles,
                                                                    std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE48_INIT_ACC(v16) Y26_STAGE48_INIT_ACC(v18)
        Y26_STAGE48_INIT_ACC(v20) Y26_STAGE48_INIT_ACC(v22)
        "vsetvli t0, zero, e8, m1\n\t"
        "mv t1, %[K]\n\t" "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t"
        "vle8.v v2, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v3, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v4, (t5)\n\t"
        "addi t5, t4, 96\n\t" "vle8.v v5, (t5)\n\t"
        Y26_STAGE48_DOT(v16, v0, v2) Y26_STAGE48_DOT(v18, v0, v3)
        Y26_STAGE48_DOT(v20, v0, v4) Y26_STAGE48_DOT(v22, v0, v5)
        "addi t3, t3, 32\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE48_STORE(v16) Y26_STAGE48_STORE(v18)
        Y26_STAGE48_STORE(v20) Y26_STAGE48_STORE(v22)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5",
          "v0", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23");
}

extern "C" __attribute__((noinline)) void y26_stage48_kernel_m8n16(const std::int8_t* a,
                                                                    const std::int8_t* b,
                                                                    int k_tiles,
                                                                    std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE48_INIT_ACC(v16) Y26_STAGE48_INIT_ACC(v18)
        Y26_STAGE48_INIT_ACC(v20) Y26_STAGE48_INIT_ACC(v22)
        Y26_STAGE48_INIT_ACC(v24) Y26_STAGE48_INIT_ACC(v26)
        Y26_STAGE48_INIT_ACC(v28) Y26_STAGE48_INIT_ACC(v30)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t"
        "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "vle8.v v2, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v3, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v4, (t5)\n\t"
        "addi t5, t4, 96\n\t" "vle8.v v5, (t5)\n\t"
        Y26_STAGE48_DOT(v16, v0, v2) Y26_STAGE48_DOT(v18, v1, v2)
        Y26_STAGE48_DOT(v20, v0, v3) Y26_STAGE48_DOT(v22, v1, v3)
        Y26_STAGE48_DOT(v24, v0, v4) Y26_STAGE48_DOT(v26, v1, v4)
        Y26_STAGE48_DOT(v28, v0, v5) Y26_STAGE48_DOT(v30, v1, v5)
        "addi t3, t3, 64\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE48_STORE(v16) Y26_STAGE48_STORE(v18)
        Y26_STAGE48_STORE(v20) Y26_STAGE48_STORE(v22)
        Y26_STAGE48_STORE(v24) Y26_STAGE48_STORE(v26)
        Y26_STAGE48_STORE(v28) Y26_STAGE48_STORE(v30)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v1",
          "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

extern "C" __attribute__((noinline)) void y26_stage48_kernel_m12n16(const std::int8_t* a,
                                                                     const std::int8_t* b,
                                                                     int k_tiles,
                                                                     std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE48_INIT_ACC(v8) Y26_STAGE48_INIT_ACC(v10)
        Y26_STAGE48_INIT_ACC(v12) Y26_STAGE48_INIT_ACC(v14)
        Y26_STAGE48_INIT_ACC(v16) Y26_STAGE48_INIT_ACC(v18)
        Y26_STAGE48_INIT_ACC(v20) Y26_STAGE48_INIT_ACC(v22)
        Y26_STAGE48_INIT_ACC(v24) Y26_STAGE48_INIT_ACC(v26)
        Y26_STAGE48_INIT_ACC(v28) Y26_STAGE48_INIT_ACC(v30)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t"
        "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "addi t5, t3, 64\n\t" "vle8.v v2, (t5)\n\t"
        "vle8.v v4, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v5, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v6, (t5)\n\t"
        "addi t5, t4, 96\n\t" "vle8.v v7, (t5)\n\t"
        Y26_STAGE48_DOT(v8, v0, v4) Y26_STAGE48_DOT(v10, v1, v4)
        Y26_STAGE48_DOT(v12, v2, v4) Y26_STAGE48_DOT(v14, v0, v5)
        Y26_STAGE48_DOT(v16, v1, v5) Y26_STAGE48_DOT(v18, v2, v5)
        Y26_STAGE48_DOT(v20, v0, v6) Y26_STAGE48_DOT(v22, v1, v6)
        Y26_STAGE48_DOT(v24, v2, v6) Y26_STAGE48_DOT(v26, v0, v7)
        Y26_STAGE48_DOT(v28, v1, v7) Y26_STAGE48_DOT(v30, v2, v7)
        "addi t3, t3, 96\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE48_STORE(v8) Y26_STAGE48_STORE(v10)
        Y26_STAGE48_STORE(v12) Y26_STAGE48_STORE(v14)
        Y26_STAGE48_STORE(v16) Y26_STAGE48_STORE(v18)
        Y26_STAGE48_STORE(v20) Y26_STAGE48_STORE(v22)
        Y26_STAGE48_STORE(v24) Y26_STAGE48_STORE(v26)
        Y26_STAGE48_STORE(v28) Y26_STAGE48_STORE(v30)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v1", "v2",
          "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

extern "C" __attribute__((noinline)) void y26_stage53_kernel_m12n4(const std::int8_t* a,
                                                                     const std::int8_t* b,
                                                                     int k_tiles,
                                                                     std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE48_INIT_ACC(v8) Y26_STAGE48_INIT_ACC(v10) Y26_STAGE48_INIT_ACC(v12)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t"
        "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "addi t5, t3, 64\n\t" "vle8.v v2, (t5)\n\t"
        "vle8.v v4, (t4)\n\t"
        Y26_STAGE48_DOT(v8, v0, v4) Y26_STAGE48_DOT(v10, v1, v4)
        Y26_STAGE48_DOT(v12, v2, v4)
        "addi t3, t3, 96\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE48_STORE(v8) Y26_STAGE48_STORE(v10) Y26_STAGE48_STORE(v12)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5",
          "v0", "v1", "v2", "v4", "v8", "v9", "v10", "v11", "v12", "v13");
}

extern "C" __attribute__((noinline)) void y26_stage53_kernel_m12n8(const std::int8_t* a,
                                                                     const std::int8_t* b,
                                                                     int k_tiles,
                                                                     std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE48_INIT_ACC(v8) Y26_STAGE48_INIT_ACC(v10) Y26_STAGE48_INIT_ACC(v12)
        Y26_STAGE48_INIT_ACC(v14) Y26_STAGE48_INIT_ACC(v16) Y26_STAGE48_INIT_ACC(v18)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t"
        "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "addi t5, t3, 64\n\t" "vle8.v v2, (t5)\n\t"
        "vle8.v v4, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v5, (t5)\n\t"
        Y26_STAGE48_DOT(v8, v0, v4) Y26_STAGE48_DOT(v10, v1, v4)
        Y26_STAGE48_DOT(v12, v2, v4) Y26_STAGE48_DOT(v14, v0, v5)
        Y26_STAGE48_DOT(v16, v1, v5) Y26_STAGE48_DOT(v18, v2, v5)
        "addi t3, t3, 96\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE48_STORE(v8) Y26_STAGE48_STORE(v10) Y26_STAGE48_STORE(v12)
        Y26_STAGE48_STORE(v14) Y26_STAGE48_STORE(v16) Y26_STAGE48_STORE(v18)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5",
          "v0", "v1", "v2", "v4", "v5", "v8", "v9", "v10", "v11",
          "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
}

extern "C" __attribute__((noinline)) void y26_stage48_load_vlse64_4(const std::int8_t* source,
                                                                     std::int8_t* destination) {
    __asm__ volatile(
        "vsetivli t0, 4, e64, m1, ta, ma\n\t"
        "li t1, 16\n\t"
        "vlse64.v v0, (%[source]), t1\n\t"
        "vse64.v v0, (%[destination])\n\t"
        :
        : [source] "r"(source), [destination] "r"(destination)
        : "memory", "t0", "t1", "v0");
}

extern "C" __attribute__((noinline)) void y26_stage48_load_vlseg2e64_4(const std::int8_t* source,
                                                                        std::int8_t* destination) {
    __asm__ volatile(
        "vsetivli t0, 4, e64, m1, ta, ma\n\t"
        "vlseg2e64.v v0, (%[source])\n\t"
        "vse64.v v0, (%[destination])\n\t"
        :
        : [source] "r"(source), [destination] "r"(destination)
        : "memory", "t0", "v0", "v1");
}

#undef Y26_STAGE48_INIT_ACC
#undef Y26_STAGE48_DOT
#undef Y26_STAGE48_STORE
#endif

void run_block(ComputeRoute route,
               MBlock block,
               const std::int8_t* a,
               const std::int8_t* b,
               std::int32_t* c) {
    const int rows = rows_for(block);
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (route == ComputeRoute::ime) {
        if (block == MBlock::m4) {
            y26_stage48_kernel_m4n16(a, b, kKTiles, c);
        } else if (block == MBlock::m8) {
            y26_stage48_kernel_m8n16(a, b, kKTiles, c);
        } else {
            y26_stage48_kernel_m12n16(a, b, kKTiles, c);
        }
        return;
    }
#else
    (void)route;
#endif
    scalar_block(a, b, kKTiles, rows, c);
}

std::uint64_t repeated_byte(std::int8_t value) {
    return static_cast<std::uint64_t>(static_cast<std::uint8_t>(value)) * 0x0101010101010101ULL;
}

void copy_c8(const std::int8_t* source, std::int8_t* destination) {
    std::uint64_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    std::memcpy(destination, &value, sizeof(value));
}

struct PackCounters {
    std::uint64_t vector_groups = 0;
    std::uint64_t scalar_c8_groups = 0;
    std::uint64_t border_chunks = 0;
};

std::size_t input_offset(int channel_block, int y, int x) {
    return (((static_cast<std::size_t>(channel_block) * kInputH + y) * kInputW + x) * 8U);
}

void pack_group_scalar(const std::int8_t* input,
                       int channel_block,
                       int input_y,
                       int input_x,
                       std::int8_t* destination) {
    for (int row = 0; row < 4; ++row) {
        copy_c8(input + input_offset(channel_block, input_y, input_x + row * kStride), destination + row * 8);
    }
}

void pack_group_vector(const std::int8_t* input,
                       int channel_block,
                       int input_y,
                       int input_x,
                       LoadStrategy strategy,
                       std::int8_t* destination) {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    const std::int8_t* source = input + input_offset(channel_block, input_y, input_x);
    if (strategy == LoadStrategy::rvv_vlse64) {
        y26_stage48_load_vlse64_4(source, destination);
        return;
    }
    if (strategy == LoadStrategy::rvv_vlseg2e64) {
        y26_stage48_load_vlseg2e64_4(source, destination);
        return;
    }
#else
    (void)strategy;
#endif
    pack_group_scalar(input, channel_block, input_y, input_x, destination);
}

void pack_a_direct(const std::int8_t* input,
                   int input_zero_point,
                   int m_begin,
                   int rows,
                   int valid_rows,
                   LoadStrategy strategy,
                   std::int8_t* panel,
                   PackCounters* counters) {
    const std::int8_t padding = int8_v1::signed_storage(static_cast<std::uint8_t>(input_zero_point));
    const std::uint64_t padding_word = repeated_byte(padding);
    constexpr int channel_blocks = kInputC / 8;
    for (int tile = 0; tile < kKTiles; ++tile) {
        const int channel_block = tile % channel_blocks;
        const int kernel_position = tile / channel_blocks;
        const int kernel_y = kernel_position / kKernelW;
        const int kernel_x = kernel_position % kKernelW;
        for (int group = 0; group < rows; group += 4) {
            std::int8_t* destination = panel + (static_cast<std::size_t>(tile) * rows + group) * 8U;
            const int flat = m_begin + group;
            const int output_y = flat / kOutputW;
            const int output_x = flat % kOutputW;
            const int input_y = output_y * kStride + kernel_y - kPad;
            const int input_x = output_x * kStride + kernel_x - kPad;
            const bool complete = group + 4 <= valid_rows && flat + 3 < kOutputH * kOutputW &&
                                  output_x + 3 < kOutputW && input_y >= 0 && input_y < kInputH &&
                                  input_x >= 0 && input_x + 3 * kStride < kInputW;
            const bool segment_safe = complete && input_x + 3 * kStride + 1 < kInputW;
            const bool vector = complete && strategy == LoadStrategy::rvv_vlse64;
            const bool segment = segment_safe && strategy == LoadStrategy::rvv_vlseg2e64;
            if (vector || segment) {
                pack_group_vector(input, channel_block, input_y, input_x, strategy, destination);
                if (counters != nullptr) {
                    ++counters->vector_groups;
                }
                continue;
            }
            if (complete) {
                pack_group_scalar(input, channel_block, input_y, input_x, destination);
                if (counters != nullptr) {
                    ++counters->scalar_c8_groups;
                }
                continue;
            }
            for (int row = 0; row < 4; ++row) {
                const int position = group + row;
                const int output_flat = m_begin + position;
                std::int8_t* chunk = destination + row * 8;
                if (position >= valid_rows || output_flat >= kOutputH * kOutputW) {
                    std::memcpy(chunk, &padding_word, sizeof(padding_word));
                    if (counters != nullptr) {
                        ++counters->border_chunks;
                    }
                    continue;
                }
                const int oy = output_flat / kOutputW;
                const int ox = output_flat % kOutputW;
                const int iy = oy * kStride + kernel_y - kPad;
                const int ix = ox * kStride + kernel_x - kPad;
                if (iy < 0 || iy >= kInputH || ix < 0 || ix >= kInputW) {
                    std::memcpy(chunk, &padding_word, sizeof(padding_word));
                    if (counters != nullptr) {
                        ++counters->border_chunks;
                    }
                } else {
                    copy_c8(input + input_offset(channel_block, iy, ix), chunk);
                    if (counters != nullptr) {
                        ++counters->scalar_c8_groups;
                    }
                }
            }
        }
    }
}

struct WorkerScratch {
    std::vector<std::int8_t> a_panel;
    std::array<std::int32_t, 12 * kNBlock> c_tile {};
    int observed_cpu = -1;
    bool affinity_set = false;
};

class WorkerPool {
public:
    using Job = void (*)(void*, int, WorkerScratch&);

    explicit WorkerPool(int requested) : count_(std::clamp(requested, 1, kMaximumWorkers)) {
        scratch_.resize(static_cast<std::size_t>(count_));
        for (WorkerScratch& scratch : scratch_) {
            scratch.a_panel.resize(static_cast<std::size_t>(12) * kKernelK);
        }
        threads_.reserve(static_cast<std::size_t>(count_));
        for (int index = 0; index < count_; ++index) {
            threads_.emplace_back([this, index]() { worker_loop(index); });
        }
        std::unique_lock lock(mutex_);
        ready_cv_.wait(lock, [this]() { return ready_ == count_; });
    }

    ~WorkerPool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    int capacity() const noexcept { return count_; }

    bool affinity_ok() const noexcept {
        for (const WorkerScratch& scratch : scratch_) {
            if (!scratch.affinity_set || scratch.observed_cpu < 0 || scratch.observed_cpu > 3) {
                return false;
            }
        }
        return true;
    }

    void dispatch(int active, Job job, void* context) {
        {
            std::lock_guard lock(mutex_);
            active_ = std::clamp(active, 1, count_);
            job_ = job;
            context_ = context;
            completed_ = 0;
            ++generation_;
        }
        start_cv_.notify_all();
        std::unique_lock lock(mutex_);
        done_cv_.wait(lock, [this]() { return completed_ == count_; });
    }

private:
    void worker_loop(int index) {
        WorkerScratch& scratch = scratch_[static_cast<std::size_t>(index)];
        scratch.affinity_set = pin_current_thread(index);
        scratch.observed_cpu = current_cpu();
        {
            std::lock_guard lock(mutex_);
            ++ready_;
        }
        ready_cv_.notify_one();
        std::uint64_t local_generation = 0;
        for (;;) {
            Job job = nullptr;
            void* context = nullptr;
            int active = 0;
            {
                std::unique_lock lock(mutex_);
                start_cv_.wait(lock, [&]() { return generation_ != local_generation; });
                local_generation = generation_;
                if (stopping_) {
                    return;
                }
                job = job_;
                context = context_;
                active = active_;
            }
            if (index < active && job != nullptr) {
                scratch.observed_cpu = current_cpu();
                job(context, index, scratch);
            }
            {
                std::lock_guard lock(mutex_);
                ++completed_;
            }
            done_cv_.notify_one();
        }
    }

    int count_ = 0;
    std::vector<std::thread> threads_;
    std::vector<WorkerScratch> scratch_;
    mutable std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::condition_variable ready_cv_;
    std::uint64_t generation_ = 0;
    int active_ = 0;
    int completed_ = 0;
    int ready_ = 0;
    Job job_ = nullptr;
    void* context_ = nullptr;
    bool stopping_ = false;
};

}  // namespace

struct Model5DirectConv::Impl {
    std::filesystem::path package;
    std::vector<std::int8_t> packed_weights;
    std::vector<std::int32_t> weight_sums;
    std::vector<std::int32_t> bias;
    std::vector<int8_v1::RequantAsset> requant;
    std::array<std::int8_t, 256> silu_lut {};
    std::unique_ptr<WorkerPool> pool;
    int input_zero_point = 0;
    std::uint64_t accumulator_bound = 0;
    std::string error;
    bool ready = false;
};

namespace {

struct RunContext {
    Model5DirectConv::Impl* conv = nullptr;
    const std::int8_t* input = nullptr;
    std::int8_t* output = nullptr;
    RunOptions options;
    std::array<int, kMaximumWorkers> status {};
    std::array<double, kMaximumWorkers> total_us {};
    std::array<double, kMaximumWorkers> pack_us {};
    std::array<double, kMaximumWorkers> kernel_us {};
    std::array<double, kMaximumWorkers> epilogue_us {};
    std::array<PackCounters, kMaximumWorkers> counters {};
};

void store_tile(const Model5DirectConv::Impl& conv,
                std::int8_t* output,
                const std::int32_t* accumulators,
                int rows,
                int valid_rows,
                int m_begin,
                int n_begin) {
    const int row_groups = rows / 4;
    const std::int64_t correction = 128 - static_cast<std::int64_t>(conv.input_zero_point);
    for (int output_lane = 0; output_lane < kNBlock; ++output_lane) {
        const int output_channel = n_begin + output_lane;
        if (output_channel >= kOutputC) {
            continue;
        }
        for (int row = 0; row < valid_rows; ++row) {
            const int output_group = output_lane / 4;
            const int output_inner = output_lane % 4;
            const int row_group = row / 4;
            const int row_inner = row % 4;
            const std::int32_t raw = accumulators[
                (output_group * row_groups + row_group) * 16 + row_inner * 4 + output_inner];
            const std::int64_t corrected = static_cast<std::int64_t>(raw) +
                correction * conv.weight_sums[static_cast<std::size_t>(output_channel)] +
                conv.bias[static_cast<std::size_t>(output_channel)];
            std::uint8_t conv_code = 0;
            if (!int8_v1::requantize_u8(
                    corrected, conv.requant[static_cast<std::size_t>(output_channel)], &conv_code)) {
                continue;
            }
            const int flat = m_begin + row;
            const int output_y = flat / kOutputW;
            const int output_x = flat % kOutputW;
            const int channel_block = output_channel / 8;
            const int channel_inner = output_channel % 8;
            const std::size_t offset =
                (((static_cast<std::size_t>(channel_block) * kOutputH + output_y) * kOutputW + output_x) * 8U) +
                channel_inner;
            output[offset] = conv.silu_lut[conv_code];
        }
    }
}

void run_worker(void* opaque, int worker_index, WorkerScratch& scratch) {
    auto& context = *static_cast<RunContext*>(opaque);
    auto& conv = *context.conv;
    const int requested_rows = rows_for(context.options.m_block);
    const int output_m = kOutputH * kOutputW;
    const int m_blocks = (output_m + requested_rows - 1) / requested_rows;
    int m_block_begin = 0;
    int m_block_end = m_blocks;
    int n_block_begin = 0;
    int n_block_end = kNBlocks;
    if (context.options.partition == PartitionPolicy::spatial) {
        m_block_begin = m_blocks * worker_index / context.options.workers;
        m_block_end = m_blocks * (worker_index + 1) / context.options.workers;
    } else {
        n_block_begin = kNBlocks * worker_index / context.options.workers;
        n_block_end = kNBlocks * (worker_index + 1) / context.options.workers;
    }
    if (context.options.route == ComputeRoute::ime && !y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        context.status[static_cast<std::size_t>(worker_index)] = Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
        return;
    }
    const auto worker_begin = Clock::now();
    for (int block = m_block_begin; block < m_block_end; ++block) {
        const int m_begin = block * requested_rows;
        const int valid_rows = std::min(requested_rows, output_m - m_begin);
        MBlock actual_block = context.options.m_block;
        if (context.options.m_block == MBlock::m12 && valid_rows <= 8) {
            actual_block = valid_rows <= 4 ? MBlock::m4 : MBlock::m8;
        } else if (context.options.m_block == MBlock::m8 && valid_rows <= 4) {
            actual_block = MBlock::m4;
        }
        const int rows = rows_for(actual_block);
        PackCounters* counters = context.options.profile_phases
            ? &context.counters[static_cast<std::size_t>(worker_index)]
            : nullptr;
        const auto pack_begin = context.options.profile_phases ? Clock::now() : worker_begin;
        pack_a_direct(context.input,
                      conv.input_zero_point,
                      m_begin,
                      rows,
                      valid_rows,
                      context.options.load_strategy,
                      scratch.a_panel.data(),
                      counters);
        const auto pack_end = context.options.profile_phases ? Clock::now() : worker_begin;
        if (context.options.profile_phases) {
            context.pack_us[static_cast<std::size_t>(worker_index)] += elapsed_us(pack_begin, pack_end);
        }
        for (int n_block = n_block_begin; n_block < n_block_end; ++n_block) {
            const std::int8_t* weights = conv.packed_weights.data() +
                static_cast<std::size_t>(n_block) * kKTiles * kNBlock * 8U;
            const auto kernel_begin = context.options.profile_phases ? Clock::now() : worker_begin;
            run_block(context.options.route, actual_block, scratch.a_panel.data(), weights, scratch.c_tile.data());
            const auto kernel_end = context.options.profile_phases ? Clock::now() : worker_begin;
            if (context.options.profile_phases) {
                context.kernel_us[static_cast<std::size_t>(worker_index)] += elapsed_us(kernel_begin, kernel_end);
            }
            const auto epilogue_begin = context.options.profile_phases ? Clock::now() : worker_begin;
            store_tile(conv,
                       context.output,
                       scratch.c_tile.data(),
                       rows,
                       valid_rows,
                       m_begin,
                       n_block * kNBlock);
            const auto epilogue_end = context.options.profile_phases ? Clock::now() : worker_begin;
            if (context.options.profile_phases) {
                context.epilogue_us[static_cast<std::size_t>(worker_index)] +=
                    elapsed_us(epilogue_begin, epilogue_end);
            }
        }
    }
    context.total_us[static_cast<std::size_t>(worker_index)] = elapsed_us(worker_begin, Clock::now());
    context.status[static_cast<std::size_t>(worker_index)] = Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

Model5DirectConv::Model5DirectConv() : impl_(std::make_unique<Impl>()) {}
Model5DirectConv::~Model5DirectConv() = default;
Model5DirectConv::Model5DirectConv(Model5DirectConv&&) noexcept = default;
Model5DirectConv& Model5DirectConv::operator=(Model5DirectConv&&) noexcept = default;

int Model5DirectConv::prepare(const std::filesystem::path& package_dir, int worker_capacity) {
    if (worker_capacity < 1 || worker_capacity > kMaximumWorkers) {
        impl_->error = "worker capacity must be in [1,4]";
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    try {
        auto prepared = std::make_unique<Impl>();
        prepared->package = std::filesystem::canonical(package_dir);
        const auto meta = read_single_tsv_row(prepared->package / "model5_meta.tsv");
        if (text_field(meta, "contract_id") != int8_v1::kContractId ||
            text_field(meta, "profile_id") != int8_v1::kGeneralProfile ||
            text_field(meta, "layout_id") != int8_v1::kNchwc8LayoutId) {
            throw std::runtime_error("unsupported integer contract/profile/layout");
        }
        const std::array<std::pair<const char*, int>, 11> required {{
            {"input_h", kInputH}, {"input_w", kInputW}, {"input_c", kInputC},
            {"output_h", kOutputH}, {"output_w", kOutputW}, {"output_c", kOutputC},
            {"kernel_h", kKernelH}, {"kernel_w", kKernelW}, {"stride_h", kStride},
            {"stride_w", kStride}, {"int32_safe", 1},
        }};
        for (const auto& [name, expected] : required) {
            if (integer_field(meta, name) != expected) {
                throw std::runtime_error(std::string("model5 metadata mismatch: ") + name);
            }
        }
        if (integer_field(meta, "pad_h") != kPad || integer_field(meta, "pad_w") != kPad ||
            integer_field(meta, "k") != kKernelK || integer_field(meta, "k_tiles") != kKTiles ||
            integer_field(meta, "n_blocks") != kNBlocks) {
            throw std::runtime_error("model5 geometry metadata mismatch");
        }
        prepared->input_zero_point = static_cast<int>(integer_field(meta, "input_zero_point"));
        prepared->accumulator_bound = static_cast<std::uint64_t>(integer_field(meta, "accumulator_absolute_bound"));
        prepared->packed_weights = read_binary<std::int8_t>(
            prepared->package / "weights_packed_n16k8_s8.bin", kNBlocks * kKTiles * kNBlock * 8U);
        prepared->weight_sums = read_binary<std::int32_t>(prepared->package / "weight_sums_i32.bin", kOutputC);
        prepared->bias = read_binary<std::int32_t>(prepared->package / "bias_i32.bin", kOutputC);
        const auto multipliers = read_binary<std::int64_t>(prepared->package / "requant_multiplier_i64.bin", kOutputC);
        const auto shifts = read_binary<std::int32_t>(prepared->package / "requant_right_shift_i32.bin", kOutputC);
        const auto lut = read_binary<std::int8_t>(prepared->package / "silu_lut_s8.bin", 256);
        std::copy(lut.begin(), lut.end(), prepared->silu_lut.begin());
        const int output_zero_point = static_cast<int>(integer_field(meta, "conv_output_zero_point"));
        prepared->requant.resize(kOutputC);
        for (int channel = 0; channel < kOutputC; ++channel) {
            auto& asset = prepared->requant[static_cast<std::size_t>(channel)];
            asset.multiplier = multipliers[static_cast<std::size_t>(channel)];
            asset.right_shift = shifts[static_cast<std::size_t>(channel)];
            asset.output_zero_point = output_zero_point;
            asset.clamp_min = 0;
            asset.clamp_max = 255;
            if (!int8_v1::valid_requant_asset(asset)) {
                throw std::runtime_error("invalid requant asset");
            }
        }
        prepared->pool = std::make_unique<WorkerPool>(worker_capacity);
        prepared->ready = true;
        impl_ = std::move(prepared);
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::exception& error) {
        impl_->error = error.what();
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

int Model5DirectConv::run(const std::int8_t* input_nchwc8_s8,
                          std::int8_t* output_nchwc8_s8,
                          const RunOptions& options,
                          Timing* timing) {
    if (!impl_ || !impl_->ready || !impl_->pool || input_nchwc8_s8 == nullptr || output_nchwc8_s8 == nullptr ||
        options.workers < 1 || options.workers > impl_->pool->capacity() || rows_for(options.m_block) == 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (options.route == ComputeRoute::ime) {
        if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
            return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
        }
        if (y26_k1x_ime_probe_once() != Y26_VMADOT_STATUS_SUCCESS) {
            return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
        }
    }
    RunContext context;
    context.conv = impl_.get();
    context.input = input_nchwc8_s8;
    context.output = output_nchwc8_s8;
    context.options = options;
    const auto begin = Clock::now();
    impl_->pool->dispatch(options.workers, run_worker, &context);
    const auto end = Clock::now();
    int status = Y26_CONV_STATUS_SUCCESS;
    for (int worker = 0; worker < options.workers; ++worker) {
        if (context.status[static_cast<std::size_t>(worker)] != Y26_CONV_STATUS_SUCCESS) {
            status = context.status[static_cast<std::size_t>(worker)];
        }
    }
    if (timing != nullptr) {
        *timing = {};
        timing->total_us = elapsed_us(begin, end);
        timing->workers = options.workers;
        timing->affinity_ok = impl_->pool->affinity_ok() ? 1 : 0;
        timing->min_worker_us = std::numeric_limits<double>::max();
        for (int worker = 0; worker < options.workers; ++worker) {
            const std::size_t index = static_cast<std::size_t>(worker);
            timing->direct_a_delivery_us += context.pack_us[index];
            timing->vmadot_us += context.kernel_us[index];
            timing->scalar_epilogue_us += context.epilogue_us[index];
            timing->min_worker_us = std::min(timing->min_worker_us, context.total_us[index]);
            timing->max_worker_us = std::max(timing->max_worker_us, context.total_us[index]);
            timing->vector_groups += context.counters[index].vector_groups;
            timing->scalar_c8_groups += context.counters[index].scalar_c8_groups;
            timing->border_chunks += context.counters[index].border_chunks;
        }
        timing->barrier_us = std::max(0.0, timing->total_us - timing->max_worker_us);
    }
    return status;
}

int Model5DirectConv::debug_pack_a(const std::int8_t* input_nchwc8_s8,
                                   int m_begin,
                                   MBlock m_block,
                                   LoadStrategy strategy,
                                   std::int8_t* panel,
                                   std::size_t panel_bytes) const {
    if (!impl_ || !impl_->ready || input_nchwc8_s8 == nullptr || panel == nullptr || m_begin < 0 ||
        m_begin >= kOutputH * kOutputW) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int rows = rows_for(m_block);
    if (rows == 0 || panel_bytes < static_cast<std::size_t>(rows) * kKernelK) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const int valid_rows = std::min(rows, kOutputH * kOutputW - m_begin);
    pack_a_direct(input_nchwc8_s8,
                  impl_->input_zero_point,
                  m_begin,
                  rows,
                  valid_rows,
                  strategy,
                  panel,
                  nullptr);
    return Y26_CONV_STATUS_SUCCESS;
}

std::size_t Model5DirectConv::input_bytes() const noexcept { return kInputH * kInputW * kInputC; }
std::size_t Model5DirectConv::output_bytes() const noexcept { return kOutputH * kOutputW * kOutputC; }
std::size_t Model5DirectConv::packed_weight_bytes() const noexcept {
    return impl_ ? impl_->packed_weights.size() + impl_->weight_sums.size() * sizeof(std::int32_t) : 0;
}
std::size_t Model5DirectConv::per_worker_workspace_bytes() const noexcept {
    return static_cast<std::size_t>(12) * kKernelK + 12U * kNBlock * sizeof(std::int32_t);
}
std::uint64_t Model5DirectConv::macs() const noexcept {
    return static_cast<std::uint64_t>(kOutputH) * kOutputW * kOutputC * kKernelK;
}
bool Model5DirectConv::affinity_ok() const noexcept { return impl_ && impl_->pool && impl_->pool->affinity_ok(); }
const std::string& Model5DirectConv::last_error() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->error : empty;
}

void nchw_u8_to_nchwc8_s8(const std::uint8_t* input,
                           std::int8_t* output,
                           int batches,
                           int channels,
                           int height,
                           int width) {
    if (input == nullptr || output == nullptr || batches <= 0 || channels <= 0 || channels % 8 != 0 ||
        height <= 0 || width <= 0) {
        return;
    }
    for (int batch = 0; batch < batches; ++batch) {
        for (int channel = 0; channel < channels; ++channel) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const std::size_t source =
                        ((static_cast<std::size_t>(batch) * channels + channel) * height + y) * width + x;
                    std::size_t destination = 0;
                    if (int8_v1::nchwc8_offset(
                            batch, channel, y, x, batches, channels, height, width, &destination)) {
                        output[destination] = int8_v1::signed_storage(input[source]);
                    }
                }
            }
        }
    }
}

void nchwc8_s8_to_nchw_u8(const std::int8_t* input,
                           std::uint8_t* output,
                           int batches,
                           int channels,
                           int height,
                           int width) {
    if (input == nullptr || output == nullptr || batches <= 0 || channels <= 0 || channels % 8 != 0 ||
        height <= 0 || width <= 0) {
        return;
    }
    for (int batch = 0; batch < batches; ++batch) {
        for (int channel = 0; channel < channels; ++channel) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    std::size_t source = 0;
                    if (int8_v1::nchwc8_offset(batch, channel, y, x, batches, channels, height, width, &source)) {
                        const std::size_t destination =
                            ((static_cast<std::size_t>(batch) * channels + channel) * height + y) * width + x;
                        output[destination] = int8_v1::semantic_code(input[source]);
                    }
                }
            }
        }
    }
}

const char* compute_route_name(ComputeRoute route) noexcept {
    return route == ComputeRoute::scalar ? "scalar" : "ime";
}

const char* m_block_name(MBlock block) noexcept {
    switch (block) {
        case MBlock::m4: return "m4n16";
        case MBlock::m8: return "m8n16";
        case MBlock::m12: return "m12n16";
    }
    return "unknown";
}

const char* load_strategy_name(LoadStrategy strategy) noexcept {
    switch (strategy) {
        case LoadStrategy::c8_u64: return "four_u64_c8";
        case LoadStrategy::rvv_vlse64: return "rvv_vlse64_c8x4";
        case LoadStrategy::rvv_vlseg2e64: return "rvv_vlseg2e64_c8x4";
    }
    return "unknown";
}

const char* partition_policy_name(PartitionPolicy policy) noexcept {
    return policy == PartitionPolicy::spatial ? "spatial" : "output_channel";
}

}  // namespace y26::stage48
