#include "y26_k1x_full_executor.h"

#include "y26_k1x_int8_v1.h"
#include "y26_k1x_package.h"
#include "y26_k1x_stage49_slice.h"
#include "y26_k1x_stage51_q62.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#if defined(__linux__)
#include <sys/resource.h>
#endif

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
extern "C" void y26_stage48_kernel_m12n16(
    const std::int8_t*, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage48_kernel_m8n16(
    const std::int8_t*, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage53_kernel_m12n4(
    const std::int8_t*, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage53_kernel_m12n8(
    const std::int8_t*, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage54_kernel_direct_1x1_m12n16(
    const std::int8_t*, std::ptrdiff_t, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage48_load_vlse64_4(const std::int8_t*, std::int8_t*);
extern "C" void y26_stage49_load_vlseg2_pair_4(
    const std::int8_t*, std::int8_t*, std::int8_t*);
extern "C" void y26_stage49_load_contiguous_c8x4(const std::int8_t*, std::int8_t*);
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace y26::stage52 {
namespace {

using Clock = std::chrono::steady_clock;
using Row = std::unordered_map<std::string, std::string>;
__extension__ using UnsignedInt128 = unsigned __int128;
constexpr int kDenseM = 12;
constexpr int kDenseN = 16;

#if !defined(Y26_K1X_ENABLE_IME_ASM) || !defined(__riscv)
void scalar_m12n16(const std::int8_t* a, const std::int8_t* b,
                   int k_tiles, std::int32_t* c) noexcept {
    std::fill(c, c + kDenseM * kDenseN, 0);
    for (int tile = 0; tile < k_tiles; ++tile) {
        const std::int8_t* at = a + static_cast<std::size_t>(tile) * kDenseM * 8U;
        const std::int8_t* bt = b + static_cast<std::size_t>(tile) * kDenseN * 8U;
        for (int output_group = 0; output_group < 4; ++output_group) {
            for (int row_group = 0; row_group < 3; ++row_group) {
                std::int32_t* result = c + (output_group * 3 + row_group) * 16;
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
#endif

void run_m12n16(const std::int8_t* a, const std::int8_t* b,
                int k_tiles, std::int32_t* c) noexcept {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    y26_stage48_kernel_m12n16(a, b, k_tiles, c);
#else
    scalar_m12n16(a, b, k_tiles, c);
#endif
}

void run_m12n_live(const std::int8_t* a, const std::int8_t* b,
                   int k_tiles, int live_channels, std::int32_t* c) noexcept {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (live_channels <= 4) {
        y26_stage53_kernel_m12n4(a, b, k_tiles, c);
    } else if (live_channels <= 8) {
        y26_stage53_kernel_m12n8(a, b, k_tiles, c);
    } else {
        y26_stage48_kernel_m12n16(a, b, k_tiles, c);
    }
#else
    (void)live_channels;
    scalar_m12n16(a, b, k_tiles, c);
#endif
}

void run_m8n16(const std::int8_t* a, const std::int8_t* b,
               int k_tiles, std::int32_t* c) noexcept {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    y26_stage48_kernel_m8n16(a, b, k_tiles, c);
#else
    std::fill(c, c + 8 * kDenseN, 0);
    for (int tile = 0; tile < k_tiles; ++tile) {
        const auto* at = a + static_cast<std::size_t>(tile) * 8U * 8U;
        const auto* bt = b + static_cast<std::size_t>(tile) * kDenseN * 8U;
        for (int output_group = 0; output_group < 4; ++output_group) {
            for (int row_group = 0; row_group < 2; ++row_group) {
                auto* result = c + (output_group * 2 + row_group) * 16;
                for (int row = 0; row < 4; ++row) {
                    for (int output = 0; output < 4; ++output) {
                        for (int lane = 0; lane < 8; ++lane) {
                            result[row * 4 + output] +=
                                static_cast<std::int32_t>(at[(row_group * 4 + row) * 8 + lane]) *
                                static_cast<std::int32_t>(bt[(output_group * 4 + output) * 8 + lane]);
                        }
                    }
                }
            }
        }
    }
#endif
}

void transform_lut_rvv(const std::int8_t* source, std::int8_t* destination,
                       const std::int8_t* lut, std::size_t count) noexcept {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    __asm__ volatile(
        "li t1, 128\n\t"
        "1:\n\t"
        "beqz %[count], 2f\n\t"
        "vsetvli t0, %[count], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[source])\n\t"
        "vxor.vx v0, v0, t1\n\t"
        "vluxei8.v v1, (%[lut]), v0\n\t"
        "vse8.v v1, (%[destination])\n\t"
        "add %[source], %[source], t0\n\t"
        "add %[destination], %[destination], t0\n\t"
        "sub %[count], %[count], t0\n\t"
        "j 1b\n\t"
        "2:\n\t"
        : [source] "+r"(source), [destination] "+r"(destination), [count] "+r"(count)
        : [lut] "r"(lut)
        : "memory", "t0", "t1", "v0", "v1");
#else
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] = lut[int8_v1::semantic_code(source[index])];
    }
#endif
}

void transform_lut2_rvv(const std::int8_t* left, const std::int8_t* right,
                        std::int8_t* destination, const std::int8_t* lut,
                        std::size_t count) noexcept {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    __asm__ volatile(
        "li t1, 128\n\t"
        "1:\n\t"
        "beqz %[count], 2f\n\t"
        "vsetvli t0, %[count], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[left])\n\t"
        "vle8.v v1, (%[right])\n\t"
        "vxor.vx v0, v0, t1\n\t"
        "vxor.vx v1, v1, t1\n\t"
        "vwaddu.vx v2, v0, zero\n\t"
        "vwaddu.vx v4, v1, zero\n\t"
        "vsetvli zero, t0, e16, m2, ta, ma\n\t"
        "vsll.vi v2, v2, 8\n\t"
        "vor.vv v2, v2, v4\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"
        "vluxei16.v v6, (%[lut]), v2\n\t"
        "vse8.v v6, (%[destination])\n\t"
        "add %[left], %[left], t0\n\t"
        "add %[right], %[right], t0\n\t"
        "add %[destination], %[destination], t0\n\t"
        "sub %[count], %[count], t0\n\t"
        "j 1b\n\t"
        "2:\n\t"
        : [left] "+r"(left), [right] "+r"(right),
          [destination] "+r"(destination), [count] "+r"(count)
        : [lut] "r"(lut)
        : "memory", "t0", "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint8_t left_code = int8_v1::semantic_code(left[index]);
        const std::uint8_t right_code = int8_v1::semantic_code(right[index]);
        destination[index] = lut[static_cast<std::size_t>(left_code) * 256U + right_code];
    }
#endif
}

void softmax_max_sum_rvv(const std::int8_t* source, std::size_t count,
                         const std::uint64_t* exp_q48,
                         std::uint8_t* maximum, std::uint64_t* sum) noexcept {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    const std::int8_t* max_source = source;
    std::size_t max_count = count;
    std::uint64_t max_value = 0;
    __asm__ volatile(
        "vsetivli zero, 1, e8, m1, ta, ma\n\t"
        "vmv.s.x v2, zero\n\t"
        "li t1, 128\n\t"
        "1:\n\t"
        "beqz %[count], 2f\n\t"
        "vsetvli t0, %[count], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[source])\n\t"
        "vxor.vx v0, v0, t1\n\t"
        "vredmaxu.vs v2, v0, v2\n\t"
        "add %[source], %[source], t0\n\t"
        "sub %[count], %[count], t0\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "vmv.x.s %[maximum], v2\n\t"
        : [source] "+r"(max_source), [count] "+r"(max_count),
          [maximum] "=r"(max_value)
        :
        : "memory", "t0", "t1", "v0", "v2");

    const std::int8_t* sum_source = source;
    std::size_t sum_count = count;
    std::uint64_t sum_value = 0;
    __asm__ volatile(
        "vsetivli zero, 1, e64, m1, ta, ma\n\t"
        "vmv.s.x v4, zero\n\t"
        "li t1, 128\n\t"
        "1:\n\t"
        "beqz %[count], 2f\n\t"
        "vsetvli t0, %[count], e8, mf2, ta, ma\n\t"
        "vle8.v v0, (%[source])\n\t"
        "vxor.vx v0, v0, t1\n\t"
        "vrsub.vx v0, v0, %[maximum]\n\t"
        "vwaddu.vx v2, v0, zero\n\t"
        "vsetvli zero, t0, e16, m1, ta, ma\n\t"
        "vsll.vi v2, v2, 3\n\t"
        "vsetvli zero, t0, e64, m4, ta, ma\n\t"
        "vluxei16.v v8, (%[lut]), v2\n\t"
        "vredsum.vs v4, v8, v4\n\t"
        "add %[source], %[source], t0\n\t"
        "sub %[count], %[count], t0\n\t"
        "j 1b\n\t"
        "2:\n\t"
        "vmv.x.s %[sum], v4\n\t"
        : [source] "+r"(sum_source), [count] "+r"(sum_count), [sum] "=r"(sum_value)
        : [maximum] "r"(max_value), [lut] "r"(exp_q48)
        : "memory", "t0", "t1", "v0", "v2", "v4", "v8", "v9", "v10", "v11");
    *maximum = static_cast<std::uint8_t>(max_value);
    *sum = sum_value;
#else
    std::uint8_t max_value = 0;
    for (std::size_t index = 0; index < count; ++index) {
        max_value = std::max(max_value, int8_v1::semantic_code(source[index]));
    }
    std::uint64_t sum_value = 0;
    for (std::size_t index = 0; index < count; ++index) {
        sum_value += exp_q48[max_value - int8_v1::semantic_code(source[index])];
    }
    *maximum = max_value;
    *sum = sum_value;
#endif
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) noexcept {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

std::int8_t quantize_input_f32(float value) noexcept {
    const double scaled = static_cast<double>(value) * 255.0;
    const double floor_value = std::floor(scaled);
    const double fraction = scaled - floor_value;
    std::int64_t rounded = static_cast<std::int64_t>(floor_value);
    if (fraction > 0.5 || (fraction == 0.5 && (rounded & 1))) ++rounded;
    return int8_v1::signed_storage(static_cast<std::uint8_t>(
        std::clamp<std::int64_t>(rounded, 0, 255)));
}

std::vector<std::string> split(std::string_view text, char delimiter) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = text.find(delimiter, begin);
        result.emplace_back(text.substr(begin, end == std::string_view::npos ? end : end - begin));
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return result;
}

std::vector<Row> read_tsv(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::string line;
    if (!stream || !std::getline(stream, line)) throw std::runtime_error("cannot read TSV: " + path.string());
    const auto header = split(line, '\t');
    std::vector<Row> rows;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        const auto values = split(line, '\t');
        if (values.size() != header.size()) throw std::runtime_error("malformed TSV: " + path.string());
        Row& row = rows.emplace_back();
        for (std::size_t index = 0; index < header.size(); ++index) row.emplace(header[index], values[index]);
    }
    return rows;
}

const std::string& value(const Row& row, const char* field) {
    const auto found = row.find(field);
    if (found == row.end()) throw std::runtime_error(std::string("missing field: ") + field);
    return found->second;
}

std::string optional_value(const Row& row, const std::string& field, std::string fallback = {}) {
    const auto found = row.find(field);
    return found == row.end() || found->second.empty() ? std::move(fallback) : found->second;
}

std::int64_t parse_i64(std::string_view text, const char* field) {
    std::int64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) {
        throw std::runtime_error(std::string("invalid integer: ") + field);
    }
    return result;
}

int integer(const Row& row, const char* field, int fallback = 0) {
    const auto found = row.find(field);
    if (found == row.end() || found->second.empty()) return fallback;
    const std::int64_t parsed = parse_i64(found->second, field);
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("integer out of range: ") + field);
    }
    return static_cast<int>(parsed);
}

std::size_t size_field(const Row& row, const char* field) {
    const std::int64_t parsed = parse_i64(value(row, field), field);
    if (parsed < 0) throw std::runtime_error(std::string("negative size: ") + field);
    return static_cast<std::size_t>(parsed);
}

std::vector<int> parse_ints(std::string_view text, char delimiter) {
    if (text.empty()) return {};
    std::vector<int> result;
    for (const std::string& item : split(text, delimiter)) {
        result.push_back(static_cast<int>(parse_i64(item, "integer list")));
    }
    return result;
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open asset: " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) != count * sizeof(T)) {
        throw std::runtime_error("asset size mismatch: " + path.string());
    }
    stream.seekg(0);
    std::vector<T> result(count);
    if (bytes > 0 && !stream.read(reinterpret_cast<char*>(result.data()), bytes)) {
        throw std::runtime_error("cannot read asset: " + path.string());
    }
    return result;
}

bool pin_thread(int cpu) noexcept {
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

int current_cpu() noexcept {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

class WorkerPool {
public:
    using Job = void (*)(void*, int, int);

    WorkerPool(int workers, int cpu_begin, SchedulerMode scheduler)
        : count_(workers), cpu_begin_(cpu_begin), scheduler_(scheduler) {
        threads_.reserve(static_cast<std::size_t>(count_));
        affinity_.resize(static_cast<std::size_t>(count_));
        for (int worker = 0; worker < count_; ++worker) {
            threads_.emplace_back([this, worker]() { loop(worker); });
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
        for (std::thread& thread : threads_) if (thread.joinable()) thread.join();
    }

    void dispatch(int active, Job job, void* context) {
        active = std::clamp(active, 1, count_);
        {
            std::lock_guard lock(mutex_);
            active_ = active;
            completed_ = 0;
            job_ = job;
            context_ = context;
            ++generation_;
        }
        start_cv_.notify_all();
        std::unique_lock lock(mutex_);
        complete_cv_.wait(lock, [this, active]() { return completed_ == active; });
    }

    bool affinity_ok() const noexcept {
        return std::all_of(affinity_.begin(), affinity_.end(), [](int value) { return value == 1; });
    }

private:
    void loop(int worker) {
        bool ok = pin_thread(cpu_begin_ + worker);
#if defined(__linux__)
        if (scheduler_ == SchedulerMode::rr20) {
            sched_param parameter {};
            parameter.sched_priority = 20;
            ok = ok && pthread_setschedparam(pthread_self(), SCHED_RR, &parameter) == 0;
        }
#endif
        ok = ok && current_cpu() == cpu_begin_ + worker;
        affinity_[static_cast<std::size_t>(worker)] = ok ? 1 : 0;
        std::unique_lock lock(mutex_);
        ++ready_;
        ready_cv_.notify_one();
        std::uint64_t observed = generation_;
        for (;;) {
            start_cv_.wait(lock, [this, observed]() { return stopping_ || generation_ != observed; });
            if (stopping_) return;
            observed = generation_;
            const int active = active_;
            Job job = job_;
            void* context = context_;
            if (worker >= active) continue;
            lock.unlock();
            job(context, worker, active);
            lock.lock();
            ++completed_;
            if (completed_ == active) complete_cv_.notify_one();
        }
    }

    int count_ = 0;
    int cpu_begin_ = 0;
    SchedulerMode scheduler_ = SchedulerMode::safe;
    std::vector<std::thread> threads_;
    std::vector<int> affinity_;
    mutable std::mutex mutex_;
    std::condition_variable ready_cv_;
    std::condition_variable start_cv_;
    std::condition_variable complete_cv_;
    int ready_ = 0;
    int active_ = 0;
    int completed_ = 0;
    std::uint64_t generation_ = 0;
    bool stopping_ = false;
    Job job_ = nullptr;
    void* context_ = nullptr;
};

enum class Layout { feature_nchwc8, linear };

struct Tensor {
    int id = -1;
    std::string name;
    int rank = 0;
    std::array<int, 4> dims {1, 1, 1, 1};
    std::size_t logical_elements = 0;
    std::size_t storage_bytes = 0;
    std::size_t offset = 0;
    Layout layout = Layout::linear;
    int zero_point = 0;
};

struct Conv {
    int output_c = 0;
    int input_c = 0;
    int group = 1;
    int kernel_h = 0;
    int kernel_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int pad_top = 0;
    int pad_left = 0;
    std::uint64_t accumulator_bound = 0;
    bool e2c_compatible = false;
    std::vector<std::int8_t> weights;
    std::vector<std::int32_t> bias;
    std::vector<std::int64_t> multiplier;
    std::vector<std::int32_t> shift;
    std::vector<std::int8_t> packed_weights;
    std::vector<std::int64_t> weight_sums;
    std::vector<std::int64_t> corrected_bias;
    std::vector<std::int64_t> multiplier_m63;
    std::vector<std::int8_t> depthwise_weights_c8;
    std::vector<std::int32_t> depthwise_corrected_bias_i32;
    std::vector<std::int8_t> stem_weights_tap_major;
    std::vector<std::int32_t> stem_corrected_bias_i32;
    int k_tiles = 0;
    int n_blocks = 0;
    int input_blocks = 0;
    bool dense_ime_eligible = false;
    bool depthwise_rvv_eligible = false;
    bool rgb_stem_rvv_eligible = false;
    bool stage55_family_a_weight_stationary = false;
    bool stage55_family_b_m8 = false;
};

enum class BranchTransform { copy, split, reshape_split, resize, pool0, pool1, pool2, pool3 };

struct Branch {
    int input_slot = 0;
    BranchTransform transform = BranchTransform::copy;
    int axis = 1;
    int part = 0;
    int parts = 1;
    std::vector<std::int8_t> lut;
};

enum class OpKind {
    input_quant, conv_dense, conv_grouped, lut1, lut2, split,
    reshape, transpose, reshape_split_transpose, concat, resize,
    matmul, softmax_transpose,
};

struct Operation {
    int index = -1;
    OpKind kind = OpKind::input_quant;
    std::string name;
    int output = -1;
    std::vector<int> inputs;
    Conv conv;
    std::vector<std::int8_t> lut;
    int axis = 1;
    int part = 0;
    int parts = 1;
    int split_offset = 0;
    std::vector<int> source_shape;
    std::vector<int> perm;
    std::vector<Branch> branches;
    std::int64_t multiplier = 0;
    std::int64_t multiplier_m63 = 0;
    int right_shift = 0;
    int left_zero_point = 0;
    int right_zero_point = 0;
    int output_zero_point = 0;
    std::uint64_t softmax_reciprocal_q32 = 0;
    std::vector<std::uint64_t> exp_q48;
    bool matmul_ime_eligible = false;
    int fused_lut_output = -1;
    std::vector<std::int8_t> fused_lut;
    bool skip_when_fused = false;
};

struct OperationProfileSample {
    int full_operation_index = -1;
    int resident_operation_index = -1;
    std::string kind;
    std::string source;
    std::string name;
    double wall_us = 0.0;
};

const char* operation_kind_name(OpKind kind) noexcept {
    switch (kind) {
        case OpKind::input_quant: return "input_quant";
        case OpKind::conv_dense: return "conv_dense";
        case OpKind::conv_grouped: return "conv_grouped";
        case OpKind::lut1: return "lut1";
        case OpKind::lut2: return "lut2";
        case OpKind::split: return "split";
        case OpKind::reshape: return "reshape";
        case OpKind::transpose: return "transpose";
        case OpKind::reshape_split_transpose: return "reshape_split_transpose";
        case OpKind::concat: return "concat";
        case OpKind::resize: return "resize";
        case OpKind::matmul: return "matmul";
        case OpKind::softmax_transpose: return "softmax_transpose";
    }
    return "unknown";
}

struct HeadScale {
    int resolution = 0;
    int stride = 0;
    int reg_tensor = -1;
    int cls_tensor = -1;
    std::vector<std::int32_t> reg_q16;
    std::vector<std::uint32_t> cls_q24;
};

std::size_t ravel(const std::array<int, 4>& index, const Tensor& tensor) noexcept {
    std::size_t result = 0;
    for (int axis = 0; axis < tensor.rank; ++axis) {
        result = result * static_cast<std::size_t>(tensor.dims[static_cast<std::size_t>(axis)]) +
                 static_cast<std::size_t>(index[static_cast<std::size_t>(axis)]);
    }
    return result;
}

std::array<int, 4> unravel(std::size_t flat, const Tensor& tensor) noexcept {
    std::array<int, 4> index {};
    for (int axis = tensor.rank - 1; axis >= 0; --axis) {
        const std::size_t dim = static_cast<std::size_t>(tensor.dims[static_cast<std::size_t>(axis)]);
        index[static_cast<std::size_t>(axis)] = static_cast<int>(flat % dim);
        flat /= dim;
    }
    return index;
}

std::size_t physical_offset(const Tensor& tensor, std::size_t logical_flat) noexcept {
    if (tensor.layout == Layout::linear || tensor.rank != 4) return tensor.offset + logical_flat;
    const auto index = unravel(logical_flat, tensor);
    const std::size_t channel = static_cast<std::size_t>(index[1]);
    return tensor.offset + ((((static_cast<std::size_t>(index[0]) *
        ((static_cast<std::size_t>(tensor.dims[1]) + 7U) / 8U) + channel / 8U) *
        static_cast<std::size_t>(tensor.dims[2]) + static_cast<std::size_t>(index[2])) *
        static_cast<std::size_t>(tensor.dims[3]) + static_cast<std::size_t>(index[3])) * 8U +
        channel % 8U);
}

std::uint8_t semantic(const std::vector<std::int8_t>& arena, const Tensor& tensor,
                      std::size_t logical_flat) noexcept {
    return int8_v1::semantic_code(arena[physical_offset(tensor, logical_flat)]);
}

void store_semantic(std::vector<std::int8_t>& arena, const Tensor& tensor,
                    std::size_t logical_flat, std::uint8_t code) noexcept {
    arena[physical_offset(tensor, logical_flat)] = int8_v1::signed_storage(code);
}

std::uint64_t fnv1a64(const float* values, std::size_t count) noexcept {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values);
    std::uint64_t hash = UINT64_C(1469598103934665603);
    for (std::size_t index = 0; index < count * sizeof(float); ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::int64_t round_divide_even(UnsignedInt128 numerator, UnsignedInt128 denominator) noexcept {
    const UnsignedInt128 quotient = numerator / denominator;
    const UnsignedInt128 remainder = numerator % denominator;
    const UnsignedInt128 doubled = remainder * 2U;
    return static_cast<std::int64_t>(quotient +
        (doubled > denominator || (doubled == denominator && (quotient & 1U)) ? 1U : 0U));
}

OpKind parse_kind(std::string_view value) {
    if (value == "input_quant") return OpKind::input_quant;
    if (value == "conv_dense") return OpKind::conv_dense;
    if (value == "conv_grouped") return OpKind::conv_grouped;
    if (value == "lut1") return OpKind::lut1;
    if (value == "lut2") return OpKind::lut2;
    if (value == "split") return OpKind::split;
    if (value == "reshape") return OpKind::reshape;
    if (value == "transpose") return OpKind::transpose;
    if (value == "reshape_split_transpose" || value == "split_transpose") {
        return OpKind::reshape_split_transpose;
    }
    if (value == "concat") return OpKind::concat;
    if (value == "resize") return OpKind::resize;
    if (value == "matmul") return OpKind::matmul;
    if (value == "softmax_transpose") return OpKind::softmax_transpose;
    throw std::runtime_error("unsupported full-graph operation kind: " + std::string(value));
}

BranchTransform parse_branch_transform(std::string_view value) {
    if (value.empty() || value == "copy") return BranchTransform::copy;
    if (value == "split") return BranchTransform::split;
    if (value == "reshape_split") return BranchTransform::reshape_split;
    if (value == "resize") return BranchTransform::resize;
    if (value == "pool0") return BranchTransform::pool0;
    if (value == "pool1") return BranchTransform::pool1;
    if (value == "pool2") return BranchTransform::pool2;
    if (value == "pool3") return BranchTransform::pool3;
    throw std::runtime_error("unsupported Concat branch transform");
}

}  // namespace

struct FullExecutor::Impl {
    using OperationRunner = void (*)(Impl*, const Operation&, const float*,
                                     const std::uint8_t*, int);

    struct DenseScratch {
        std::vector<std::int8_t> a;
        alignas(32) std::array<std::int32_t, kDenseM * kDenseN> c {};
        alignas(32) std::array<std::int64_t, kDenseM> row_sums {};
    };

    struct AttentionProfile {
        std::atomic<std::uint64_t> matmul_pack_ns {0};
        std::atomic<std::uint64_t> matmul_compute_ns {0};
        std::atomic<std::uint64_t> softmax_max_sum_ns {0};
        std::atomic<std::uint64_t> softmax_normalize_transpose_ns {0};
    };

    std::filesystem::path package;
    std::string manifest;
    RunConfig config;
    std::vector<Tensor> tensors;
    std::vector<Operation> operations;
    std::vector<OperationRunner> operation_runners;
    std::vector<std::int8_t> arena;
    std::vector<std::vector<std::int8_t>> captured;
    struct CoreBridge {
        int full_tensor = -1;
        int core_tensor = -1;
        std::vector<std::int8_t> diagnostic_snapshot;
    };
    std::array<CoreBridge, 6> core_bridges;
    std::array<HeadScale, 3> head;
    std::array<float, 300 * 6> last_output {};
    std::unique_ptr<WorkerPool> pool;
    std::vector<DenseScratch> dense_scratch;
    std::vector<std::int8_t> matmul_packed_right;
    std::vector<std::int64_t> matmul_right_sums;
    std::unique_ptr<stage49::PersistentSlice> optimized_core;
    std::vector<std::int8_t> core_input_diagnostic_snapshot;
    int core_start_operation = -1;
    int core_end_operation = -1;
    int core_input_tensor = -1;
    int core_output_tensor = -1;
    int optimized_core_last_operation = -1;
    std::size_t core_scratch_offset = 0;
    std::size_t core_scratch_bytes = 0;
    std::size_t total_weight_bytes = 0;
    std::string error;
    const float* current_float_input = nullptr;
    bool controller_affinity_ok = false;
    bool e2c2_enabled = true;
    bool small_n_enabled = true;
    bool rgb_stem_enabled = true;
    bool fused_lut_enabled = false;
    bool direct_1x1_enabled = false;
    bool e2c3_enabled = false;
    bool e2c4_enabled = false;
    bool dense_m8_enabled = false;
    bool dense_weight_stationary_enabled = false;
    bool stage55_dense_family_a_enabled = false;
    bool stage55_dense_family_b_enabled = false;
    int dense_partition = 0;
    bool depthwise_v2_enabled = false;
    bool depthwise_x2_enabled = false;
    bool depthwise_border_v2_enabled = false;
    bool stage55_depthwise_e2c4_enabled = false;
    bool lut2_rvv_enabled = false;
    bool input_rvv_v2_enabled = false;
    bool input_compact_c3_enabled = false;
    bool input_stem_fused_enabled = false;
    bool attention_v2_enabled = false;
    bool attention_subphase_profile_enabled = false;
    bool head_v2_enabled = false;
    bool dense_pack_rvv_enabled = false;
    bool static_schedule_enabled = false;
    std::vector<int> static_batch_end;
    bool ready = false;
    std::uint64_t profile_run_sequence = 0;
    std::uint64_t attention_profile_run_sequence = 0;
    std::shared_ptr<AttentionProfile> attention_profile = std::make_shared<AttentionProfile>();

    int full_operation_index(std::string_view name) const noexcept {
        const auto found = std::find_if(operations.begin(), operations.end(),
            [name](const Operation& operation) { return operation.name == name; });
        return found == operations.end() ? -1 : found->index;
    }

    void dispatch_workers(int active, WorkerPool::Job job, void* context) {
        if (optimized_core && config.scheduler == SchedulerMode::safe) {
            if (optimized_core->dispatch_external(active, job, context) != 0) {
                throw std::runtime_error("resident worker dispatch failed");
            }
            return;
        }
        if (!pool) throw std::runtime_error("full worker pool is unavailable");
        pool->dispatch(active, job, context);
    }

    bool worker_affinity_ok() const noexcept {
        if (optimized_core && config.scheduler == SchedulerMode::safe) {
            return optimized_core->worker_affinity_ok();
        }
        return pool && pool->affinity_ok();
    }

    void run_optimized_core(stage49::SliceTiming* timing) {
        if (!optimized_core || core_input_tensor < 0 || core_output_tensor < 0) {
            throw std::runtime_error("optimized core is unavailable");
        }
        const Tensor& input = tensor(core_input_tensor);
        auto* input_data = arena.data() + input.offset;
        const int core_input = optimized_core->input_tensor_id();
        optimized_core->clear_external_tensor_bindings();
        if (std::getenv("Y26_STAGE53_CAPTURE_RESIDENT") != nullptr) {
            core_input_diagnostic_snapshot.assign(
                input_data, input_data + input.storage_bytes);
        }
        if (optimized_core->bind_external_arena(
                arena.data() + core_scratch_offset, core_scratch_bytes) != 0 ||
            optimized_core->tensor_bytes(core_input) != input.storage_bytes ||
            optimized_core->bind_external_tensor(
                core_input, input_data, input.storage_bytes) != 0) {
            throw std::runtime_error("optimized core input contract mismatch");
        }
        for (const CoreBridge& bridge : core_bridges) {
            if (bridge.full_tensor < 0 || bridge.core_tensor < 0 ||
                optimized_core->tensor_bytes(bridge.core_tensor) != tensor(bridge.full_tensor).storage_bytes) {
                throw std::runtime_error("optimized core live-out contract mismatch");
            }
            const Tensor& full = tensor(bridge.full_tensor);
            if (optimized_core->bind_external_tensor(
                    bridge.core_tensor, arena.data() + full.offset, full.storage_bytes) != 0) {
                throw std::runtime_error("optimized core live-out binding failed");
            }
        }
        stage49::RunOptions options;
        options.route = stage49::ComputeRoute::ime;
        options.kernel = stage49::KernelShape::m12n16;
        options.load = stage49::LoadStrategy::vlseg2_pair_vlse;
        options.epilogue = stage49::EpilogueStrategy::rvv_q62;
        options.partition = stage49::PartitionPolicy::spatial;
        options.nonconv = stage49::NonConvStrategy::explicit_rvv_lut;
        options.scheduler = stage49::SchedulerStrategy::active_workers_complete;
        options.workers = config.workers;
        const char* perf_group = std::getenv("Y26_STAGE55_PERF_GROUP");
        options.counter_collection_already_started = perf_group != nullptr && perf_group[0] != '\0';
        if (optimized_core->run_range_resident(0, optimized_core_last_operation, options, timing) != 0) {
            throw std::runtime_error("optimized core execution failed: " + optimized_core->last_error());
        }
        if (std::getenv("Y26_STAGE53_CAPTURE_RESIDENT") != nullptr) {
            for (CoreBridge& bridge : core_bridges) {
                const Tensor& full = tensor(bridge.full_tensor);
                bridge.diagnostic_snapshot.assign(
                    arena.data() + full.offset, arena.data() + full.offset + full.storage_bytes);
            }
        }
    }

    const Tensor& tensor(int id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= tensors.size()) throw std::runtime_error("invalid tensor id");
        return tensors[static_cast<std::size_t>(id)];
    }

    std::uint8_t code(int id, std::size_t flat) const noexcept {
        return semantic(arena, tensors[static_cast<std::size_t>(id)], flat);
    }

    void set_code(int id, std::size_t flat, std::uint8_t value) noexcept {
        store_semantic(arena, tensors[static_cast<std::size_t>(id)], flat, value);
    }

    void prepare_dense_conv(Operation& operation) {
        Conv& conv = operation.conv;
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        const bool supported_input_channels = conv.input_c % 8 == 0 || conv.input_c == 3;
        if (operation.kind != OpKind::conv_dense || conv.group != 1 || !supported_input_channels ||
            input.rank != 4 || output.rank != 4 || input.layout != Layout::feature_nchwc8 ||
            output.layout != Layout::feature_nchwc8) {
            return;
        }
        conv.input_blocks = (conv.input_c + 7) / 8;
        conv.k_tiles = conv.kernel_h * conv.kernel_w * conv.input_blocks;
        conv.n_blocks = (conv.output_c + kDenseN - 1) / kDenseN;
        std::vector<std::int8_t> recomputed_packed(
            static_cast<std::size_t>(conv.n_blocks) * conv.k_tiles * kDenseN * 8U, 0);
        std::vector<std::int64_t> recomputed_weight_sums(static_cast<std::size_t>(conv.output_c), 0);
        for (int output_channel = 0; output_channel < conv.output_c; ++output_channel) {
            for (int kernel_y = 0; kernel_y < conv.kernel_h; ++kernel_y) {
                for (int kernel_x = 0; kernel_x < conv.kernel_w; ++kernel_x) {
                    for (int input_channel = 0; input_channel < conv.input_c; ++input_channel) {
                        const std::size_t source =
                            (((static_cast<std::size_t>(output_channel) * conv.input_c + input_channel) *
                               conv.kernel_h + kernel_y) * conv.kernel_w + kernel_x);
                        const int tile = (kernel_y * conv.kernel_w + kernel_x) * conv.input_blocks + input_channel / 8;
                        const std::size_t destination =
                            (((static_cast<std::size_t>(output_channel / kDenseN) * conv.k_tiles + tile) *
                               kDenseN + output_channel % kDenseN) * 8U + input_channel % 8);
                        const std::int8_t weight = conv.weights[source];
                        recomputed_packed[destination] = weight;
                        recomputed_weight_sums[static_cast<std::size_t>(output_channel)] += weight;
                    }
                }
            }
        }
        if (conv.packed_weights != recomputed_packed || conv.weight_sums != recomputed_weight_sums ||
            conv.corrected_bias.size() != static_cast<std::size_t>(conv.output_c) ||
            conv.multiplier_m63.size() != static_cast<std::size_t>(conv.output_c)) {
            throw std::runtime_error("offline dense Conv package asset mismatch");
        }
        const std::int64_t correction = 128 - static_cast<std::int64_t>(input.zero_point);
        bool all_e2c = conv.e2c_compatible;
        for (int channel = 0; channel < conv.output_c; ++channel) {
            const std::int64_t expected_corrected_bias =
                conv.bias[static_cast<std::size_t>(channel)] +
                correction * recomputed_weight_sums[static_cast<std::size_t>(channel)];
            if (conv.corrected_bias[static_cast<std::size_t>(channel)] != expected_corrected_bias) {
                throw std::runtime_error("offline corrected bias mismatch");
            }
            const std::int64_t multiplier = conv.multiplier[static_cast<std::size_t>(channel)];
            if (conv.shift[static_cast<std::size_t>(channel)] != 62 || multiplier <= 0 ||
                multiplier > std::numeric_limits<std::int64_t>::max() / 2) {
                all_e2c = false;
                if (conv.multiplier_m63[static_cast<std::size_t>(channel)] != 0) {
                    throw std::runtime_error("offline M63 fallback asset mismatch");
                }
            } else {
                if (conv.multiplier_m63[static_cast<std::size_t>(channel)] != multiplier * 2) {
                    throw std::runtime_error("offline M63 asset mismatch");
                }
            }
        }
        if (!all_e2c) throw std::runtime_error("dense Conv is not fully E2c compatible");
        conv.dense_ime_eligible = true;
        if (conv.input_c == 3 && conv.output_c == 16 && conv.kernel_h == 3 &&
            conv.kernel_w == 3 && conv.stride_h == 2 && conv.stride_w == 2) {
            conv.stem_weights_tap_major.resize(27U * 16U);
            conv.stem_corrected_bias_i32.resize(16U);
            for (int channel = 0; channel < 16; ++channel) {
                const std::int64_t corrected = conv.corrected_bias[static_cast<std::size_t>(channel)];
                if (corrected < std::numeric_limits<std::int32_t>::min() ||
                    corrected > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("RGB stem corrected bias exceeds int32");
                }
                conv.stem_corrected_bias_i32[static_cast<std::size_t>(channel)] =
                    static_cast<std::int32_t>(corrected);
                for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                    for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                        for (int input_channel = 0; input_channel < 3; ++input_channel) {
                            const int tap = (kernel_y * 3 + kernel_x) * 3 + input_channel;
                            const std::size_t source_index =
                                (((static_cast<std::size_t>(channel) * 3U + input_channel) * 3U +
                                  kernel_y) * 3U + kernel_x);
                            conv.stem_weights_tap_major[
                                static_cast<std::size_t>(tap) * 16U + channel] =
                                conv.weights[source_index];
                        }
                    }
                }
            }
            conv.rgb_stem_rvv_eligible = true;
        }
    }

    void prepare_depthwise_conv(Operation& operation) {
        Conv& conv = operation.conv;
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        if (operation.kind != OpKind::conv_grouped || conv.group != conv.input_c ||
            conv.group != conv.output_c || conv.input_c % 8 != 0 ||
            conv.kernel_h != 3 || conv.kernel_w != 3 ||
            conv.stride_h != 1 || conv.stride_w != 1 ||
            input.rank != 4 || output.rank != 4 ||
            input.layout != Layout::feature_nchwc8 ||
            output.layout != Layout::feature_nchwc8) {
            return;
        }
        if (conv.accumulator_bound > static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("depthwise accumulator exceeds int32");
        }
        conv.depthwise_weights_c8.resize(conv.weights.size());
        conv.depthwise_corrected_bias_i32.resize(static_cast<std::size_t>(conv.output_c));
        const int channel_blocks = conv.output_c / 8;
        for (int channel_block = 0; channel_block < channel_blocks; ++channel_block) {
            for (int tap = 0; tap < 9; ++tap) {
                for (int lane = 0; lane < 8; ++lane) {
                    const int channel = channel_block * 8 + lane;
                    conv.depthwise_weights_c8[
                        (static_cast<std::size_t>(channel_block) * 9U + tap) * 8U + lane] =
                        conv.weights[static_cast<std::size_t>(channel) * 9U + tap];
                }
            }
        }
        conv.multiplier_m63.resize(static_cast<std::size_t>(conv.output_c));
        const std::int64_t input_correction = 128 - static_cast<std::int64_t>(input.zero_point);
        for (int channel = 0; channel < conv.output_c; ++channel) {
            std::int64_t weight_sum = 0;
            for (int tap = 0; tap < 9; ++tap) {
                weight_sum += conv.weights[static_cast<std::size_t>(channel) * 9U + tap];
            }
            const std::int64_t corrected_bias =
                conv.bias[static_cast<std::size_t>(channel)] + input_correction * weight_sum;
            if (corrected_bias < std::numeric_limits<std::int32_t>::min() ||
                corrected_bias > std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error("depthwise corrected bias exceeds int32");
            }
            conv.depthwise_corrected_bias_i32[static_cast<std::size_t>(channel)] =
                static_cast<std::int32_t>(corrected_bias);
            const std::int64_t multiplier = conv.multiplier[static_cast<std::size_t>(channel)];
            if (conv.shift[static_cast<std::size_t>(channel)] != 62 || multiplier <= 0 ||
                multiplier > std::numeric_limits<std::int64_t>::max() / 2) {
                throw std::runtime_error("depthwise Conv is not Q62 vsmul-compatible");
            }
            conv.multiplier_m63[static_cast<std::size_t>(channel)] = multiplier * 2;
        }
        conv.depthwise_rvv_eligible = true;
    }

    void pack_dense_a(const Conv& conv, const Tensor& input, const std::int8_t* input_data,
                      int output_w, int m_begin, int valid_rows, int m_block,
                      std::int8_t* panel) const noexcept {
        const int input_blocks = conv.input_blocks;
        const std::int8_t padding = int8_v1::signed_storage(static_cast<std::uint8_t>(input.zero_point));
        for (int tile = 0; tile < conv.k_tiles; ++tile) {
            const int channel_block = tile % input_blocks;
            const int kernel_position = tile / input_blocks;
            const int kernel_y = kernel_position / conv.kernel_w;
            const int kernel_x = kernel_position % conv.kernel_w;
            for (int group = 0; group < m_block; group += 4) {
                std::int8_t* destination =
                    panel + (static_cast<std::size_t>(tile) * m_block + group) * 8U;
                const int flat = m_begin + group;
                const int output_y = flat / output_w;
                const int output_x = flat % output_w;
                const int input_y = output_y * conv.stride_h - conv.pad_top + kernel_y;
                const int input_x = output_x * conv.stride_w - conv.pad_left + kernel_x;
                const bool complete = group + 4 <= valid_rows && output_x + 3 < output_w &&
                    input_y >= 0 && input_y < input.dims[2] && input_x >= 0 &&
                    input_x + 3 * conv.stride_w < input.dims[3] &&
                    (conv.stride_w == 1 || conv.stride_w == 2);
                if (complete) {
                    const std::size_t source =
                        (((static_cast<std::size_t>(channel_block) * input.dims[2] + input_y) *
                           input.dims[3] + input_x) * 8U);
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
                    if (conv.stride_w == 1) {
                        y26_stage49_load_contiguous_c8x4(input_data + source, destination);
                    } else {
                        y26_stage48_load_vlse64_4(input_data + source, destination);
                    }
#else
                    for (int row = 0; row < 4; ++row) {
                        std::memcpy(destination + row * 8,
                                    input_data + source + static_cast<std::size_t>(row * conv.stride_w) * 8U, 8);
                    }
#endif
                    continue;
                }
                for (int row = 0; row < 4; ++row) {
                    std::int8_t* row_destination = destination + row * 8;
                    if (group + row >= valid_rows) {
                        std::fill(row_destination, row_destination + 8, padding);
                        continue;
                    }
                    const int row_flat = m_begin + group + row;
                    const int row_output_y = row_flat / output_w;
                    const int row_output_x = row_flat % output_w;
                    const int row_input_y = row_output_y * conv.stride_h - conv.pad_top + kernel_y;
                    const int row_input_x = row_output_x * conv.stride_w - conv.pad_left + kernel_x;
                    if (row_input_y < 0 || row_input_y >= input.dims[2] ||
                        row_input_x < 0 || row_input_x >= input.dims[3]) {
                        std::fill(row_destination, row_destination + 8, padding);
                    } else {
                        const std::size_t source =
                            (((static_cast<std::size_t>(channel_block) * input.dims[2] + row_input_y) *
                               input.dims[3] + row_input_x) * 8U);
                        std::memcpy(row_destination, input_data + source, 8);
                    }
                }
            }
        }
    }

    void pack_dense_a_p3(const Conv& conv, const Tensor& input,
                         const std::int8_t* input_data, int output_w,
                         int m_begin, int valid_rows, int m_block,
                         std::int8_t* panel) const noexcept {
        if (!dense_pack_rvv_enabled || m_block != kDenseM || conv.kernel_h != 3 ||
            conv.kernel_w != 3 || conv.stride_h != 2 || conv.stride_w != 2) {
            pack_dense_a(conv, input, input_data, output_w, m_begin, valid_rows, m_block, panel);
            return;
        }
        for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
            for (int channel_block = 0; channel_block < conv.input_blocks; ++channel_block) {
                const int tile0 = kernel_y * 3 * conv.input_blocks + channel_block;
                const int tile1 = tile0 + conv.input_blocks;
                const int tile2 = tile1 + conv.input_blocks;
                for (int group = 0; group < m_block; group += 4) {
                    const int flat = m_begin + group;
                    const int output_y = flat / output_w;
                    const int output_x = flat % output_w;
                    const int input_y = output_y * 2 - conv.pad_top + kernel_y;
                    const int input_x = output_x * 2 - conv.pad_left;
                    const bool complete = group + 4 <= valid_rows &&
                        output_x + 3 < output_w && input_y >= 0 && input_y < input.dims[2] &&
                        input_x >= 0 && input_x + 8 < input.dims[3];
                    if (!complete) {
                        pack_dense_a(conv, input, input_data, output_w, m_begin,
                                     valid_rows, m_block, panel);
                        return;
                    }
                    const std::size_t source_offset =
                        ((static_cast<std::size_t>(channel_block) * input.dims[2] + input_y) *
                         input.dims[3] + input_x) * 8U;
                    const auto* source = input_data + source_offset;
                    auto* destination0 = panel +
                        (static_cast<std::size_t>(tile0) * m_block + group) * 8U;
                    auto* destination1 = panel +
                        (static_cast<std::size_t>(tile1) * m_block + group) * 8U;
                    auto* destination2 = panel +
                        (static_cast<std::size_t>(tile2) * m_block + group) * 8U;
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
                    y26_stage49_load_vlseg2_pair_4(source, destination0, destination1);
                    y26_stage48_load_vlse64_4(source + 16, destination2);
#else
                    for (int row = 0; row < 4; ++row) {
                        std::memcpy(destination0 + row * 8, source + row * 16, 8);
                        std::memcpy(destination1 + row * 8, source + row * 16 + 8, 8);
                        std::memcpy(destination2 + row * 8, source + row * 16 + 16, 8);
                    }
#endif
                }
            }
        }
    }

    struct ConvJob { Impl* self; const Operation* op; std::atomic<int> status {0}; };
    enum class RangeKind { input_quant, input_rgb, lut, transform, concat_branch, resize, matmul, softmax };
    struct RangeJob {
        Impl* self = nullptr;
        const Operation* op = nullptr;
        const Branch* branch = nullptr;
        const float* input = nullptr;
        const std::uint8_t* rgb = nullptr;
        int rgb_stride = 0;
        std::size_t total = 0;
        int axis_offset = 0;
        RangeKind kind = RangeKind::lut;
        std::atomic<int> status {0};
    };

    struct StaticBarrier {
        std::atomic<int> arrivals {0};
        std::atomic<std::uint64_t> epoch {0};

        void wait(int workers) noexcept {
            const std::uint64_t observed = epoch.load(std::memory_order_acquire);
            if (arrivals.fetch_add(1, std::memory_order_acq_rel) + 1 == workers) {
                arrivals.store(0, std::memory_order_relaxed);
                epoch.fetch_add(1, std::memory_order_release);
                return;
            }
            unsigned spins = 0;
            while (epoch.load(std::memory_order_acquire) == observed) {
                if ((++spins & 4095U) == 0U) std::this_thread::yield();
            }
        }
    };

    struct StaticScheduleJob {
        Impl* self = nullptr;
        std::size_t begin = 0;
        std::size_t end = 0;
        const float* input = nullptr;
        const std::uint8_t* rgb = nullptr;
        int rgb_stride = 0;
        StaticBarrier barrier;
        std::atomic<int> status {0};
    };

    static bool static_schedule_eligible(OpKind kind) noexcept {
        switch (kind) {
            case OpKind::input_quant:
            case OpKind::conv_dense:
            case OpKind::conv_grouped:
            case OpKind::lut1:
            case OpKind::lut2:
            case OpKind::split:
            case OpKind::reshape:
            case OpKind::transpose:
            case OpKind::reshape_split_transpose:
            case OpKind::resize:
                return true;
            case OpKind::concat:
            case OpKind::matmul:
            case OpKind::softmax_transpose:
                return false;
        }
        return false;
    }

    std::size_t static_range_total(const Operation& operation) const {
        const Tensor& output = tensor(operation.output);
        if (operation.kind == OpKind::input_quant) {
            return output.layout == Layout::feature_nchwc8 && output.rank == 4 &&
                    output.dims[0] == 1 && output.dims[1] <= 8
                ? static_cast<std::size_t>(output.dims[2]) * output.dims[3]
                : output.logical_elements;
        }
        if (operation.kind == OpKind::lut1) {
            const Tensor& input = tensor(operation.inputs[0]);
            return input.layout == output.layout && input.storage_bytes == output.storage_bytes
                ? output.storage_bytes : output.logical_elements;
        }
        if (operation.kind == OpKind::lut2) {
            const Tensor& left = tensor(operation.inputs[0]);
            const Tensor& right = tensor(operation.inputs[1]);
            const bool raw = left.layout == Layout::feature_nchwc8 &&
                right.layout == left.layout && output.layout == left.layout &&
                left.rank == output.rank && right.rank == output.rank &&
                left.dims == output.dims && right.dims == output.dims &&
                left.storage_bytes == output.storage_bytes &&
                right.storage_bytes == output.storage_bytes;
            return raw ? output.storage_bytes : output.logical_elements;
        }
        if (operation.kind == OpKind::split || operation.kind == OpKind::reshape ||
            operation.kind == OpKind::transpose ||
            operation.kind == OpKind::reshape_split_transpose) {
            const Tensor& input = tensor(operation.inputs[0]);
            const bool raw_split = operation.kind == OpKind::split && operation.axis == 1 &&
                input.layout == Layout::feature_nchwc8 && output.layout == input.layout &&
                input.rank == 4 && output.rank == 4 && operation.split_offset % 8 == 0 &&
                input.dims[2] == output.dims[2] && input.dims[3] == output.dims[3];
            const bool raw_reshape = operation.kind == OpKind::reshape &&
                input.layout == output.layout && input.storage_bytes == output.storage_bytes;
            return raw_split || raw_reshape ? output.storage_bytes : output.logical_elements;
        }
        return output.logical_elements;
    }

    void run_static_operation_chunk(const Operation& operation, const float* input,
                                    const std::uint8_t* rgb, int rgb_stride,
                                    int worker, int workers) {
        if (operation.skip_when_fused && fused_lut_enabled && !config.capture_boundaries) return;
        if (operation.kind == OpKind::input_quant) {
            if (input_stem_fused_enabled && !config.capture_boundaries && input != nullptr) return;
            const std::size_t total = static_range_total(operation);
            const std::size_t begin = total * static_cast<std::size_t>(worker) /
                                      static_cast<std::size_t>(workers);
            const std::size_t end = total * static_cast<std::size_t>(worker + 1) /
                                    static_cast<std::size_t>(workers);
            if (rgb == nullptr) run_input_chunk(operation, input, begin, end);
            else run_input_rgb_chunk(operation, rgb, rgb_stride, begin, end);
            return;
        }
        if (operation.kind == OpKind::conv_dense || operation.kind == OpKind::conv_grouped) {
            if (config.compute == ComputeMode::optimized && rgb_stem_enabled &&
                operation.conv.rgb_stem_rvv_eligible) {
                run_rgb_stem_chunk(operation, worker, workers);
            } else if (config.compute == ComputeMode::optimized && operation.conv.dense_ime_eligible) {
                run_dense_conv_chunk(operation, worker, workers);
            } else if (config.compute == ComputeMode::optimized &&
                       operation.kind == OpKind::conv_grouped &&
                       operation.conv.group == operation.conv.input_c &&
                       operation.conv.group == operation.conv.output_c) {
                run_depthwise_conv_chunk(operation, worker, workers);
            } else {
                run_conv_chunk(operation, worker, workers);
            }
            return;
        }
        const std::size_t total = static_range_total(operation);
        const std::size_t begin = total * static_cast<std::size_t>(worker) /
                                  static_cast<std::size_t>(workers);
        const std::size_t end = total * static_cast<std::size_t>(worker + 1) /
                                static_cast<std::size_t>(workers);
        switch (operation.kind) {
            case OpKind::lut1:
            case OpKind::lut2: run_lut_chunk(operation, begin, end); break;
            case OpKind::split:
            case OpKind::reshape:
            case OpKind::transpose:
            case OpKind::reshape_split_transpose:
                run_transform_chunk(operation, begin, end);
                break;
            case OpKind::resize: run_resize_chunk(operation, begin, end); break;
            default: throw std::runtime_error("unsupported prepared-schedule operation");
        }
    }

    static void static_schedule_job(void* opaque, int worker, int workers) {
        auto& job = *static_cast<StaticScheduleJob*>(opaque);
        for (std::size_t index = job.begin; index <= job.end; ++index) {
            if (job.status.load(std::memory_order_relaxed) == 0) {
                try {
                    job.self->run_static_operation_chunk(
                        job.self->operations[index], job.input, job.rgb, job.rgb_stride,
                        worker, workers);
                } catch (...) {
                    job.status.store(1, std::memory_order_relaxed);
                }
            }
            job.barrier.wait(workers);
        }
    }

    void dispatch_static_batch(std::size_t begin, std::size_t end, const float* input,
                               const std::uint8_t* rgb, int rgb_stride) {
        StaticScheduleJob job;
        job.self = this;
        job.begin = begin;
        job.end = end;
        job.input = input;
        job.rgb = rgb;
        job.rgb_stride = rgb_stride;
        dispatch_workers(config.workers, static_schedule_job, &job);
        if (job.status.load(std::memory_order_relaxed) != 0) {
            throw std::runtime_error("prepared static schedule worker failed");
        }
    }

    static void range_job(void* opaque, int worker, int workers) {
        auto& job = *static_cast<RangeJob*>(opaque);
        if (job.status.load(std::memory_order_relaxed) != 0) return;
        try {
            const std::size_t begin = job.total * static_cast<std::size_t>(worker) /
                                      static_cast<std::size_t>(workers);
            const std::size_t end = job.total * static_cast<std::size_t>(worker + 1) /
                                    static_cast<std::size_t>(workers);
            switch (job.kind) {
                case RangeKind::input_quant: job.self->run_input_chunk(*job.op, job.input, begin, end); break;
                case RangeKind::input_rgb:
                    job.self->run_input_rgb_chunk(*job.op, job.rgb, job.rgb_stride, begin, end);
                    break;
                case RangeKind::lut: job.self->run_lut_chunk(*job.op, begin, end); break;
                case RangeKind::transform: job.self->run_transform_chunk(*job.op, begin, end); break;
                case RangeKind::concat_branch:
                    job.self->run_concat_branch_chunk(*job.op, *job.branch, job.axis_offset, begin, end);
                    break;
                case RangeKind::resize: job.self->run_resize_chunk(*job.op, begin, end); break;
                case RangeKind::matmul: job.self->run_matmul_chunk(*job.op, begin, end); break;
                case RangeKind::softmax: job.self->run_softmax_chunk(*job.op, begin, end); break;
            }
        } catch (...) {
            job.status.store(1, std::memory_order_relaxed);
        }
    }

    void dispatch_range(RangeJob& job) {
        dispatch_workers(config.workers, range_job, &job);
        if (job.status.load(std::memory_order_relaxed) != 0) {
            throw std::runtime_error("parallel operator worker failed");
        }
    }

    static void conv_job(void* opaque, int worker, int workers) {
        auto& job = *static_cast<ConvJob*>(opaque);
        if (job.status.load(std::memory_order_relaxed) != 0) return;
        try {
            if (job.self->config.compute == ComputeMode::optimized &&
                job.self->rgb_stem_enabled && job.op->conv.rgb_stem_rvv_eligible) {
                job.self->run_rgb_stem_chunk(*job.op, worker, workers);
            } else if (job.self->config.compute == ComputeMode::optimized &&
                       job.op->conv.dense_ime_eligible) {
                job.self->run_dense_conv_chunk(*job.op, worker, workers);
            } else if (job.self->config.compute == ComputeMode::optimized &&
                       job.op->kind == OpKind::conv_grouped &&
                       job.op->conv.group == job.op->conv.input_c &&
                       job.op->conv.group == job.op->conv.output_c) {
                job.self->run_depthwise_conv_chunk(*job.op, worker, workers);
            } else {
                job.self->run_conv_chunk(*job.op, worker, workers);
            }
        } catch (...) {
            job.status.store(1, std::memory_order_relaxed);
        }
    }

    void run_rgb_stem_chunk(const Operation& operation, int worker, int workers) {
        const Conv& conv = operation.conv;
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        const auto* input_data = arena.data() + input.offset;
        auto* output_data = arena.data() + output.offset;
        const int output_h = output.dims[2];
        const int output_w = output.dims[3];
        const int input_h = input.dims[2];
        const int input_w = input.dims[3];
        const std::size_t input_pixel_stride =
            input_compact_c3_enabled && !config.capture_boundaries ? 3U : 8U;
        const bool fused_float_input = input_stem_fused_enabled && !config.capture_boundaries &&
            current_float_input != nullptr;
        const std::size_t input_plane = static_cast<std::size_t>(input_h) * input_w;
        const int total = output_h * output_w;
        const int begin = total * worker / workers;
        const int end = total * (worker + 1) / workers;
        const std::int8_t padding = int8_v1::signed_storage(
            static_cast<std::uint8_t>(input.zero_point));
        stage51::VectorFixedPointState vector_state;
        if (!stage51::begin_q62_vector_rne(&vector_state)) {
            throw std::runtime_error("cannot establish RGB stem Q62 vector state");
        }
        alignas(64) std::array<std::int32_t, 16> accumulator {};
        alignas(32) std::array<std::int64_t, 4> corrected {};
        alignas(32) std::array<std::int64_t, 4> multipliers {};
        for (int spatial = begin; spatial < end; ++spatial) {
            const int output_y = spatial / output_w;
            const int output_x = spatial % output_w;
#if defined(__riscv_vector)
            constexpr std::size_t kChannels = 16;
            vint32m2_t sum = __riscv_vle32_v_i32m2(
                conv.stem_corrected_bias_i32.data(), kChannels);
            for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                const int input_y = output_y * 2 - conv.pad_top + kernel_y;
                for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                    const int input_x = output_x * 2 - conv.pad_left + kernel_x;
                    for (int input_channel = 0; input_channel < 3; ++input_channel) {
                        const bool valid = input_y >= 0 && input_y < input_h &&
                            input_x >= 0 && input_x < input_w;
                        std::int8_t input_value = padding;
                        if (valid) {
                            input_value = fused_float_input
                                ? quantize_input_f32(current_float_input[
                                      static_cast<std::size_t>(input_channel) * input_plane +
                                      static_cast<std::size_t>(input_y) * input_w + input_x])
                                : input_data[(static_cast<std::size_t>(input_y) * input_w + input_x) *
                                                 input_pixel_stride +
                                             static_cast<std::size_t>(input_channel)];
                        }
                        const int tap = (kernel_y * 3 + kernel_x) * 3 + input_channel;
                        const vint8mf2_t weight_i8 = __riscv_vle8_v_i8mf2(
                            conv.stem_weights_tap_major.data() + static_cast<std::size_t>(tap) * 16U,
                            kChannels);
                        const vint32m2_t weight_i32 =
                            __riscv_vsext_vf4_i32m2(weight_i8, kChannels);
                        sum = __riscv_vmacc_vx_i32m2(
                            sum, static_cast<std::int32_t>(input_value), weight_i32, kChannels);
                    }
                }
            }
            __riscv_vse32_v_i32m2(accumulator.data(), sum, kChannels);
#else
            for (int output_channel = 0; output_channel < 16; ++output_channel) {
                std::int64_t sum = conv.stem_corrected_bias_i32[
                    static_cast<std::size_t>(output_channel)];
                for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                    const int input_y = output_y * 2 - conv.pad_top + kernel_y;
                    for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                        const int input_x = output_x * 2 - conv.pad_left + kernel_x;
                        for (int input_channel = 0; input_channel < 3; ++input_channel) {
                            const bool valid = input_y >= 0 && input_y < input_h &&
                                input_x >= 0 && input_x < input_w;
                            std::int8_t input_value = padding;
                            if (valid) {
                                input_value = fused_float_input
                                    ? quantize_input_f32(current_float_input[
                                          static_cast<std::size_t>(input_channel) * input_plane +
                                          static_cast<std::size_t>(input_y) * input_w + input_x])
                                    : input_data[(static_cast<std::size_t>(input_y) * input_w + input_x) *
                                                     input_pixel_stride +
                                                 static_cast<std::size_t>(input_channel)];
                            }
                            const int tap = (kernel_y * 3 + kernel_x) * 3 + input_channel;
                            sum += static_cast<std::int32_t>(input_value) *
                                conv.stem_weights_tap_major[
                                    static_cast<std::size_t>(tap) * 16U + output_channel];
                        }
                    }
                }
                accumulator[static_cast<std::size_t>(output_channel)] =
                    static_cast<std::int32_t>(sum);
            }
#endif
            for (int group = 0; group < 4; ++group) {
                for (int lane = 0; lane < 4; ++lane) {
                    const int channel = group * 4 + lane;
                    corrected[static_cast<std::size_t>(lane)] =
                        accumulator[static_cast<std::size_t>(channel)];
                    multipliers[static_cast<std::size_t>(lane)] =
                        conv.multiplier_m63[static_cast<std::size_t>(channel)];
                }
                const std::size_t destination =
                    ((static_cast<std::size_t>((group * 4) / 8) * total + spatial) * 8U +
                     static_cast<std::size_t>((group * 4) % 8));
                stage51::q62_vsmul_m63_i64x4_to_s8(
                    corrected.data(), multipliers.data(), output.zero_point,
                    output_data + destination);
            }
        }
        const auto result = stage51::end_q62_vector_rne(&vector_state);
        if (!result.restored || result.saturated) {
            throw std::runtime_error("RGB stem vector state restoration failed");
        }
    }

    void store_dense_block(const Operation& operation, const Conv& conv,
                           const Tensor& output, std::int8_t* output_data,
                           DenseScratch& scratch, int m_block, int m_begin,
                           int valid_rows, int n_block, bool fuse_lut) {
        const int output_m = output.dims[2] * output.dims[3];
        const int row_groups = m_block / 4;
        for (int output_group = 0; output_group < 4;) {
            const int channel_begin = n_block * kDenseN + output_group * 4;
            if (channel_begin >= conv.output_c) break;
            const int paired_channels = std::min(8, conv.output_c - channel_begin);
            if (e2c3_enabled && conv.e2c_compatible && e2c2_enabled &&
                paired_channels == 8 && output_group + 1 < 4) {
                for (int row = 0; row < valid_rows; ++row) {
                    const int row_group = row / 4;
                    const int row_inner = row % 4;
                    const std::int32_t* raw_low = scratch.c.data() +
                        (output_group * row_groups + row_group) * 16 + row_inner * 4;
                    const std::int32_t* raw_high = scratch.c.data() +
                        ((output_group + 1) * row_groups + row_group) * 16 + row_inner * 4;
                    const std::size_t destination =
                        ((static_cast<std::size_t>(channel_begin / 8) * output_m +
                          m_begin + row) * 8U);
                    if (e2c4_enabled) {
                        if (fuse_lut) {
                            stage51::q62_e2c4_i32x4x2_bias_lut_to_s8(
                                raw_low, raw_high,
                                conv.corrected_bias.data() + channel_begin,
                                conv.multiplier_m63.data() + channel_begin,
                                output.zero_point, operation.fused_lut.data(),
                                output_data + destination);
                        } else {
                            stage51::q62_e2c4_i32x4x2_bias_to_s8(
                                raw_low, raw_high,
                                conv.corrected_bias.data() + channel_begin,
                                conv.multiplier_m63.data() + channel_begin,
                                output.zero_point, output_data + destination);
                        }
                    } else {
                        alignas(64) std::array<std::int64_t, 8> corrected {};
                        for (int lane = 0; lane < 4; ++lane) {
                            corrected[static_cast<std::size_t>(lane)] =
                                static_cast<std::int64_t>(raw_low[lane]) +
                                conv.corrected_bias[static_cast<std::size_t>(channel_begin + lane)];
                            corrected[static_cast<std::size_t>(lane + 4)] =
                                static_cast<std::int64_t>(raw_high[lane]) +
                                conv.corrected_bias[static_cast<std::size_t>(channel_begin + lane + 4)];
                        }
                        if (fuse_lut) {
                            stage51::q62_vsmul_m63_i64x8_lut_to_s8(
                                corrected.data(), conv.multiplier_m63.data() + channel_begin,
                                output.zero_point, operation.fused_lut.data(),
                                output_data + destination);
                        } else {
                            stage51::q62_vsmul_m63_i64x8_to_s8(
                                corrected.data(), conv.multiplier_m63.data() + channel_begin,
                                output.zero_point, output_data + destination);
                        }
                    }
                }
                output_group += 2;
                continue;
            }
            const int valid_channels = std::min(4, conv.output_c - channel_begin);
            for (int row = 0; row < valid_rows; ++row) {
                const int row_group = row / 4;
                const int row_inner = row % 4;
                const std::int32_t* raw = scratch.c.data() +
                    (output_group * row_groups + row_group) * 16 + row_inner * 4;
                alignas(32) std::array<std::int64_t, 4> corrected {};
                alignas(32) std::array<std::int64_t, 4> multipliers {};
                alignas(32) std::array<std::int64_t, 4> rounded {};
                for (int lane = 0; lane < valid_channels; ++lane) {
                    const int channel = channel_begin + lane;
                    corrected[static_cast<std::size_t>(lane)] =
                        static_cast<std::int64_t>(raw[lane]) +
                        conv.corrected_bias[static_cast<std::size_t>(channel)];
                    multipliers[static_cast<std::size_t>(lane)] =
                        conv.multiplier_m63[static_cast<std::size_t>(channel)];
                }
                if (conv.e2c_compatible && valid_channels == 4) {
                    if (e2c2_enabled) {
                        const std::size_t destination =
                            ((static_cast<std::size_t>(channel_begin / 8) * output_m +
                              m_begin + row) * 8U + static_cast<std::size_t>(channel_begin % 8));
                        if (fuse_lut) {
                            alignas(8) std::array<std::int8_t, 4> quantized_storage {};
                            stage51::q62_vsmul_m63_i64x4_to_s8(
                                corrected.data(), multipliers.data(), output.zero_point,
                                quantized_storage.data());
                            for (int lane = 0; lane < 4; ++lane) {
                                output_data[destination + static_cast<std::size_t>(lane)] =
                                    operation.fused_lut[int8_v1::semantic_code(
                                        quantized_storage[static_cast<std::size_t>(lane)])];
                            }
                        } else {
                            stage51::q62_vsmul_m63_i64x4_to_s8(
                                corrected.data(), multipliers.data(), output.zero_point,
                                output_data + destination);
                        }
                        continue;
                    }
                    stage51::q62_vsmul_m63_i64x4(
                        corrected.data(), multipliers.data(), rounded.data());
                }
                for (int lane = 0; lane < valid_channels; ++lane) {
                    const int channel = channel_begin + lane;
                    std::uint8_t quantized = 0;
                    if (conv.e2c_compatible && valid_channels == 4) {
                        quantized = static_cast<std::uint8_t>(std::clamp<std::int64_t>(
                            rounded[static_cast<std::size_t>(lane)] + output.zero_point, 0, 255));
                    } else {
                        const int8_v1::RequantAsset asset {
                            conv.multiplier[static_cast<std::size_t>(channel)],
                            conv.shift[static_cast<std::size_t>(channel)], output.zero_point, 0, 255,
                        };
                        if (!int8_v1::requantize_u8(
                                corrected[static_cast<std::size_t>(lane)], asset, &quantized)) {
                            throw std::runtime_error("dense Conv requantization failed");
                        }
                    }
                    const std::size_t destination =
                        ((static_cast<std::size_t>(channel / 8) * output_m + m_begin + row) * 8U +
                         static_cast<std::size_t>(channel % 8));
                    output_data[destination] = fuse_lut
                        ? operation.fused_lut[quantized]
                        : int8_v1::signed_storage(quantized);
                }
            }
            ++output_group;
        }
    }

    void run_dense_conv_chunk(const Operation& operation, int worker, int workers) {
        const Conv& conv = operation.conv;
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        const bool fuse_lut = fused_lut_enabled && !config.capture_boundaries &&
            operation.fused_lut_output >= 0;
        const Tensor& store_output = fuse_lut ? tensor(operation.fused_lut_output) : output;
        const auto* input_data = arena.data() + input.offset;
        auto* output_data = arena.data() + store_output.offset;
        DenseScratch& scratch = dense_scratch[static_cast<std::size_t>(worker)];
        const int output_m = output.dims[2] * output.dims[3];
        const bool use_m8 = (dense_m8_enabled || conv.stage55_family_b_m8) &&
            conv.output_c % kDenseN == 0;
        const int m_block = use_m8 ? 8 : kDenseM;
        const int tiles = (output_m + m_block - 1) / m_block;
        int tile_begin = tiles * worker / workers;
        int tile_end = tiles * (worker + 1) / workers;
        int n_begin = 0;
        int n_end = conv.n_blocks;
        if (dense_partition == 1) {
            tile_begin = 0;
            tile_end = tiles;
            n_begin = conv.n_blocks * worker / workers;
            n_end = conv.n_blocks * (worker + 1) / workers;
        } else if (dense_partition == 2 && workers >= 4) {
            const int spatial_group = worker / 2;
            const int output_group = worker % 2;
            tile_begin = tiles * spatial_group / 2;
            tile_end = tiles * (spatial_group + 1) / 2;
            n_begin = conv.n_blocks * output_group / 2;
            n_end = conv.n_blocks * (output_group + 1) / 2;
        }
        stage51::VectorFixedPointState vector_state;
        if (conv.e2c_compatible && !stage51::begin_q62_vector_rne(&vector_state)) {
            throw std::runtime_error("cannot establish Q62 vector state");
        }
        const bool direct_1x1_shape = direct_1x1_enabled && m_block == kDenseM &&
            conv.kernel_h == 1 && conv.kernel_w == 1 && conv.stride_h == 1 &&
            conv.stride_w == 1 && conv.pad_top == 0 && conv.pad_left == 0 &&
            conv.input_c % 8 == 0 && conv.output_c % kDenseN == 0 &&
            input.dims[2] == output.dims[2] && input.dims[3] == output.dims[3];
        const auto execute_block = [&](int m_begin, int valid_rows,
                                       [[maybe_unused]] bool direct_1x1_tile, int n_block) {
            const std::int8_t* packed = conv.packed_weights.data() +
                static_cast<std::size_t>(n_block) * conv.k_tiles * kDenseN * 8U;
            const int live_channels = std::min(kDenseN, conv.output_c - n_block * kDenseN);
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
            if (direct_1x1_tile) {
                const auto channel_block_stride = static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(input.dims[2]) * input.dims[3] * 8U);
                y26_stage54_kernel_direct_1x1_m12n16(
                    input_data + static_cast<std::size_t>(m_begin) * 8U,
                    channel_block_stride, packed, conv.k_tiles, scratch.c.data());
            } else
#endif
            if (m_block == 8) {
                run_m8n16(scratch.a.data(), packed, conv.k_tiles, scratch.c.data());
            } else if (small_n_enabled) {
                run_m12n_live(
                    scratch.a.data(), packed, conv.k_tiles, live_channels, scratch.c.data());
            } else {
                run_m12n16(scratch.a.data(), packed, conv.k_tiles, scratch.c.data());
            }
            store_dense_block(operation, conv, output, output_data, scratch,
                              m_block, m_begin, valid_rows, n_block, fuse_lut);
        };
        const bool use_weight_stationary =
            dense_weight_stationary_enabled || conv.stage55_family_a_weight_stationary;
        if (use_weight_stationary && direct_1x1_shape) {
            for (int n_block = n_begin; n_block < n_end; ++n_block) {
                for (int tile = tile_begin; tile < tile_end; ++tile) {
                    const int m_begin = tile * m_block;
                    const int valid_rows = std::min(m_block, output_m - m_begin);
                    const bool direct_1x1_tile = valid_rows == kDenseM;
                    if (!direct_1x1_tile) {
                        pack_dense_a_p3(conv, input, input_data, output.dims[3], m_begin,
                                        valid_rows, m_block, scratch.a.data());
                    }
                    execute_block(m_begin, valid_rows, direct_1x1_tile, n_block);
                }
            }
        } else {
            for (int tile = tile_begin; tile < tile_end; ++tile) {
                const int m_begin = tile * m_block;
                const int valid_rows = std::min(m_block, output_m - m_begin);
                const bool direct_1x1_tile = direct_1x1_shape && valid_rows == kDenseM;
                if (!direct_1x1_tile) {
                    pack_dense_a_p3(conv, input, input_data, output.dims[3], m_begin,
                                    valid_rows, m_block, scratch.a.data());
                }
                for (int n_block = n_begin; n_block < n_end; ++n_block) {
                    execute_block(m_begin, valid_rows, direct_1x1_tile, n_block);
                }
            }
        }
        if (conv.e2c_compatible) {
            const auto result = stage51::end_q62_vector_rne(&vector_state);
            if (!result.restored || result.saturated) {
                throw std::runtime_error("Q62 vector state restoration failed");
            }
        }
    }

    void run_conv_chunk(const Operation& operation, int worker, int workers) {
        const Conv& conv = operation.conv;
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        const int output_h = output.dims[2];
        const int output_w = output.dims[3];
        const int total = output.dims[1] * output_h * output_w;
        const int begin = total * worker / workers;
        const int end = total * (worker + 1) / workers;
        const int output_per_group = conv.output_c / conv.group;
        const int input_per_group = conv.input_c / conv.group;
        for (int linear = begin; linear < end; ++linear) {
            int temporary = linear;
            const int output_x = temporary % output_w;
            temporary /= output_w;
            const int output_y = temporary % output_h;
            const int output_channel = temporary / output_h;
            const int group_index = output_channel / output_per_group;
            std::int64_t accumulator = conv.bias[static_cast<std::size_t>(output_channel)];
            for (int input_local = 0; input_local < input_per_group; ++input_local) {
                const int input_channel = group_index * input_per_group + input_local;
                for (int kernel_y = 0; kernel_y < conv.kernel_h; ++kernel_y) {
                    const int input_y = output_y * conv.stride_h - conv.pad_top + kernel_y;
                    if (input_y < 0 || input_y >= input.dims[2]) continue;
                    for (int kernel_x = 0; kernel_x < conv.kernel_w; ++kernel_x) {
                        const int input_x = output_x * conv.stride_w - conv.pad_left + kernel_x;
                        if (input_x < 0 || input_x >= input.dims[3]) continue;
                        const std::size_t input_flat =
                            (static_cast<std::size_t>(input_channel) * input.dims[2] + input_y) * input.dims[3] + input_x;
                        const std::size_t weight_index =
                            (((static_cast<std::size_t>(output_channel) * input_per_group + input_local) *
                               conv.kernel_h + kernel_y) * conv.kernel_w + kernel_x);
                        accumulator += (static_cast<int>(code(operation.inputs[0], input_flat)) - input.zero_point) *
                                       static_cast<int>(conv.weights[weight_index]);
                    }
                }
            }
            const int8_v1::RequantAsset asset {
                conv.multiplier[static_cast<std::size_t>(output_channel)],
                conv.shift[static_cast<std::size_t>(output_channel)], output.zero_point, 0, 255,
            };
            std::uint8_t quantized = 0;
            if (!int8_v1::requantize_u8(accumulator, asset, &quantized)) {
                throw std::runtime_error("Conv requantization failed");
            }
            const std::size_t output_flat =
                (static_cast<std::size_t>(output_channel) * output_h + output_y) * output_w + output_x;
            set_code(operation.output, output_flat, quantized);
        }
    }

    void run_depthwise_conv_chunk(const Operation& operation, int worker, int workers) {
        const Conv& conv = operation.conv;
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        if (!conv.depthwise_rvv_eligible) {
            run_conv_chunk(operation, worker, workers);
            return;
        }
        const int output_h = output.dims[2];
        const int output_w = output.dims[3];
        const int input_h = input.dims[2];
        const int input_w = input.dims[3];
        const int channel_blocks = conv.output_c / 8;
        const int total_rows = channel_blocks * output_h;
        const int begin = total_rows * worker / workers;
        const int end = total_rows * (worker + 1) / workers;
        const auto* input_data = arena.data() + input.offset;
        const bool fuse_lut = depthwise_v2_enabled && fused_lut_enabled &&
            !config.capture_boundaries && operation.fused_lut_output >= 0;
        const Tensor& store_output = fuse_lut ? tensor(operation.fused_lut_output) : output;
        auto* output_data = arena.data() + store_output.offset;
        const std::int64_t input_correction = 128 - input.zero_point;
        stage51::VectorFixedPointState vector_state;
        if (!stage51::begin_q62_vector_rne(&vector_state)) {
            throw std::runtime_error("cannot establish depthwise Q62 vector state");
        }
        alignas(32) std::array<std::int32_t, 8> accumulator {};
        alignas(32) [[maybe_unused]] std::array<std::int32_t, 8> accumulator_pair {};
        alignas(32) std::array<std::int64_t, 4> corrected {};
        alignas(64) std::array<std::int64_t, 8> corrected_c8 {};
        alignas(64) const std::array<std::int64_t, 8> zero_bias {};
        alignas(8) std::array<std::int8_t, 8> padding_c8 {};
        padding_c8.fill(int8_v1::signed_storage(static_cast<std::uint8_t>(input.zero_point)));
        for (int row_task = begin; row_task < end; ++row_task) {
            const int channel_block = row_task / output_h;
            const int output_y = row_task % output_h;
            const auto* packed_weights = conv.depthwise_weights_c8.data() +
                static_cast<std::size_t>(channel_block) * 9U * 8U;
            const auto* bias = conv.bias.data() + static_cast<std::size_t>(channel_block) * 8U;
            [[maybe_unused]] const auto* corrected_bias = conv.depthwise_corrected_bias_i32.data() +
                static_cast<std::size_t>(channel_block) * 8U;
            const auto* multipliers = conv.multiplier_m63.data() +
                static_cast<std::size_t>(channel_block) * 8U;
            const auto store_c8 = [&](const std::int32_t* values, int output_x) {
                const std::size_t output_physical =
                    ((static_cast<std::size_t>(channel_block) * output_h + output_y) * output_w +
                     output_x) * 8U;
                if (stage55_depthwise_e2c4_enabled) {
                    if (fuse_lut) {
                        stage51::q62_e2c4_i32x4x2_bias_lut_to_s8(
                            values, values + 4, zero_bias.data(), multipliers,
                            output.zero_point, operation.fused_lut.data(),
                            output_data + output_physical);
                    } else {
                        stage51::q62_e2c4_i32x4x2_bias_to_s8(
                            values, values + 4, zero_bias.data(), multipliers,
                            output.zero_point, output_data + output_physical);
                    }
                } else {
                    for (int lane = 0; lane < 8; ++lane) {
                        corrected_c8[static_cast<std::size_t>(lane)] = values[lane];
                    }
                    if (fuse_lut) {
                        stage51::q62_vsmul_m63_i64x8_lut_to_s8(
                            corrected_c8.data(), multipliers, output.zero_point,
                            operation.fused_lut.data(), output_data + output_physical);
                    } else {
                        stage51::q62_vsmul_m63_i64x8_to_s8(
                            corrected_c8.data(), multipliers, output.zero_point,
                            output_data + output_physical);
                    }
                }
            };
            for (int output_x = 0; output_x < output_w; ++output_x) {
                [[maybe_unused]] const bool interior =
                    output_y > 0 && output_y + 1 < output_h &&
                    output_x > 0 && output_x + 1 < output_w &&
                    input_h == output_h && input_w == output_w &&
                    conv.pad_top == 1 && conv.pad_left == 1;
#if defined(__riscv_vector)
                if (depthwise_v2_enabled && depthwise_x2_enabled && interior &&
                    output_x + 2 < output_w) {
                    constexpr std::size_t kLanes = 8;
                    vint32m1_t sum0 = __riscv_vle32_v_i32m1(corrected_bias, kLanes);
                    vint32m1_t sum1 = sum0;
                    for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                        for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                            const int tap = kernel_y * 3 + kernel_x;
                            const std::size_t input_physical =
                                ((static_cast<std::size_t>(channel_block) * input_h +
                                  output_y + kernel_y - 1) * input_w +
                                  output_x + kernel_x - 1) * 8U;
                            const vint8mf4_t weight_i8 = __riscv_vle8_v_i8mf4(
                                packed_weights + static_cast<std::size_t>(tap) * 8U, kLanes);
                            const vint32m1_t weight_i32 =
                                __riscv_vsext_vf4_i32m1(weight_i8, kLanes);
                            const vint32m1_t input0_i32 = __riscv_vsext_vf4_i32m1(
                                __riscv_vle8_v_i8mf4(input_data + input_physical, kLanes), kLanes);
                            const vint32m1_t input1_i32 = __riscv_vsext_vf4_i32m1(
                                __riscv_vle8_v_i8mf4(input_data + input_physical + 8U, kLanes), kLanes);
                            sum0 = __riscv_vmacc_vv_i32m1(sum0, input0_i32, weight_i32, kLanes);
                            sum1 = __riscv_vmacc_vv_i32m1(sum1, input1_i32, weight_i32, kLanes);
                        }
                    }
                    __riscv_vse32_v_i32m1(accumulator.data(), sum0, kLanes);
                    __riscv_vse32_v_i32m1(accumulator_pair.data(), sum1, kLanes);
                    store_c8(accumulator.data(), output_x);
                    store_c8(accumulator_pair.data(), output_x + 1);
                    ++output_x;
                    continue;
                }
                if (interior || (depthwise_v2_enabled && depthwise_border_v2_enabled)) {
                    constexpr std::size_t kLanes = 8;
                    vint32m1_t sum = __riscv_vle32_v_i32m1(
                        depthwise_v2_enabled ? corrected_bias : bias, kLanes);
                    for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                        for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                            const int tap = kernel_y * 3 + kernel_x;
                            const int input_y = output_y - conv.pad_top + kernel_y;
                            const int input_x = output_x - conv.pad_left + kernel_x;
                            const bool valid = input_y >= 0 && input_y < input_h &&
                                input_x >= 0 && input_x < input_w;
                            const std::int8_t* input_pointer = padding_c8.data();
                            if (valid) {
                                const std::size_t input_physical =
                                    ((static_cast<std::size_t>(channel_block) * input_h + input_y) *
                                     input_w + input_x) * 8U;
                                input_pointer = input_data + input_physical;
                            }
                            const vint8mf4_t input_i8 =
                                __riscv_vle8_v_i8mf4(input_pointer, kLanes);
                            const vint8mf4_t weight_i8 = __riscv_vle8_v_i8mf4(
                                packed_weights + static_cast<std::size_t>(tap) * 8U, kLanes);
                            vint32m1_t input_i32 = __riscv_vsext_vf4_i32m1(input_i8, kLanes);
                            if (!depthwise_v2_enabled) {
                                input_i32 = __riscv_vadd_vx_i32m1(
                                    input_i32, static_cast<std::int32_t>(input_correction), kLanes);
                            }
                            const vint32m1_t weight_i32 =
                                __riscv_vsext_vf4_i32m1(weight_i8, kLanes);
                            sum = __riscv_vmacc_vv_i32m1(sum, input_i32, weight_i32, kLanes);
                        }
                    }
                    __riscv_vse32_v_i32m1(accumulator.data(), sum, kLanes);
                } else
#endif
                {
                    for (int lane = 0; lane < 8; ++lane) {
                        std::int64_t sum = bias[lane];
                        for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                            const int input_y = output_y - conv.pad_top + kernel_y;
                            if (input_y < 0 || input_y >= input_h) continue;
                            for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                                const int input_x = output_x - conv.pad_left + kernel_x;
                                if (input_x < 0 || input_x >= input_w) continue;
                                const int tap = kernel_y * 3 + kernel_x;
                                const std::size_t input_physical =
                                    ((static_cast<std::size_t>(channel_block) * input_h + input_y) *
                                     input_w + input_x) * 8U + static_cast<std::size_t>(lane);
                                sum += (static_cast<std::int64_t>(input_data[input_physical]) +
                                        input_correction) *
                                       packed_weights[static_cast<std::size_t>(tap) * 8U + lane];
                            }
                        }
                        accumulator[static_cast<std::size_t>(lane)] = static_cast<std::int32_t>(sum);
                    }
                }
                if (depthwise_v2_enabled) {
                    store_c8(accumulator.data(), output_x);
                } else for (int half = 0; half < 2; ++half) {
                    const std::size_t output_physical =
                        ((static_cast<std::size_t>(channel_block) * output_h + output_y) * output_w +
                         output_x) * 8U;
                    for (int lane = 0; lane < 4; ++lane) {
                        corrected[static_cast<std::size_t>(lane)] =
                            accumulator[static_cast<std::size_t>(half * 4 + lane)];
                    }
                    stage51::q62_vsmul_m63_i64x4_to_s8(
                        corrected.data(), multipliers + half * 4, output.zero_point,
                        output_data + output_physical + static_cast<std::size_t>(half * 4));
                }
            }
        }
        const auto result = stage51::end_q62_vector_rne(&vector_state);
        if (!result.restored || result.saturated) {
            throw std::runtime_error("depthwise Q62 vector state restoration failed");
        }
    }

    void run_conv(const Operation& operation) {
        ConvJob job {this, &operation};
        dispatch_workers(config.workers, conv_job, &job);
        if (job.status.load(std::memory_order_relaxed) != 0) throw std::runtime_error("Conv worker failed");
    }

    void run_lut_chunk(const Operation& operation, std::size_t begin, std::size_t end) {
        const Tensor& output = tensor(operation.output);
        const Tensor& input = tensor(operation.inputs[0]);
        if (operation.kind == OpKind::lut1 && input.layout == output.layout &&
            input.storage_bytes == output.storage_bytes) {
            transform_lut_rvv(arena.data() + input.offset + begin,
                              arena.data() + output.offset + begin,
                              operation.lut.data(), end - begin);
            return;
        }
        if (operation.kind == OpKind::lut2) {
            const Tensor& right = tensor(operation.inputs[1]);
            const bool direct = input.layout == Layout::feature_nchwc8 &&
                right.layout == input.layout && output.layout == input.layout &&
                input.rank == output.rank && right.rank == output.rank &&
                input.dims == output.dims && right.dims == output.dims &&
                input.storage_bytes == output.storage_bytes &&
                right.storage_bytes == output.storage_bytes;
            if (direct) {
                const auto* left_data = arena.data() + input.offset;
                const auto* right_data = arena.data() + right.offset;
                auto* output_data = arena.data() + output.offset;
                if (lut2_rvv_enabled) {
                    transform_lut2_rvv(left_data + begin, right_data + begin,
                                       output_data + begin, operation.lut.data(), end - begin);
                    return;
                }
                std::size_t index = begin;
                for (; index + 4U <= end; index += 4U) {
                    if (index + 64U < end) {
                        __builtin_prefetch(left_data + index + 64U, 0, 1);
                        __builtin_prefetch(right_data + index + 64U, 0, 1);
                    }
                    for (std::size_t lane = 0; lane < 4U; ++lane) {
                        const std::size_t position = index + lane;
                        const std::uint8_t left = int8_v1::semantic_code(left_data[position]);
                        const std::uint8_t right_code = int8_v1::semantic_code(right_data[position]);
                        output_data[position] = operation.lut[
                            static_cast<std::size_t>(left) * 256U + right_code];
                    }
                }
                for (; index < end; ++index) {
                    const std::uint8_t left = int8_v1::semantic_code(left_data[index]);
                    const std::uint8_t right_code = int8_v1::semantic_code(right_data[index]);
                    output_data[index] = operation.lut[
                        static_cast<std::size_t>(left) * 256U + right_code];
                }
                return;
            }
        }
        for (std::size_t index = begin; index < end; ++index) {
            const std::uint8_t left = code(operation.inputs[0], index);
            std::size_t lut_index = left;
            if (operation.kind == OpKind::lut2) {
                lut_index = static_cast<std::size_t>(left) * 256U + code(operation.inputs[1], index);
            }
            set_code(operation.output, index, int8_v1::semantic_code(operation.lut[lut_index]));
        }
    }

    void run_lut(const Operation& operation) {
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        bool raw = operation.kind == OpKind::lut1 && input.layout == output.layout &&
                   input.storage_bytes == output.storage_bytes;
        if (operation.kind == OpKind::lut2) {
            const Tensor& right = tensor(operation.inputs[1]);
            raw = input.layout == Layout::feature_nchwc8 && right.layout == input.layout &&
                output.layout == input.layout && input.rank == output.rank &&
                right.rank == output.rank && input.dims == output.dims &&
                right.dims == output.dims && input.storage_bytes == output.storage_bytes &&
                right.storage_bytes == output.storage_bytes;
        }
        RangeJob job {this, &operation, nullptr, nullptr, nullptr, 0,
                      raw ? output.storage_bytes : output.logical_elements, 0, RangeKind::lut};
        dispatch_range(job);
    }

    std::size_t map_split_source(const Operation& operation, std::size_t output_flat,
                                 bool transpose) const {
        const Tensor& source = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        auto output_index = unravel(output_flat, output);
        std::array<int, 4> split_index = output_index;
        if (transpose) {
            split_index = {};
            for (int axis = 0; axis < output.rank; ++axis) {
                split_index[static_cast<std::size_t>(operation.perm[static_cast<std::size_t>(axis)])] =
                    output_index[static_cast<std::size_t>(axis)];
            }
        }
        const int axis = operation.axis < 0 ? output.rank + operation.axis : operation.axis;
        split_index[static_cast<std::size_t>(axis)] += operation.split_offset;
        if (!operation.source_shape.empty()) {
            Tensor virtual_source = source;
            virtual_source.rank = static_cast<int>(operation.source_shape.size());
            virtual_source.dims = {1, 1, 1, 1};
            virtual_source.logical_elements = 1;
            for (int dim = 0; dim < virtual_source.rank; ++dim) {
                virtual_source.dims[static_cast<std::size_t>(dim)] =
                    operation.source_shape[static_cast<std::size_t>(dim)];
                virtual_source.logical_elements *= static_cast<std::size_t>(
                    virtual_source.dims[static_cast<std::size_t>(dim)]);
            }
            if (virtual_source.logical_elements != source.logical_elements) {
                throw std::runtime_error("split source reshape element mismatch");
            }
            return ravel(split_index, virtual_source);
        }
        return ravel(split_index, source);
    }

    void run_transform_chunk(const Operation& operation, std::size_t begin, std::size_t end) {
        const Tensor& source = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        const bool attention_split =
            (operation.kind == OpKind::split ||
             operation.kind == OpKind::reshape_split_transpose) &&
            source.layout == Layout::feature_nchwc8 && output.layout == Layout::linear &&
            source.rank == 4 && output.rank == 4 && operation.axis == 2 &&
            operation.source_shape.size() == 4 && operation.source_shape[0] == 1 &&
            source.dims[0] == 1 &&
            operation.source_shape[1] * operation.source_shape[2] == source.dims[1] &&
            operation.source_shape[3] == source.dims[2] * source.dims[3] &&
            output.dims[0] == 1 && output.dims[1] == operation.source_shape[1];
        if (attention_split) {
            const int channels_per_head = operation.source_shape[2];
            const int spatial_count = operation.source_shape[3];
            const auto* source_data = arena.data() + source.offset;
            auto* output_data = arena.data() + output.offset;
            for (std::size_t index = begin; index < end; ++index) {
                std::size_t temporary = index;
                int local_channel = 0;
                int spatial = 0;
                int head = 0;
                if (operation.kind == OpKind::reshape_split_transpose) {
                    local_channel = static_cast<int>(temporary %
                        static_cast<std::size_t>(output.dims[3]));
                    temporary /= static_cast<std::size_t>(output.dims[3]);
                    spatial = static_cast<int>(temporary %
                        static_cast<std::size_t>(output.dims[2]));
                    head = static_cast<int>(temporary /
                        static_cast<std::size_t>(output.dims[2]));
                } else {
                    spatial = static_cast<int>(temporary %
                        static_cast<std::size_t>(output.dims[3]));
                    temporary /= static_cast<std::size_t>(output.dims[3]);
                    local_channel = static_cast<int>(temporary %
                        static_cast<std::size_t>(output.dims[2]));
                    head = static_cast<int>(temporary /
                        static_cast<std::size_t>(output.dims[2]));
                }
                const int source_channel =
                    head * channels_per_head + operation.split_offset + local_channel;
                const std::size_t source_physical =
                    (static_cast<std::size_t>(source_channel / 8) * spatial_count + spatial) * 8U +
                    static_cast<std::size_t>(source_channel % 8);
                const std::uint8_t source_code =
                    int8_v1::semantic_code(source_data[source_physical]);
                output_data[index] = operation.lut[source_code];
            }
            return;
        }
        if (operation.kind == OpKind::split && operation.axis == 1 &&
            source.layout == Layout::feature_nchwc8 && output.layout == Layout::feature_nchwc8 &&
            source.rank == 4 && output.rank == 4 && operation.split_offset % 8 == 0 &&
            source.dims[2] == output.dims[2] && source.dims[3] == output.dims[3]) {
            const std::size_t spatial_bytes =
                static_cast<std::size_t>(source.dims[2]) * source.dims[3] * 8U;
            const auto* source_data = arena.data() + source.offset +
                static_cast<std::size_t>(operation.split_offset / 8) * spatial_bytes + begin;
            auto* destination_data = arena.data() + output.offset + begin;
            transform_lut_rvv(source_data, destination_data, operation.lut.data(), end - begin);
            return;
        }
        if (operation.kind == OpKind::reshape && source.layout == output.layout &&
            source.storage_bytes == output.storage_bytes) {
            transform_lut_rvv(arena.data() + source.offset + begin,
                              arena.data() + output.offset + begin,
                              operation.lut.data(), end - begin);
            return;
        }
        for (std::size_t index = begin; index < end; ++index) {
            std::size_t source_index = index;
            if (operation.kind == OpKind::split) {
                source_index = map_split_source(operation, index, false);
            } else if (operation.kind == OpKind::transpose) {
                const auto output_index = unravel(index, output);
                std::array<int, 4> source_index_array {};
                for (int axis = 0; axis < output.rank; ++axis) {
                    source_index_array[static_cast<std::size_t>(operation.perm[static_cast<std::size_t>(axis)])] =
                        output_index[static_cast<std::size_t>(axis)];
                }
                source_index = ravel(source_index_array, source);
            } else if (operation.kind == OpKind::reshape_split_transpose) {
                source_index = map_split_source(operation, index, true);
            }
            const std::uint8_t source_code = code(operation.inputs[0], source_index);
            set_code(operation.output, index, int8_v1::semantic_code(operation.lut[source_code]));
        }
    }

    void run_transform(const Operation& operation) {
        const Tensor& source = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        const bool raw_split = operation.kind == OpKind::split && operation.axis == 1 &&
            source.layout == Layout::feature_nchwc8 && output.layout == Layout::feature_nchwc8 &&
            source.rank == 4 && output.rank == 4 && operation.split_offset % 8 == 0 &&
            source.dims[2] == output.dims[2] && source.dims[3] == output.dims[3];
        const bool raw_reshape = operation.kind == OpKind::reshape && source.layout == output.layout &&
            source.storage_bytes == output.storage_bytes;
        RangeJob job {this, &operation, nullptr, nullptr, nullptr, 0,
                      raw_split || raw_reshape ? output.storage_bytes : output.logical_elements,
                      0, RangeKind::transform};
        dispatch_range(job);
    }

    std::uint8_t pooled_code(int tensor_id, int channel, int y, int x, int depth) const {
        const Tensor& source = tensor(tensor_id);
        if (depth == 0) {
            const std::size_t flat = (static_cast<std::size_t>(channel) * source.dims[2] + y) * source.dims[3] + x;
            return code(tensor_id, flat);
        }
        std::uint8_t maximum = 0;
        for (int kernel_y = -2; kernel_y <= 2; ++kernel_y) {
            const int input_y = y + kernel_y;
            if (input_y < 0 || input_y >= source.dims[2]) continue;
            for (int kernel_x = -2; kernel_x <= 2; ++kernel_x) {
                const int input_x = x + kernel_x;
                if (input_x < 0 || input_x >= source.dims[3]) continue;
                maximum = std::max(maximum, pooled_code(tensor_id, channel, input_y, input_x, depth - 1));
            }
        }
        return maximum;
    }

    std::array<int, 4> concat_branch_dims(const Operation& operation, const Branch& branch) const {
        const Tensor& output = tensor(operation.output);
        const Tensor& source = tensor(operation.inputs[static_cast<std::size_t>(branch.input_slot)]);
        std::array<int, 4> dimensions = source.dims;
        if (branch.transform == BranchTransform::split || branch.transform == BranchTransform::reshape_split) {
            dimensions[static_cast<std::size_t>(branch.axis)] /= branch.parts;
        }
        if (branch.transform == BranchTransform::resize) {
            dimensions[2] = output.dims[2];
            dimensions[3] = output.dims[3];
        }
        return dimensions;
    }

    bool concat_resize_c8_eligible(const Operation& operation, const Branch& branch,
                                   int axis_offset) const {
        if (branch.transform != BranchTransform::resize) return false;
        const Tensor& output = tensor(operation.output);
        const Tensor& source = tensor(operation.inputs[static_cast<std::size_t>(branch.input_slot)]);
        const std::array<int, 4> branch_dims = concat_branch_dims(operation, branch);
        const int axis = operation.axis < 0 ? output.rank + operation.axis : operation.axis;
        return axis == 1 && output.rank == 4 && source.rank == 4 &&
            output.layout == Layout::feature_nchwc8 && source.layout == Layout::feature_nchwc8 &&
            output.dims[0] == 1 && source.dims[0] == 1 && branch_dims[1] == source.dims[1] &&
            branch_dims[1] % 8 == 0 && axis_offset % 8 == 0 &&
            source.dims[2] > 0 && source.dims[3] > 0 &&
            output.dims[2] % source.dims[2] == 0 &&
            output.dims[3] % source.dims[3] == 0;
    }

    void run_concat_branch_chunk(const Operation& operation, const Branch& branch, int axis_offset,
                                 std::size_t begin, std::size_t end) {
        const Tensor& output = tensor(operation.output);
        const int input_id = operation.inputs[static_cast<std::size_t>(branch.input_slot)];
        const Tensor& source = tensor(input_id);
        const std::array<int, 4> branch_dims = concat_branch_dims(operation, branch);
        const int axis = operation.axis < 0 ? output.rank + operation.axis : operation.axis;
        if (concat_resize_c8_eligible(operation, branch, axis_offset)) {
            const int scale_y = output.dims[2] / source.dims[2];
            const int scale_x = output.dims[3] / source.dims[3];
            const std::size_t output_spatial_bytes =
                static_cast<std::size_t>(output.dims[2]) * output.dims[3] * 8U;
            const std::size_t destination_base = output.offset +
                static_cast<std::size_t>(axis_offset / 8) * output_spatial_bytes;
            const auto* source_data = arena.data() + source.offset;
            auto* destination_data = arena.data() + destination_base;
            alignas(32) std::array<std::int8_t, 32> mapped {};
            for (std::size_t group = begin; group < end;) {
                const std::size_t batch = std::min<std::size_t>(4, end - group);
                if (branch.lut.empty()) {
                    std::memcpy(mapped.data(), source_data + group * 8U, batch * 8U);
                } else {
                    transform_lut_rvv(source_data + group * 8U, mapped.data(),
                                      branch.lut.data(), batch * 8U);
                }
                for (std::size_t item = 0; item < batch; ++item) {
                    const std::size_t source_group = group + item;
                    std::size_t temporary = source_group;
                    const int source_x = static_cast<int>(temporary %
                        static_cast<std::size_t>(source.dims[3]));
                    temporary /= static_cast<std::size_t>(source.dims[3]);
                    const int source_y = static_cast<int>(temporary %
                        static_cast<std::size_t>(source.dims[2]));
                    const int channel_block = static_cast<int>(temporary /
                        static_cast<std::size_t>(source.dims[2]));
                    const auto* value = mapped.data() + item * 8U;
                    for (int repeat_y = 0; repeat_y < scale_y; ++repeat_y) {
                        const int output_y = source_y * scale_y + repeat_y;
                        for (int repeat_x = 0; repeat_x < scale_x; ++repeat_x) {
                            const int output_x = source_x * scale_x + repeat_x;
                            const std::size_t destination_group =
                                ((static_cast<std::size_t>(channel_block) * output.dims[2] + output_y) *
                                 output.dims[3] + output_x) * 8U;
                            std::memcpy(destination_data + destination_group, value, 8U);
                        }
                    }
                }
                group += batch;
            }
            return;
        }
        const bool blocked_fast_path = axis == 1 && output.rank == 4 && source.rank == 4 &&
            output.layout == Layout::feature_nchwc8 && source.layout == Layout::feature_nchwc8 &&
            output.dims[0] == 1 && source.dims[0] == 1 && branch_dims[1] % 8 == 0 &&
            axis_offset % 8 == 0;
        if (blocked_fast_path) {
            const std::size_t output_spatial_bytes =
                static_cast<std::size_t>(output.dims[2]) * output.dims[3] * 8U;
            const std::size_t destination_base = output.offset +
                static_cast<std::size_t>(axis_offset / 8) * output_spatial_bytes;
            const int split_channel =
                (branch.transform == BranchTransform::split ||
                 branch.transform == BranchTransform::reshape_split)
                    ? branch.part * branch_dims[1]
                    : 0;
            const std::size_t source_spatial_bytes =
                static_cast<std::size_t>(source.dims[2]) * source.dims[3] * 8U;
            const std::size_t source_base = source.offset +
                static_cast<std::size_t>(split_channel / 8) * source_spatial_bytes;
            const bool direct = branch.transform == BranchTransform::copy ||
                branch.transform == BranchTransform::split ||
                branch.transform == BranchTransform::reshape_split ||
                branch.transform == BranchTransform::pool0;
            if (direct && source.dims[2] == output.dims[2] &&
                source.dims[3] == output.dims[3]) {
                const auto* source_data = arena.data() + source_base + begin;
                auto* destination_data = arena.data() + destination_base + begin;
                const std::size_t bytes = end - begin;
                const auto source_address = reinterpret_cast<std::uintptr_t>(source_data);
                const auto destination_address = reinterpret_cast<std::uintptr_t>(destination_data);
                const bool overlaps = source_address < destination_address + bytes &&
                    destination_address < source_address + bytes;
                if (!overlaps || source_data == destination_data) {
                    if (branch.lut.empty()) {
                        std::memmove(destination_data, source_data, bytes);
                    } else {
                        transform_lut_rvv(source_data, destination_data, branch.lut.data(), bytes);
                    }
                    return;
                }
            }
            if (branch.transform == BranchTransform::resize) {
                auto* destination_data = arena.data() + destination_base;
                const auto* source_data = arena.data() + source_base;
                for (std::size_t flat = begin; flat < end; ++flat) {
                    std::size_t temporary = flat;
                    const int channel_inner = static_cast<int>(temporary % 8U);
                    temporary /= 8U;
                    const int x = static_cast<int>(temporary % static_cast<std::size_t>(output.dims[3]));
                    temporary /= static_cast<std::size_t>(output.dims[3]);
                    const int y = static_cast<int>(temporary % static_cast<std::size_t>(output.dims[2]));
                    const int channel_block = static_cast<int>(temporary / static_cast<std::size_t>(output.dims[2]));
                    const int source_y = y * source.dims[2] / output.dims[2];
                    const int source_x = x * source.dims[3] / output.dims[3];
                    const std::size_t source_physical =
                        ((static_cast<std::size_t>(channel_block) * source.dims[2] + source_y) *
                         source.dims[3] + source_x) * 8U + channel_inner;
                    const std::uint8_t source_code = int8_v1::semantic_code(source_data[source_physical]);
                    destination_data[flat] = branch.lut.empty()
                        ? int8_v1::signed_storage(source_code)
                        : branch.lut[source_code];
                }
                return;
            }
            const int pool_depth = branch.transform == BranchTransform::pool1 ? 1 :
                branch.transform == BranchTransform::pool2 ? 2 :
                branch.transform == BranchTransform::pool3 ? 3 : -1;
            if (pool_depth > 0 && source.dims[2] == output.dims[2] &&
                source.dims[3] == output.dims[3]) {
                auto* output_data = arena.data();
                const std::size_t previous_base = output.offset +
                    static_cast<std::size_t>((axis_offset - branch_dims[1]) / 8) * output_spatial_bytes;
                for (std::size_t flat = begin; flat < end; ++flat) {
                    std::size_t temporary = flat;
                    const int channel_inner = static_cast<int>(temporary % 8U);
                    temporary /= 8U;
                    const int x = static_cast<int>(temporary % static_cast<std::size_t>(output.dims[3]));
                    temporary /= static_cast<std::size_t>(output.dims[3]);
                    const int y = static_cast<int>(temporary % static_cast<std::size_t>(output.dims[2]));
                    const int channel_block = static_cast<int>(temporary / static_cast<std::size_t>(output.dims[2]));
                    std::uint8_t maximum = 0;
                    for (int kernel_y = -2; kernel_y <= 2; ++kernel_y) {
                        const int input_y = y + kernel_y;
                        if (input_y < 0 || input_y >= output.dims[2]) continue;
                        for (int kernel_x = -2; kernel_x <= 2; ++kernel_x) {
                            const int input_x = x + kernel_x;
                            if (input_x < 0 || input_x >= output.dims[3]) continue;
                            const std::size_t source_physical = previous_base +
                                ((static_cast<std::size_t>(channel_block) * output.dims[2] + input_y) *
                                 output.dims[3] + input_x) * 8U + channel_inner;
                            maximum = std::max(maximum,
                                int8_v1::semantic_code(output_data[source_physical]));
                        }
                    }
                    output_data[destination_base + flat] = branch.lut.empty()
                        ? int8_v1::signed_storage(maximum)
                        : branch.lut[maximum];
                }
                return;
            }
        }
        Tensor branch_tensor = source;
        branch_tensor.dims = branch_dims;
        branch_tensor.logical_elements = 1;
        for (int dim = 0; dim < branch_tensor.rank; ++dim) {
            branch_tensor.logical_elements *= branch_dims[static_cast<std::size_t>(dim)];
        }
        for (std::size_t flat = begin; flat < end; ++flat) {
            auto branch_index = unravel(flat, branch_tensor);
            auto source_index = branch_index;
            if (branch.transform == BranchTransform::split || branch.transform == BranchTransform::reshape_split) {
                source_index[static_cast<std::size_t>(branch.axis)] +=
                    branch.part * branch_dims[static_cast<std::size_t>(branch.axis)];
            }
            if (branch.transform == BranchTransform::resize) {
                source_index[2] = branch_index[2] * source.dims[2] / branch_dims[2];
                source_index[3] = branch_index[3] * source.dims[3] / branch_dims[3];
            }
            std::uint8_t source_code = 0;
            const int pool_depth = branch.transform == BranchTransform::pool0 ? 0 :
                branch.transform == BranchTransform::pool1 ? 1 :
                branch.transform == BranchTransform::pool2 ? 2 :
                branch.transform == BranchTransform::pool3 ? 3 : -1;
            if (pool_depth >= 0) {
                if (pool_depth == 0) {
                    source_code = pooled_code(input_id, source_index[1], source_index[2], source_index[3], 0);
                } else {
                    const int previous_channel = axis_offset - branch_dims[1] + source_index[1];
                    for (int kernel_y = -2; kernel_y <= 2; ++kernel_y) {
                        const int input_y = source_index[2] + kernel_y;
                        if (input_y < 0 || input_y >= output.dims[2]) continue;
                        for (int kernel_x = -2; kernel_x <= 2; ++kernel_x) {
                            const int input_x = source_index[3] + kernel_x;
                            if (input_x < 0 || input_x >= output.dims[3]) continue;
                            const std::size_t previous_flat =
                                (static_cast<std::size_t>(previous_channel) * output.dims[2] + input_y) *
                                output.dims[3] + input_x;
                            source_code = std::max(source_code, code(operation.output, previous_flat));
                        }
                    }
                }
            } else {
                source_code = code(input_id, ravel(source_index, source));
            }
            auto output_index = branch_index;
            output_index[static_cast<std::size_t>(axis)] += axis_offset;
            const std::uint8_t result = branch.lut.empty() ? source_code :
                int8_v1::semantic_code(branch.lut[source_code]);
            set_code(operation.output, ravel(output_index, output), result);
        }
    }

    void run_concat(const Operation& operation) {
        const Tensor& output = tensor(operation.output);
        int axis_offset = 0;
        for (const Branch& branch : operation.branches) {
            const std::array<int, 4> branch_dims = concat_branch_dims(operation, branch);
            const int axis = operation.axis < 0 ? output.rank + operation.axis : operation.axis;
            std::size_t elements = 1;
            for (int dim = 0; dim < output.rank; ++dim) {
                elements *= static_cast<std::size_t>(branch_dims[static_cast<std::size_t>(dim)]);
            }
            if (concat_resize_c8_eligible(operation, branch, axis_offset)) {
                const Tensor& source = tensor(
                    operation.inputs[static_cast<std::size_t>(branch.input_slot)]);
                elements = static_cast<std::size_t>(source.dims[1] / 8) *
                    source.dims[2] * source.dims[3];
            }
            RangeJob job {this, &operation, &branch, nullptr, nullptr, 0, elements, axis_offset,
                          RangeKind::concat_branch};
            dispatch_range(job);
            axis_offset += branch_dims[static_cast<std::size_t>(axis)];
        }
        if (axis_offset != output.dims[static_cast<std::size_t>(operation.axis < 0 ? output.rank + operation.axis : operation.axis)]) {
            throw std::runtime_error("Concat axis coverage mismatch");
        }
    }

    void run_resize_chunk(const Operation& operation, std::size_t begin, std::size_t end) {
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        for (std::size_t flat = begin; flat < end; ++flat) {
            auto index = unravel(flat, output);
            auto source_index = index;
            source_index[2] = index[2] * input.dims[2] / output.dims[2];
            source_index[3] = index[3] * input.dims[3] / output.dims[3];
            const std::uint8_t source = code(operation.inputs[0], ravel(source_index, input));
            set_code(operation.output, flat, int8_v1::semantic_code(operation.lut[source]));
        }
    }

    void run_resize(const Operation& operation) {
        RangeJob job {this, &operation, nullptr, nullptr, nullptr, 0,
                      tensor(operation.output).logical_elements,
                      0, RangeKind::resize};
        dispatch_range(job);
    }

    void run_matmul_chunk(const Operation& operation, std::size_t begin, std::size_t end) {
        const Tensor& left = tensor(operation.inputs[0]);
        const Tensor& right = tensor(operation.inputs[1]);
        const int m = left.dims[static_cast<std::size_t>(left.rank - 2)];
        const int k = left.dims[static_cast<std::size_t>(left.rank - 1)];
        const int n = right.dims[static_cast<std::size_t>(right.rank - 1)];
        for (std::size_t output_index = begin; output_index < end; ++output_index) {
                    const int column = static_cast<int>(output_index % static_cast<std::size_t>(n));
                    const std::size_t outer = output_index / static_cast<std::size_t>(n);
                    const int row = static_cast<int>(outer % static_cast<std::size_t>(m));
                    const int batch = static_cast<int>(outer / static_cast<std::size_t>(m));
                    std::int64_t accumulator = 0;
                    for (int inner = 0; inner < k; ++inner) {
                        const std::size_t left_index = (static_cast<std::size_t>(batch) * m + row) * k + inner;
                        const std::size_t right_index = (static_cast<std::size_t>(batch) * k + inner) * n + column;
                        accumulator += (static_cast<int>(code(operation.inputs[0], left_index)) - operation.left_zero_point) *
                                       (static_cast<int>(code(operation.inputs[1], right_index)) - operation.right_zero_point);
                    }
                    std::uint8_t result = 0;
                    const int8_v1::RequantAsset asset {operation.multiplier, operation.right_shift,
                                                       operation.output_zero_point, 0, 255};
                    if (!int8_v1::requantize_u8(accumulator, asset, &result)) {
                        throw std::runtime_error("MatMul requantization failed");
                    }
                    set_code(operation.output, output_index, result);
        }
    }

    struct MatmulImeJob {
        Impl* self;
        const Operation* op;
        std::atomic<int> status {0};
    };

    static void matmul_ime_job(void* opaque, int worker, int workers) {
        auto& job = *static_cast<MatmulImeJob*>(opaque);
        if (job.status.load(std::memory_order_relaxed) != 0) return;
        try {
            job.self->run_matmul_ime_chunk(*job.op, worker, workers);
        } catch (...) {
            job.status.store(1, std::memory_order_relaxed);
        }
    }

    void pack_matmul_right(const Operation& operation) {
        const Tensor& right = tensor(operation.inputs[1]);
        const int k = right.dims[static_cast<std::size_t>(right.rank - 2)];
        const int n = right.dims[static_cast<std::size_t>(right.rank - 1)];
        const int k_tiles = k / 8;
        const int n_blocks = n / kDenseN;
        const std::size_t matrix_elements = static_cast<std::size_t>(k) * n;
        const int batches = static_cast<int>(right.logical_elements / matrix_elements);
        const auto* source = arena.data() + right.offset;
        std::fill(matmul_right_sums.begin(),
                  matmul_right_sums.begin() + static_cast<std::size_t>(batches) * n, 0);
        for (int batch = 0; batch < batches; ++batch) {
            for (int n_block = 0; n_block < n_blocks; ++n_block) {
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                    for (int output = 0; output < kDenseN; ++output) {
                        const int column = n_block * kDenseN + output;
                        std::int8_t* destination = matmul_packed_right.data() +
                            ((((static_cast<std::size_t>(batch) * n_blocks + n_block) * k_tiles + k_tile) *
                               kDenseN + output) * 8U);
                        for (int lane = 0; lane < 8; ++lane) {
                            const int inner = k_tile * 8 + lane;
                            const std::int8_t raw = source[
                                (static_cast<std::size_t>(batch) * k + inner) * n + column];
                            destination[lane] = raw;
                            matmul_right_sums[static_cast<std::size_t>(batch) * n + column] += raw;
                        }
                    }
                }
            }
        }
    }

    void run_matmul_ime_chunk(const Operation& operation, int worker, int workers) {
        const Tensor& left = tensor(operation.inputs[0]);
        const Tensor& right = tensor(operation.inputs[1]);
        const Tensor& output = tensor(operation.output);
        const int m = left.dims[static_cast<std::size_t>(left.rank - 2)];
        const int k = left.dims[static_cast<std::size_t>(left.rank - 1)];
        const int n = right.dims[static_cast<std::size_t>(right.rank - 1)];
        const int k_tiles = k / 8;
        const int n_blocks = n / kDenseN;
        const int batches = static_cast<int>(left.logical_elements / (static_cast<std::size_t>(m) * k));
        const int tiles_per_batch = (m + kDenseM - 1) / kDenseM;
        const int total_tiles = batches * tiles_per_batch;
        const int tile_begin = total_tiles * worker / workers;
        const int tile_end = total_tiles * (worker + 1) / workers;
        const auto* left_data = arena.data() + left.offset;
        auto* output_data = arena.data() + output.offset;
        DenseScratch& scratch = dense_scratch[static_cast<std::size_t>(worker)];
        const std::int64_t left_correction = 128 - operation.left_zero_point;
        const std::int64_t right_correction = 128 - operation.right_zero_point;
        const std::int64_t constant_correction =
            static_cast<std::int64_t>(k) * left_correction * right_correction;
        stage51::VectorFixedPointState vector_state;
        if (!stage51::begin_q62_vector_rne(&vector_state)) {
            throw std::runtime_error("cannot establish MatMul Q62 vector state");
        }
        for (int tile = tile_begin; tile < tile_end; ++tile) {
            const int batch = tile / tiles_per_batch;
            const int m_begin = (tile % tiles_per_batch) * kDenseM;
            const int valid_rows = std::min(kDenseM, m - m_begin);
            std::fill(scratch.row_sums.begin(), scratch.row_sums.end(), 0);
            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
                std::int8_t* destination = scratch.a.data() +
                    static_cast<std::size_t>(k_tile) * kDenseM * 8U;
                for (int row = 0; row < kDenseM; ++row) {
                    if (row >= valid_rows) {
                        std::fill(destination + row * 8, destination + (row + 1) * 8, 0);
                        continue;
                    }
                    const std::int8_t* source = left_data +
                        (static_cast<std::size_t>(batch) * m + m_begin + row) * k + k_tile * 8;
                    std::memcpy(destination + row * 8, source, 8);
                    for (int lane = 0; lane < 8; ++lane) {
                        scratch.row_sums[static_cast<std::size_t>(row)] += source[lane];
                    }
                }
            }
            for (int n_block = 0; n_block < n_blocks; ++n_block) {
                const std::int8_t* packed = matmul_packed_right.data() +
                    (static_cast<std::size_t>(batch) * n_blocks + n_block) * k_tiles * kDenseN * 8U;
                run_m12n16(scratch.a.data(), packed, k_tiles, scratch.c.data());
                for (int output_group = 0; output_group < 4; ++output_group) {
                    const int column_begin = n_block * kDenseN + output_group * 4;
                    alignas(32) std::array<std::int64_t, 4> multipliers {
                        operation.multiplier_m63, operation.multiplier_m63,
                        operation.multiplier_m63, operation.multiplier_m63,
                    };
                    for (int row = 0; row < valid_rows; ++row) {
                        const int row_group = row / 4;
                        const int row_inner = row % 4;
                        const std::int32_t* raw = scratch.c.data() +
                            (output_group * 3 + row_group) * 16 + row_inner * 4;
                        alignas(32) std::array<std::int64_t, 4> corrected {};
                        alignas(32) std::array<std::int64_t, 4> rounded {};
                        for (int lane = 0; lane < 4; ++lane) {
                            const int column = column_begin + lane;
                            corrected[static_cast<std::size_t>(lane)] =
                                static_cast<std::int64_t>(raw[lane]) +
                                right_correction * scratch.row_sums[static_cast<std::size_t>(row)] +
                                left_correction * matmul_right_sums[static_cast<std::size_t>(batch) * n + column] +
                                constant_correction;
                        }
                        stage51::q62_vsmul_m63_i64x4(
                            corrected.data(), multipliers.data(), rounded.data());
                        for (int lane = 0; lane < 4; ++lane) {
                            const int column = column_begin + lane;
                            const std::uint8_t quantized = static_cast<std::uint8_t>(
                                std::clamp<std::int64_t>(rounded[static_cast<std::size_t>(lane)] +
                                    operation.output_zero_point, 0, 255));
                            output_data[(static_cast<std::size_t>(batch) * m + m_begin + row) * n + column] =
                                int8_v1::signed_storage(quantized);
                        }
                    }
                }
            }
        }
        const auto result = stage51::end_q62_vector_rne(&vector_state);
        if (!result.restored || result.saturated) {
            throw std::runtime_error("MatMul Q62 vector state restoration failed");
        }
    }

    void run_matmul_ime(const Operation& operation) {
        const auto pack_begin = attention_subphase_profile_enabled ? Clock::now() : Clock::time_point {};
        pack_matmul_right(operation);
        const auto compute_begin = attention_subphase_profile_enabled ? Clock::now() : Clock::time_point {};
        MatmulImeJob job {this, &operation};
        dispatch_workers(config.workers, matmul_ime_job, &job);
        if (job.status.load(std::memory_order_relaxed) != 0) {
            throw std::runtime_error("MatMul IME worker failed");
        }
        if (attention_subphase_profile_enabled) {
            const auto end = Clock::now();
            attention_profile->matmul_pack_ns.fetch_add(
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    compute_begin - pack_begin).count()), std::memory_order_relaxed);
            attention_profile->matmul_compute_ns.fetch_add(
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - compute_begin).count()), std::memory_order_relaxed);
        }
    }

    void run_matmul(const Operation& operation) {
        if (config.compute == ComputeMode::optimized && operation.matmul_ime_eligible) {
            run_matmul_ime(operation);
            return;
        }
        RangeJob job {this, &operation, nullptr, nullptr, nullptr, 0,
                      tensor(operation.output).logical_elements,
                      0, RangeKind::matmul};
        dispatch_range(job);
    }

    void run_softmax_chunk(const Operation& operation, std::size_t begin, std::size_t end) {
        const Tensor& input = tensor(operation.inputs[0]);
        const Tensor& output = tensor(operation.output);
        const int width = input.dims[static_cast<std::size_t>(input.rank - 1)];
        const bool direct_transpose = input.layout == Layout::linear &&
            output.layout == Layout::linear && input.rank == 4 && output.rank == 4 &&
            operation.perm.size() == 4 && operation.perm[0] == 0 &&
            operation.perm[1] == 1 && operation.perm[2] == 3 && operation.perm[3] == 2 &&
            output.dims[0] == input.dims[0] && output.dims[1] == input.dims[1] &&
            output.dims[2] == input.dims[3] && output.dims[3] == input.dims[2];
        if (direct_transpose) {
            const int rows_per_matrix = input.dims[2];
            const auto* source = arena.data() + input.offset;
            auto* destination = arena.data() + output.offset;
            std::uint64_t max_sum_ns = 0;
            std::uint64_t normalize_transpose_ns = 0;
            for (std::size_t row = begin; row < end; ++row) {
                const auto max_sum_begin = attention_subphase_profile_enabled
                    ? Clock::now() : Clock::time_point {};
                const auto* source_row = source + row * static_cast<std::size_t>(width);
                std::uint8_t maximum = 0;
                std::uint64_t sum = 0;
                if (attention_v2_enabled) {
                    softmax_max_sum_rvv(source_row, static_cast<std::size_t>(width),
                                        operation.exp_q48.data(), &maximum, &sum);
                } else {
                    for (int column = 0; column < width; ++column) {
                        maximum = std::max(maximum, int8_v1::semantic_code(source_row[column]));
                    }
                    for (int column = 0; column < width; ++column) {
                        const std::uint8_t difference = static_cast<std::uint8_t>(
                            maximum - int8_v1::semantic_code(source_row[column]));
                        sum += operation.exp_q48[difference];
                    }
                }
                const auto normalize_begin = attention_subphase_profile_enabled
                    ? Clock::now() : Clock::time_point {};
                if (attention_subphase_profile_enabled) {
                    max_sum_ns += static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            normalize_begin - max_sum_begin).count());
                }

                // Exact quotient work is shared by every equal score in a row.
                std::array<std::uint8_t, 256> quantized_by_difference {};
                std::array<std::uint8_t, 256> valid {};
                const std::size_t matrix = row / static_cast<std::size_t>(rows_per_matrix);
                const std::size_t row_in_matrix = row % static_cast<std::size_t>(rows_per_matrix);
                for (int column = 0; column < width; ++column) {
                    const std::uint8_t difference = static_cast<std::uint8_t>(
                        maximum - int8_v1::semantic_code(source_row[column]));
                    if (valid[difference] == 0) {
                        const UnsignedInt128 numerator =
                            static_cast<UnsignedInt128>(operation.exp_q48[difference]) *
                            operation.softmax_reciprocal_q32;
                        const UnsignedInt128 denominator = static_cast<UnsignedInt128>(sum) << 32U;
                        const std::int64_t quantized =
                            round_divide_even(numerator, denominator) + operation.output_zero_point;
                        quantized_by_difference[difference] = static_cast<std::uint8_t>(
                            std::clamp<std::int64_t>(quantized, 0, 255));
                        valid[difference] = 1;
                    }
                    const std::size_t destination_flat =
                        (matrix * static_cast<std::size_t>(width) + static_cast<std::size_t>(column)) *
                        static_cast<std::size_t>(rows_per_matrix) + row_in_matrix;
                    destination[destination_flat] =
                        int8_v1::signed_storage(quantized_by_difference[difference]);
                }
                if (attention_subphase_profile_enabled) {
                    normalize_transpose_ns += static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            Clock::now() - normalize_begin).count());
                }
            }
            if (attention_subphase_profile_enabled) {
                attention_profile->softmax_max_sum_ns.fetch_add(max_sum_ns, std::memory_order_relaxed);
                attention_profile->softmax_normalize_transpose_ns.fetch_add(
                    normalize_transpose_ns, std::memory_order_relaxed);
            }
            return;
        }
        std::array<int, 4> source_index {};
        for (std::size_t row = begin; row < end; ++row) {
            std::uint8_t maximum = 0;
            for (int column = 0; column < width; ++column) {
                maximum = std::max(maximum, code(operation.inputs[0], row * width + column));
            }
            std::uint64_t sum = 0;
            for (int column = 0; column < width; ++column) {
                sum += operation.exp_q48[static_cast<std::size_t>(maximum - code(operation.inputs[0], row * width + column))];
            }
            for (int column = 0; column < width; ++column) {
                const std::uint64_t exponent = operation.exp_q48[
                    static_cast<std::size_t>(maximum - code(operation.inputs[0], row * width + column))];
                const UnsignedInt128 numerator = static_cast<UnsignedInt128>(exponent) *
                                                 operation.softmax_reciprocal_q32;
                const UnsignedInt128 denominator = static_cast<UnsignedInt128>(sum) << 32U;
                const std::int64_t quantized = round_divide_even(numerator, denominator) + operation.output_zero_point;
                const std::uint8_t result = static_cast<std::uint8_t>(std::clamp<std::int64_t>(quantized, 0, 255));
                const std::size_t source_flat = row * width + column;
                source_index = unravel(source_flat, input);
                std::array<int, 4> destination_index {};
                for (int axis = 0; axis < output.rank; ++axis) {
                    destination_index[static_cast<std::size_t>(axis)] =
                        source_index[static_cast<std::size_t>(operation.perm[static_cast<std::size_t>(axis)])];
                }
                set_code(operation.output, ravel(destination_index, output), result);
            }
        }
    }

    void run_softmax(const Operation& operation) {
        const Tensor& input = tensor(operation.inputs[0]);
        const int width = input.dims[static_cast<std::size_t>(input.rank - 1)];
        RangeJob job {this, &operation, nullptr, nullptr, nullptr, 0,
                      input.logical_elements / static_cast<std::size_t>(width),
                      0, RangeKind::softmax};
        dispatch_range(job);
    }

    void run_input_chunk(const Operation& operation, const float* input,
                         std::size_t begin, std::size_t end) {
        const Tensor& output = tensor(operation.output);
        if (output.layout == Layout::feature_nchwc8 && output.rank == 4 &&
            output.dims[0] == 1 && output.dims[1] > 0 && output.dims[1] <= 8) {
            const std::size_t plane = static_cast<std::size_t>(output.dims[2]) * output.dims[3];
            auto* destination = arena.data() + output.offset;
            const bool compact_c3 = input_compact_c3_enabled && !config.capture_boundaries &&
                output.dims[1] == 3;
            const std::size_t pixel_stride = compact_c3 ? 3U : 8U;
#if defined(__riscv_vector)
            alignas(32) std::array<std::int32_t, 8> converted {};
            std::size_t spatial = begin;
            while (spatial < end) {
                const std::size_t vl = __riscv_vsetvl_e32m1(
                    std::min<std::size_t>(8, end - spatial));
                for (int channel = 0; channel < output.dims[1]; ++channel) {
                    vfloat32m1_t value = __riscv_vle32_v_f32m1(
                        input + static_cast<std::size_t>(channel) * plane + spatial, vl);
                    value = __riscv_vfmul_vf_f32m1(value, 255.0F, vl);
                    vint32m1_t code = __riscv_vfcvt_x_f_v_i32m1_rm(
                        value, __RISCV_FRM_RNE, vl);
                    code = __riscv_vmax_vx_i32m1(code, 0, vl);
                    code = __riscv_vmin_vx_i32m1(code, 255, vl);
                    if (input_rvv_v2_enabled) {
                        const vuint32m1_t unsigned_code = __riscv_vreinterpret_v_i32m1_u32m1(code);
                        const vuint16mf2_t code_u16 =
                            __riscv_vncvt_x_x_w_u16mf2(unsigned_code, vl);
                        vuint8mf4_t code_u8 = __riscv_vncvt_x_x_w_u8mf4(code_u16, vl);
                        code_u8 = __riscv_vxor_vx_u8mf4(code_u8, 128U, vl);
                        __riscv_vsse8_v_u8mf4(
                            reinterpret_cast<std::uint8_t*>(destination) + spatial * pixel_stride +
                                static_cast<std::size_t>(channel),
                            static_cast<std::ptrdiff_t>(pixel_stride), code_u8, vl);
                    } else {
                        __riscv_vse32_v_i32m1(converted.data(), code, vl);
                        for (std::size_t lane = 0; lane < vl; ++lane) {
                            destination[(spatial + lane) * pixel_stride +
                                        static_cast<std::size_t>(channel)] =
                                int8_v1::signed_storage(
                                    static_cast<std::uint8_t>(converted[lane]));
                        }
                    }
                }
                if (input_rvv_v2_enabled && !compact_c3) {
                    const vuint8mf4_t padding = __riscv_vmv_v_x_u8mf4(128U, vl);
                    for (int channel = output.dims[1]; channel < 8; ++channel) {
                        __riscv_vsse8_v_u8mf4(
                            reinterpret_cast<std::uint8_t*>(destination) + spatial * pixel_stride +
                                static_cast<std::size_t>(channel),
                            static_cast<std::ptrdiff_t>(pixel_stride), padding, vl);
                    }
                } else if (!compact_c3) {
                    for (std::size_t lane = 0; lane < vl; ++lane) {
                        std::fill(destination + (spatial + lane) * pixel_stride + output.dims[1],
                                  destination + (spatial + lane + 1U) * pixel_stride,
                                  int8_v1::signed_storage(0));
                    }
                }
                spatial += vl;
            }
#else
            for (std::size_t spatial = begin; spatial < end; ++spatial) {
                for (int channel = 0; channel < output.dims[1]; ++channel) {
                    const double scaled = static_cast<double>(
                        input[static_cast<std::size_t>(channel) * plane + spatial]) * 255.0;
                    const double floor_value = std::floor(scaled);
                    const double fraction = scaled - floor_value;
                    std::int64_t rounded = static_cast<std::int64_t>(floor_value);
                    if (fraction > 0.5 || (fraction == 0.5 && (rounded & 1))) ++rounded;
                    destination[spatial * pixel_stride + static_cast<std::size_t>(channel)] =
                        int8_v1::signed_storage(static_cast<std::uint8_t>(
                            std::clamp<std::int64_t>(rounded, 0, 255)));
                }
                if (!compact_c3) {
                    std::fill(destination + spatial * pixel_stride + output.dims[1],
                              destination + (spatial + 1U) * pixel_stride,
                              int8_v1::signed_storage(0));
                }
            }
#endif
            return;
        }
        for (std::size_t flat = begin; flat < end; ++flat) {
            const double scaled = static_cast<double>(input[flat]) * 255.0;
            const double floor_value = std::floor(scaled);
            const double fraction = scaled - floor_value;
            std::int64_t rounded = static_cast<std::int64_t>(floor_value);
            if (fraction > 0.5 || (fraction == 0.5 && (rounded & 1))) ++rounded;
            set_code(operation.output, flat,
                     static_cast<std::uint8_t>(std::clamp<std::int64_t>(rounded, 0, 255)));
        }
    }

    void run_input_rgb_chunk(const Operation& operation, const std::uint8_t* rgb,
                             int stride, std::size_t begin, std::size_t end) {
        const Tensor& output = tensor(operation.output);
        if (output.layout == Layout::feature_nchwc8 && output.rank == 4 &&
            output.dims[0] == 1 && output.dims[1] == 3) {
            auto* destination = arena.data() + output.offset;
            const bool compact_c3 = input_compact_c3_enabled && !config.capture_boundaries;
            const std::size_t pixel_stride = compact_c3 ? 3U : 8U;
            const std::size_t width = static_cast<std::size_t>(output.dims[3]);
            for (std::size_t spatial = begin; spatial < end; ++spatial) {
                const std::size_t y = spatial / width;
                const std::size_t x = spatial % width;
                const auto* source = rgb + y * static_cast<std::size_t>(stride) + x * 3U;
                auto* output_pixel = destination + spatial * pixel_stride;
                output_pixel[0] = int8_v1::signed_storage(source[0]);
                output_pixel[1] = int8_v1::signed_storage(source[1]);
                output_pixel[2] = int8_v1::signed_storage(source[2]);
                if (!compact_c3) {
                    std::fill(output_pixel + 3, output_pixel + 8, int8_v1::signed_storage(0));
                }
            }
            return;
        }
        constexpr std::size_t kPlane = 640U * 640U;
        for (std::size_t flat = begin; flat < end; ++flat) {
            const std::size_t channel = flat / kPlane;
            const std::size_t spatial = flat % kPlane;
            const std::size_t y = spatial / 640U;
            const std::size_t x = spatial % 640U;
            set_code(operation.output, flat,
                     rgb[y * static_cast<std::size_t>(stride) + x * 3U + channel]);
        }
    }

    static void execute_input(Impl* self, const Operation& operation, const float* input,
                              const std::uint8_t* rgb, int rgb_stride) {
        if (self->input_stem_fused_enabled && !self->config.capture_boundaries &&
            input != nullptr) {
            return;
        }
        const Tensor& output = self->tensor(operation.output);
        RangeJob job;
        job.self = self;
        job.op = &operation;
        job.input = input;
        job.rgb = rgb;
        job.rgb_stride = rgb_stride;
        job.total = output.layout == Layout::feature_nchwc8 && output.rank == 4 &&
                    output.dims[0] == 1 && output.dims[1] <= 8
            ? static_cast<std::size_t>(output.dims[2]) * output.dims[3]
            : output.logical_elements;
        job.kind = rgb == nullptr ? RangeKind::input_quant : RangeKind::input_rgb;
        self->dispatch_range(job);
    }

    static void execute_conv(Impl* self, const Operation& operation, const float*,
                             const std::uint8_t*, int) {
        self->run_conv(operation);
    }

    static void execute_lut(Impl* self, const Operation& operation, const float*,
                            const std::uint8_t*, int) {
        self->run_lut(operation);
    }

    static void execute_transform(Impl* self, const Operation& operation, const float*,
                                  const std::uint8_t*, int) {
        self->run_transform(operation);
    }

    static void execute_concat(Impl* self, const Operation& operation, const float*,
                               const std::uint8_t*, int) {
        self->run_concat(operation);
    }

    static void execute_resize(Impl* self, const Operation& operation, const float*,
                               const std::uint8_t*, int) {
        self->run_resize(operation);
    }

    static void execute_matmul(Impl* self, const Operation& operation, const float*,
                               const std::uint8_t*, int) {
        self->run_matmul(operation);
    }

    static void execute_softmax(Impl* self, const Operation& operation, const float*,
                                const std::uint8_t*, int) {
        self->run_softmax(operation);
    }

    static OperationRunner runner_for(OpKind kind) {
        switch (kind) {
            case OpKind::input_quant: return execute_input;
            case OpKind::conv_dense:
            case OpKind::conv_grouped: return execute_conv;
            case OpKind::lut1:
            case OpKind::lut2: return execute_lut;
            case OpKind::split:
            case OpKind::reshape:
            case OpKind::transpose:
            case OpKind::reshape_split_transpose: return execute_transform;
            case OpKind::concat: return execute_concat;
            case OpKind::resize: return execute_resize;
            case OpKind::matmul: return execute_matmul;
            case OpKind::softmax_transpose: return execute_softmax;
        }
        throw std::runtime_error("operation runner is unavailable");
    }

    struct Point {
        std::array<std::int32_t, 4> box_q16 {};
        std::uint32_t best_score_q24 = 0;
        int best_class = 0;
        int point_index = 0;
        int scale_index = 0;
        int local_index = 0;
    };

    struct Candidate {
        std::uint32_t score_q24;
        int point_slot;
        int class_index;
    };

    void decode_head(float* output) {
        std::array<Point, 8400> points {};
        int global = 0;
        for (int scale_index = 0; scale_index < 3; ++scale_index) {
            const HeadScale& scale = head[static_cast<std::size_t>(scale_index)];
            const Tensor& reg_tensor = tensor(scale.reg_tensor);
            const Tensor& cls_tensor = tensor(scale.cls_tensor);
            if (reg_tensor.layout != Layout::feature_nchwc8 ||
                cls_tensor.layout != Layout::feature_nchwc8 ||
                reg_tensor.rank != 4 || cls_tensor.rank != 4) {
                throw std::runtime_error("head tensors require NCHWc8 physical layout");
            }
            const auto* reg_data = arena.data() + reg_tensor.offset;
            const auto* cls_data = arena.data() + cls_tensor.offset;
            const int pixels = scale.resolution * scale.resolution;
            const auto direct_code = [pixels](const std::int8_t* data, int channel, int spatial) {
                const std::size_t offset =
                    (static_cast<std::size_t>(channel / 8) * pixels + spatial) * 8U +
                    static_cast<std::size_t>(channel % 8);
                return int8_v1::semantic_code(data[offset]);
            };
            const int global_begin = global;
            for (int local = 0; local < pixels; ++local, ++global) {
                const int y = local / scale.resolution;
                const int x = local % scale.resolution;
                Point& point = points[static_cast<std::size_t>(global)];
                point.point_index = global;
                point.scale_index = scale_index;
                point.local_index = local;
                const std::int32_t anchor_x = (2 * x + 1) << 15;
                const std::int32_t anchor_y = (2 * y + 1) << 15;
                const auto reg = [&](int channel) {
                    return scale.reg_q16[static_cast<std::size_t>(channel) * 256U +
                        direct_code(reg_data, channel, local)];
                };
                point.box_q16 = {
                    (anchor_x - reg(0)) * scale.stride,
                    (anchor_y - reg(1)) * scale.stride,
                    (anchor_x + reg(2)) * scale.stride,
                    (anchor_y + reg(3)) * scale.stride,
                };
            }
            for (int class_block = 0; class_block < 10; ++class_block) {
                const auto* class_block_data = cls_data +
                    static_cast<std::size_t>(class_block) * pixels * 8U;
                for (int local = 0; local < pixels; ++local) {
                    Point& point = points[static_cast<std::size_t>(global_begin + local)];
                    const auto* codes = class_block_data + static_cast<std::size_t>(local) * 8U;
                    for (int lane = 0; lane < 8; ++lane) {
                        const int class_index = class_block * 8 + lane;
                        const std::uint32_t score = scale.cls_q24[
                            int8_v1::semantic_code(codes[lane])];
                        if (score > point.best_score_q24) {
                            point.best_score_q24 = score;
                            point.best_class = class_index;
                        }
                    }
                }
            }
        }
        auto point_compare = [](const Point& left, const Point& right) {
            return left.best_score_q24 != right.best_score_q24
                ? left.best_score_q24 > right.best_score_q24
                : left.point_index < right.point_index;
        };
        std::partial_sort(points.begin(), points.begin() + 300, points.end(), point_compare);
        auto candidate_compare = [](const Candidate& left, const Candidate& right) {
            if (left.score_q24 != right.score_q24) return left.score_q24 > right.score_q24;
            if (left.point_slot != right.point_slot) return left.point_slot < right.point_slot;
            return left.class_index < right.class_index;
        };
        std::array<Candidate, 300 * 80> candidate_storage;
        Candidate* candidates = candidate_storage.data();
        std::size_t candidate_count = 0;
        std::array<Candidate, 300> candidate_heap;
        std::size_t heap_size = 0;
        for (int slot = 0; slot < 300; ++slot) {
            const Point& point = points[static_cast<std::size_t>(slot)];
            const HeadScale& scale = head[static_cast<std::size_t>(point.scale_index)];
            const int pixels = scale.resolution * scale.resolution;
            const Tensor& cls_tensor = tensor(scale.cls_tensor);
            const auto* cls_data = arena.data() + cls_tensor.offset;
            for (int class_index = 0; class_index < 80; ++class_index) {
                const std::size_t physical =
                    (static_cast<std::size_t>(class_index / 8) * pixels + point.local_index) * 8U +
                    static_cast<std::size_t>(class_index % 8);
                const Candidate candidate {
                    scale.cls_q24[int8_v1::semantic_code(cls_data[physical])], slot, class_index};
                if (!head_v2_enabled) {
                    candidate_storage[candidate_count++] = candidate;
                } else if (heap_size < candidate_heap.size()) {
                    candidate_heap[heap_size++] = candidate;
                    if (heap_size == candidate_heap.size()) {
                        std::make_heap(candidate_heap.begin(), candidate_heap.end(), candidate_compare);
                    }
                } else if (candidate_compare(candidate, candidate_heap.front())) {
                    std::pop_heap(candidate_heap.begin(), candidate_heap.end(), candidate_compare);
                    candidate_heap.back() = candidate;
                    std::push_heap(candidate_heap.begin(), candidate_heap.end(), candidate_compare);
                }
            }
        }
        if (head_v2_enabled) {
            std::sort_heap(candidate_heap.begin(), candidate_heap.end(), candidate_compare);
            candidates = candidate_heap.data();
            candidate_count = candidate_heap.size();
        } else {
            std::partial_sort(candidate_storage.begin(), candidate_storage.begin() + 300,
                              candidate_storage.begin() + static_cast<std::ptrdiff_t>(candidate_count),
                              candidate_compare);
        }
        for (int detection = 0; detection < 300; ++detection) {
            const Candidate& candidate = candidates[static_cast<std::size_t>(detection)];
            const Point& point = points[static_cast<std::size_t>(candidate.point_slot)];
            float* row = output + static_cast<std::size_t>(detection) * 6U;
            for (int coordinate = 0; coordinate < 4; ++coordinate) {
                row[coordinate] = static_cast<float>(point.box_q16[static_cast<std::size_t>(coordinate)]) / 65536.0F;
            }
            row[4] = static_cast<float>(candidate.score_q24) / 16777216.0F;
            row[5] = static_cast<float>(candidate.class_index);
        }
    }
};

FullExecutor::FullExecutor() : impl_(std::make_unique<Impl>()) {}
FullExecutor::~FullExecutor() = default;
FullExecutor::FullExecutor(FullExecutor&&) noexcept = default;
FullExecutor& FullExecutor::operator=(FullExecutor&&) noexcept = default;

int FullExecutor::prepare(const std::filesystem::path& package_dir,
                          const std::string& trusted_manifest_sha256,
                          const RunConfig& config) {
    if (config.workers < 1 || config.workers > 4 || config.worker_cpu_begin != 0 ||
        config.controller_cpu < 0) return 1;
    impl_->error.clear();
    try {
        const auto verified = int8_v1::verify_package(
            package_dir, trusted_manifest_sha256, int8_v1::kContractId,
            kFullGraphProfileId, int8_v1::kNchwc8LayoutId, 2);
        if (!verified.ok) throw std::runtime_error("package verification failed: " + verified.error);
        Impl prepared;
        prepared.package = std::filesystem::canonical(package_dir);
        prepared.manifest = verified.manifest_sha256;
        prepared.config = config;
        prepared.controller_affinity_ok = pin_thread(config.controller_cpu);
        const char* e2c2_environment = std::getenv("Y26_STAGE52_E2C2");
        prepared.e2c2_enabled = e2c2_environment == nullptr ||
            std::string_view(e2c2_environment) != "0";
        const char* small_n_environment = std::getenv("Y26_STAGE53_SMALL_N");
        prepared.small_n_enabled = small_n_environment == nullptr ||
            std::string_view(small_n_environment) != "0";
        const char* rgb_stem_environment = std::getenv("Y26_STAGE53_RGB_STEM");
        prepared.rgb_stem_enabled = rgb_stem_environment == nullptr ||
            std::string_view(rgb_stem_environment) != "0";
        const char* fused_lut_environment = std::getenv("Y26_STAGE53_FUSED_LUT");
        prepared.fused_lut_enabled = fused_lut_environment != nullptr &&
            std::string_view(fused_lut_environment) == "1";
        const char* direct_1x1_environment = std::getenv("Y26_STAGE54_DIRECT_1X1");
        prepared.direct_1x1_enabled = direct_1x1_environment != nullptr &&
            std::string_view(direct_1x1_environment) == "1";
        const char* e2c3_environment = std::getenv("Y26_STAGE54_E2C3");
        prepared.e2c3_enabled = e2c3_environment != nullptr &&
            std::string_view(e2c3_environment) == "1";
        const char* e2c4_environment = std::getenv("Y26_STAGE55_E2C4");
        prepared.e2c4_enabled = e2c4_environment != nullptr &&
            std::string_view(e2c4_environment) == "1";
        const char* dense_m8_environment = std::getenv("Y26_STAGE54_DENSE_M8");
        prepared.dense_m8_enabled = dense_m8_environment != nullptr &&
            std::string_view(dense_m8_environment) == "1";
        const char* dense_weight_stationary_environment =
            std::getenv("Y26_STAGE54_DENSE_WEIGHT_STATIONARY");
        prepared.dense_weight_stationary_enabled =
            dense_weight_stationary_environment != nullptr &&
            std::string_view(dense_weight_stationary_environment) == "1";
        prepared.stage55_dense_family_a_enabled =
            std::getenv("Y26_STAGE55_DENSE_FAMILY_A") != nullptr;
        prepared.stage55_dense_family_b_enabled =
            std::getenv("Y26_STAGE55_DENSE_FAMILY_B") != nullptr;
        const char* dense_partition_environment = std::getenv("Y26_STAGE54_DENSE_PARTITION");
        if (dense_partition_environment != nullptr) {
            const std::string_view partition(dense_partition_environment);
            if (partition == "output") prepared.dense_partition = 1;
            else if (partition == "2d") prepared.dense_partition = 2;
        }
        const char* dense_pack_rvv_environment = std::getenv("Y26_STAGE54_DENSE_PACK_RVV");
        prepared.dense_pack_rvv_enabled = dense_pack_rvv_environment != nullptr &&
            std::string_view(dense_pack_rvv_environment) == "1";
        const char* static_schedule_environment = std::getenv("Y26_STAGE54_STATIC_SCHEDULE");
        prepared.static_schedule_enabled = static_schedule_environment != nullptr &&
            std::string_view(static_schedule_environment) == "1";
        const char* depthwise_v2_environment = std::getenv("Y26_STAGE54_DEPTHWISE_V2");
        prepared.depthwise_v2_enabled = depthwise_v2_environment != nullptr &&
            std::string_view(depthwise_v2_environment) == "1";
        const char* depthwise_x2_environment = std::getenv("Y26_STAGE54_DEPTHWISE_X2");
        prepared.depthwise_x2_enabled = depthwise_x2_environment != nullptr &&
            std::string_view(depthwise_x2_environment) == "1";
        const char* depthwise_border_v2_environment =
            std::getenv("Y26_STAGE54_DEPTHWISE_BORDER_V2");
        prepared.depthwise_border_v2_enabled = depthwise_border_v2_environment != nullptr &&
            std::string_view(depthwise_border_v2_environment) == "1";
        prepared.stage55_depthwise_e2c4_enabled =
            std::getenv("Y26_STAGE55_DEPTHWISE_E2C4") != nullptr;
        const char* lut2_rvv_environment = std::getenv("Y26_STAGE54_LUT2_RVV");
        prepared.lut2_rvv_enabled = lut2_rvv_environment != nullptr &&
            std::string_view(lut2_rvv_environment) == "1";
        const char* input_rvv_v2_environment = std::getenv("Y26_STAGE54_INPUT_RVV_V2");
        prepared.input_rvv_v2_enabled = input_rvv_v2_environment != nullptr &&
            std::string_view(input_rvv_v2_environment) == "1";
        const char* input_compact_c3_environment = std::getenv("Y26_STAGE54_INPUT_COMPACT_C3");
        prepared.input_compact_c3_enabled = input_compact_c3_environment != nullptr &&
            std::string_view(input_compact_c3_environment) == "1";
        const char* input_stem_fused_environment = std::getenv("Y26_STAGE54_INPUT_STEM_FUSED");
        prepared.input_stem_fused_enabled = input_stem_fused_environment != nullptr &&
            std::string_view(input_stem_fused_environment) == "1";
        const char* attention_v2_environment = std::getenv("Y26_STAGE54_ATTENTION_V2");
        prepared.attention_v2_enabled = attention_v2_environment != nullptr &&
            std::string_view(attention_v2_environment) == "1";
        prepared.attention_subphase_profile_enabled =
            std::getenv("Y26_STAGE55_ATTENTION_PROFILE") != nullptr;
        const char* head_v2_environment = std::getenv("Y26_STAGE54_HEAD_V2");
        prepared.head_v2_enabled = head_v2_environment != nullptr &&
            std::string_view(head_v2_environment) == "1";
        std::size_t arena_bytes = 0;
        for (const Row& row : read_tsv(prepared.package / "tensors.tsv")) {
            Tensor tensor;
            tensor.id = integer(row, "id");
            tensor.name = value(row, "name");
            tensor.rank = integer(row, "rank");
            tensor.dims = {integer(row, "dim0"), integer(row, "dim1"),
                           integer(row, "dim2"), integer(row, "dim3")};
            tensor.logical_elements = size_field(row, "logical_elements");
            tensor.storage_bytes = size_field(row, "storage_bytes");
            tensor.offset = size_field(row, "arena_offset");
            tensor.zero_point = integer(row, "zero_point");
            tensor.layout = value(row, "layout") == int8_v1::kNchwc8LayoutId
                ? Layout::feature_nchwc8 : Layout::linear;
            if (tensor.id != static_cast<int>(prepared.tensors.size()) || tensor.rank < 1 || tensor.rank > 4 ||
                tensor.offset > std::numeric_limits<std::size_t>::max() - tensor.storage_bytes) {
                throw std::runtime_error("invalid full-graph tensor descriptor");
            }
            arena_bytes = std::max(arena_bytes, tensor.offset + tensor.storage_bytes);
            prepared.tensors.push_back(std::move(tensor));
        }
        prepared.arena.resize(arena_bytes);
        if (config.capture_boundaries) {
            prepared.captured.resize(prepared.tensors.size());
            for (const Tensor& tensor : prepared.tensors) {
                prepared.captured[static_cast<std::size_t>(tensor.id)].resize(tensor.logical_elements);
            }
        }
        for (const Row& row : read_tsv(prepared.package / "operations.tsv")) {
            Operation operation;
            operation.index = integer(row, "index");
            operation.kind = parse_kind(value(row, "kind"));
            operation.name = value(row, "name");
            operation.output = integer(row, "output");
            operation.inputs = parse_ints(optional_value(row, "inputs"), ',');
            if (operation.index != static_cast<int>(prepared.operations.size())) {
                throw std::runtime_error("operation index mismatch");
            }
            if (operation.kind == OpKind::conv_dense || operation.kind == OpKind::conv_grouped) {
                Conv& conv = operation.conv;
                conv.output_c = integer(row, "output_c");
                conv.input_c = integer(row, "input_c");
                conv.group = integer(row, "group");
                conv.kernel_h = integer(row, "kernel_h");
                conv.kernel_w = integer(row, "kernel_w");
                conv.stride_h = integer(row, "stride_h");
                conv.stride_w = integer(row, "stride_w");
                conv.pad_top = integer(row, "pad_top");
                conv.pad_left = integer(row, "pad_left");
                conv.accumulator_bound = static_cast<std::uint64_t>(parse_i64(value(row, "accumulator_bound"), "bound"));
                conv.e2c_compatible = integer(row, "e2c_compatible") != 0;
                const std::size_t weight_count = static_cast<std::size_t>(conv.output_c) *
                    (conv.input_c / conv.group) * conv.kernel_h * conv.kernel_w;
                conv.weights = read_binary<std::int8_t>(prepared.package / value(row, "weight_file"), weight_count);
                conv.bias = read_binary<std::int32_t>(prepared.package / value(row, "bias_file"), conv.output_c);
                conv.multiplier = read_binary<std::int64_t>(prepared.package / value(row, "multiplier_file"), conv.output_c);
                conv.shift = read_binary<std::int32_t>(prepared.package / value(row, "shift_file"), conv.output_c);
                if (operation.kind == OpKind::conv_dense) {
                    const int input_blocks = (conv.input_c + 7) / 8;
                    const int k_tiles = conv.kernel_h * conv.kernel_w * input_blocks;
                    const int output_blocks = (conv.output_c + kDenseN - 1) / kDenseN;
                    conv.packed_weights = read_binary<std::int8_t>(
                        prepared.package / value(row, "packed_weight_file"),
                        static_cast<std::size_t>(output_blocks) * k_tiles * kDenseN * 8U);
                    conv.weight_sums = read_binary<std::int64_t>(
                        prepared.package / value(row, "weight_sum_file"), conv.output_c);
                    conv.corrected_bias = read_binary<std::int64_t>(
                        prepared.package / value(row, "corrected_bias_file"), conv.output_c);
                    conv.multiplier_m63 = read_binary<std::int64_t>(
                        prepared.package / value(row, "multiplier_m63_file"), conv.output_c);
                }
                prepared.total_weight_bytes += conv.weights.size();
            } else if (operation.kind == OpKind::lut1 || operation.kind == OpKind::lut2 ||
                       operation.kind == OpKind::split || operation.kind == OpKind::reshape ||
                       operation.kind == OpKind::transpose || operation.kind == OpKind::reshape_split_transpose ||
                       operation.kind == OpKind::resize) {
                const std::size_t count = operation.kind == OpKind::lut2 ? 65536U : 256U;
                operation.lut = read_binary<std::int8_t>(prepared.package / value(row, "lut_file"), count);
                operation.axis = integer(row, "axis", 1);
                operation.part = integer(row, "part", 0);
                operation.parts = integer(row, "parts", 1);
                const Tensor& split_output = prepared.tensors[static_cast<std::size_t>(operation.output)];
                const int split_axis = operation.axis < 0 ? split_output.rank + operation.axis : operation.axis;
                operation.split_offset = integer(row, "split_offset", operation.part *
                    split_output.dims[static_cast<std::size_t>(split_axis)]);
                operation.source_shape = parse_ints(optional_value(row, "source_shape"), 'x');
                operation.perm = parse_ints(optional_value(row, "perm"), ',');
            } else if (operation.kind == OpKind::concat) {
                operation.axis = integer(row, "axis", 1);
                const int branch_count = integer(row, "branch_count");
                for (int branch_index = 0; branch_index < branch_count; ++branch_index) {
                    Branch branch;
                    const std::string prefix = "branch" + std::to_string(branch_index) + "_";
                    branch.input_slot = integer(row, (prefix + "input_slot").c_str());
                    branch.transform = parse_branch_transform(optional_value(row, prefix + "transform", "copy"));
                    branch.axis = integer(row, (prefix + "axis").c_str(), 1);
                    branch.part = integer(row, (prefix + "part").c_str(), 0);
                    branch.parts = integer(row, (prefix + "parts").c_str(), 1);
                    const std::string file = optional_value(row, prefix + "lut_file", "-");
                    if (file != "-") branch.lut = read_binary<std::int8_t>(prepared.package / file, 256);
                    operation.branches.push_back(std::move(branch));
                }
            } else if (operation.kind == OpKind::matmul) {
                operation.multiplier = parse_i64(value(row, "multiplier"), "multiplier");
                operation.right_shift = integer(row, "right_shift");
                operation.left_zero_point = integer(row, "left_zero_point");
                operation.right_zero_point = integer(row, "right_zero_point");
                operation.output_zero_point = integer(row, "output_zero_point");
                const Tensor& left = prepared.tensors[static_cast<std::size_t>(operation.inputs[0])];
                const Tensor& right = prepared.tensors[static_cast<std::size_t>(operation.inputs[1])];
                const Tensor& output = prepared.tensors[static_cast<std::size_t>(operation.output)];
                const int k = left.dims[static_cast<std::size_t>(left.rank - 1)];
                const int n = right.dims[static_cast<std::size_t>(right.rank - 1)];
                operation.matmul_ime_eligible = left.layout == Layout::linear &&
                    right.layout == Layout::linear && output.layout == Layout::linear &&
                    k % 8 == 0 && n % kDenseN == 0 && operation.right_shift == 62 &&
                    operation.multiplier > 0 &&
                    operation.multiplier <= std::numeric_limits<std::int64_t>::max() / 2;
                if (operation.matmul_ime_eligible) operation.multiplier_m63 = operation.multiplier * 2;
            } else if (operation.kind == OpKind::softmax_transpose) {
                operation.axis = integer(row, "axis", -1);
                operation.perm = parse_ints(value(row, "perm"), ',');
                operation.output_zero_point = integer(row, "output_zero_point");
                operation.softmax_reciprocal_q32 = static_cast<std::uint64_t>(
                    parse_i64(value(row, "output_reciprocal_q32"), "softmax reciprocal"));
                operation.exp_q48 = read_binary<std::uint64_t>(prepared.package / value(row, "exp_file"), 256);
            }
            if (operation.kind == OpKind::conv_dense) {
                prepared.prepare_dense_conv(operation);
                Conv& conv = operation.conv;
                const Tensor& output = prepared.tensor(operation.output);
                const int output_m = output.dims[2] * output.dims[3];
                conv.stage55_family_a_weight_stationary =
                    prepared.stage55_dense_family_a_enabled && conv.kernel_h == 1 &&
                    conv.kernel_w == 1 && conv.stride_h == 1 && conv.stride_w == 1 &&
                    conv.input_c <= 96 && output_m >= 6400 && conv.output_c % kDenseN == 0;
                conv.stage55_family_b_m8 = prepared.stage55_dense_family_b_enabled &&
                    conv.kernel_h == 3 && conv.kernel_w == 3 && output_m <= 1600 &&
                    conv.k_tiles * 8 >= 576 && conv.output_c % kDenseN == 0;
                prepared.total_weight_bytes += operation.conv.packed_weights.size();
            } else if (operation.kind == OpKind::conv_grouped) {
                prepared.prepare_depthwise_conv(operation);
                prepared.total_weight_bytes += operation.conv.depthwise_weights_c8.size();
            }
            prepared.operation_runners.push_back(Impl::runner_for(operation.kind));
            prepared.operations.push_back(std::move(operation));
        }
        std::vector<int> tensor_input_uses(prepared.tensors.size(), 0);
        for (const Operation& operation : prepared.operations) {
            for (int input_id : operation.inputs) {
                if (input_id >= 0 && input_id < static_cast<int>(tensor_input_uses.size())) {
                    ++tensor_input_uses[static_cast<std::size_t>(input_id)];
                }
            }
        }
        for (std::size_t index = 0; index + 1 < prepared.operations.size(); ++index) {
            Operation& conv_operation = prepared.operations[index];
            Operation& lut_operation = prepared.operations[index + 1];
            const bool fusable_conv = conv_operation.kind == OpKind::conv_dense ||
                (prepared.depthwise_v2_enabled && conv_operation.kind == OpKind::conv_grouped &&
                 conv_operation.conv.depthwise_rvv_eligible);
            if (!fusable_conv || lut_operation.kind != OpKind::lut1 ||
                lut_operation.inputs.size() != 1 || lut_operation.inputs[0] != conv_operation.output ||
                tensor_input_uses[static_cast<std::size_t>(conv_operation.output)] != 1 ||
                lut_operation.lut.size() != 256) {
                continue;
            }
            const Tensor& preactivation = prepared.tensor(conv_operation.output);
            const Tensor& activation = prepared.tensor(lut_operation.output);
            if (preactivation.layout != activation.layout ||
                preactivation.dims != activation.dims ||
                preactivation.storage_bytes != activation.storage_bytes) {
                continue;
            }
            const auto overlaps_activation = [&](const Tensor& candidate) {
                return activation.offset < candidate.offset + candidate.storage_bytes &&
                    candidate.offset < activation.offset + activation.storage_bytes;
            };
            if (std::any_of(conv_operation.inputs.begin(), conv_operation.inputs.end(),
                    [&](int input_id) {
                        return overlaps_activation(prepared.tensor(input_id));
                    })) {
                continue;
            }
            conv_operation.fused_lut_output = lut_operation.output;
            conv_operation.fused_lut = lut_operation.lut;
            lut_operation.skip_when_fused = true;
        }
        std::size_t maximum_panel_bytes = 0;
        std::size_t maximum_matmul_right_bytes = 0;
        std::size_t maximum_matmul_right_sums = 0;
        for (const Operation& operation : prepared.operations) {
            if (operation.conv.dense_ime_eligible) {
                maximum_panel_bytes = std::max(maximum_panel_bytes,
                    static_cast<std::size_t>(operation.conv.k_tiles) * kDenseM * 8U);
            }
            if (operation.matmul_ime_eligible) {
                const Tensor& left = prepared.tensor(operation.inputs[0]);
                const Tensor& right = prepared.tensor(operation.inputs[1]);
                const int k = left.dims[static_cast<std::size_t>(left.rank - 1)];
                const int n = right.dims[static_cast<std::size_t>(right.rank - 1)];
                const std::size_t matrices = right.logical_elements /
                    (static_cast<std::size_t>(k) * n);
                maximum_panel_bytes = std::max(maximum_panel_bytes,
                    static_cast<std::size_t>(k / 8) * kDenseM * 8U);
                maximum_matmul_right_bytes = std::max(maximum_matmul_right_bytes,
                    right.logical_elements);
                maximum_matmul_right_sums = std::max(maximum_matmul_right_sums,
                    matrices * static_cast<std::size_t>(n));
            }
        }
        prepared.dense_scratch.resize(static_cast<std::size_t>(config.workers));
        for (Impl::DenseScratch& scratch : prepared.dense_scratch) {
            scratch.a.resize(maximum_panel_bytes);
        }
        prepared.matmul_packed_right.resize(maximum_matmul_right_bytes);
        prepared.matmul_right_sums.resize(maximum_matmul_right_sums);
        for (const Row& row : read_tsv(prepared.package / "head_assets.tsv")) {
            const int index = integer(row, "scale_index");
            if (index < 0 || index >= 3) throw std::runtime_error("invalid head scale");
            HeadScale& scale = prepared.head[static_cast<std::size_t>(index)];
            scale.resolution = integer(row, "resolution");
            scale.stride = integer(row, "stride");
            scale.reg_tensor = integer(row, "reg_tensor");
            scale.cls_tensor = integer(row, "cls_tensor");
            scale.reg_q16 = read_binary<std::int32_t>(prepared.package / value(row, "reg_lut_file"), 4U * 256U);
            scale.cls_q24 = read_binary<std::uint32_t>(prepared.package / value(row, "cls_lut_file"), 256);
        }
        const auto find_tensor = [&](std::string_view name) {
            const auto found = std::find_if(prepared.tensors.begin(), prepared.tensors.end(),
                [&](const Tensor& candidate) { return candidate.name == name; });
            return found == prepared.tensors.end() ? -1 : found->id;
        };
        prepared.core_input_tensor = find_tensor(
            "/model.4/cv2/conv/Conv_output_0_QuantizeLinear_Output");
        prepared.core_output_tensor = find_tensor(
            "/model.9/Add_output_0_QuantizeLinear_Output");
        const int core_first_output = find_tensor(
            "/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output");
        for (const Operation& operation : prepared.operations) {
            if (operation.output == core_first_output) prepared.core_start_operation = operation.index;
            if (operation.output == prepared.core_output_tensor) prepared.core_end_operation = operation.index;
        }
        if (prepared.core_start_operation > 0) {
            Operation& core_entry = prepared.operations[
                static_cast<std::size_t>(prepared.core_start_operation)];
            Operation& producer = prepared.operations[
                static_cast<std::size_t>(prepared.core_start_operation - 1)];
            if (core_entry.skip_when_fused &&
                producer.fused_lut_output == core_entry.output) {
                producer.fused_lut_output = -1;
                producer.fused_lut.clear();
                core_entry.skip_when_fused = false;
            }
        }
        const std::filesystem::path core_package = prepared.package / "optimized_core";
        if (config.compute == ComputeMode::optimized && std::filesystem::exists(core_package / "package.json")) {
            prepared.optimized_core = std::make_unique<stage49::PersistentSlice>();
            const std::string core_manifest = int8_v1::sha256_file(core_package / "asset_hashes.tsv");
            if (prepared.optimized_core->prepare_with_contract(
                    core_package, core_manifest, config.workers,
                    int8_v1::kContractId, int8_v1::kGeneralProfile, false) != 0 ||
                prepared.core_start_operation < 0 || prepared.core_end_operation < prepared.core_start_operation) {
                throw std::runtime_error("cannot prepare optimized resident core");
            }
            prepared.optimized_core_last_operation = 35;
            const std::array<std::string_view, 6> full_names {
                "/model.4/cv2/act/Mul_output_0_QuantizeLinear_Output",
                "/model.6/cv2/act/Mul_output_0_QuantizeLinear_Output",
                "/model.8/cv2/act/Mul_output_0_QuantizeLinear_Output",
                "/model.9/Concat_output_0_QuantizeLinear_Output",
                "/model.9/cv2/conv/Conv_output_0_QuantizeLinear_Output",
                "/model.9/Add_output_0_QuantizeLinear_Output",
            };
            const std::array<std::string_view, 6> core_keys {
                "model4.postact", "model.6.output", "model.8.output", "model.9.concat",
                "model.9.cv2_preact", "model.9.output",
            };
            for (std::size_t index = 0; index < prepared.core_bridges.size(); ++index) {
                prepared.core_bridges[index].full_tensor = find_tensor(full_names[index]);
                prepared.core_bridges[index].core_tensor =
                    prepared.optimized_core->tensor_id_for_key(std::string(core_keys[index]));
            }
            if (prepared.optimized_core->operation_count() <= prepared.optimized_core_last_operation ||
                std::any_of(prepared.core_bridges.begin(), prepared.core_bridges.end(),
                    [](const Impl::CoreBridge& bridge) {
                        return bridge.full_tensor < 0 || bridge.core_tensor < 0;
                    })) {
                throw std::runtime_error("optimized model4-to-model9 boundary is absent");
            }
            prepared.core_scratch_offset = (prepared.arena.size() + 63U) & ~std::size_t {63U};
            prepared.core_scratch_bytes = prepared.optimized_core->arena_bytes();
            prepared.arena.resize(prepared.core_scratch_offset + prepared.core_scratch_bytes);
        }
        prepared.static_batch_end.assign(prepared.operations.size(), -1);
        if (prepared.static_schedule_enabled) {
            std::size_t cursor = 0;
            while (cursor < prepared.operations.size()) {
                if (prepared.core_start_operation >= 0 &&
                    cursor == static_cast<std::size_t>(prepared.core_start_operation)) {
                    cursor = static_cast<std::size_t>(prepared.core_end_operation + 1);
                    continue;
                }
                if (!Impl::static_schedule_eligible(prepared.operations[cursor].kind)) {
                    ++cursor;
                    continue;
                }
                const std::size_t begin = cursor;
                while (cursor < prepared.operations.size() &&
                       Impl::static_schedule_eligible(prepared.operations[cursor].kind) &&
                       !(prepared.core_start_operation >= 0 &&
                         cursor == static_cast<std::size_t>(prepared.core_start_operation))) {
                    ++cursor;
                }
                if (cursor - begin >= 2U) {
                    prepared.static_batch_end[begin] = static_cast<int>(cursor - 1U);
                }
            }
        }
        if (!prepared.optimized_core || config.scheduler != SchedulerMode::safe) {
            prepared.pool = std::make_unique<WorkerPool>(
                config.workers, config.worker_cpu_begin, config.scheduler);
        }
        if (!prepared.controller_affinity_ok || !prepared.worker_affinity_ok()) {
            throw std::runtime_error("executor CPU affinity or scheduler policy could not be established");
        }
        prepared.ready = true;
        *impl_ = std::move(prepared);
        return 0;
    } catch (const std::exception& error) {
        impl_->error = error.what();
        return 2;
    }
}

int FullExecutor::run_preprocessed(const float* input, std::size_t input_count,
                                   float* output, std::size_t output_count,
                                   RunTiming* timing) {
    if (input == nullptr || input_count != 3U * 640U * 640U) return 1;
    return run_input_surface(input, nullptr, 0, output, output_count, timing);
}

int FullExecutor::run_input_surface(const float* input, const std::uint8_t* rgb,
                                    int rgb_stride, float* output,
                                    std::size_t output_count, RunTiming* timing) {
    if (!impl_ || !impl_->ready || output == nullptr || output_count != 300U * 6U ||
        ((input == nullptr) == (rgb == nullptr))) return 1;
    try {
        struct ActiveWindowGuard {
            stage49::PersistentSlice* core = nullptr;
            explicit ActiveWindowGuard(stage49::PersistentSlice* selected) : core(selected) {
                if (core != nullptr) core->begin_active_window();
            }
            ~ActiveWindowGuard() {
                if (core != nullptr) core->end_active_window();
            }
        } active_window(impl_->optimized_core.get());
        impl_->current_float_input = input;
        RunTiming local;
        const bool profile_operations = std::getenv("Y26_STAGE52_PROFILE_OPS") != nullptr ||
                                        std::getenv("Y26_STAGE53_PROFILE_OPS") != nullptr;
        if (impl_->attention_subphase_profile_enabled) {
            impl_->attention_profile->matmul_pack_ns.store(0, std::memory_order_relaxed);
            impl_->attention_profile->matmul_compute_ns.store(0, std::memory_order_relaxed);
            impl_->attention_profile->softmax_max_sum_ns.store(0, std::memory_order_relaxed);
            impl_->attention_profile->softmax_normalize_transpose_ns.store(0, std::memory_order_relaxed);
        }
        int diagnostic_stop_after = -1;
        if (const char* stop = std::getenv("Y26_STAGE54_STOP_AFTER_OP"); stop != nullptr) {
            const char* end = stop + std::strlen(stop);
            const auto parsed = std::from_chars(stop, end, diagnostic_stop_after);
            if (parsed.ec != std::errc {} || parsed.ptr != end || diagnostic_stop_after < 0) {
                throw std::runtime_error("invalid Y26_STAGE54_STOP_AFTER_OP");
            }
        }
        bool diagnostic_stopped = false;
        std::vector<OperationProfileSample> profile_samples;
        if (profile_operations) profile_samples.reserve(impl_->operations.size() + 8U);
        const char* perf_group = std::getenv("Y26_STAGE55_PERF_GROUP");
        if (perf_group != nullptr && perf_group[0] != '\0' && impl_->optimized_core) {
            impl_->optimized_core->begin_worker_counter_collection(perf_group);
        }
        const std::clock_t process_cpu_begin = std::clock();
#if defined(__linux__)
        rusage usage_begin {};
        rusage usage_end {};
        const bool usage_available = getrusage(RUSAGE_SELF, &usage_begin) == 0;
#endif
        const auto total_begin = Clock::now();
        for (std::size_t operation_index = 0; operation_index < impl_->operations.size(); ++operation_index) {
            const Operation& operation = impl_->operations[operation_index];
            const int static_end = impl_->static_schedule_enabled && !profile_operations &&
                    !impl_->config.capture_boundaries
                ? impl_->static_batch_end[operation_index] : -1;
            if (static_end >= static_cast<int>(operation_index)) {
                impl_->dispatch_static_batch(operation_index, static_cast<std::size_t>(static_end),
                                             input, rgb, rgb_stride);
                operation_index = static_cast<std::size_t>(static_end);
                continue;
            }
            if (operation.skip_when_fused && impl_->fused_lut_enabled &&
                !impl_->config.capture_boundaries) {
                if (profile_operations) {
                    profile_samples.push_back(OperationProfileSample {
                        operation.index, -1, "lut1", "fused_into_dense_conv",
                        operation.name, 0.0});
                }
                continue;
            }
            const auto begin = Clock::now();
            if (impl_->optimized_core && !impl_->config.capture_boundaries &&
                operation.index == impl_->core_start_operation) {
                stage49::SliceTiming resident_timing;
                impl_->run_optimized_core(profile_operations ? &resident_timing : nullptr);
                operation_index = static_cast<std::size_t>(impl_->core_end_operation);
                const double duration = elapsed_us(begin, Clock::now());
                local.resident_core_us += duration;
                if (profile_operations) {
                    double resident_operation_sum_us = 0.0;
                    for (const stage49::OperationTiming& row : resident_timing.operations) {
                        profile_samples.push_back(OperationProfileSample {
                            impl_->full_operation_index(row.name), row.operation_index,
                            row.kind, "resident_core", row.name, row.wall_us});
                        resident_operation_sum_us += row.wall_us;
                    }
                    profile_samples.push_back(OperationProfileSample {
                        -1, -1, "bridge", "resident_bridge", "resident_core_bridge_overhead",
                        std::max(0.0, duration - resident_operation_sum_us)});
                }
                continue;
            }
            impl_->operation_runners[operation_index](
                impl_.get(), operation, input, rgb, rgb_stride);
            const double duration = elapsed_us(begin, Clock::now());
            if (profile_operations) {
                profile_samples.push_back(OperationProfileSample {
                    operation.index, -1, operation_kind_name(operation.kind),
                    "full_executor", operation.name, duration});
            }
            if (operation.kind == OpKind::input_quant) local.input_quantize_us += duration;
            if (operation.kind == OpKind::conv_dense) local.dense_conv_us += duration;
            if (operation.kind == OpKind::matmul || operation.kind == OpKind::softmax_transpose) {
                local.attention_us += duration;
            }
            if (operation.kind == OpKind::conv_grouped) local.depthwise_us += duration;
            if (operation.kind == OpKind::lut1 || operation.kind == OpKind::lut2) local.lut_us += duration;
            if (operation.kind == OpKind::concat) local.concat_us += duration;
            if (operation.kind == OpKind::split || operation.kind == OpKind::reshape ||
                operation.kind == OpKind::transpose || operation.kind == OpKind::reshape_split_transpose ||
                operation.kind == OpKind::resize) {
                local.transform_us += duration;
            }
            if (impl_->config.capture_boundaries) {
                const Tensor& tensor = impl_->tensor(operation.output);
                auto& captured = impl_->captured[static_cast<std::size_t>(operation.output)];
                if (captured.size() != tensor.logical_elements) {
                    throw std::runtime_error("captured boundary size mismatch");
                }
                for (std::size_t index = 0; index < tensor.logical_elements; ++index) {
                    captured[index] = int8_v1::signed_storage(impl_->code(operation.output, index));
                }
            }
            if (operation.index == diagnostic_stop_after) {
                diagnostic_stopped = true;
                break;
            }
        }
        if (diagnostic_stopped) {
            std::fill(output, output + output_count, 0.0F);
        } else {
            const auto head_begin = Clock::now();
            impl_->decode_head(output);
            local.head_us = elapsed_us(head_begin, Clock::now());
            if (profile_operations) {
                profile_samples.push_back(OperationProfileSample {
                    -1, -1, "head_decode", "full_executor", "final_head_decode", local.head_us});
            }
        }
        local.total_us = elapsed_us(total_begin, Clock::now());
        const std::clock_t process_cpu_end = std::clock();
        if (process_cpu_begin != static_cast<std::clock_t>(-1) &&
            process_cpu_end != static_cast<std::clock_t>(-1)) {
            local.process_cpu_us = static_cast<double>(process_cpu_end - process_cpu_begin) *
                                   1000000.0 / CLOCKS_PER_SEC;
        }
#if defined(__linux__)
        if (usage_available && getrusage(RUSAGE_SELF, &usage_end) == 0) {
            local.voluntary_context_switches = static_cast<std::uint64_t>(
                std::max<long>(0, usage_end.ru_nvcsw - usage_begin.ru_nvcsw));
            local.involuntary_context_switches = static_cast<std::uint64_t>(
                std::max<long>(0, usage_end.ru_nivcsw - usage_begin.ru_nivcsw));
        }
#endif
        local.output_hash = fnv1a64(output, output_count);
        local.affinity_ok = impl_->controller_affinity_ok && impl_->worker_affinity_ok() ? 1 : 0;
        if (perf_group != nullptr && perf_group[0] != '\0' && impl_->optimized_core) {
            for (const stage49::WorkerCounter& counter : impl_->optimized_core->worker_counters()) {
                std::fprintf(stderr,
                    "stage55_worker_counter\t%d\t%d\t%d\t%s\t%s\t%d\t%llu\t%llu\t%llu\t%llu\t%llu\n",
                    counter.worker, counter.worker_tid, counter.worker_cpu,
                    counter.event.c_str(), counter.status.c_str(), counter.error_number,
                    static_cast<unsigned long long>(counter.event_id),
                    static_cast<unsigned long long>(counter.iterations),
                    static_cast<unsigned long long>(counter.count),
                    static_cast<unsigned long long>(counter.time_enabled),
                    static_cast<unsigned long long>(counter.time_running));
            }
        }
        std::copy(output, output + output_count, impl_->last_output.begin());
        if (profile_operations) {
            const std::uint64_t run_id = impl_->profile_run_sequence++;
            double operation_sum_us = 0.0;
            for (const OperationProfileSample& sample : profile_samples) {
                operation_sum_us += sample.wall_us;
                std::fprintf(stderr,
                    "stage53_op\t%llu\t%d\t%d\t%s\t%s\t%.3f\t%s\n",
                    static_cast<unsigned long long>(run_id), sample.full_operation_index,
                    sample.resident_operation_index, sample.kind.c_str(), sample.source.c_str(),
                    sample.wall_us, sample.name.c_str());
            }
            std::fprintf(stderr, "stage53_profile_run\t%llu\t%.3f\t%.3f\t%zu\n",
                static_cast<unsigned long long>(run_id), local.total_us, operation_sum_us,
                profile_samples.size());
        }
        if (impl_->attention_subphase_profile_enabled) {
            const auto run_id = impl_->attention_profile_run_sequence++;
            std::fprintf(stderr, "stage55_attention_phase\t%llu\tmatmul_pack\t%llu\n",
                static_cast<unsigned long long>(run_id),
                static_cast<unsigned long long>(impl_->attention_profile->matmul_pack_ns.load(
                    std::memory_order_relaxed)));
            std::fprintf(stderr, "stage55_attention_phase\t%llu\tmatmul_compute\t%llu\n",
                static_cast<unsigned long long>(run_id),
                static_cast<unsigned long long>(impl_->attention_profile->matmul_compute_ns.load(
                    std::memory_order_relaxed)));
            std::fprintf(stderr, "stage55_attention_phase\t%llu\tsoftmax_max_sum\t%llu\n",
                static_cast<unsigned long long>(run_id),
                static_cast<unsigned long long>(impl_->attention_profile->softmax_max_sum_ns.load(
                    std::memory_order_relaxed)));
            std::fprintf(stderr,
                "stage55_attention_phase\t%llu\tsoftmax_normalize_transpose\t%llu\n",
                static_cast<unsigned long long>(run_id),
                static_cast<unsigned long long>(
                    impl_->attention_profile->softmax_normalize_transpose_ns.load(
                        std::memory_order_relaxed)));
        }
        if (timing != nullptr) *timing = local;
        impl_->current_float_input = nullptr;
        return 0;
    } catch (const std::exception& error) {
        impl_->current_float_input = nullptr;
        impl_->error = error.what();
        return 3;
    }
}

int FullExecutor::run_rgb(const std::uint8_t* rgb, int width, int height, int stride,
                          float* output, std::size_t output_count, RunTiming* timing) {
    if (!impl_ || rgb == nullptr || width <= 0 || height <= 0 || stride < width * 3) return 1;
    // This bounded API route intentionally accepts only the already-letterboxed
    // 640x640 RGB surface.  The CLI performs file decode and letterbox explicitly.
    if (width != 640 || height != 640) {
        impl_->error = "run_rgb requires a 640x640 letterboxed RGB image";
        return 4;
    }
    return run_input_surface(nullptr, rgb, stride, output, output_count, timing);
}

int FullExecutor::diagnostic_benchmark_conv_shape(
    int operation_index, int output_h, int output_w, int output_channels,
    int warmup, int runs, DiagnosticConvShapeResult* result) {
    if (!impl_ || !impl_->ready || result == nullptr || output_h <= 0 || output_w <= 0 ||
        warmup < 0 || runs <= 0) {
        return 1;
    }
    try {
        const auto found = std::find_if(
            impl_->operations.begin(), impl_->operations.end(),
            [operation_index](const Operation& operation) {
                return operation.index == operation_index;
            });
        if (found == impl_->operations.end() ||
            (found->kind != OpKind::conv_dense && found->kind != OpKind::conv_grouped)) {
            throw std::runtime_error("diagnostic operation is not a Conv");
        }
        Operation& operation = *found;
        Conv& conv = operation.conv;
        if (conv.rgb_stem_rvv_eligible) {
            throw std::runtime_error("diagnostic lattice excludes the dedicated RGB stem");
        }
        Tensor& input = impl_->tensors[static_cast<std::size_t>(operation.inputs[0])];
        Tensor& output = impl_->tensors[static_cast<std::size_t>(operation.output)];
        if (input.rank != 4 || output.rank != 4 ||
            input.layout != Layout::feature_nchwc8 ||
            output.layout != Layout::feature_nchwc8) {
            throw std::runtime_error("diagnostic Conv is not NCHWc8");
        }
        const bool depthwise = operation.kind == OpKind::conv_grouped;
        const int selected_channels = output_channels > 0 ? output_channels : conv.output_c;
        if (selected_channels <= 0 || selected_channels > conv.output_c ||
            selected_channels % 4 != 0 || (depthwise && selected_channels != conv.output_c)) {
            throw std::runtime_error("invalid diagnostic output-channel subset");
        }
        const int input_h = output_h * conv.stride_h;
        const int input_w = output_w * conv.stride_w;
        if (output_h > output.dims[2] || output_w > output.dims[3] ||
            input_h > input.dims[2] || input_w > input.dims[3]) {
            throw std::runtime_error("diagnostic dimensions may only shrink a prepared operation");
        }

        const Tensor saved_input = input;
        const Tensor saved_output = output;
        Tensor* fused_output = nullptr;
        Tensor saved_fused_output;
        if (operation.fused_lut_output >= 0 && operation.fused_lut_output != operation.output) {
            fused_output = &impl_->tensors[static_cast<std::size_t>(operation.fused_lut_output)];
            saved_fused_output = *fused_output;
        }
        const int saved_output_c = conv.output_c;
        const int saved_n_blocks = conv.n_blocks;
        const bool saved_family_a = conv.stage55_family_a_weight_stationary;
        const bool saved_family_b = conv.stage55_family_b_m8;
        struct RestoreDescriptors {
            Tensor& input;
            Tensor& output;
            Tensor saved_input;
            Tensor saved_output;
            Tensor* fused_output;
            Tensor saved_fused_output;
            Conv& conv;
            int output_c;
            int n_blocks;
            bool family_a;
            bool family_b;
            ~RestoreDescriptors() {
                input = std::move(saved_input);
                output = std::move(saved_output);
                if (fused_output != nullptr) *fused_output = std::move(saved_fused_output);
                conv.output_c = output_c;
                conv.n_blocks = n_blocks;
                conv.stage55_family_a_weight_stationary = family_a;
                conv.stage55_family_b_m8 = family_b;
            }
        } restore {input, output, saved_input, saved_output, fused_output, saved_fused_output,
                   conv, saved_output_c, saved_n_blocks, saved_family_a, saved_family_b};

        const auto set_feature_shape = [](Tensor& tensor, int channels, int height, int width) {
            tensor.dims[1] = channels;
            tensor.dims[2] = height;
            tensor.dims[3] = width;
            tensor.logical_elements = static_cast<std::size_t>(channels) * height * width;
            tensor.storage_bytes = static_cast<std::size_t>((channels + 7) / 8) *
                height * width * 8U;
        };
        set_feature_shape(input, conv.input_c, input_h, input_w);
        set_feature_shape(output, selected_channels, output_h, output_w);
        if (fused_output != nullptr) {
            set_feature_shape(*fused_output, selected_channels, output_h, output_w);
        }
        if (input.storage_bytes > saved_input.storage_bytes ||
            output.storage_bytes > saved_output.storage_bytes ||
            (fused_output != nullptr &&
             fused_output->storage_bytes > saved_fused_output.storage_bytes)) {
            throw std::runtime_error("diagnostic shape exceeds prepared arena allocation");
        }
        conv.output_c = selected_channels;
        conv.n_blocks = (selected_channels + kDenseN - 1) / kDenseN;
        const int output_m = output_h * output_w;
        conv.stage55_family_a_weight_stationary = impl_->stage55_dense_family_a_enabled &&
            conv.kernel_h == 1 && conv.kernel_w == 1 && conv.stride_h == 1 &&
            conv.stride_w == 1 && conv.input_c <= 96 && output_m >= 6400 &&
            selected_channels % kDenseN == 0;
        conv.stage55_family_b_m8 = impl_->stage55_dense_family_b_enabled &&
            conv.kernel_h == 3 && conv.kernel_w == 3 && output_m <= 1600 &&
            conv.k_tiles * 8 >= 576 && selected_channels % kDenseN == 0;

        auto* input_data = impl_->arena.data() + input.offset;
        for (std::size_t index = 0; index < input.storage_bytes; ++index) {
            input_data[index] = static_cast<std::int8_t>((index * 37U + 11U) & 0xffU);
        }
        Tensor& stored = fused_output != nullptr ? *fused_output : output;
        auto* stored_data = impl_->arena.data() + stored.offset;
        const auto clear_output = [&]() {
            std::fill(stored_data, stored_data + stored.storage_bytes, std::int8_t {0});
        };

        struct ActiveWindowGuard {
            stage49::PersistentSlice* core;
            explicit ActiveWindowGuard(stage49::PersistentSlice* value) : core(value) {
                if (core != nullptr) core->begin_active_window();
            }
            ~ActiveWindowGuard() {
                if (core != nullptr) core->end_active_window();
            }
        } active_window(impl_->optimized_core.get());

        for (int index = 0; index < warmup; ++index) {
            clear_output();
            impl_->run_conv(operation);
        }
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(runs));
        std::uint64_t expected_hash = 0;
        bool deterministic = true;
        for (int index = 0; index < runs; ++index) {
            clear_output();
            const auto begin = Clock::now();
            impl_->run_conv(operation);
            const auto end = Clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
            std::uint64_t hash = 1469598103934665603ULL;
            for (std::size_t byte = 0; byte < stored.storage_bytes; ++byte) {
                hash ^= static_cast<std::uint8_t>(stored_data[byte]);
                hash *= 1099511628211ULL;
            }
            if (index == 0) expected_hash = hash;
            deterministic = deterministic && hash == expected_hash;
        }
        std::vector<double> ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        const auto percentile = [&](double fraction) {
            const double position = fraction * static_cast<double>(ordered.size() - 1U);
            const std::size_t lower = static_cast<std::size_t>(position);
            const std::size_t upper = std::min(lower + 1U, ordered.size() - 1U);
            const double part = position - static_cast<double>(lower);
            return ordered[lower] + (ordered[upper] - ordered[lower]) * part;
        };
        result->operation_index = operation.index;
        result->operation_name = operation.name;
        result->operation_kind = operation_kind_name(operation.kind);
        result->output_h = output_h;
        result->output_w = output_w;
        result->output_c = selected_channels;
        result->input_c = conv.input_c;
        result->kernel_h = conv.kernel_h;
        result->kernel_w = conv.kernel_w;
        result->stride_h = conv.stride_h;
        result->stride_w = conv.stride_w;
        result->m = output_m;
        result->n = selected_channels;
        result->k = conv.input_c * conv.kernel_h * conv.kernel_w;
        result->working_set_bytes = input.storage_bytes + stored.storage_bytes;
        result->packed_weight_bytes = conv.packed_weights.empty()
            ? conv.depthwise_weights_c8.size() : conv.packed_weights.size();
        result->mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) /
            static_cast<double>(samples.size());
        result->median_us = percentile(0.5);
        result->p95_us = percentile(0.95);
        result->maximum_us = ordered.back();
        result->output_hash = expected_hash;
        result->deterministic = deterministic;
        if (!deterministic) throw std::runtime_error("diagnostic Conv output is not deterministic");
        return 0;
    } catch (const std::exception& error) {
        impl_->error = error.what();
        return 2;
    }
}

int FullExecutor::copy_boundary(int tensor_id, std::uint8_t* output, std::size_t bytes) const {
    if (!impl_ || !impl_->ready || output == nullptr || tensor_id < 0 ||
        static_cast<std::size_t>(tensor_id) >= impl_->tensors.size()) return 1;
    const Tensor& tensor = impl_->tensor(tensor_id);
    if (bytes != tensor.logical_elements) return 1;
    if (impl_->config.capture_boundaries) {
        const auto& captured = impl_->captured[static_cast<std::size_t>(tensor_id)];
        if (captured.size() != bytes) return 2;
        for (std::size_t index = 0; index < bytes; ++index) {
            output[index] = int8_v1::semantic_code(captured[index]);
        }
        return 0;
    }
    if (tensor_id == impl_->core_input_tensor &&
        impl_->core_input_diagnostic_snapshot.size() == bytes) {
        for (std::size_t index = 0; index < bytes; ++index) {
            const std::size_t offset = physical_offset(tensor, index) - tensor.offset;
            output[index] = int8_v1::semantic_code(
                impl_->core_input_diagnostic_snapshot[offset]);
        }
        return 0;
    }
    for (const Impl::CoreBridge& bridge : impl_->core_bridges) {
        if (tensor_id == bridge.full_tensor && bridge.diagnostic_snapshot.size() == bytes) {
            for (std::size_t index = 0; index < bytes; ++index) {
                const std::size_t offset = physical_offset(tensor, index) - tensor.offset;
                output[index] = int8_v1::semantic_code(bridge.diagnostic_snapshot[offset]);
            }
            return 0;
        }
    }
    for (std::size_t index = 0; index < bytes; ++index) output[index] = impl_->code(tensor_id, index);
    return 0;
}

int FullExecutor::copy_output(float* output, std::size_t output_count) const {
    if (!impl_ || !impl_->ready || output == nullptr || output_count != impl_->last_output.size()) {
        return 1;
    }
    std::copy(impl_->last_output.begin(), impl_->last_output.end(), output);
    return 0;
}

int FullExecutor::tensor_id_for_name(const std::string& name) const noexcept {
    if (!impl_) return -1;
    const auto found = std::find_if(impl_->tensors.begin(), impl_->tensors.end(),
        [&](const Tensor& tensor) { return tensor.name == name; });
    return found == impl_->tensors.end() ? -1 : found->id;
}

std::size_t FullExecutor::tensor_bytes(int tensor_id) const noexcept {
    if (!impl_ || !impl_->ready || tensor_id < 0 ||
        static_cast<std::size_t>(tensor_id) >= impl_->tensors.size()) {
        return 0;
    }
    return impl_->tensors[static_cast<std::size_t>(tensor_id)].logical_elements;
}

int FullExecutor::operation_count() const noexcept { return impl_ ? static_cast<int>(impl_->operations.size()) : 0; }
int FullExecutor::tensor_count() const noexcept { return impl_ ? static_cast<int>(impl_->tensors.size()) : 0; }
std::size_t FullExecutor::arena_bytes() const noexcept { return impl_ ? impl_->arena.size() : 0; }
std::size_t FullExecutor::packed_weight_bytes() const noexcept { return impl_ ? impl_->total_weight_bytes : 0; }
const std::string& FullExecutor::package_manifest_sha256() const noexcept { return impl_->manifest; }
const std::string& FullExecutor::last_error() const noexcept { return impl_->error; }

const char* scheduler_mode_name(SchedulerMode value) noexcept {
    return value == SchedulerMode::safe ? "safe" : "rr20";
}

const char* compute_mode_name(ComputeMode value) noexcept {
    return value == ComputeMode::scalar ? "scalar" : "optimized";
}

}  // namespace y26::stage52
