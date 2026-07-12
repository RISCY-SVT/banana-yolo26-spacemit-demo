#include "y26_k1x_stage47_aot.h"

#include "y26_k1x_activation.h"
#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace y26::stage47 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int kMaximumWorkers = 4;
constexpr int kNBlock = 16;
constexpr int kExactMultiplierBits = 62;
__extension__ typedef __int128 SignedInt128;
__extension__ typedef unsigned __int128 UnsignedInt128;

struct ExactRequantParams {
    std::int64_t multiplier_q62 = 0;
    int exponent = 0;
    int output_zero_point_u8 = 0;
};

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

int rows_for(KernelShape shape) {
    switch (shape) {
        case KernelShape::scalar:
        case KernelShape::m4n16:
            return 4;
        case KernelShape::m8n16:
            return 8;
        case KernelShape::m12n16:
            return 12;
    }
    return 0;
}

int align_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
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

std::int8_t signed_storage(std::uint8_t code) {
    return static_cast<std::int8_t>(static_cast<int>(code) - 128);
}

std::uint8_t unsigned_code(std::int8_t storage) {
    return static_cast<std::uint8_t>(static_cast<int>(storage) + 128);
}

ExactRequantParams exact_requant_params(double multiplier, int output_zero_point_u8) {
    ExactRequantParams result;
    result.output_zero_point_u8 = output_zero_point_u8;
    if (multiplier == 0.0) {
        return result;
    }
    int exponent = 0;
    const double fraction = std::frexp(multiplier, &exponent);
    result.multiplier_q62 = static_cast<std::int64_t>(std::llround(std::ldexp(fraction, kExactMultiplierBits)));
    result.exponent = exponent;
    return result;
}

std::int64_t round_shift_right_even_128(SignedInt128 value, int shift) {
    if (shift <= 0) {
        return static_cast<std::int64_t>(value << -shift);
    }
    const bool negative = value < 0;
    const UnsignedInt128 magnitude = negative ? static_cast<UnsignedInt128>(-value)
                                               : static_cast<UnsignedInt128>(value);
    if (shift >= 127) {
        return 0;
    }
    UnsignedInt128 quotient = magnitude >> shift;
    const UnsignedInt128 mask = (static_cast<UnsignedInt128>(1) << shift) - 1;
    const UnsignedInt128 remainder = magnitude & mask;
    const UnsignedInt128 half = static_cast<UnsignedInt128>(1) << (shift - 1);
    if (remainder > half || (remainder == half && (quotient & 1U) != 0)) {
        ++quotient;
    }
    const std::int64_t signed_value = static_cast<std::int64_t>(quotient);
    return negative ? -signed_value : signed_value;
}

std::uint8_t exact_requant(std::int32_t value, const ExactRequantParams& params) {
    const SignedInt128 product = static_cast<SignedInt128>(value) * params.multiplier_q62;
    const std::int64_t rounded = round_shift_right_even_128(product, kExactMultiplierBits - params.exponent);
    return static_cast<std::uint8_t>(std::clamp<std::int64_t>(
        rounded + params.output_zero_point_u8, 0, 255));
}

void scalar_block(const std::int8_t* a,
                  const std::int8_t* b,
                  int k_tiles,
                  int rows,
                  std::int32_t* c) {
    std::fill(c, c + static_cast<std::size_t>(rows) * kNBlock, 0);
    const int row_groups = rows / 4;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const std::int8_t* at = a + static_cast<std::size_t>(kt) * rows * 8;
        const std::int8_t* bt = b + static_cast<std::size_t>(kt) * kNBlock * 8;
        for (int ng = 0; ng < 4; ++ng) {
            for (int rg = 0; rg < row_groups; ++rg) {
                std::int32_t* tile = c + static_cast<std::size_t>(ng * row_groups + rg) * 16U;
                for (int r = 0; r < 4; ++r) {
                    for (int n = 0; n < 4; ++n) {
                        std::int32_t sum = tile[r * 4 + n];
                        for (int kk = 0; kk < 8; ++kk) {
                            sum += static_cast<std::int32_t>(at[(rg * 4 + r) * 8 + kk]) *
                                   static_cast<std::int32_t>(bt[(ng * 4 + n) * 8 + kk]);
                        }
                        tile[r * 4 + n] = sum;
                    }
                }
            }
        }
    }
}

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)

#define Y26_STAGE47_INIT_ACC(reg) "vxor.vv " #reg ", " #reg ", " #reg "\n\t"
#define Y26_STAGE47_DOT(acc, a, b) "smt.vmadot " #acc ", " #a ", " #b "\n\t"
#define Y26_STAGE47_STORE(reg) "vse32.v " #reg ", (t2)\n\t" "addi t2, t2, 64\n\t"

__attribute__((noinline)) void kernel_m4n16(const std::int8_t* a,
                                            const std::int8_t* b,
                                            int k_tiles,
                                            std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE47_INIT_ACC(v16) Y26_STAGE47_INIT_ACC(v18)
        Y26_STAGE47_INIT_ACC(v20) Y26_STAGE47_INIT_ACC(v22)
        "vsetvli t0, zero, e8, m1\n\t"
        "mv t1, %[K]\n\t" "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t"
        "vle8.v v2, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v3, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v4, (t5)\n\t"
        "addi t5, t4, 96\n\t" "vle8.v v5, (t5)\n\t"
        Y26_STAGE47_DOT(v16, v0, v2) Y26_STAGE47_DOT(v18, v0, v3)
        Y26_STAGE47_DOT(v20, v0, v4) Y26_STAGE47_DOT(v22, v0, v5)
        "addi t3, t3, 32\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE47_STORE(v16) Y26_STAGE47_STORE(v18)
        Y26_STAGE47_STORE(v20) Y26_STAGE47_STORE(v22)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5",
          "v0", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23");
}

__attribute__((noinline)) void kernel_m8n16(const std::int8_t* a,
                                            const std::int8_t* b,
                                            int k_tiles,
                                            std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE47_INIT_ACC(v16) Y26_STAGE47_INIT_ACC(v18)
        Y26_STAGE47_INIT_ACC(v20) Y26_STAGE47_INIT_ACC(v22)
        Y26_STAGE47_INIT_ACC(v24) Y26_STAGE47_INIT_ACC(v26)
        Y26_STAGE47_INIT_ACC(v28) Y26_STAGE47_INIT_ACC(v30)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t"
        "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "vle8.v v2, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v3, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v4, (t5)\n\t"
        "addi t5, t4, 96\n\t" "vle8.v v5, (t5)\n\t"
        Y26_STAGE47_DOT(v16, v0, v2) Y26_STAGE47_DOT(v18, v1, v2)
        Y26_STAGE47_DOT(v20, v0, v3) Y26_STAGE47_DOT(v22, v1, v3)
        Y26_STAGE47_DOT(v24, v0, v4) Y26_STAGE47_DOT(v26, v1, v4)
        Y26_STAGE47_DOT(v28, v0, v5) Y26_STAGE47_DOT(v30, v1, v5)
        "addi t3, t3, 64\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE47_STORE(v16) Y26_STAGE47_STORE(v18)
        Y26_STAGE47_STORE(v20) Y26_STAGE47_STORE(v22)
        Y26_STAGE47_STORE(v24) Y26_STAGE47_STORE(v26)
        Y26_STAGE47_STORE(v28) Y26_STAGE47_STORE(v30)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v1",
          "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

__attribute__((noinline)) void kernel_m12n16(const std::int8_t* a,
                                             const std::int8_t* b,
                                             int k_tiles,
                                             std::int32_t* c) {
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2\n\t"
        Y26_STAGE47_INIT_ACC(v8) Y26_STAGE47_INIT_ACC(v10)
        Y26_STAGE47_INIT_ACC(v12) Y26_STAGE47_INIT_ACC(v14)
        Y26_STAGE47_INIT_ACC(v16) Y26_STAGE47_INIT_ACC(v18)
        Y26_STAGE47_INIT_ACC(v20) Y26_STAGE47_INIT_ACC(v22)
        Y26_STAGE47_INIT_ACC(v24) Y26_STAGE47_INIT_ACC(v26)
        Y26_STAGE47_INIT_ACC(v28) Y26_STAGE47_INIT_ACC(v30)
        "vsetvli t0, zero, e8, m1\n\t" "mv t1, %[K]\n\t"
        "mv t3, %[A]\n\t" "mv t4, %[B]\n\t"
        "1:\n\t"
        "vle8.v v0, (t3)\n\t" "addi t5, t3, 32\n\t" "vle8.v v1, (t5)\n\t"
        "addi t5, t3, 64\n\t" "vle8.v v2, (t5)\n\t"
        "vle8.v v4, (t4)\n\t" "addi t5, t4, 32\n\t" "vle8.v v5, (t5)\n\t"
        "addi t5, t4, 64\n\t" "vle8.v v6, (t5)\n\t"
        "addi t5, t4, 96\n\t" "vle8.v v7, (t5)\n\t"
        Y26_STAGE47_DOT(v8, v0, v4) Y26_STAGE47_DOT(v10, v1, v4)
        Y26_STAGE47_DOT(v12, v2, v4) Y26_STAGE47_DOT(v14, v0, v5)
        Y26_STAGE47_DOT(v16, v1, v5) Y26_STAGE47_DOT(v18, v2, v5)
        Y26_STAGE47_DOT(v20, v0, v6) Y26_STAGE47_DOT(v22, v1, v6)
        Y26_STAGE47_DOT(v24, v2, v6) Y26_STAGE47_DOT(v26, v0, v7)
        Y26_STAGE47_DOT(v28, v1, v7) Y26_STAGE47_DOT(v30, v2, v7)
        "addi t3, t3, 96\n\t" "addi t4, t4, 128\n\t"
        "addi t1, t1, -1\n\t" "bnez t1, 1b\n\t"
        "vsetvli t0, zero, e32, m2\n\t" "mv t2, %[C]\n\t"
        Y26_STAGE47_STORE(v8) Y26_STAGE47_STORE(v10)
        Y26_STAGE47_STORE(v12) Y26_STAGE47_STORE(v14)
        Y26_STAGE47_STORE(v16) Y26_STAGE47_STORE(v18)
        Y26_STAGE47_STORE(v20) Y26_STAGE47_STORE(v22)
        Y26_STAGE47_STORE(v24) Y26_STAGE47_STORE(v26)
        Y26_STAGE47_STORE(v28) Y26_STAGE47_STORE(v30)
        :
        : [A] "r"(a), [B] "r"(b), [K] "r"(k_tiles), [C] "r"(c)
        : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "v0", "v1", "v2",
          "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
          "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

#undef Y26_STAGE47_INIT_ACC
#undef Y26_STAGE47_DOT
#undef Y26_STAGE47_STORE
#endif

void run_block(KernelShape shape,
               const std::int8_t* a,
               const std::int8_t* b,
               int k_tiles,
               std::int32_t* c) {
    const int rows = rows_for(shape);
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (shape == KernelShape::m4n16) {
        kernel_m4n16(a, b, k_tiles, c);
        return;
    }
    if (shape == KernelShape::m8n16) {
        kernel_m8n16(a, b, k_tiles, c);
        return;
    }
    if (shape == KernelShape::m12n16) {
        kernel_m12n16(a, b, k_tiles, c);
        return;
    }
#endif
    scalar_block(a, b, k_tiles, rows, c);
}

struct WorkerScratch {
    std::vector<std::int8_t> a_panel;
    std::array<std::int32_t, 12 * kNBlock> c_tile {};
    int observed_cpu = -1;
    bool affinity_set = false;
};

}  // namespace

struct WorkerPool::Impl {
    using Job = void (*)(void*, int, WorkerScratch&);

    explicit Impl(int requested) : count(std::clamp(requested, 1, kMaximumWorkers)) {
        scratch.resize(static_cast<std::size_t>(count));
        threads.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            threads.emplace_back([this, index]() { worker_loop(index); });
        }
        std::unique_lock lock(mutex);
        ready_cv.wait(lock, [this]() { return ready == count; });
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            ++generation;
        }
        start_cv.notify_all();
        for (std::thread& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void worker_loop(int index) {
        WorkerScratch& state = scratch[static_cast<std::size_t>(index)];
        state.affinity_set = pin_current_thread(index);
        state.observed_cpu = current_cpu();
        {
            std::lock_guard lock(mutex);
            ++ready;
        }
        ready_cv.notify_one();
        std::uint64_t local_generation = 0;
        for (;;) {
            Job local_job = nullptr;
            void* local_context = nullptr;
            int local_active = 0;
            {
                std::unique_lock lock(mutex);
                start_cv.wait(lock, [&]() { return generation != local_generation; });
                local_generation = generation;
                if (stopping) {
                    return;
                }
                local_job = job;
                local_context = context;
                local_active = active;
            }
            if (index < local_active && local_job != nullptr) {
                state.observed_cpu = current_cpu();
                local_job(local_context, index, state);
            }
            {
                std::lock_guard lock(mutex);
                ++completed;
            }
            done_cv.notify_one();
        }
    }

    void ensure_workspace(std::size_t a_bytes) {
        for (WorkerScratch& worker : scratch) {
            if (worker.a_panel.size() < a_bytes) {
                worker.a_panel.resize(a_bytes);
            }
        }
    }

    void dispatch(int active_workers, Job new_job, void* new_context) {
        {
            std::lock_guard lock(mutex);
            active = std::clamp(active_workers, 1, count);
            job = new_job;
            context = new_context;
            completed = 0;
            ++generation;
        }
        start_cv.notify_all();
        std::unique_lock lock(mutex);
        done_cv.wait(lock, [this]() { return completed == count; });
    }

    bool affinity_ok() const {
        for (int index = 0; index < count; ++index) {
            const WorkerScratch& worker = scratch[static_cast<std::size_t>(index)];
            if (!worker.affinity_set || worker.observed_cpu < 0 || worker.observed_cpu > 3) {
                return false;
            }
        }
        return true;
    }

    int count = 0;
    std::vector<std::thread> threads;
    std::vector<WorkerScratch> scratch;
    mutable std::mutex mutex;
    std::condition_variable start_cv;
    std::condition_variable done_cv;
    std::condition_variable ready_cv;
    std::uint64_t generation = 0;
    int active = 0;
    int completed = 0;
    int ready = 0;
    Job job = nullptr;
    void* context = nullptr;
    bool stopping = false;
};

WorkerPool::WorkerPool(int workers) : impl_(std::make_unique<Impl>(workers)) {}
WorkerPool::~WorkerPool() = default;
int WorkerPool::capacity() const noexcept { return impl_ ? impl_->count : 0; }
bool WorkerPool::affinity_ok() const noexcept { return impl_ && impl_->affinity_ok(); }

struct IntegratedConv::Impl {
    struct Segment {
        OutputSegmentSpec spec;
        std::vector<ExactRequantParams> requant;
        std::array<std::int8_t, 256> lut {};
    };

    ConvSpec spec;
    int kernel_k = 0;
    int k_padded = 0;
    int k_tiles = 0;
    int n_padded = 0;
    int n_blocks = 0;
    std::vector<std::int8_t> packed_weights;
    std::vector<std::int32_t> weight_sums;
    std::vector<float> owned_weight_scales;
    std::vector<std::int32_t> owned_bias;
    std::vector<std::int64_t> exact_code_thresholds;
    std::vector<Segment> segments;
    std::uint64_t total_macs = 0;
    bool ready = false;
};

namespace {

bool valid_tensor(const TensorSpec& tensor) {
    return tensor.h > 0 && tensor.w > 0 && tensor.c > 0 && tensor.scale > 0.0f &&
           tensor.zero_point_u8 >= 0 && tensor.zero_point_u8 <= 255;
}

bool valid_conv(const ConvSpec& spec) {
    if (!valid_tensor(spec.input) || spec.output_h <= 0 || spec.output_w <= 0 || spec.output_c <= 0 ||
        (spec.kernel_h != 1 && spec.kernel_h != 3) || spec.kernel_h != spec.kernel_w ||
        (spec.stride_h != 1 && spec.stride_h != 2) || spec.stride_h != spec.stride_w ||
        spec.pad_h < 0 || spec.pad_w < 0 || spec.group != 1 || spec.conv_output_scale <= 0.0f ||
        spec.conv_output_zero_point_u8 < 0 || spec.conv_output_zero_point_u8 > 255 ||
        spec.weights_ohwi_s8 == nullptr || spec.weight_scales == nullptr || spec.bias_i32 == nullptr ||
        spec.weight_scale_count < static_cast<std::size_t>(spec.output_c) ||
        spec.bias_count < static_cast<std::size_t>(spec.output_c) || spec.segments.empty() ||
        spec.segments.size() > 2) {
        return false;
    }
    const std::size_t expected = static_cast<std::size_t>(spec.output_c) * spec.kernel_h *
                                 spec.kernel_w * spec.input.c;
    if (spec.weight_count < expected) {
        return false;
    }
    int covered = 0;
    for (const OutputSegmentSpec& segment : spec.segments) {
        if (!valid_tensor(segment.output) || segment.channel_begin != covered || segment.channel_count <= 0 ||
            segment.channel_begin + segment.channel_count > spec.output_c ||
            segment.output.h != spec.output_h || segment.output.w != spec.output_w ||
            segment.output.c != segment.channel_count) {
            return false;
        }
        covered += segment.channel_count;
    }
    return covered == spec.output_c;
}

void pack_weights(IntegratedConv::Impl& impl) {
    impl.packed_weights.assign(static_cast<std::size_t>(impl.n_blocks) * impl.k_tiles * kNBlock * 8, 0);
    impl.weight_sums.assign(static_cast<std::size_t>(impl.spec.output_c), 0);
    for (int oc = 0; oc < impl.spec.output_c; ++oc) {
        std::int32_t sum = 0;
        for (int k = 0; k < impl.kernel_k; ++k) {
            sum += impl.spec.weights_ohwi_s8[static_cast<std::size_t>(oc) * impl.kernel_k + k];
        }
        impl.weight_sums[static_cast<std::size_t>(oc)] = sum;
    }
    for (int nb = 0; nb < impl.n_blocks; ++nb) {
        for (int kt = 0; kt < impl.k_tiles; ++kt) {
            std::int8_t* destination = impl.packed_weights.data() +
                (static_cast<std::size_t>(nb) * impl.k_tiles + kt) * kNBlock * 8;
            for (int n = 0; n < kNBlock; ++n) {
                const int oc = nb * kNBlock + n;
                for (int kk = 0; kk < 8; ++kk) {
                    const int k = kt * 8 + kk;
                    destination[n * 8 + kk] =
                        oc < impl.spec.output_c && k < impl.kernel_k
                            ? impl.spec.weights_ohwi_s8[static_cast<std::size_t>(oc) * impl.kernel_k + k]
                            : std::int8_t{0};
                }
            }
        }
    }
}

std::uint8_t reference_conv_code(const IntegratedConv::Impl& impl, int channel, std::int32_t value) {
    const float accumulator_scale = impl.spec.input.scale * impl.spec.weight_scales[channel];
    const float dequantized = static_cast<float>(value) * accumulator_scale;
    return y26_quantize_u8_nearest_even_f32(
        dequantized, impl.spec.conv_output_scale, impl.spec.conv_output_zero_point_u8);
}

void build_exact_code_thresholds(IntegratedConv::Impl& impl) {
    impl.exact_code_thresholds.assign(static_cast<std::size_t>(impl.spec.output_c) * 256U,
                                      std::numeric_limits<std::int64_t>::max());
    const int original_round = std::fegetround();
    std::fesetround(FE_TONEAREST);
    for (int channel = 0; channel < impl.spec.output_c; ++channel) {
        std::int64_t* thresholds = impl.exact_code_thresholds.data() + static_cast<std::size_t>(channel) * 256U;
        thresholds[0] = std::numeric_limits<std::int64_t>::min();
        const std::uint8_t maximum_code = reference_conv_code(
            impl, channel, std::numeric_limits<std::int32_t>::max());
        for (int code = 1; code <= maximum_code; ++code) {
            std::int64_t lower = std::numeric_limits<std::int32_t>::min();
            std::int64_t upper = std::numeric_limits<std::int32_t>::max();
            while (lower < upper) {
                const std::int64_t midpoint = lower + (upper - lower) / 2;
                if (reference_conv_code(impl, channel, static_cast<std::int32_t>(midpoint)) >= code) {
                    upper = midpoint;
                } else {
                    lower = midpoint + 1;
                }
            }
            thresholds[code] = lower;
        }
    }
    if (original_round != -1) {
        std::fesetround(original_round);
    }
}

std::uint8_t corrected_exact_code(const IntegratedConv::Impl& conv,
                                  int channel,
                                  std::int32_t value,
                                  const ExactRequantParams& params) {
    int code = exact_requant(value, params);
    const std::int64_t* thresholds = conv.exact_code_thresholds.data() + static_cast<std::size_t>(channel) * 256U;
    while (code > 0 && static_cast<std::int64_t>(value) < thresholds[code]) {
        --code;
    }
    while (code < 255 && static_cast<std::int64_t>(value) >= thresholds[code + 1]) {
        ++code;
    }
    return static_cast<std::uint8_t>(code);
}

void pack_a(const IntegratedConv::Impl& impl,
            const std::int8_t* input,
            int m_begin,
            int rows,
            std::int8_t* panel) {
    const std::int8_t padding = signed_storage(static_cast<std::uint8_t>(impl.spec.input.zero_point_u8));
    for (int kt = 0; kt < impl.k_tiles; ++kt) {
        for (int r = 0; r < rows; ++r) {
            const int m = m_begin + r;
            const int oy = m / impl.spec.output_w;
            const int ox = m % impl.spec.output_w;
            for (int kk = 0; kk < 8; ++kk) {
                const int k = kt * 8 + kk;
                std::int8_t value = padding;
                if (m < impl.spec.output_h * impl.spec.output_w && k < impl.kernel_k) {
                    const int ic = k % impl.spec.input.c;
                    const int kernel_position = k / impl.spec.input.c;
                    const int ky = kernel_position / impl.spec.kernel_w;
                    const int kx = kernel_position % impl.spec.kernel_w;
                    const int iy = oy * impl.spec.stride_h + ky - impl.spec.pad_h;
                    const int ix = ox * impl.spec.stride_w + kx - impl.spec.pad_w;
                    if (iy >= 0 && iy < impl.spec.input.h && ix >= 0 && ix < impl.spec.input.w) {
                        value = input[(static_cast<std::size_t>(iy) * impl.spec.input.w + ix) *
                                      impl.spec.input.c + ic];
                    }
                }
                panel[(static_cast<std::size_t>(kt) * rows + r) * 8 + kk] = value;
            }
        }
    }
}

struct ConvRunContext {
    const IntegratedConv::Impl* conv = nullptr;
    const std::int8_t* input = nullptr;
    std::array<std::int8_t*, 2> outputs {};
    KernelShape kernel = KernelShape::scalar;
    PartitionPolicy partition = PartitionPolicy::spatial;
    int active_workers = 1;
    bool profile = false;
    std::array<int, kMaximumWorkers> status {};
    std::array<double, kMaximumWorkers> total_us {};
    std::array<double, kMaximumWorkers> pack_us {};
    std::array<double, kMaximumWorkers> kernel_us {};
    std::array<double, kMaximumWorkers> epilogue_us {};
};

const IntegratedConv::Impl::Segment* segment_for(const IntegratedConv::Impl& conv, int oc, int* local_channel) {
    for (const auto& segment : conv.segments) {
        const int begin = segment.spec.channel_begin;
        if (oc >= begin && oc < begin + segment.spec.channel_count) {
            *local_channel = oc - begin;
            return &segment;
        }
    }
    return nullptr;
}

void store_tile(const IntegratedConv::Impl& conv,
                const std::array<std::int8_t*, 2>& outputs,
                const std::int32_t* c,
                int rows,
                int valid_rows,
                int m_begin,
                int n_begin) {
    const int row_groups = rows / 4;
    const std::int64_t correction = 128 - static_cast<std::int64_t>(conv.spec.input.zero_point_u8);
    for (int n = 0; n < kNBlock; ++n) {
        const int oc = n_begin + n;
        if (oc >= conv.spec.output_c) {
            continue;
        }
        int local_channel = 0;
        const auto* segment = segment_for(conv, oc, &local_channel);
        if (segment == nullptr) {
            continue;
        }
        const std::size_t segment_index = static_cast<std::size_t>(segment - conv.segments.data());
        std::int8_t* output = outputs[segment_index];
        for (int r = 0; r < valid_rows; ++r) {
            const int ng = n / 4;
            const int nc = n % 4;
            const int rg = r / 4;
            const int rr = r % 4;
            const std::int32_t raw = c[(ng * row_groups + rg) * 16 + rr * 4 + nc];
            const std::int64_t corrected64 = static_cast<std::int64_t>(raw) +
                correction * conv.weight_sums[static_cast<std::size_t>(oc)] +
                conv.spec.bias_i32[oc];
            const std::int32_t corrected = static_cast<std::int32_t>(corrected64);
            const std::uint8_t conv_code = corrected_exact_code(
                conv, oc, corrected, segment->requant[static_cast<std::size_t>(oc)]);
            const std::int8_t result = segment->spec.silu ? segment->lut[conv_code] : signed_storage(conv_code);
            output[static_cast<std::size_t>(m_begin + r) * segment->spec.channel_count + local_channel] = result;
        }
    }
}

void run_worker_job(void* opaque, int worker_index, WorkerScratch& scratch) {
    auto& context = *static_cast<ConvRunContext*>(opaque);
    const auto& conv = *context.conv;
    const int rows = rows_for(context.kernel);
    const int output_m = conv.spec.output_h * conv.spec.output_w;
    const int m_blocks = (output_m + rows - 1) / rows;
    const int n_blocks = conv.n_blocks;
    int m_block_begin = 0;
    int m_block_end = m_blocks;
    int n_block_begin = 0;
    int n_block_end = n_blocks;
    if (context.partition == PartitionPolicy::spatial) {
        m_block_begin = m_blocks * worker_index / context.active_workers;
        m_block_end = m_blocks * (worker_index + 1) / context.active_workers;
    } else {
        n_block_begin = n_blocks * worker_index / context.active_workers;
        n_block_end = n_blocks * (worker_index + 1) / context.active_workers;
    }
    if (context.kernel != KernelShape::scalar && !y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        context.status[static_cast<std::size_t>(worker_index)] = Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
        return;
    }
    const auto worker_begin = Clock::now();
    for (int mb = m_block_begin; mb < m_block_end; ++mb) {
        const int m_begin = mb * rows;
        const int valid_rows = std::min(rows, output_m - m_begin);
        KernelShape tile_kernel = context.kernel;
        int tile_rows = rows;
        if (context.kernel == KernelShape::m12n16 && valid_rows <= 8) {
            tile_kernel = valid_rows <= 4 ? KernelShape::m4n16 : KernelShape::m8n16;
            tile_rows = rows_for(tile_kernel);
        } else if (context.kernel == KernelShape::m8n16 && valid_rows <= 4) {
            tile_kernel = KernelShape::m4n16;
            tile_rows = rows_for(tile_kernel);
        }
        const auto pack_begin = context.profile ? Clock::now() : worker_begin;
        pack_a(conv, context.input, m_begin, tile_rows, scratch.a_panel.data());
        const auto pack_end = context.profile ? Clock::now() : worker_begin;
        if (context.profile) {
            context.pack_us[static_cast<std::size_t>(worker_index)] += elapsed_us(pack_begin, pack_end);
        }
        for (int nb = n_block_begin; nb < n_block_end; ++nb) {
            const std::int8_t* packed_b = conv.packed_weights.data() +
                static_cast<std::size_t>(nb) * conv.k_tiles * kNBlock * 8;
            const auto kernel_begin = context.profile ? Clock::now() : worker_begin;
            run_block(tile_kernel, scratch.a_panel.data(), packed_b, conv.k_tiles, scratch.c_tile.data());
            const auto kernel_end = context.profile ? Clock::now() : worker_begin;
            if (context.profile) {
                context.kernel_us[static_cast<std::size_t>(worker_index)] += elapsed_us(kernel_begin, kernel_end);
            }
            const auto epilogue_begin = context.profile ? Clock::now() : worker_begin;
            store_tile(conv, context.outputs, scratch.c_tile.data(), tile_rows, valid_rows, m_begin, nb * kNBlock);
            const auto epilogue_end = context.profile ? Clock::now() : worker_begin;
            if (context.profile) {
                context.epilogue_us[static_cast<std::size_t>(worker_index)] += elapsed_us(epilogue_begin, epilogue_end);
            }
        }
    }
    context.total_us[static_cast<std::size_t>(worker_index)] = elapsed_us(worker_begin, Clock::now());
    context.status[static_cast<std::size_t>(worker_index)] = Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

IntegratedConv::IntegratedConv() : impl_(std::make_unique<Impl>()) {}
IntegratedConv::~IntegratedConv() = default;
IntegratedConv::IntegratedConv(IntegratedConv&&) noexcept = default;
IntegratedConv& IntegratedConv::operator=(IntegratedConv&&) noexcept = default;

int IntegratedConv::prepare(const ConvSpec& spec) {
    if (!valid_conv(spec)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    try {
        Impl prepared;
        prepared.spec = spec;
        prepared.owned_weight_scales.assign(spec.weight_scales, spec.weight_scales + spec.output_c);
        prepared.owned_bias.assign(spec.bias_i32, spec.bias_i32 + spec.output_c);
        prepared.spec.weight_scales = prepared.owned_weight_scales.data();
        prepared.spec.bias_i32 = prepared.owned_bias.data();
        prepared.kernel_k = spec.kernel_h * spec.kernel_w * spec.input.c;
        prepared.k_padded = align_up(prepared.kernel_k, 8);
        prepared.k_tiles = prepared.k_padded / 8;
        prepared.n_padded = align_up(spec.output_c, kNBlock);
        prepared.n_blocks = prepared.n_padded / kNBlock;
        pack_weights(prepared);
        build_exact_code_thresholds(prepared);
        for (const OutputSegmentSpec& output : spec.segments) {
            Impl::Segment segment;
            segment.spec = output;
            segment.requant.resize(static_cast<std::size_t>(spec.output_c));
            for (int oc = 0; oc < spec.output_c; ++oc) {
                const double multiplier = static_cast<double>(spec.input.scale) * spec.weight_scales[oc] /
                                          static_cast<double>(spec.conv_output_scale);
                segment.requant[static_cast<std::size_t>(oc)] =
                    exact_requant_params(multiplier, spec.conv_output_zero_point_u8);
            }
            if (output.silu &&
                y26_build_silu_u8_to_s8_lut(spec.conv_output_scale,
                                            spec.conv_output_zero_point_u8,
                                            output.output.scale,
                                            output.output.zero_point_u8,
                                            segment.lut.data()) != Y26_CONV_STATUS_SUCCESS) {
                return Y26_CONV_STATUS_INVALID_ARGUMENT;
            }
            prepared.segments.push_back(std::move(segment));
        }
        prepared.total_macs = static_cast<std::uint64_t>(spec.output_h) * spec.output_w * spec.output_c *
                              spec.kernel_h * spec.kernel_w * spec.input.c;
        prepared.ready = true;
        *impl_ = std::move(prepared);
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

int IntegratedConv::run(WorkerPool& pool,
                        const std::int8_t* input_nhwc_s8,
                        const std::array<std::int8_t*, 2>& outputs_nhwc_s8,
                        std::size_t output_count,
                        const RunOptions& options,
                        IntegratedTiming* timing) const {
    if (!impl_ || !impl_->ready || !pool.impl_ || input_nhwc_s8 == nullptr ||
        output_count != impl_->segments.size() || options.workers < 1 ||
        options.workers > pool.impl_->count || rows_for(options.kernel) == 0) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t index = 0; index < output_count; ++index) {
        if (outputs_nhwc_s8[index] == nullptr) {
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
    }
    if (options.kernel != KernelShape::scalar) {
        if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
            return Y26_CONV_STATUS_NOT_BUILT_WITH_IME;
        }
        if (y26_k1x_ime_probe_once() != Y26_VMADOT_STATUS_SUCCESS) {
            return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
        }
    }
    const int rows = rows_for(options.kernel);
    pool.impl_->ensure_workspace(static_cast<std::size_t>(rows) * impl_->k_padded);
    ConvRunContext context;
    context.conv = impl_.get();
    context.input = input_nhwc_s8;
    context.kernel = options.kernel;
    context.partition = options.partition;
    context.active_workers = options.workers;
    context.profile = options.profile_phases;
    for (std::size_t index = 0; index < output_count; ++index) {
        context.outputs[index] = outputs_nhwc_s8[index];
    }
    const auto begin = Clock::now();
    pool.impl_->dispatch(options.workers, run_worker_job, &context);
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
        timing->affinity_ok = pool.impl_->affinity_ok() ? 1 : 0;
        timing->min_worker_us = std::numeric_limits<double>::max();
        for (int worker = 0; worker < options.workers; ++worker) {
            const std::size_t index = static_cast<std::size_t>(worker);
            timing->gather_pack_us += context.pack_us[index];
            timing->vmadot_us += context.kernel_us[index];
            timing->fused_epilogue_us += context.epilogue_us[index];
            timing->max_worker_us = std::max(timing->max_worker_us, context.total_us[index]);
            timing->min_worker_us = std::min(timing->min_worker_us, context.total_us[index]);
        }
        timing->barrier_us = std::max(0.0, timing->total_us - timing->max_worker_us);
    }
    return status;
}

std::size_t IntegratedConv::prepared_weight_bytes() const noexcept {
    return impl_ ? impl_->packed_weights.size() + impl_->weight_sums.size() * sizeof(std::int32_t) : 0;
}

std::size_t IntegratedConv::per_worker_workspace_bytes(KernelShape shape) const noexcept {
    return impl_ ? static_cast<std::size_t>(rows_for(shape)) * impl_->k_padded + 12U * 16U * sizeof(std::int32_t) : 0;
}

std::uint64_t IntegratedConv::macs() const noexcept { return impl_ ? impl_->total_macs : 0; }

void nchw_u8_to_nhwc_s8(const std::uint8_t* input, std::int8_t* output, int h, int w, int c) {
    if (input == nullptr || output == nullptr || h <= 0 || w <= 0 || c <= 0) {
        return;
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int channel = 0; channel < c; ++channel) {
                output[(static_cast<std::size_t>(y) * w + x) * c + channel] =
                    signed_storage(input[(static_cast<std::size_t>(channel) * h + y) * w + x]);
            }
        }
    }
}

void nhwc_s8_to_nchw_u8(const std::int8_t* input, std::uint8_t* output, int h, int w, int c) {
    if (input == nullptr || output == nullptr || h <= 0 || w <= 0 || c <= 0) {
        return;
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int channel = 0; channel < c; ++channel) {
                output[(static_cast<std::size_t>(channel) * h + y) * w + x] =
                    unsigned_code(input[(static_cast<std::size_t>(y) * w + x) * c + channel]);
            }
        }
    }
}

const char* kernel_shape_name(KernelShape shape) noexcept {
    switch (shape) {
        case KernelShape::scalar: return "scalar";
        case KernelShape::m4n16: return "m4n16";
        case KernelShape::m8n16: return "m8n16";
        case KernelShape::m12n16: return "m12n16";
    }
    return "unknown";
}

const char* partition_policy_name(PartitionPolicy policy) noexcept {
    switch (policy) {
        case PartitionPolicy::spatial: return "spatial";
        case PartitionPolicy::output_channel: return "output_channel";
    }
    return "unknown";
}

}  // namespace y26::stage47
