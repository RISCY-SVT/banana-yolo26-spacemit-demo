#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv) && defined(__linux__)
#include <csetjmp>
#include <csignal>
#include <sched.h>
#include <sys/ucontext.h>
#include <ucontext.h>
#endif

namespace {

struct Options {
    int iterations = 1;
    int warmup = 0;
    int repeats = 1;
    std::string only_case = "all";
};

struct Stat {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct TrapInfo {
    int trapped = 0;
    int si_code = 0;
    int cpu = -1;
    std::uintptr_t pc = 0;
    std::uint32_t insn32 = 0;
};

constexpr int kStage35OracleMismatch = 5;

struct CaseResult {
    const char* name = "";
    const char* emission = "";
    int accumulator_groups = 1;
    int vmadots_per_iteration = 1;
    int status = 0;
    int mismatches = 0;
    std::uint64_t checksum = 0;
    TrapInfo trap {};
    Stat cycles_per_iteration {};
    double cycles_per_vmadot = 0.0;
};

Options parse_options(int argc, char** argv) {
    Options options {};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--iterations") {
            options.iterations = std::atoi(require_value("--iterations"));
        } else if (arg == "--warmup") {
            options.warmup = std::atoi(require_value("--warmup"));
        } else if (arg == "--repeats") {
            options.repeats = std::atoi(require_value("--repeats"));
        } else if (arg == "--case") {
            options.only_case = require_value("--case");
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    options.iterations = std::max(1, options.iterations);
    options.warmup = std::max(0, options.warmup);
    options.repeats = std::max(1, options.repeats);
    return options;
}

Stat summarize(const std::vector<double>& values) {
    Stat stat {};
    if (values.empty()) {
        return stat;
    }
    stat.min = std::numeric_limits<double>::infinity();
    stat.max = 0.0;
    for (double value : values) {
        stat.mean += value;
        stat.min = std::min(stat.min, value);
        stat.max = std::max(stat.max, value);
    }
    stat.mean /= static_cast<double>(values.size());
    if (values.size() > 1) {
        double var = 0.0;
        for (double value : values) {
            const double delta = value - stat.mean;
            var += delta * delta;
        }
        stat.stddev = std::sqrt(var / static_cast<double>(values.size() - 1));
    }
    return stat;
}

std::uint64_t checksum_words(const std::int32_t* values, std::size_t count) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < count; ++i) {
        hash ^= static_cast<std::uint32_t>(values[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

int compare_block(const std::int32_t* actual, const std::int32_t* expected, int multiplier) {
    int mismatches = 0;
    for (int i = 0; i < 16; ++i) {
        if (actual[i] != expected[i] * multiplier) {
            ++mismatches;
        }
    }
    return mismatches;
}

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)

sigjmp_buf g_stage35_sigill_jump;
volatile sig_atomic_t g_stage35_armed = 0;
TrapInfo g_stage35_trap {};

extern "C" void y26_stage35_standalone_named_v28(const std::int8_t* a,
                                                  const std::int8_t* b,
                                                  std::int32_t* c);
extern "C" void y26_stage35_standalone_raw_v28(const std::int8_t* a,
                                                const std::int8_t* b,
                                                std::int32_t* c);

asm(R"(
    .text
    .align 2
    .globl y26_stage35_standalone_named_v28
    .type y26_stage35_standalone_named_v28, @function
y26_stage35_standalone_named_v28:
    vsetvli      t0, zero, e32, m2
    vxor.vv      v28, v28, v28
    vsetvli      t0, zero, e8, m1
    vle8.v       v0, (a0)
    vle8.v       v1, (a1)
    smt.vmadot   v28, v0, v1
    vsetvli      t0, zero, e32, m2
    vse32.v      v28, (a2)
    ret
    .size y26_stage35_standalone_named_v28, .-y26_stage35_standalone_named_v28

    .globl y26_stage35_standalone_raw_v28
    .type y26_stage35_standalone_raw_v28, @function
y26_stage35_standalone_raw_v28:
    vsetvli      t0, zero, e32, m2
    vxor.vv      v28, v28, v28
    vsetvli      t0, zero, e8, m1
    vle8.v       v0, (a0)
    vle8.v       v1, (a1)
    .word        0xe2103e2b
    vsetvli      t0, zero, e32, m2
    vse32.v      v28, (a2)
    ret
    .size y26_stage35_standalone_raw_v28, .-y26_stage35_standalone_raw_v28
)");

void stage35_sigill_handler(int /*signo*/, siginfo_t* info, void* uctx) {
    g_stage35_trap.trapped = 1;
    g_stage35_trap.si_code = info != nullptr ? info->si_code : 0;
#if defined(__linux__)
    g_stage35_trap.cpu = sched_getcpu();
#endif
    auto* context = reinterpret_cast<ucontext_t*>(uctx);
    if (context != nullptr) {
        g_stage35_trap.pc = static_cast<std::uintptr_t>(context->uc_mcontext.__gregs[REG_PC]);
        if (g_stage35_trap.pc != 0) {
            g_stage35_trap.insn32 = *reinterpret_cast<const std::uint32_t*>(g_stage35_trap.pc);
        }
    }
    if (g_stage35_armed != 0) {
        siglongjmp(g_stage35_sigill_jump, 1);
    }
}

template <typename Fn>
TrapInfo run_trap_guarded(Fn&& fn) {
    struct sigaction old_action {};
    struct sigaction new_action {};
    new_action.sa_sigaction = stage35_sigill_handler;
    new_action.sa_flags = SA_SIGINFO;
    sigemptyset(&new_action.sa_mask);
    g_stage35_trap = TrapInfo {};
    if (sigaction(SIGILL, &new_action, &old_action) != 0) {
        TrapInfo trap {};
        trap.trapped = 1;
        trap.si_code = -1;
        return trap;
    }
    g_stage35_armed = 1;
    if (sigsetjmp(g_stage35_sigill_jump, 1) == 0) {
        fn();
    }
    g_stage35_armed = 0;
    sigaction(SIGILL, &old_action, nullptr);
    return g_stage35_trap;
}

__attribute__((noinline)) void named_exact_v28(int iterations,
                                               const std::int8_t* a,
                                               const std::int8_t* b,
                                               std::int32_t* c) {
    for (int i = 0; i < iterations; ++i) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v28, v28, v28           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadot   v28, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    }
}

__attribute__((noinline)) void raw_exact_v28(int iterations,
                                             const std::int8_t* a,
                                             const std::int8_t* b,
                                             std::int32_t* c) {
    for (int i = 0; i < iterations; ++i) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v28, v28, v28           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            ".word        0xe2103e2b              \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    }
}

__attribute__((noinline)) void raw_dep_chain_v28(int iterations,
                                                 const std::int8_t* a,
                                                 const std::int8_t* b,
                                                 std::int32_t* c) {
    __asm__ volatile(
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        ".word        0xe2103e2b              \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C] "r"(c), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29");
}

__attribute__((noinline)) void raw_load_included_v28(int iterations,
                                                     const std::int8_t* a,
                                                     const std::int8_t* b,
                                                     std::int32_t* c) {
    __asm__ volatile(
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        ".word        0xe2103e2b              \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C] "r"(c), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29");
}

__attribute__((noinline)) void named_single_v24(int iterations,
                                                const std::int8_t* a,
                                                const std::int8_t* b,
                                                std::int32_t* c) {
    for (int i = 0; i < iterations; ++i) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v24, v24, v24           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadot   v24, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v24, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v24", "v25");
    }
}

__attribute__((noinline)) void named_single_v20(int iterations,
                                                const std::int8_t* a,
                                                const std::int8_t* b,
                                                std::int32_t* c) {
    for (int i = 0; i < iterations; ++i) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v20, v20, v20           \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadot   v20, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v20, (%[C])             \n\t"
            :
            : [A] "r"(a), [B] "r"(b), [C] "r"(c)
            : "cc", "memory", "t0", "v0", "v1", "v20", "v21");
    }
}

__attribute__((noinline)) void raw_two_acc_v28_v30(int iterations,
                                                   const std::int8_t* a,
                                                   const std::int8_t* b,
                                                   std::int32_t* c0,
                                                   std::int32_t* c1) {
    __asm__ volatile(
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "vxor.vv      v30, v30, v30           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        ".word        0xe2103e2b              \n\t"
        ".word        0xe2103f2b              \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C0])            \n\t"
        "vse32.v      v30, (%[C1])            \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C0] "r"(c0), [C1] "r"(c1), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29", "v30", "v31");
}

__attribute__((noinline)) void raw_four_acc_v20_v22_v24_v26(int iterations,
                                                            const std::int8_t* a,
                                                            const std::int8_t* b,
                                                            std::int32_t* c0,
                                                            std::int32_t* c1,
                                                            std::int32_t* c2,
                                                            std::int32_t* c3) {
    __asm__ volatile(
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v20, v20, v20           \n\t"
        "vxor.vv      v22, v22, v22           \n\t"
        "vxor.vv      v24, v24, v24           \n\t"
        "vxor.vv      v26, v26, v26           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        ".word        0xe2103a2b              \n\t"
        ".word        0xe2103b2b              \n\t"
        ".word        0xe2103c2b              \n\t"
        ".word        0xe2103d2b              \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v20, (%[C0])            \n\t"
        "vse32.v      v22, (%[C1])            \n\t"
        "vse32.v      v24, (%[C2])            \n\t"
        "vse32.v      v26, (%[C3])            \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C0] "r"(c0), [C1] "r"(c1), [C2] "r"(c2), [C3] "r"(c3),
          [N] "r"(iterations)
        : "cc",
          "memory",
          "t0",
          "t1",
          "v0",
          "v1",
          "v20",
          "v21",
          "v22",
          "v23",
          "v24",
          "v25",
          "v26",
          "v27");
}

__attribute__((noinline)) void raw_six_acc_v16_v18_v20_v22_v24_v26(int iterations,
                                                                   const std::int8_t* a,
                                                                   const std::int8_t* b,
                                                                   std::int32_t* c0,
                                                                   std::int32_t* c1,
                                                                   std::int32_t* c2,
                                                                   std::int32_t* c3,
                                                                   std::int32_t* c4,
                                                                   std::int32_t* c5) {
    __asm__ volatile(
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v16, v16, v16           \n\t"
        "vxor.vv      v18, v18, v18           \n\t"
        "vxor.vv      v20, v20, v20           \n\t"
        "vxor.vv      v22, v22, v22           \n\t"
        "vxor.vv      v24, v24, v24           \n\t"
        "vxor.vv      v26, v26, v26           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        ".word        0xe210382b              \n\t"
        ".word        0xe210392b              \n\t"
        ".word        0xe2103a2b              \n\t"
        ".word        0xe2103b2b              \n\t"
        ".word        0xe2103c2b              \n\t"
        ".word        0xe2103d2b              \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v16, (%[C0])            \n\t"
        "vse32.v      v18, (%[C1])            \n\t"
        "vse32.v      v20, (%[C2])            \n\t"
        "vse32.v      v22, (%[C3])            \n\t"
        "vse32.v      v24, (%[C4])            \n\t"
        "vse32.v      v26, (%[C5])            \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C0] "r"(c0), [C1] "r"(c1), [C2] "r"(c2), [C3] "r"(c3),
          [C4] "r"(c4), [C5] "r"(c5), [N] "r"(iterations)
        : "cc",
          "memory",
          "t0",
          "t1",
          "v0",
          "v1",
          "v16",
          "v17",
          "v18",
          "v19",
          "v20",
          "v21",
          "v22",
          "v23",
          "v24",
          "v25",
          "v26",
          "v27");
}

#endif

template <typename Fn>
CaseResult run_case(const char* name,
                    const char* emission,
                    int accumulator_groups,
                    int vmadots_per_iteration,
                    int expected_multiplier,
                    const Options& options,
                    Fn&& fn,
                    const std::array<std::int32_t, 16>& scalar,
                    std::array<std::int32_t, 96>& output) {
    CaseResult result {};
    result.name = name;
    result.emission = emission;
    result.accumulator_groups = accumulator_groups;
    result.vmadots_per_iteration = vmadots_per_iteration;

    std::vector<double> cycles;
    cycles.reserve(static_cast<std::size_t>(options.repeats));
    for (int i = 0; i < options.warmup; ++i) {
        std::fill(output.begin(), output.end(), 0);
    #if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
        const TrapInfo trap = run_trap_guarded([&]() { fn(std::max(1, options.iterations / 100), output.data()); });
        if (trap.trapped != 0) {
            result.trap = trap;
            result.status = Y26_VMADOT_STATUS_SIGILL_CAUGHT;
            return result;
        }
    #else
        fn(std::max(1, options.iterations / 100), output.data());
    #endif
    }
    for (int r = 0; r < options.repeats; ++r) {
        std::fill(output.begin(), output.end(), 0);
    #if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
        std::chrono::steady_clock::time_point begin {};
        std::chrono::steady_clock::time_point end {};
        const TrapInfo trap = run_trap_guarded([&]() {
            begin = std::chrono::steady_clock::now();
            fn(options.iterations, output.data());
            end = std::chrono::steady_clock::now();
        });
        if (trap.trapped != 0) {
            result.trap = trap;
            result.status = Y26_VMADOT_STATUS_SIGILL_CAUGHT;
            return result;
        }
        cycles.push_back(std::chrono::duration<double, std::nano>(end - begin).count() /
                         static_cast<double>(options.iterations));
    #else
        fn(options.iterations, output.data());
        cycles.push_back(0.0);
    #endif
        result.checksum ^= checksum_words(output.data(), output.size()) + static_cast<std::uint64_t>(r);
    }
    result.cycles_per_iteration = summarize(cycles);
    result.cycles_per_vmadot =
        vmadots_per_iteration > 0 ? result.cycles_per_iteration.mean / static_cast<double>(vmadots_per_iteration) : 0.0;
    result.mismatches = compare_block(output.data(), scalar.data(), expected_multiplier);
    result.status = result.mismatches == 0 ? Y26_VMADOT_STATUS_SUCCESS : kStage35OracleMismatch;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    alignas(64) std::array<std::int8_t, 32> a {};
    alignas(64) std::array<std::int8_t, 32> b {};
    alignas(64) std::array<std::int32_t, 96> output {};
    alignas(64) std::array<std::int32_t, 16> scalar {};
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<std::int8_t>((static_cast<int>(i) * 7) % 251 - 125);
        b[i] = static_cast<std::int8_t>(123 - ((static_cast<int>(i) * 11) % 247));
    }
    y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), scalar.data(), false);

    std::cout << "stage35_vmadot_sigill"
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " repeats=" << options.repeats
              << " case=" << options.only_case
              << " benchmark_scope=vmadot_microkernel_only_not_yolo26_inference"
              << "\n";

    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        std::cout << "ime_status=not-built\n";
        return 0;
    }

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    const int probe_status = y26_k1x_ime_probe_once();
    const bool hotpath_allowed = y26_k1x_ime_hotpath_allowed_on_current_cpu();
    const auto snapshot = y26_k1x_ime_runtime_state_snapshot();
    std::cout << "probe_status=" << probe_status
              << " probe_cpu=" << snapshot.probe_cpu
              << " capability=" << snapshot.capability
              << " hotpath_allowed=" << (hotpath_allowed ? 1 : 0)
              << " current_cpu=" << sched_getcpu()
              << "\n";
    if (probe_status != Y26_VMADOT_STATUS_SUCCESS || !hotpath_allowed) {
        return 1;
    }

    auto wants = [&](const char* name) {
        return options.only_case.empty() || options.only_case == "all" || options.only_case == name;
    };

    std::vector<CaseResult> results;
    if (wants("case0_existing_helper_call") || wants("A0_existing_helper")) {
        results.push_back(run_case(
            "case0_existing_helper_call", "public_helper_named", 1, 1, 1, options,
            [&](int iterations, std::int32_t* dst) {
                for (int i = 0; i < iterations; ++i) {
                    y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(a.data(), b.data(), dst, false);
                }
            },
            scalar, output));
    }
    if (wants("case1_stage34_exact_single_wrapper_shape_named")) {
        results.push_back(run_case(
            "case1_stage34_exact_single_wrapper_shape_named", "inline_named", 1, 1, 1, options,
            [&](int iterations, std::int32_t* dst) { named_exact_v28(iterations, a.data(), b.data(), dst); },
            scalar, output));
    }
    if (wants("case2_stage34_exact_single_wrapper_shape_raw_same_as_helper")) {
        results.push_back(run_case(
            "case2_stage34_exact_single_wrapper_shape_raw_same_as_helper", "inline_raw_word_0xe2103e2b", 1, 1, 1,
            options,
            [&](int iterations, std::int32_t* dst) { raw_exact_v28(iterations, a.data(), b.data(), dst); },
            scalar, output));
    }
    if (wants("case3_standalone_S_known_good_bytes")) {
        results.push_back(run_case(
            "case3_standalone_S_known_good_bytes", "standalone_raw_word_0xe2103e2b", 1, 1, 1, options,
            [&](int iterations, std::int32_t* dst) {
                for (int i = 0; i < iterations; ++i) {
                    y26_stage35_standalone_raw_v28(a.data(), b.data(), dst);
                }
            },
            scalar, output));
    }
    if (wants("case4_standalone_S_named_v28_v0_v1")) {
        results.push_back(run_case(
            "case4_standalone_S_named_v28_v0_v1", "standalone_named", 1, 1, 1, options,
            [&](int iterations, std::int32_t* dst) {
                for (int i = 0; i < iterations; ++i) {
                    y26_stage35_standalone_named_v28(a.data(), b.data(), dst);
                }
            },
            scalar, output));
    }
    if (wants("case5_standalone_S_raw_word_same_as_case4")) {
        results.push_back(run_case(
            "case5_standalone_S_raw_word_same_as_case4", "standalone_raw_word_0xe2103e2b", 1, 1, 1, options,
            [&](int iterations, std::int32_t* dst) {
                for (int i = 0; i < iterations; ++i) {
                    y26_stage35_standalone_raw_v28(a.data(), b.data(), dst);
                }
            },
            scalar, output));
    }
    if (wants("case6_v24_v0_v1")) {
        results.push_back(run_case(
            "case6_v24_v0_v1", "inline_named_v24", 1, 1, 1, options,
            [&](int iterations, std::int32_t* dst) { named_single_v24(iterations, a.data(), b.data(), dst); },
            scalar, output));
    }
    if (wants("case7_v20_v0_v1")) {
        results.push_back(run_case(
            "case7_v20_v0_v1", "inline_named_v20", 1, 1, 1, options,
            [&](int iterations, std::int32_t* dst) { named_single_v20(iterations, a.data(), b.data(), dst); },
            scalar, output));
    }
    if (wants("case8_two_accumulators_v28_v30") || wants("A3_raw_independent_2_accumulators")) {
        results.push_back(run_case(
            "case8_two_accumulators_v28_v30", "inline_raw_words_v28_v30", 2, 2, options.iterations, options,
            [&](int iterations, std::int32_t* dst) {
                raw_two_acc_v28_v30(iterations, a.data(), b.data(), dst, dst + 16);
            },
            scalar, output));
    }
    if (wants("case9_four_accumulators_v20_v22_v24_v26") || wants("A4_raw_independent_4_accumulators")) {
        results.push_back(run_case(
            "case9_four_accumulators_v20_v22_v24_v26", "inline_raw_words_v20_v22_v24_v26", 4, 4,
            options.iterations, options,
            [&](int iterations, std::int32_t* dst) {
                raw_four_acc_v20_v22_v24_v26(iterations, a.data(), b.data(), dst, dst + 16, dst + 32, dst + 48);
            },
            scalar, output));
    }
    if (wants("A5_raw_independent_6_accumulators_if_register_safe")) {
        results.push_back(run_case(
            "A5_raw_independent_6_accumulators_if_register_safe", "inline_raw_words_v16_v18_v20_v22_v24_v26", 6, 6,
            options.iterations, options,
            [&](int iterations, std::int32_t* dst) {
                raw_six_acc_v16_v18_v20_v22_v24_v26(iterations, a.data(), b.data(), dst, dst + 16, dst + 32,
                                                     dst + 48, dst + 64, dst + 80);
            },
            scalar, output));
    }
    if (wants("A1_raw_single_acc_dependent_chain")) {
        results.push_back(run_case(
            "A1_raw_single_acc_dependent_chain", "inline_raw_word_0xe2103e2b", 1, 1, options.iterations, options,
            [&](int iterations, std::int32_t* dst) { raw_dep_chain_v28(iterations, a.data(), b.data(), dst); },
            scalar, output));
    }
    if (wants("A2_raw_single_acc_load_included")) {
        results.push_back(run_case(
            "A2_raw_single_acc_load_included", "inline_raw_word_0xe2103e2b_load_each", 1, 1, options.iterations,
            options,
            [&](int iterations, std::int32_t* dst) { raw_load_included_v28(iterations, a.data(), b.data(), dst); },
            scalar, output));
    }

    if (results.empty()) {
        std::cerr << "no case selected by --case=" << options.only_case << "\n";
        return 2;
    }

    std::cout << "case\temission\tstatus\tmismatches\taccumulator_groups\tvmadots_per_iteration\tmean_ns_per_iteration\tstddev\tmin\tmax\tns_per_vmadot\tchecksum\ttrap\tsi_code\tcpu\tpc\tfaulting_insn32_hex\n";
    int failed = 0;
    for (const CaseResult& result : results) {
        if (result.status != Y26_VMADOT_STATUS_SUCCESS) {
            failed = 1;
        }
        std::cout << result.name << "\t"
                  << result.emission << "\t"
                  << result.status << "\t"
                  << result.mismatches << "\t"
                  << result.accumulator_groups << "\t"
                  << result.vmadots_per_iteration << "\t"
                  << result.cycles_per_iteration.mean << "\t"
                  << result.cycles_per_iteration.stddev << "\t"
                  << result.cycles_per_iteration.min << "\t"
                  << result.cycles_per_iteration.max << "\t"
                  << result.cycles_per_vmadot << "\t"
                  << result.checksum << "\t"
                  << result.trap.trapped << "\t"
                  << result.trap.si_code << "\t"
                  << result.trap.cpu << "\t0x"
                  << std::hex << result.trap.pc << "\t0x" << result.trap.insn32 << std::dec << "\n";
    }
    return failed;
#else
    (void)output;
    std::cout << "ime_status=not-riscv-asm-build\n";
    return 0;
#endif
}
