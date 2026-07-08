#include "y26_k1x_vmadot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv) && defined(__linux__)
#include <sched.h>
#endif

namespace {

struct Options {
    int iterations = 1000000;
    int warmup = 1000;
    int repeats = 5;
    std::string only_case = "probe_only_no_vmadot";
};

struct Stat {
    double mean = 0.0;
    double stddev = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct CaseResult {
    const char* name = "";
    int accumulator_groups = 1;
    int loads_per_iteration = 0;
    std::uint64_t guard = 0;
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

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
inline std::uint64_t rdcycle() {
    std::uint64_t value = 0;
    __asm__ volatile("rdcycle %0" : "=r"(value));
    return value;
}

__attribute__((noinline)) void vmadot_dep_chain(int iterations,
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
        "smt.vmadot   v28, v0, v1             \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C] "r"(c), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29");
}

__attribute__((noinline)) void vmadot_exact_single_wrapper_shape(int iterations,
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

__attribute__((noinline)) void vmadot_independent_2(int iterations,
                                                     const std::int8_t* a,
                                                     const std::int8_t* b,
                                                     std::int32_t* c0,
                                                     std::int32_t* c1) {
    __asm__ volatile(
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v24, v24, v24           \n\t"
        "vxor.vv      v26, v26, v26           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        "smt.vmadot   v24, v0, v1             \n\t"
        "smt.vmadot   v26, v0, v1             \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v24, (%[C0])            \n\t"
        "vse32.v      v26, (%[C1])            \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C0] "r"(c0), [C1] "r"(c1), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v24", "v25", "v26", "v27");
}

__attribute__((noinline)) void vmadot_independent_2_high(int iterations,
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
        "smt.vmadot   v28, v0, v1             \n\t"
        "smt.vmadot   v30, v0, v1             \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C0])            \n\t"
        "vse32.v      v30, (%[C1])            \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C0] "r"(c0), [C1] "r"(c1), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29", "v30", "v31");
}

__attribute__((noinline)) void vmadot_independent_4(int iterations,
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
        "smt.vmadot   v20, v0, v1             \n\t"
        "smt.vmadot   v22, v0, v1             \n\t"
        "smt.vmadot   v24, v0, v1             \n\t"
        "smt.vmadot   v26, v0, v1             \n\t"
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

__attribute__((noinline)) void vmadot_independent_6(int iterations,
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
        "vxor.vv      v20, v20, v20           \n\t"
        "vxor.vv      v22, v22, v22           \n\t"
        "vxor.vv      v24, v24, v24           \n\t"
        "vxor.vv      v26, v26, v26           \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "vxor.vv      v30, v30, v30           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        "smt.vmadot   v20, v0, v1             \n\t"
        "smt.vmadot   v22, v0, v1             \n\t"
        "smt.vmadot   v24, v0, v1             \n\t"
        "smt.vmadot   v26, v0, v1             \n\t"
        "smt.vmadot   v28, v0, v1             \n\t"
        "smt.vmadot   v30, v0, v1             \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v20, (%[C0])            \n\t"
        "vse32.v      v22, (%[C1])            \n\t"
        "vse32.v      v24, (%[C2])            \n\t"
        "vse32.v      v26, (%[C3])            \n\t"
        "vse32.v      v28, (%[C4])            \n\t"
        "vse32.v      v30, (%[C5])            \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C0] "r"(c0), [C1] "r"(c1), [C2] "r"(c2), [C3] "r"(c3),
          [C4] "r"(c4), [C5] "r"(c5), [N] "r"(iterations)
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
          "v27",
          "v28",
          "v29",
          "v30",
          "v31");
}

__attribute__((noinline)) void vmadot_load_included_1(int iterations,
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
        "smt.vmadot   v28, v0, v1             \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C] "r"(c), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29");
}

__attribute__((noinline)) void vmadot_safe_vset_each_1(int iterations,
                                                        const std::int8_t* a,
                                                        const std::int8_t* b,
                                                        std::int32_t* c) {
    __asm__ volatile(
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vle8.v       v0, (%[A])              \n\t"
        "vle8.v       v1, (%[B])              \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "smt.vmadot   v28, v0, v1             \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C] "r"(c), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29");
}

__attribute__((noinline)) void vmadot_safe_vset_each_2_high(int iterations,
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
        "mv           t1, %[N]                \n\t"
        "1:                                      \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "smt.vmadot   v28, v0, v1             \n\t"
        "smt.vmadot   v30, v0, v1             \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "addi         t1, t1, -1              \n\t"
        "bnez         t1, 1b                  \n\t"
        "vse32.v      v28, (%[C0])            \n\t"
        "vse32.v      v30, (%[C1])            \n\t"
        :
        : [A] "r"(a), [B] "r"(b), [C0] "r"(c0), [C1] "r"(c1), [N] "r"(iterations)
        : "cc", "memory", "t0", "t1", "v0", "v1", "v28", "v29", "v30", "v31");
}
#endif

template <typename Fn>
CaseResult run_case(const char* name,
                    int accumulator_groups,
                    int loads_per_iteration,
                    const Options& options,
                    Fn&& fn,
                    std::array<std::int32_t, 96>& output) {
    for (int i = 0; i < options.warmup; ++i) {
        fn(std::max(1, options.iterations / 100), output.data());
    }
    std::vector<double> cycles;
    cycles.reserve(static_cast<std::size_t>(options.repeats));
    std::uint64_t guard = 0;
    for (int r = 0; r < options.repeats; ++r) {
        std::fill(output.begin(), output.end(), 0);
    #if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
        const std::uint64_t begin = rdcycle();
        fn(options.iterations, output.data());
        const std::uint64_t end = rdcycle();
        cycles.push_back(static_cast<double>(end - begin) / static_cast<double>(options.iterations));
    #else
        fn(options.iterations, output.data());
        cycles.push_back(0.0);
    #endif
        guard ^= checksum_words(output.data(), output.size()) + static_cast<std::uint64_t>(r);
    }
    CaseResult result {};
    result.name = name;
    result.accumulator_groups = accumulator_groups;
    result.loads_per_iteration = loads_per_iteration;
    result.guard = guard;
    result.cycles_per_iteration = summarize(cycles);
    result.cycles_per_vmadot =
        accumulator_groups > 0 ? result.cycles_per_iteration.mean / static_cast<double>(accumulator_groups) : 0.0;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    alignas(64) std::array<std::int8_t, 32> a {};
    alignas(64) std::array<std::int8_t, 32> b {};
    alignas(64) std::array<std::int32_t, 96> output {};
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<std::int8_t>((static_cast<int>(i) * 7) % 251 - 125);
        b[i] = static_cast<std::int8_t>(123 - ((static_cast<int>(i) * 11) % 247));
    }

    std::cout << "stage34_vmadot_throughput"
              << " iterations=" << options.iterations
              << " warmup=" << options.warmup
              << " repeats=" << options.repeats
              << " case=" << (options.only_case.empty() ? "all" : options.only_case)
              << " benchmark_scope=vmadot_microkernel_only_not_yolo26_inference"
              << "\n";
    std::cout.flush();

    if (!y26_vmadot_4x4x8_ime_available_buildtime()) {
        (void)output;
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
              << " hotpath_allowed=" << (hotpath_allowed ? 1 : 0);
    #if defined(__linux__)
    std::cout << " current_cpu=" << sched_getcpu();
    #endif
    std::cout << "\n";
    std::cout.flush();
    if (probe_status != Y26_VMADOT_STATUS_SUCCESS || !hotpath_allowed) {
        return 1;
    }

    auto wants = [&](const char* name) {
        return options.only_case.empty() || options.only_case == "all" || options.only_case == name;
    };

    std::vector<CaseResult> results;
    if (wants("probe_only_no_vmadot")) {
        CaseResult probe_only {};
        probe_only.name = "probe_only_no_vmadot";
        probe_only.accumulator_groups = 0;
        results.push_back(probe_only);
    }
    if (wants("dependent_chain_1acc_loadfree")) {
        results.push_back(run_case(
        "dependent_chain_1acc_loadfree", 1, 0, options,
            [&](int iterations, std::int32_t* dst) { vmadot_dep_chain(iterations, a.data(), b.data(), dst); }, output));
    }
    if (wants("exact_single_wrapper_shape")) {
        results.push_back(run_case(
            "exact_single_wrapper_shape", 1, 2, options,
            [&](int iterations, std::int32_t* dst) {
                vmadot_exact_single_wrapper_shape(iterations, a.data(), b.data(), dst);
            },
            output));
    }
    if (wants("independent_2acc_loadfree")) {
        results.push_back(run_case(
            "independent_2acc_loadfree", 2, 0, options,
            [&](int iterations, std::int32_t* dst) {
                vmadot_independent_2(iterations, a.data(), b.data(), dst, dst + 16);
            },
            output));
    }
    if (wants("independent_2acc_high_loadfree")) {
        results.push_back(run_case(
            "independent_2acc_high_loadfree", 2, 0, options,
            [&](int iterations, std::int32_t* dst) {
                vmadot_independent_2_high(iterations, a.data(), b.data(), dst, dst + 16);
            },
            output));
    }
    if (wants("independent_4acc_loadfree")) {
        results.push_back(run_case(
            "independent_4acc_loadfree", 4, 0, options,
            [&](int iterations, std::int32_t* dst) {
                vmadot_independent_4(iterations, a.data(), b.data(), dst, dst + 16, dst + 32, dst + 48);
            },
            output));
    }
    if (wants("independent_6acc_loadfree")) {
        results.push_back(run_case(
            "independent_6acc_loadfree", 6, 0, options,
            [&](int iterations, std::int32_t* dst) {
                vmadot_independent_6(iterations, a.data(), b.data(), dst, dst + 16, dst + 32, dst + 48, dst + 64, dst + 80);
            },
            output));
    }
    if (wants("load_included_1acc")) {
        results.push_back(run_case(
            "load_included_1acc", 1, 2, options,
            [&](int iterations, std::int32_t* dst) { vmadot_load_included_1(iterations, a.data(), b.data(), dst); },
            output));
    }
    if (wants("safe_vset_each_1acc")) {
        results.push_back(run_case(
            "safe_vset_each_1acc", 1, 0, options,
            [&](int iterations, std::int32_t* dst) { vmadot_safe_vset_each_1(iterations, a.data(), b.data(), dst); },
            output));
    }
    if (wants("safe_vset_each_2acc_high")) {
        results.push_back(run_case(
            "safe_vset_each_2acc_high", 2, 0, options,
            [&](int iterations, std::int32_t* dst) {
                vmadot_safe_vset_each_2_high(iterations, a.data(), b.data(), dst, dst + 16);
            },
            output));
    }
    if (results.empty()) {
        std::cerr << "no case selected by --case=" << options.only_case << "\n";
        return 2;
    }

    std::cout << "case\taccumulator_groups\tloads_per_iteration\tmean_cycles_per_iteration\tstddev\tmin\tmax\tmean_cycles_per_vmadot\tguard\n";
    for (const CaseResult& result : results) {
        std::cout << result.name << "\t"
                  << result.accumulator_groups << "\t"
                  << result.loads_per_iteration << "\t"
                  << result.cycles_per_iteration.mean << "\t"
                  << result.cycles_per_iteration.stddev << "\t"
                  << result.cycles_per_iteration.min << "\t"
                  << result.cycles_per_iteration.max << "\t"
                  << result.cycles_per_vmadot << "\t"
                  << result.guard << "\n";
    }

    if (options.only_case.empty() || options.only_case == "all") {
        double dep = 0.0;
        double best_independent = std::numeric_limits<double>::infinity();
        for (const CaseResult& result : results) {
            if (std::strcmp(result.name, "dependent_chain_1acc_loadfree") == 0) {
                dep = result.cycles_per_vmadot;
            }
            if (result.accumulator_groups > 1) {
                best_independent = std::min(best_independent, result.cycles_per_vmadot);
            }
        }
        if (!std::isfinite(best_independent)) {
            best_independent = 0.0;
        }
        std::cout << "best_independent_cycles_per_vmadot=" << best_independent
                  << " dependent_cycles_per_vmadot=" << dep
                  << " independent_speedup_vs_dependent=" << (best_independent > 0.0 ? dep / best_independent : 0.0)
                  << " note=selected-subset-microdiagnostic-not-model-fps"
                  << "\n";
    }
    return 0;
#else
    std::cout << "ime_status=not-riscv-asm-build\n";
    return 0;
#endif
}
