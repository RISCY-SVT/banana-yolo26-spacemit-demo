#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

#if defined(__linux__)
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(__riscv) && defined(__linux__)
#include <sys/ucontext.h>
#include <ucontext.h>
#endif

namespace {

struct VectorState {
    std::uint64_t vl = 0;
    std::uint64_t vtype = 0;
    std::uint64_t vstart = 0;
    std::uint64_t vcsr = 0;
    std::uint64_t executed = 0;
};

static_assert(offsetof(VectorState, vl) == 0);
static_assert(offsetof(VectorState, vtype) == 8);
static_assert(offsetof(VectorState, vstart) == 16);
static_assert(offsetof(VectorState, vcsr) == 24);
static_assert(offsetof(VectorState, executed) == 32);

struct ProbeResult {
    char name[8] {};
    std::int32_t requested_cpu = -1;
    std::int32_t observed_cpu = -1;
    std::int32_t trapped = 0;
    std::int32_t signal_number = 0;
    std::int32_t si_code = 0;
    std::int32_t child_exit = 0;
    std::uint64_t fault_address = 0;
    std::uint64_t fault_pc = 0;
    std::uint32_t instruction = 0;
    std::uint32_t mismatch_count = 0;
    VectorState setup {};
    VectorState trap {};
    VectorState after {};
    std::uint64_t input_hash = 0;
    std::uint64_t output_hash = 0;
    std::uint64_t oracle_hash = 0;
};

constexpr std::uint64_t kFnvOffset = UINT64_C(1469598103934665603);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

[[maybe_unused]] std::uint64_t hash_bytes(const void* data, std::size_t size) noexcept {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

#if defined(__riscv) && defined(__linux__)

sigjmp_buf g_sigill_jump;
volatile sig_atomic_t g_sigill_armed = 0;
ProbeResult* g_active_result = nullptr;

VectorState read_vector_state() noexcept {
    VectorState state {};
    asm volatile("csrr %0, vl" : "=r"(state.vl));
    asm volatile("csrr %0, vtype" : "=r"(state.vtype));
    asm volatile("csrr %0, vstart" : "=r"(state.vstart));
    asm volatile("csrr %0, vcsr" : "=r"(state.vcsr));
    return state;
}

void sigill_handler(int signal_number, siginfo_t* info, void* opaque_context) {
    ProbeResult* result = g_active_result;
    if (result != nullptr) {
        result->trapped = 1;
        result->signal_number = signal_number;
        result->si_code = info != nullptr ? info->si_code : 0;
        result->fault_address = info != nullptr
            ? reinterpret_cast<std::uintptr_t>(info->si_addr)
            : 0;
        result->observed_cpu = sched_getcpu();
        result->trap = read_vector_state();
        auto* context = reinterpret_cast<ucontext_t*>(opaque_context);
        if (context != nullptr) {
            result->fault_pc = static_cast<std::uint64_t>(
                context->uc_mcontext.__gregs[REG_PC]);
            if (result->fault_pc != 0) {
                result->instruction = *reinterpret_cast<const volatile std::uint32_t*>(
                    static_cast<std::uintptr_t>(result->fault_pc));
            }
        }
    }
    if (g_sigill_armed != 0) siglongjmp(g_sigill_jump, 1);
    _exit(128 + SIGILL);
}

bool install_sigill_handler(struct sigaction* previous) noexcept {
    struct sigaction action {};
    action.sa_sigaction = sigill_handler;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    return sigaction(SIGILL, &action, previous) == 0;
}

bool pin_current_thread(int cpu) noexcept {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

#define Y26_SNAPSHOT_VECTOR_STATE \
    "csrr t2, vl\n\t" \
    "sd t2, 0(%[state])\n\t" \
    "csrr t2, vtype\n\t" \
    "sd t2, 8(%[state])\n\t" \
    "csrr t3, vstart\n\t" \
    "sd t3, 16(%[state])\n\t" \
    "csrr t3, vcsr\n\t" \
    "sd t3, 24(%[state])\n\t" \
    "bltz t2, 9f\n\t"

#define Y26_MARK_EXECUTED \
    "li t2, 1\n\t" \
    "sd t2, 32(%[state])\n\t" \
    "9:\n\t"

extern "C" __attribute__((noinline)) void y26_stage55_probe_i0(
    const std::uint8_t* indices, const std::uint8_t* table,
    std::uint8_t* output, std::size_t count, VectorState* state) noexcept {
    asm volatile(
        "vsetvli t0, %[count], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[indices])\n\t"
        Y26_SNAPSHOT_VECTOR_STATE
        "vluxei8.v v2, (%[table]), v0\n\t"
        "vse8.v v2, (%[output])\n\t"
        Y26_MARK_EXECUTED
        :
        : [indices] "r"(indices), [table] "r"(table), [output] "r"(output),
          [count] "r"(count), [state] "r"(state)
        : "memory", "t0", "t2", "t3", "v0", "v2");
}

extern "C" __attribute__((noinline)) void y26_stage55_probe_i1(
    const std::uint8_t* left, const std::uint8_t* right,
    const std::uint8_t* table, std::uint8_t* output,
    std::size_t count, VectorState* state) noexcept {
    asm volatile(
        "vsetvli t0, %[count], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[left])\n\t"
        "vle8.v v1, (%[right])\n\t"
        "vwaddu.vx v2, v0, zero\n\t"
        "vwaddu.vx v4, v1, zero\n\t"
        "vsetvli zero, t0, e16, m2, ta, ma\n\t"
        "vsll.vi v2, v2, 8\n\t"
        "vor.vv v2, v2, v4\n\t"
        "vsetvli zero, t0, e8, m1, ta, ma\n\t"
        Y26_SNAPSHOT_VECTOR_STATE
        "vluxei16.v v6, (%[table]), v2\n\t"
        "vse8.v v6, (%[output])\n\t"
        Y26_MARK_EXECUTED
        :
        : [left] "r"(left), [right] "r"(right), [table] "r"(table),
          [output] "r"(output), [count] "r"(count), [state] "r"(state)
        : "memory", "t0", "t2", "t3", "v0", "v1", "v2", "v3",
          "v4", "v5", "v6");
}

extern "C" __attribute__((noinline)) void y26_stage55_probe_i2(
    const std::uint8_t* offsets, const std::uint16_t* table,
    std::uint16_t* output, std::size_t count, VectorState* state) noexcept {
    asm volatile(
        "vsetvli t0, %[count], e8, mf2, ta, ma\n\t"
        "vle8.v v0, (%[offsets])\n\t"
        "vsetvli zero, t0, e16, m1, ta, ma\n\t"
        Y26_SNAPSHOT_VECTOR_STATE
        "vluxei8.v v2, (%[table]), v0\n\t"
        "vse16.v v2, (%[output])\n\t"
        Y26_MARK_EXECUTED
        :
        : [offsets] "r"(offsets), [table] "r"(table), [output] "r"(output),
          [count] "r"(count), [state] "r"(state)
        : "memory", "t0", "t2", "t3", "v0", "v2");
}

extern "C" __attribute__((noinline)) void y26_stage55_probe_i3(
    const std::uint8_t* offsets, const std::uint32_t* table,
    std::uint32_t* output, std::size_t count, VectorState* state) noexcept {
    asm volatile(
        "vsetvli t0, %[count], e8, mf2, ta, ma\n\t"
        "vle8.v v0, (%[offsets])\n\t"
        "vsetvli zero, t0, e32, m2, ta, ma\n\t"
        Y26_SNAPSHOT_VECTOR_STATE
        "vluxei8.v v4, (%[table]), v0\n\t"
        "vse32.v v4, (%[output])\n\t"
        Y26_MARK_EXECUTED
        :
        : [offsets] "r"(offsets), [table] "r"(table), [output] "r"(output),
          [count] "r"(count), [state] "r"(state)
        : "memory", "t0", "t2", "t3", "v0", "v4", "v5");
}

extern "C" __attribute__((noinline)) void y26_stage55_probe_i4(
    const std::uint8_t* offsets, const std::uint64_t* table,
    std::uint64_t* output, std::size_t count, VectorState* state) noexcept {
    asm volatile(
        "vsetvli t0, %[count], e8, mf2, ta, ma\n\t"
        "vle8.v v0, (%[offsets])\n\t"
        "vsetvli zero, t0, e64, m4, ta, ma\n\t"
        Y26_SNAPSHOT_VECTOR_STATE
        "vluxei8.v v8, (%[table]), v0\n\t"
        "vse64.v v8, (%[output])\n\t"
        Y26_MARK_EXECUTED
        :
        : [offsets] "r"(offsets), [table] "r"(table), [output] "r"(output),
          [count] "r"(count), [state] "r"(state)
        : "memory", "t0", "t2", "t3", "v0", "v8", "v9", "v10", "v11");
}

extern "C" __attribute__((noinline)) void y26_stage55_probe_i5(
    const std::uint8_t* offsets, const std::uint64_t* table,
    std::uint64_t* output, std::size_t count, VectorState* state) noexcept {
    asm volatile(
        "vsetvli t0, %[count], e8, mf4, ta, ma\n\t"
        "vle8.v v0, (%[offsets])\n\t"
        "vsetvli zero, t0, e64, m2, ta, ma\n\t"
        Y26_SNAPSHOT_VECTOR_STATE
        "vluxei8.v v8, (%[table]), v0\n\t"
        "vse64.v v8, (%[output])\n\t"
        Y26_MARK_EXECUTED
        :
        : [offsets] "r"(offsets), [table] "r"(table), [output] "r"(output),
          [count] "r"(count), [state] "r"(state)
        : "memory", "t0", "t2", "t3", "v0", "v8", "v9");
}

extern "C" __attribute__((noinline)) void y26_stage55_probe_i6(
    const std::uint8_t* indices, const std::uint64_t* table,
    std::uint64_t* output, std::size_t count, VectorState* state) noexcept {
    asm volatile(
        "vsetvli t0, %[count], e8, mf2, ta, ma\n\t"
        "vle8.v v0, (%[indices])\n\t"
        "vwaddu.vx v2, v0, zero\n\t"
        "vsetvli zero, t0, e16, m1, ta, ma\n\t"
        "vsll.vi v2, v2, 3\n\t"
        "vsetvli zero, t0, e64, m4, ta, ma\n\t"
        Y26_SNAPSHOT_VECTOR_STATE
        "vluxei16.v v8, (%[table]), v2\n\t"
        "vse64.v v8, (%[output])\n\t"
        Y26_MARK_EXECUTED
        :
        : [indices] "r"(indices), [table] "r"(table), [output] "r"(output),
          [count] "r"(count), [state] "r"(state)
        : "memory", "t0", "t2", "t3", "v0", "v2", "v8", "v9", "v10", "v11");
}

#undef Y26_SNAPSHOT_VECTOR_STATE
#undef Y26_MARK_EXECUTED

template <typename Value, std::size_t Count>
void finish_result(ProbeResult* result, const std::array<Value, Count>& input,
                   const std::array<Value, Count>& output,
                   const std::array<Value, Count>& oracle) noexcept {
    result->input_hash = hash_bytes(input.data(), sizeof(input));
    result->output_hash = hash_bytes(output.data(), sizeof(output));
    result->oracle_hash = hash_bytes(oracle.data(), sizeof(oracle));
    for (std::size_t index = 0; index < Count; ++index) {
        if (output[index] != oracle[index]) ++result->mismatch_count;
    }
}

void run_i0(ProbeResult* result) noexcept {
    alignas(64) std::array<std::uint8_t, 256> table {};
    alignas(64) std::array<std::uint8_t, 32> indices {};
    alignas(64) std::array<std::uint8_t, 32> output {};
    std::array<std::uint8_t, 32> oracle {};
    for (std::size_t index = 0; index < table.size(); ++index) {
        table[index] = static_cast<std::uint8_t>((index * 73U + 19U) & 255U);
    }
    for (std::size_t index = 0; index < indices.size(); ++index) {
        indices[index] = static_cast<std::uint8_t>((index * 29U + 7U) & 255U);
        oracle[index] = table[indices[index]];
    }
    y26_stage55_probe_i0(indices.data(), table.data(), output.data(), output.size(), &result->setup);
    result->after = read_vector_state();
    finish_result(result, indices, output, oracle);
}

void run_i1(ProbeResult* result) noexcept {
    alignas(64) std::array<std::uint8_t, 65536> table {};
    alignas(64) std::array<std::uint8_t, 32> left {};
    alignas(64) std::array<std::uint8_t, 32> right {};
    alignas(64) std::array<std::uint8_t, 32> output {};
    std::array<std::uint8_t, 32> oracle {};
    for (std::size_t index = 0; index < table.size(); ++index) {
        table[index] = static_cast<std::uint8_t>((index * 17U + (index >> 8U) * 31U + 11U) & 255U);
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        left[index] = static_cast<std::uint8_t>((index * 37U + 3U) & 255U);
        right[index] = static_cast<std::uint8_t>((index * 61U + 5U) & 255U);
        oracle[index] = table[static_cast<std::size_t>(left[index]) * 256U + right[index]];
    }
    y26_stage55_probe_i1(left.data(), right.data(), table.data(), output.data(), output.size(), &result->setup);
    result->after = read_vector_state();
    std::array<std::uint8_t, 32> combined {};
    for (std::size_t index = 0; index < combined.size(); ++index) {
        combined[index] = static_cast<std::uint8_t>(left[index] ^ right[index]);
    }
    finish_result(result, combined, output, oracle);
}

void run_i6(ProbeResult* result) noexcept {
    alignas(64) std::array<std::uint64_t, 256> table {};
    alignas(64) std::array<std::uint8_t, 16> indices {};
    alignas(64) std::array<std::uint64_t, 16> output {};
    std::array<std::uint64_t, 16> oracle {};
    for (std::size_t index = 0; index < table.size(); ++index) {
        table[index] = UINT64_C(0x9e3779b97f4a7c15) * (index + 1U);
    }
    for (std::size_t index = 0; index < indices.size(); ++index) {
        indices[index] = static_cast<std::uint8_t>((index * 17U + 11U) & 255U);
        oracle[index] = table[indices[index]];
    }
    y26_stage55_probe_i6(indices.data(), table.data(), output.data(), output.size(), &result->setup);
    result->after = read_vector_state();
    std::array<std::uint64_t, 16> expanded_input {};
    for (std::size_t index = 0; index < indices.size(); ++index) {
        expanded_input[index] = indices[index];
    }
    finish_result(result, expanded_input, output, oracle);
}

template <typename Value, std::size_t Count, typename Invoke>
void run_typed(ProbeResult* result, Invoke&& invoke) noexcept {
    alignas(64) std::array<Value, 32> table {};
    alignas(64) std::array<std::uint8_t, Count> offsets {};
    alignas(64) std::array<Value, Count> output {};
    std::array<Value, Count> oracle {};
    for (std::size_t index = 0; index < table.size(); ++index) {
        table[index] = static_cast<Value>(
            UINT64_C(0x9e3779b97f4a7c15) * (index + 1U) + UINT64_C(0x102030405060708));
    }
    for (std::size_t index = 0; index < Count; ++index) {
        const std::size_t table_index = (index * 5U + 3U) % table.size();
        offsets[index] = static_cast<std::uint8_t>(table_index * sizeof(Value));
        oracle[index] = table[table_index];
    }
    invoke(offsets.data(), table.data(), output.data(), output.size(), &result->setup);
    result->after = read_vector_state();
    std::array<Value, Count> expanded_input {};
    for (std::size_t index = 0; index < Count; ++index) expanded_input[index] = offsets[index];
    finish_result(result, expanded_input, output, oracle);
}

void execute_case(std::string_view name, ProbeResult* result) noexcept {
    if (name == "I0") {
        run_i0(result);
    } else if (name == "I1") {
        run_i1(result);
    } else if (name == "I2") {
        run_typed<std::uint16_t, 16>(result, y26_stage55_probe_i2);
    } else if (name == "I3") {
        run_typed<std::uint32_t, 16>(result, y26_stage55_probe_i3);
    } else if (name == "I4") {
        run_typed<std::uint64_t, 16>(result, y26_stage55_probe_i4);
    } else if (name == "I5") {
        run_typed<std::uint64_t, 8>(result, y26_stage55_probe_i5);
    } else if (name == "I6") {
        run_i6(result);
    } else {
        result->child_exit = 64;
    }
}

#endif

void print_header() {
    std::puts(
        "case\trequested_cpu\tobserved_cpu\tstatus\texecuted\ttrapped\tsignal\tsi_code"
        "\tsi_addr\tfault_pc\tinstruction\tvl\tvtype\tvill\tvstart\tvcsr"
        "\ttrap_vl\ttrap_vtype\ttrap_vstart\ttrap_vcsr\tafter_vl\tafter_vtype"
        "\tafter_vstart\tafter_vcsr\tmismatches\tinput_hash\toutput_hash\toracle_hash"
        "\tchild_exit");
}

void print_result(const ProbeResult& result) {
    const char* status = "unsupported-host";
#if defined(__riscv) && defined(__linux__)
    if (result.trapped != 0) {
        status = "board-SIGILL-at-identified-opcode";
    } else if ((result.setup.vtype >> 63U) != 0U) {
        status = "vill-rejected";
    } else if (result.setup.executed == 0) {
        status = "not-executed";
    } else if (result.mismatch_count == 0 && result.output_hash == result.oracle_hash) {
        status = "exact";
    } else {
        status = "mismatch";
    }
#endif
    std::printf(
        "%s\t%d\t%d\t%s\t%" PRIu64 "\t%d\t%d\t%d\t0x%016" PRIx64
        "\t0x%016" PRIx64 "\t0x%08" PRIx32 "\t%" PRIu64 "\t0x%016" PRIx64
        "\t%" PRIu64 "\t%" PRIu64 "\t0x%016" PRIx64 "\t%" PRIu64
        "\t0x%016" PRIx64 "\t%" PRIu64 "\t0x%016" PRIx64 "\t%" PRIu64
        "\t0x%016" PRIx64 "\t%" PRIu64 "\t0x%016" PRIx64 "\t%u"
        "\t0x%016" PRIx64 "\t0x%016" PRIx64 "\t0x%016" PRIx64 "\t%d\n",
        result.name, result.requested_cpu, result.observed_cpu, status,
        result.setup.executed, result.trapped, result.signal_number, result.si_code,
        result.fault_address, result.fault_pc, result.instruction, result.setup.vl,
        result.setup.vtype, result.setup.vtype >> 63U, result.setup.vstart,
        result.setup.vcsr, result.trap.vl, result.trap.vtype, result.trap.vstart,
        result.trap.vcsr, result.after.vl, result.after.vtype, result.after.vstart,
        result.after.vcsr, result.mismatch_count, result.input_hash,
        result.output_hash, result.oracle_hash, result.child_exit);
}

}  // namespace

int main(int argc, char** argv) {
    int requested_cpu = 0;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--cpu" && index + 1 < argc) {
            requested_cpu = std::atoi(argv[++index]);
        } else {
            std::fprintf(stderr, "usage: %s [--cpu N]\n", argv[0]);
            return 64;
        }
    }

    constexpr std::array<std::string_view, 7> cases {"I0", "I1", "I2", "I3", "I4", "I5", "I6"};
    print_header();
    int failures = 0;
    for (const std::string_view name : cases) {
        ProbeResult result {};
        std::snprintf(result.name, sizeof(result.name), "%.*s",
                      static_cast<int>(name.size()), name.data());
        result.requested_cpu = requested_cpu;
#if defined(__riscv) && defined(__linux__)
        int descriptors[2] {-1, -1};
        if (pipe(descriptors) != 0) {
            std::perror("pipe");
            return 70;
        }
        const pid_t child = fork();
        if (child < 0) {
            std::perror("fork");
            return 70;
        }
        if (child == 0) {
            close(descriptors[0]);
            result.observed_cpu = -1;
            if (!pin_current_thread(requested_cpu)) {
                result.child_exit = errno == 0 ? 71 : errno;
            } else {
                result.observed_cpu = sched_getcpu();
                struct sigaction previous {};
                if (!install_sigill_handler(&previous)) {
                    result.child_exit = errno == 0 ? 72 : errno;
                } else {
                    g_active_result = &result;
                    g_sigill_armed = 1;
                    if (sigsetjmp(g_sigill_jump, 1) == 0) execute_case(name, &result);
                    g_sigill_armed = 0;
                    g_active_result = nullptr;
                    sigaction(SIGILL, &previous, nullptr);
                }
            }
            const ssize_t written = write(descriptors[1], &result, sizeof(result));
            close(descriptors[1]);
            _exit(written == static_cast<ssize_t>(sizeof(result)) ? 0 : 73);
        }
        close(descriptors[1]);
        const ssize_t received = read(descriptors[0], &result, sizeof(result));
        close(descriptors[0]);
        int wait_status = 0;
        waitpid(child, &wait_status, 0);
        if (received != static_cast<ssize_t>(sizeof(result))) {
            result.child_exit = WIFSIGNALED(wait_status) ? 128 + WTERMSIG(wait_status) : 74;
        } else if (WIFEXITED(wait_status) && WEXITSTATUS(wait_status) != 0) {
            result.child_exit = WEXITSTATUS(wait_status);
        }
#else
        result.child_exit = 77;
#endif
        print_result(result);
#if defined(__riscv) && defined(__linux__)
        if (result.trapped != 0 || result.setup.executed == 0 ||
            result.mismatch_count != 0 || result.output_hash != result.oracle_hash ||
            result.child_exit != 0) {
            ++failures;
        }
#endif
    }
    return failures == 0 ? 0 : 3;
}
