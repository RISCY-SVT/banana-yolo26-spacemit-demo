#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "y26_k1x_vmadot.h"

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <csetjmp>
#include <mutex>

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv) && defined(__linux__)
#include <sched.h>
#endif

namespace {

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
sigjmp_buf g_sigill_jump;
volatile sig_atomic_t g_sigill_seen = 0;

void y26_vmadot_sigill_handler(int /*signo*/, siginfo_t* /*info*/, void* /*uctx*/) {
    g_sigill_seen = 1;
    siglongjmp(g_sigill_jump, 1);
}

bool current_cpu_is_cluster0() {
#if defined(__linux__)
    const int cpu = sched_getcpu();
    return cpu >= 0 && cpu <= 3;
#else
    return false;
#endif
}

int current_cpu_or_unknown() {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

__attribute__((noinline)) void y26_vmadot_4x4x8_ime_unguarded(const std::int8_t* a_4x8_row_major,
                                                              const std::int8_t* b_4x8_transposed_nk,
                                                              std::int32_t* c_4x4_row_major,
                                                              bool accumulate) {
    if (accumulate) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vle32.v      v28, (%[C])             \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v0, (%[A])              \n\t"
            "vle8.v       v1, (%[B])              \n\t"
            "smt.vmadot   v28, v0, v1             \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a_4x8_row_major), [B] "r"(b_4x8_transposed_nk), [C] "r"(c_4x4_row_major)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    } else {
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
            : [A] "r"(a_4x8_row_major), [B] "r"(b_4x8_transposed_nk), [C] "r"(c_4x4_row_major)
            : "cc", "memory", "t0", "v0", "v1", "v28", "v29");
    }
}
#endif

std::atomic<int> g_probe_initialized {0};
std::atomic<int> g_probe_capability {Y26_IME_CAPABILITY_UNKNOWN};
std::atomic<int> g_probe_cpu {-1};
std::atomic<int> g_probe_status {Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME};
std::mutex g_probe_mutex;
thread_local bool g_thread_cluster0_hotpath_allowed = false;

bool pointers_valid(const std::int8_t* a_4x8_row_major,
                    const std::int8_t* b_4x8_transposed_nk,
                    const std::int32_t* c_4x4_row_major) {
    return a_4x8_row_major != nullptr && b_4x8_transposed_nk != nullptr && c_4x4_row_major != nullptr;
}

}  // namespace

extern "C" bool y26_vmadot_4x4x8_ime_available_buildtime() {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    return true;
#else
    return false;
#endif
}

extern "C" int y26_k1x_ime_probe_once() {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (g_probe_initialized.load(std::memory_order_acquire) != 0) {
        return g_probe_status.load(std::memory_order_acquire);
    }

    std::lock_guard<std::mutex> lock(g_probe_mutex);
    if (g_probe_initialized.load(std::memory_order_relaxed) != 0) {
        return g_probe_status.load(std::memory_order_relaxed);
    }

    g_probe_cpu.store(current_cpu_or_unknown(), std::memory_order_relaxed);
    if (!current_cpu_is_cluster0()) {
        g_probe_capability.store(Y26_IME_CAPABILITY_UNAVAILABLE, std::memory_order_release);
        g_probe_status.store(Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED, std::memory_order_release);
        g_probe_initialized.store(1, std::memory_order_release);
        return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
    }

    struct sigaction old_action {};
    struct sigaction new_action {};
    new_action.sa_sigaction = y26_vmadot_sigill_handler;
    new_action.sa_flags = SA_SIGINFO;
    sigemptyset(&new_action.sa_mask);

    if (sigaction(SIGILL, &new_action, &old_action) != 0) {
        g_probe_capability.store(Y26_IME_CAPABILITY_UNAVAILABLE, std::memory_order_release);
        g_probe_status.store(Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED, std::memory_order_release);
        g_probe_initialized.store(1, std::memory_order_release);
        return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
    }

    alignas(64) std::int8_t a[32] {};
    alignas(64) std::int8_t b[32] {};
    alignas(64) std::int32_t c[16] {};
    a[0] = 1;
    b[0] = 1;

    g_sigill_seen = 0;
    volatile sig_atomic_t sigill_status = 0;
    if (sigsetjmp(g_sigill_jump, 1) == 0) {
        y26_vmadot_4x4x8_ime_unguarded(a, b, c, false);
    } else {
        sigill_status = 1;
    }

    int status = sigill_status == 0 ? Y26_VMADOT_STATUS_SUCCESS : Y26_VMADOT_STATUS_SIGILL_CAUGHT;
    if (g_sigill_seen != 0) {
        status = Y26_VMADOT_STATUS_SIGILL_CAUGHT;
    }
    sigaction(SIGILL, &old_action, nullptr);

    if (status == Y26_VMADOT_STATUS_SUCCESS && c[0] == 1) {
        g_probe_capability.store(Y26_IME_CAPABILITY_AVAILABLE_CLUSTER0_ONLY, std::memory_order_release);
        g_thread_cluster0_hotpath_allowed = true;
    } else {
        status = Y26_VMADOT_STATUS_SIGILL_CAUGHT;
        g_probe_capability.store(Y26_IME_CAPABILITY_UNAVAILABLE, std::memory_order_release);
    }

    g_probe_status.store(status, std::memory_order_release);
    g_probe_initialized.store(1, std::memory_order_release);
    return status;
#else
    g_probe_capability.store(Y26_IME_CAPABILITY_UNAVAILABLE, std::memory_order_release);
    g_probe_status.store(Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME, std::memory_order_release);
    g_probe_initialized.store(1, std::memory_order_release);
    return Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME;
#endif
}

extern "C" bool y26_k1x_ime_available() {
    return y26_k1x_ime_probe_once() == Y26_VMADOT_STATUS_SUCCESS &&
           g_probe_capability.load(std::memory_order_acquire) == Y26_IME_CAPABILITY_AVAILABLE_CLUSTER0_ONLY;
}

extern "C" bool y26_k1x_ime_hotpath_allowed_on_current_cpu() {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (!y26_k1x_ime_available()) {
        g_thread_cluster0_hotpath_allowed = false;
        return false;
    }
    if (!current_cpu_is_cluster0()) {
        g_thread_cluster0_hotpath_allowed = false;
        return false;
    }
    g_thread_cluster0_hotpath_allowed = true;
    return true;
#else
    g_thread_cluster0_hotpath_allowed = false;
    return false;
#endif
}

extern "C" Y26ImeRuntimeStateSnapshot y26_k1x_ime_runtime_state_snapshot() {
    Y26ImeRuntimeStateSnapshot snapshot {};
    snapshot.initialized = g_probe_initialized.load(std::memory_order_acquire);
    snapshot.capability = g_probe_capability.load(std::memory_order_acquire);
    snapshot.probe_cpu = g_probe_cpu.load(std::memory_order_acquire);
    snapshot.probe_status = g_probe_status.load(std::memory_order_acquire);
    return snapshot;
}

extern "C" void y26_k1x_ime_reset_thread_hotpath_for_tests() {
    g_thread_cluster0_hotpath_allowed = false;
}

extern "C" int y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(const std::int8_t* a_4x8_row_major,
                                                             const std::int8_t* b_4x8_transposed_nk,
                                                             std::int32_t* c_4x4_row_major,
                                                             bool accumulate) {
    if (!pointers_valid(a_4x8_row_major, b_4x8_transposed_nk, c_4x4_row_major)) {
        return Y26_VMADOT_STATUS_INVALID_ARGUMENT;
    }

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    y26_vmadot_4x4x8_ime_unguarded(a_4x8_row_major, b_4x8_transposed_nk, c_4x4_row_major, accumulate);
    return Y26_VMADOT_STATUS_SUCCESS;
#else
    (void)accumulate;
    return Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME;
#endif
}

extern "C" int y26_k1x_vmadot_4x4x8_checked_cluster0_s8s8s32(const std::int8_t* a_4x8_row_major,
                                                             const std::int8_t* b_4x8_transposed_nk,
                                                             std::int32_t* c_4x4_row_major,
                                                             bool accumulate) {
    if (!pointers_valid(a_4x8_row_major, b_4x8_transposed_nk, c_4x4_row_major)) {
        return Y26_VMADOT_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
    }
    return y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(
        a_4x8_row_major, b_4x8_transposed_nk, c_4x4_row_major, accumulate);
}

extern "C" int y26_vmadot_4x4x8_ime_s8s8s32(const std::int8_t* a_4x8_row_major,
                                            const std::int8_t* b_4x8_transposed_nk,
                                            std::int32_t* c_4x4_row_major,
                                            bool accumulate) {
    if (!pointers_valid(a_4x8_row_major, b_4x8_transposed_nk, c_4x4_row_major)) {
        return Y26_VMADOT_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_k1x_ime_available()) {
        return g_probe_status.load(std::memory_order_acquire);
    }
    if (!g_thread_cluster0_hotpath_allowed) {
        if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
            return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
        }
    }
#if defined(Y26_K1X_DEBUG_CHECK_CPU) && defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (!current_cpu_is_cluster0()) {
        g_thread_cluster0_hotpath_allowed = false;
        return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
    }
#endif
    return y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(
        a_4x8_row_major, b_4x8_transposed_nk, c_4x4_row_major, accumulate);
}
