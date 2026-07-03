#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "y26_k1x_vmadot.h"

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <csetjmp>

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

}  // namespace

extern "C" bool y26_vmadot_4x4x8_ime_available_buildtime() {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    return true;
#else
    return false;
#endif
}

extern "C" int y26_vmadot_4x4x8_ime_s8s8s32(const std::int8_t* a_4x8_row_major,
                                            const std::int8_t* b_4x8_transposed_nk,
                                            std::int32_t* c_4x4_row_major,
                                            bool accumulate) {
    if (a_4x8_row_major == nullptr || b_4x8_transposed_nk == nullptr || c_4x4_row_major == nullptr) {
        return Y26_VMADOT_STATUS_INVALID_ARGUMENT;
    }

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (!current_cpu_is_cluster0()) {
        return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
    }

    struct sigaction old_action {};
    struct sigaction new_action {};
    new_action.sa_sigaction = y26_vmadot_sigill_handler;
    new_action.sa_flags = SA_SIGINFO;
    sigemptyset(&new_action.sa_mask);

    if (sigaction(SIGILL, &new_action, &old_action) != 0) {
        return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
    }

    g_sigill_seen = 0;
    int status = Y26_VMADOT_STATUS_SUCCESS;
    if (sigsetjmp(g_sigill_jump, 1) == 0) {
        y26_vmadot_4x4x8_ime_unguarded(a_4x8_row_major, b_4x8_transposed_nk, c_4x4_row_major, accumulate);
    } else {
        status = Y26_VMADOT_STATUS_SIGILL_CAUGHT;
    }

    if (g_sigill_seen != 0) {
        status = Y26_VMADOT_STATUS_SIGILL_CAUGHT;
    }
    sigaction(SIGILL, &old_action, nullptr);
    return status;
#else
    (void)accumulate;
    return Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME;
#endif
}
