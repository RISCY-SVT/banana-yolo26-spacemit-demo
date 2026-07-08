#include "y26_k1x_vmadot123_probe.h"

#include "y26_k1x_vmadot.h"

#include <cstdint>

namespace {

bool pointers_valid(const std::int8_t* a_8x8_row_major,
                    const std::int8_t* b_4x8_transposed_nk,
                    const std::int32_t* c_4x4_row_major) {
    return a_8x8_row_major != nullptr && b_4x8_transposed_nk != nullptr && c_4x4_row_major != nullptr;
}

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
__attribute__((noinline)) void y26_vmadot123_unguarded(int variant,
                                                       const std::int8_t* a_8x8_row_major,
                                                       const std::int8_t* b_4x8_transposed_nk,
                                                       std::int32_t* c_4x4_row_major,
                                                       bool accumulate) {
    if (accumulate) {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vle32.v      v28, (%[C])             \n\t"
            :
            : [C] "r"(c_4x4_row_major)
            : "cc", "memory", "t0", "v28", "v29");
    } else {
        __asm__ volatile(
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vxor.vv      v28, v28, v28           \n\t"
            :
            :
            : "cc", "memory", "t0", "v28", "v29");
    }

    switch (variant) {
    case Y26_VMADOT123_VARIANT_1:
        __asm__ volatile(
            "vsetvli      t0, zero, e8, m2        \n\t"
            "vle8.v       v8, (%[A])              \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v16, (%[B])             \n\t"
            "smt.vmadot1  v28, v8, v16            \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a_8x8_row_major), [B] "r"(b_4x8_transposed_nk), [C] "r"(c_4x4_row_major)
            : "cc", "memory", "t0", "v8", "v9", "v16", "v28", "v29");
        break;
    case Y26_VMADOT123_VARIANT_2:
        __asm__ volatile(
            "vsetvli      t0, zero, e8, m2        \n\t"
            "vle8.v       v8, (%[A])              \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v16, (%[B])             \n\t"
            "smt.vmadot2  v28, v8, v16            \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a_8x8_row_major), [B] "r"(b_4x8_transposed_nk), [C] "r"(c_4x4_row_major)
            : "cc", "memory", "t0", "v8", "v9", "v16", "v28", "v29");
        break;
    case Y26_VMADOT123_VARIANT_3:
        __asm__ volatile(
            "vsetvli      t0, zero, e8, m2        \n\t"
            "vle8.v       v8, (%[A])              \n\t"
            "vsetvli      t0, zero, e8, m1        \n\t"
            "vle8.v       v16, (%[B])             \n\t"
            "smt.vmadot3  v28, v8, v16            \n\t"
            "vsetvli      t0, zero, e32, m2       \n\t"
            "vse32.v      v28, (%[C])             \n\t"
            :
            : [A] "r"(a_8x8_row_major), [B] "r"(b_4x8_transposed_nk), [C] "r"(c_4x4_row_major)
            : "cc", "memory", "t0", "v8", "v9", "v16", "v28", "v29");
        break;
    default:
        break;
    }
}
#endif

}  // namespace

extern "C" bool y26_vmadot123_probe_available_buildtime() {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    return true;
#else
    return false;
#endif
}

extern "C" int y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(int variant,
                                                          const std::int8_t* a_8x8_row_major,
                                                          const std::int8_t* b_4x8_transposed_nk,
                                                          std::int32_t* c_4x4_row_major,
                                                          bool accumulate) {
    if (!pointers_valid(a_8x8_row_major, b_4x8_transposed_nk, c_4x4_row_major)) {
        return Y26_VMADOT_STATUS_INVALID_ARGUMENT;
    }
    if (variant < Y26_VMADOT123_VARIANT_1 || variant > Y26_VMADOT123_VARIANT_3) {
        return Y26_VMADOT_STATUS_INVALID_ARGUMENT;
    }

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    y26_vmadot123_unguarded(variant, a_8x8_row_major, b_4x8_transposed_nk, c_4x4_row_major, accumulate);
    return Y26_VMADOT_STATUS_SUCCESS;
#else
    (void)variant;
    (void)accumulate;
    return Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME;
#endif
}

extern "C" int y26_k1x_vmadot123_checked_cluster0_s8s8s32(int variant,
                                                           const std::int8_t* a_8x8_row_major,
                                                           const std::int8_t* b_4x8_transposed_nk,
                                                           std::int32_t* c_4x4_row_major,
                                                           bool accumulate) {
    if (!pointers_valid(a_8x8_row_major, b_4x8_transposed_nk, c_4x4_row_major)) {
        return Y26_VMADOT_STATUS_INVALID_ARGUMENT;
    }
    if (!y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        return Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED;
    }
    return y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(
        variant, a_8x8_row_major, b_4x8_transposed_nk, c_4x4_row_major, accumulate);
}
