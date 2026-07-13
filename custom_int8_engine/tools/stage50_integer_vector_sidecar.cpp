#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

__extension__ using Signed128 = __int128;
__extension__ using Unsigned128 = unsigned __int128;
constexpr std::size_t kLanes = 4;
constexpr std::size_t kValues = 128;

std::int64_t round_signed_product_even(Signed128 product, unsigned shift) {
    const bool negative = product < 0;
    const Unsigned128 bits = static_cast<Unsigned128>(product);
    const Unsigned128 magnitude = negative ? (~bits) + 1U : bits;
    Unsigned128 quotient = magnitude;
    if (shift != 0) {
        quotient >>= shift;
        const Unsigned128 remainder = magnitude & ((static_cast<Unsigned128>(1) << shift) - 1U);
        const Unsigned128 half = static_cast<Unsigned128>(1) << (shift - 1U);
        if (remainder > half || (remainder == half && (quotient & 1U) != 0)) ++quotient;
    }
    const std::int64_t rounded = static_cast<std::int64_t>(quotient);
    return negative ? -rounded : rounded;
}

std::int64_t round_product_even(std::int64_t value, std::int64_t multiplier, unsigned shift) {
    return round_signed_product_even(static_cast<Signed128>(value) * multiplier, shift);
}

void q31_products(const std::int32_t* values, const std::int32_t* multipliers,
                  std::int64_t* products) noexcept {
#if defined(__riscv)
    asm volatile(
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v0, (%[values])\n"
        "vle32.v v1, (%[multipliers])\n"
        "vwmul.vv v2, v0, v1\n"
        "vse64.v v2, (%[products])\n"
        :
        : [values] "r"(values), [multipliers] "r"(multipliers), [products] "r"(products)
        : "v0", "v1", "v2", "v3", "memory");
#else
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        products[lane] = static_cast<std::int64_t>(values[lane]) * multipliers[lane];
    }
#endif
}

struct LimbProducts {
    std::array<std::uint64_t, kLanes> p00 {};
    std::array<std::uint64_t, kLanes> p01 {};
    std::array<std::uint64_t, kLanes> p10 {};
    std::array<std::uint64_t, kLanes> p11 {};
};

void q62_limb_products(const std::uint32_t* a_lo, const std::uint32_t* a_hi,
                       const std::uint32_t* b_lo, const std::uint32_t* b_hi,
                       LimbProducts* output) noexcept {
#if defined(__riscv)
    asm volatile(
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v0, (%[a_lo])\n"
        "vle32.v v1, (%[a_hi])\n"
        "vle32.v v10, (%[b_lo])\n"
        "vle32.v v11, (%[b_hi])\n"
        "vwmulu.vv v2, v0, v10\n"
        "vwmulu.vv v4, v0, v11\n"
        "vwmulu.vv v6, v1, v10\n"
        "vwmulu.vv v8, v1, v11\n"
        "vse64.v v2, (%[p00])\n"
        "vse64.v v4, (%[p01])\n"
        "vse64.v v6, (%[p10])\n"
        "vse64.v v8, (%[p11])\n"
        :
        : [a_lo] "r"(a_lo), [a_hi] "r"(a_hi), [b_lo] "r"(b_lo), [b_hi] "r"(b_hi),
          [p00] "r"(output->p00.data()), [p01] "r"(output->p01.data()),
          [p10] "r"(output->p10.data()), [p11] "r"(output->p11.data())
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "memory");
#else
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        output->p00[lane] = static_cast<std::uint64_t>(a_lo[lane]) * b_lo[lane];
        output->p01[lane] = static_cast<std::uint64_t>(a_lo[lane]) * b_hi[lane];
        output->p10[lane] = static_cast<std::uint64_t>(a_hi[lane]) * b_lo[lane];
        output->p11[lane] = static_cast<std::uint64_t>(a_hi[lane]) * b_hi[lane];
    }
#endif
}

std::array<std::uint64_t, 2> reconstruct(const LimbProducts& products, std::size_t lane) {
    std::uint64_t low = products.p00[lane];
    std::uint64_t high = products.p11[lane] + (products.p01[lane] >> 32U) + (products.p10[lane] >> 32U);
    const std::uint64_t first = products.p01[lane] << 32U;
    const std::uint64_t before_first = low;
    low += first;
    high += low < before_first;
    const std::uint64_t second = products.p10[lane] << 32U;
    const std::uint64_t before_second = low;
    low += second;
    high += low < before_second;
    return {low, high};
}

}  // namespace

int main() {
    std::array<std::int32_t, kValues> values {};
    std::array<std::int32_t, kValues> q31_multipliers {};
    std::array<std::int64_t, kValues> q31_products_out {};
    for (std::size_t index = 0; index < kValues; ++index) {
        const std::int64_t magnitude = static_cast<std::int64_t>((index * 104729U) & 0x3fffffffU);
        values[index] = static_cast<std::int32_t>((index & 1U) == 0 ? magnitude : -magnitude);
        q31_multipliers[index] = static_cast<std::int32_t>((1U << 29U) + index * 7919U);
    }
    for (int delta = -2; delta <= 2; ++delta) {
        const std::size_t positive = static_cast<std::size_t>(delta + 2);
        const std::size_t negative = positive + 5U;
        values[positive] = (1 << 30) + delta;
        values[negative] = -((1 << 30) + delta);
        q31_multipliers[positive] = 1;
        q31_multipliers[negative] = 1;
    }
    bool q31_exact = true;
    for (std::size_t index = 0; index < kValues; index += kLanes) {
        q31_products(values.data() + index, q31_multipliers.data() + index, q31_products_out.data() + index);
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const std::size_t item = index + lane;
            const std::int64_t expected = static_cast<std::int64_t>(values[item]) * q31_multipliers[item];
            q31_exact = q31_exact && q31_products_out[item] == expected;
            q31_exact = q31_exact &&
                round_signed_product_even(static_cast<Signed128>(q31_products_out[item]), 31) ==
                round_product_even(values[item], q31_multipliers[item], 31);
        }
    }

    std::array<std::uint64_t, kValues> q62_values {};
    std::array<std::uint64_t, kValues> q62_multipliers {};
    bool q62_product_exact = true;
    for (std::size_t index = 0; index < kValues; ++index) {
        q62_values[index] = static_cast<std::uint64_t>((index * 130363U) & 0x7fffffffU);
        q62_multipliers[index] = (std::uint64_t{1} << 61U) + index * 104729U;
    }
    for (std::size_t index = 0; index < kValues; index += kLanes) {
        std::array<std::uint32_t, kLanes> a_lo {}, a_hi {}, b_lo {}, b_hi {};
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const std::uint64_t a = q62_values[index + lane];
            const std::uint64_t b = q62_multipliers[index + lane];
            a_lo[lane] = static_cast<std::uint32_t>(a);
            a_hi[lane] = static_cast<std::uint32_t>(a >> 32U);
            b_lo[lane] = static_cast<std::uint32_t>(b);
            b_hi[lane] = static_cast<std::uint32_t>(b >> 32U);
        }
        LimbProducts products;
        q62_limb_products(a_lo.data(), a_hi.data(), b_lo.data(), b_hi.data(), &products);
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const auto reconstructed = reconstruct(products, lane);
            const Unsigned128 expected = static_cast<Unsigned128>(q62_values[index + lane]) *
                q62_multipliers[index + lane];
            q62_product_exact = q62_product_exact && reconstructed[0] == static_cast<std::uint64_t>(expected) &&
                reconstructed[1] == static_cast<std::uint64_t>(expected >> 64U);
        }
    }

    constexpr int iterations = 20000;
    volatile std::int64_t checksum = 0;
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t index = 0; index < kValues; index += kLanes) {
            q31_products(values.data() + index, q31_multipliers.data() + index, q31_products_out.data() + index);
        }
        checksum = checksum + q31_products_out[static_cast<std::size_t>(iteration) & (kValues - 1U)];
    }
    const double elapsed_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - begin).count();
    std::cout << "rvv_execution=" <<
#if defined(__riscv)
        1
#else
        0
#endif
        << "\nq31_rvv_widen_product_exact=" << (q31_exact ? 1 : 0)
        << "\nq31_tie_neighborhood_cases=10"
        << "\nq31_full_explicit_vector_round_lut=0"
        << "\nq62_rvv_limb_product_exact=" << (q62_product_exact ? 1 : 0)
        << "\nq62_full_explicit_vector_round_lut=0"
        << "\nq31_product_probe_us=" << elapsed_us
        << "\nchecksum=" << checksum << '\n';
    return q31_exact && q62_product_exact ? 0 : 3;
}
