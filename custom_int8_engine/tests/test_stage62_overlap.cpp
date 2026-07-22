#include "y26_k1x_package.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

void check(bool value, const char* name) {
    std::cout << name << '\t' << (value ? "pass" : "fail") << '\n';
    if (!value) ++failures;
}

}  // namespace

int main() {
    using y26::int8_v1::integer_ranges_overlap;
    constexpr std::uintptr_t kBase = 0x1000;
    check(!integer_ranges_overlap(kBase, 0, kBase, 16), "zero_left");
    check(!integer_ranges_overlap(kBase, 16, kBase + 16, 16), "adjacent");
    check(integer_ranges_overlap(kBase, 16, kBase + 8, 16), "partial_overlap");
    check(integer_ranges_overlap(kBase, 64, kBase + 16, 8), "full_containment");
    check(integer_ranges_overlap(kBase + 16, 8, kBase, 64), "reverse_containment");
    check(!integer_ranges_overlap(kBase, 16, kBase + 32, 16), "disjoint");

    constexpr auto kMax = std::numeric_limits<std::uintptr_t>::max();
    check(integer_ranges_overlap(kMax - 3, 8, kBase, 8), "left_wrap_rejected");
    check(integer_ranges_overlap(kBase, 8, kMax - 3, 8), "right_wrap_rejected");
    check(!integer_ranges_overlap(kMax - 15, 8, kMax - 7, 7), "high_adjacent");
    check(integer_ranges_overlap(kMax - 15, 9, kMax - 7, 7), "high_overlap");

    std::cout << "failures\t" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
