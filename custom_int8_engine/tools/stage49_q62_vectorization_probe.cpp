#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

__extension__ using int128_t = __int128;
__extension__ using uint128_t = unsigned __int128;

std::int64_t round_q62(std::int64_t value, std::int64_t multiplier, unsigned shift) {
    const int128_t product = static_cast<int128_t>(value) * static_cast<int128_t>(multiplier);
    const bool negative = product < 0;
    const uint128_t magnitude = negative ? static_cast<uint128_t>(-product) : static_cast<uint128_t>(product);
    const uint128_t quotient = magnitude >> shift;
    const uint128_t remainder = magnitude - (quotient << shift);
    const uint128_t half = static_cast<uint128_t>(1) << (shift - 1U);
    const bool increment = remainder > half || (remainder == half && (quotient & 1U) != 0U);
    const std::int64_t rounded = static_cast<std::int64_t>(quotient + static_cast<unsigned>(increment));
    return negative ? -rounded : rounded;
}

void exact_q62_loop(const std::int64_t* values,
                    const std::int64_t* multipliers,
                    const unsigned* shifts,
                    std::int64_t* output,
                    std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = round_q62(values[i], multipliers[i], shifts[i]);
    }
}

}  // namespace

int main() {
    constexpr std::size_t count = 128;
    std::int64_t values[count]{};
    std::int64_t multipliers[count]{};
    unsigned shifts[count]{};
    std::int64_t output[count]{};
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<std::int64_t>(i) - 64;
        multipliers[i] = (std::int64_t{1} << 61) + static_cast<std::int64_t>(i * 17);
        shifts[i] = 62;
    }
    exact_q62_loop(values, multipliers, shifts, output, count);
    std::int64_t checksum = 0;
    for (const std::int64_t value : output) {
        checksum += value;
    }
    std::cout << "q62_probe_checksum=" << checksum << '\n';
    return checksum == -32 ? 0 : 1;
}
