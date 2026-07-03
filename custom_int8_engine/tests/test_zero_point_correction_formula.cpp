#include <array>
#include <cstdint>

int main() {
    const std::array<std::int32_t, 8> a {0, 1, 2, 127, 128, 129, 200, 255};
    const std::array<std::int32_t, 8> w {3, 4, 5, 126, 127, 128, 201, 250};
    constexpr std::int32_t za = 128;
    constexpr std::int32_t zw = 126;

    std::int32_t direct = 0;
    std::int32_t raw = 0;
    std::int32_t sum_a = 0;
    std::int32_t sum_w = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        direct += (a[i] - za) * (w[i] - zw);
        raw += a[i] * w[i];
        sum_a += a[i];
        sum_w += w[i];
    }

    const auto k = static_cast<std::int32_t>(a.size());
    const std::int32_t corrected = raw - zw * sum_a - za * sum_w + k * za * zw;
    return direct == corrected ? 0 : 1;
}
