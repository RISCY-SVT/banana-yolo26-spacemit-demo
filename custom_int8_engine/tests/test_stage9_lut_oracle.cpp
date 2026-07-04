#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_activation.h"

#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

std::int8_t reference_lut_value(float conv_scale, int conv_zp, float act_scale, int act_zp, int q) {
    const float x = static_cast<float>(q - conv_zp) * conv_scale;
    const float y = x / (1.0f + std::exp(-x));
    const std::uint8_t qy = y26_quantize_u8_nearest_even_f32(y, act_scale, act_zp);
    return static_cast<std::int8_t>(static_cast<int>(qy) - 128);
}

int verify_lut(const char* label, float conv_scale, int conv_zp, float act_scale, int act_zp) {
    std::int8_t lut[256] {};
    const int status = y26_build_silu_u8_to_s8_lut(conv_scale, conv_zp, act_scale, act_zp, lut);
    int mismatches = 0;
    for (int q = 0; q < 256; ++q) {
        if (lut[q] != reference_lut_value(conv_scale, conv_zp, act_scale, act_zp, q)) {
            ++mismatches;
        }
    }
    std::cout << "stage9_lut_oracle label=" << label << " status=" << status
              << " mismatches=" << mismatches << "\n";
    return status == Y26_CONV_STATUS_SUCCESS && mismatches == 0 ? 0 : 1;
}

}  // namespace

int main() {
    const auto& fixture = y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture;
    int failures = 0;
    failures += verify_lut("act0",
                           fixture.conv0_output_scale,
                           fixture.conv0_output_zero_point_u8,
                           fixture.act0_output_scale,
                           fixture.act0_output_zero_point_u8);
    failures += verify_lut("act1",
                           fixture.conv1_output_scale,
                           fixture.conv1_output_zero_point_u8,
                           fixture.act1_output_scale,
                           fixture.act1_output_zero_point_u8);
    return failures == 0 ? 0 : 1;
}
