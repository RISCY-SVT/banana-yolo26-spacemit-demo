#include "stage7_backbone_subset_fixture.h"
#include "y26_k1x_activation.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::size_t mismatches_i8(const std::vector<std::int8_t>& actual, const std::int8_t* expected) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            ++mismatches;
        }
    }
    return mismatches;
}

int verify_pack_handoff(const y26_stage7_backbone_subset_fixture::BackboneSubsetFixture& fixture) {
    const int input_h = fixture.conv2_params.input_h;
    const int input_w = fixture.conv2_params.input_w;
    const int input_c = fixture.conv2_params.input_c;
    const int output_m = input_h * input_w;
    const int k_padded = ((input_c + 7) / 8) * 8;
    const std::size_t packed_bytes =
        static_cast<std::size_t>((output_m + 3) / 4) * static_cast<std::size_t>(k_padded / 8) * 32U;
    std::vector<std::int8_t> packed(packed_bytes, 0);
    std::vector<std::int8_t> unpacked(fixture.expected_act1_count, 0);
    const int pack_status = y26_activation_packa_1x1_mmt4d_4x8_from_nhwc(
        fixture.expected_act1_s8_nhwc, input_h, input_w, input_c, packed.data(), packed.size());
    const int unpack_status = y26_activation_unpacka_1x1_mmt4d_4x8_to_nhwc(
        packed.data(), input_h, input_w, input_c, unpacked.data());
    const std::size_t mismatches = mismatches_i8(unpacked, fixture.expected_act1_s8_nhwc);
    std::cout << "stage9_pack_handoff conv2_1x1 pack_status=" << pack_status
              << " unpack_status=" << unpack_status << " packed_bytes=" << packed.size()
              << " mismatches=" << mismatches << "\n";
    return pack_status == Y26_CONV_STATUS_SUCCESS && unpack_status == Y26_CONV_STATUS_SUCCESS && mismatches == 0
               ? 0
               : 1;
}

}  // namespace

int main() {
    return verify_pack_handoff(y26_stage7_backbone_subset_fixture::kSyntheticSeededFixture);
}
