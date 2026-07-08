#pragma once

#include <cstdint>

extern "C" {

enum Y26Vmadot123Variant {
    Y26_VMADOT123_VARIANT_1 = 1,
    Y26_VMADOT123_VARIANT_2 = 2,
    Y26_VMADOT123_VARIANT_3 = 3,
};

bool y26_vmadot123_probe_available_buildtime();

int y26_k1x_vmadot123_unsafe_cluster0_s8s8s32(int variant,
                                               const std::int8_t* a_8x8_row_major,
                                               const std::int8_t* b_4x8_transposed_nk,
                                               std::int32_t* c_4x4_row_major,
                                               bool accumulate);

int y26_k1x_vmadot123_checked_cluster0_s8s8s32(int variant,
                                                const std::int8_t* a_8x8_row_major,
                                                const std::int8_t* b_4x8_transposed_nk,
                                                std::int32_t* c_4x4_row_major,
                                                bool accumulate);

}
