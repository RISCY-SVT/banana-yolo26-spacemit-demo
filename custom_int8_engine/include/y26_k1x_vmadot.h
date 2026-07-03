#pragma once

#include <cstdint>

extern "C" {

enum Y26VmadotStatusCode {
    Y26_VMADOT_STATUS_SUCCESS = 0,
    Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME = 1,
    Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED = 2,
    Y26_VMADOT_STATUS_SIGILL_CAUGHT = 3,
    Y26_VMADOT_STATUS_INVALID_ARGUMENT = 4,
};

void y26_vmadot_4x4x8_scalar_s8s8s32(const std::int8_t* a_4x8_row_major,
                                      const std::int8_t* b_4x8_transposed_nk,
                                      std::int32_t* c_4x4_row_major,
                                      bool accumulate);

bool y26_vmadot_4x4x8_ime_available_buildtime();

int y26_vmadot_4x4x8_ime_s8s8s32(const std::int8_t* a_4x8_row_major,
                                  const std::int8_t* b_4x8_transposed_nk,
                                  std::int32_t* c_4x4_row_major,
                                  bool accumulate);

}
