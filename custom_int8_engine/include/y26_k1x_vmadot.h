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

enum Y26ImeCapability {
    Y26_IME_CAPABILITY_UNKNOWN = 0,
    Y26_IME_CAPABILITY_UNAVAILABLE = 1,
    Y26_IME_CAPABILITY_AVAILABLE_CLUSTER0_ONLY = 2,
};

struct Y26ImeRuntimeStateSnapshot {
    int initialized;
    int capability;
    int probe_cpu;
    int probe_status;
};

void y26_vmadot_4x4x8_scalar_s8s8s32(const std::int8_t* a_4x8_row_major,
                                      const std::int8_t* b_4x8_transposed_nk,
                                      std::int32_t* c_4x4_row_major,
                                      bool accumulate);

bool y26_vmadot_4x4x8_ime_available_buildtime();

int y26_k1x_ime_probe_once();
bool y26_k1x_ime_available();
bool y26_k1x_ime_hotpath_allowed_on_current_cpu();
Y26ImeRuntimeStateSnapshot y26_k1x_ime_runtime_state_snapshot();
void y26_k1x_ime_reset_thread_hotpath_for_tests();

int y26_vmadot_4x4x8_ime_s8s8s32(const std::int8_t* a_4x8_row_major,
                                  const std::int8_t* b_4x8_transposed_nk,
                                  std::int32_t* c_4x4_row_major,
                                  bool accumulate);

int y26_k1x_vmadot_4x4x8_checked_cluster0_s8s8s32(const std::int8_t* a_4x8_row_major,
                                                   const std::int8_t* b_4x8_transposed_nk,
                                                   std::int32_t* c_4x4_row_major,
                                                   bool accumulate);

int y26_k1x_vmadot_4x4x8_unsafe_cluster0_s8s8s32(const std::int8_t* a_4x8_row_major,
                                                  const std::int8_t* b_4x8_transposed_nk,
                                                  std::int32_t* c_4x4_row_major,
                                                  bool accumulate);

}
