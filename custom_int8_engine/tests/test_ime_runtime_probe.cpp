#include "y26_k1x_vmadot.h"

#include <array>
#include <cstdio>

int main() {
    const bool buildtime = y26_vmadot_4x4x8_ime_available_buildtime();
    const int first_probe = y26_k1x_ime_probe_once();
    const int second_probe = y26_k1x_ime_probe_once();
    const auto snapshot = y26_k1x_ime_runtime_state_snapshot();

    std::printf("buildtime=%d\n", buildtime ? 1 : 0);
    std::printf("first_probe=%d\n", first_probe);
    std::printf("second_probe=%d\n", second_probe);
    std::printf("initialized=%d\n", snapshot.initialized);
    std::printf("capability=%d\n", snapshot.capability);
    std::printf("probe_cpu=%d\n", snapshot.probe_cpu);
    std::printf("probe_status=%d\n", snapshot.probe_status);

    if (snapshot.initialized != 1 || first_probe != second_probe || snapshot.probe_status != first_probe) {
        return 1;
    }

    if (!buildtime) {
        return (first_probe == Y26_VMADOT_STATUS_NOT_BUILT_WITH_IME &&
                snapshot.capability == Y26_IME_CAPABILITY_UNAVAILABLE)
                   ? 0
                   : 1;
    }

    if (first_probe != Y26_VMADOT_STATUS_SUCCESS) {
        if (snapshot.capability != Y26_IME_CAPABILITY_UNAVAILABLE) {
            return 1;
        }
        return first_probe == Y26_VMADOT_STATUS_RUNTIME_SAFETY_FAILED ? 0 : 1;
    }

    if (snapshot.capability != Y26_IME_CAPABILITY_AVAILABLE_CLUSTER0_ONLY || !y26_k1x_ime_available() ||
        !y26_k1x_ime_hotpath_allowed_on_current_cpu()) {
        return 1;
    }

    std::array<std::int8_t, 32> a {};
    std::array<std::int8_t, 32> b {};
    std::array<std::int32_t, 16> scalar {};
    std::array<std::int32_t, 16> ime {};
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<std::int8_t>(static_cast<int>(i) - 13);
        b[i] = static_cast<std::int8_t>(21 - static_cast<int>(i));
    }
    y26_vmadot_4x4x8_scalar_s8s8s32(a.data(), b.data(), scalar.data(), false);
    const int status = y26_vmadot_4x4x8_ime_s8s8s32(a.data(), b.data(), ime.data(), false);
    return (status == Y26_VMADOT_STATUS_SUCCESS && scalar == ime) ? 0 : 1;
}
