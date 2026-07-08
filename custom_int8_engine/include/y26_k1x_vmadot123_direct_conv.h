#pragma once

#include "y26_k1x_conv_kernels.h"

#include <cstddef>
#include <cstdint>

extern "C" {

struct Y26Vmadot123DirectConvWorkspace;

struct Y26Vmadot123DirectConvTimingUs {
    double panel_build_us;
    double kernel_compute_us;
    double correction_us;
    double writeback_us;
    double total_us;
    int output_m_step;
    int used_vmadot123;
};

Y26Vmadot123DirectConvWorkspace* y26_vmadot123_direct_conv3x3_workspace_create(
    const Y26Conv2DParams* params);

void y26_vmadot123_direct_conv3x3_workspace_destroy(Y26Vmadot123DirectConvWorkspace* workspace);

std::size_t y26_vmadot123_direct_conv3x3_workspace_bytes(
    const Y26Vmadot123DirectConvWorkspace* workspace);

int y26_vmadot123_direct_conv3x3_i8s8s32_nhwc_single_thread(
    const std::int8_t* input_nhwc_s8,
    const Y26PrepackedConvWeights* weights,
    const std::int32_t* bias_oc,
    std::int32_t* corrected_output_nhwc,
    const Y26Conv2DParams* params,
    int input_storage_zero_point_s8,
    int activation_zero_point_u8,
    Y26Vmadot123DirectConvWorkspace* workspace,
    Y26Vmadot123DirectConvTimingUs* timing);

}
