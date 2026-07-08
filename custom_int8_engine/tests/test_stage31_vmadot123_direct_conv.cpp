#include "y26_k1x_vmadot123_direct_conv.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    const Y26Conv2DParams params{4, 4, 8, 4, 1, 1, 1, 1};
    Y26Vmadot123DirectConvWorkspace* workspace =
        y26_vmadot123_direct_conv3x3_workspace_create(&params);
    if (workspace == nullptr) {
        std::fprintf(stderr, "workspace create failed\n");
        return 1;
    }
    if (y26_vmadot123_direct_conv3x3_workspace_bytes(workspace) == 0) {
        std::fprintf(stderr, "workspace bytes is zero\n");
        y26_vmadot123_direct_conv3x3_workspace_destroy(workspace);
        return 1;
    }

    std::vector<std::int8_t> input(static_cast<std::size_t>(params.input_h * params.input_w * params.input_c), 1);
    std::vector<std::int32_t> output(static_cast<std::size_t>(params.input_h * params.input_w * params.output_c), 0);
    Y26Vmadot123DirectConvTimingUs timing {};
    const int status = y26_vmadot123_direct_conv3x3_i8s8s32_nhwc_single_thread(
        input.data(), nullptr, nullptr, output.data(), &params, -128, 0, workspace, &timing);
    y26_vmadot123_direct_conv3x3_workspace_destroy(workspace);
    return status == Y26_CONV_STATUS_INVALID_ARGUMENT ? 0 : 1;
}
