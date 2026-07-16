#include "y26_k1x_executor.h"

#include <cstring>
#include <iostream>

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::cerr << "check failed: " #expression << '\n'; \
        return 1; \
    } \
} while (false)

int main() {
    static_assert(Y26_K1X_EXECUTOR_INPUT_ELEMENTS == 1228800U);
    static_assert(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS == 1800U);

    CHECK(std::strstr(y26_executor_version(), "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001") != nullptr);
    CHECK(std::strcmp(y26_executor_last_error(nullptr), "invalid executor") == 0);
    CHECK(y26_executor_get_output(nullptr, nullptr, 0) == Y26_STATUS_INVALID_ARGUMENT);
    CHECK(y26_executor_tensor_id(nullptr, "tensor") == -1);
    CHECK(y26_executor_tensor_bytes(nullptr, 0) == 0);
    CHECK(y26_executor_copy_boundary(nullptr, 0, nullptr, 0) == Y26_STATUS_INVALID_ARGUMENT);

    y26_executor* executor = y26_executor_create();
    CHECK(executor != nullptr);
    y26_executor_options options {};
    options.struct_size = sizeof(options);
    options.abi_version = Y26_K1X_EXECUTOR_ABI_VERSION;
    options.workers = 4;
    options.worker_cpu_begin = 0;
    options.controller_cpu = 4;
    options.scheduler = Y26_SCHEDULER_SAFE;
    options.flags = Y26_EXECUTOR_FLAG_NONE;

    CHECK(y26_executor_prepare(nullptr, ".", "hash", &options) == Y26_STATUS_INVALID_ARGUMENT);
    CHECK(y26_executor_prepare(executor, nullptr, "hash", &options) == Y26_STATUS_INVALID_ARGUMENT);
    CHECK(y26_executor_prepare(executor, ".", nullptr, &options) == Y26_STATUS_INVALID_ARGUMENT);
    options.struct_size = 0;
    CHECK(y26_executor_prepare(executor, ".", "hash", &options) == Y26_STATUS_INVALID_ARGUMENT);
    options.struct_size = sizeof(options);
    options.scheduler = static_cast<y26_scheduler>(99);
    CHECK(y26_executor_prepare(executor, ".", "hash", &options) == Y26_STATUS_INVALID_ARGUMENT);
    options.scheduler = Y26_SCHEDULER_SAFE;
    options.flags = 0x80000000U;
    CHECK(y26_executor_prepare(executor, ".", "hash", &options) == Y26_STATUS_INVALID_ARGUMENT);

    float output[Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS] {};
    CHECK(y26_executor_run_preprocessed(executor, nullptr, 0, output,
                                        Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, nullptr) ==
          Y26_STATUS_INVALID_STATE);
    CHECK(y26_executor_get_output(executor, output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS) ==
          Y26_STATUS_INVALID_STATE);
    y26_executor_destroy(executor);
    y26_executor_destroy(nullptr);
    return 0;
}
