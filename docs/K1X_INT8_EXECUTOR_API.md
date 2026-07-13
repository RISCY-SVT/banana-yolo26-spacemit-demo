# K1X INT8 Executor API

The public C ABI is declared in `y26_k1x_executor.h`. C++ exceptions are caught
inside the implementation and never cross this boundary.

```c
y26_executor* y26_executor_create(void);
y26_status y26_executor_prepare(y26_executor*, const char* package_dir,
                                const char* trusted_manifest_sha256,
                                const y26_executor_options*);
y26_status y26_executor_run_preprocessed(y26_executor*, const float*, size_t,
                                         float*, size_t, y26_run_timing*);
y26_status y26_executor_run_rgb(y26_executor*, const uint8_t*, int, int, int,
                                float*, size_t, y26_run_timing*);
y26_status y26_executor_get_output(const y26_executor*, float*, size_t);
int y26_executor_tensor_id(const y26_executor*, const char*);
size_t y26_executor_tensor_bytes(const y26_executor*, int);
y26_status y26_executor_copy_boundary(const y26_executor*, int, uint8_t*, size_t);
void y26_executor_destroy(y26_executor*);
const char* y26_executor_last_error(const y26_executor*);
const char* y26_executor_version(void);
```

One handle owns one arena, prepared package, worker pool, and output buffer.
Calls on one handle must be serialized. Use one independently prepared handle
per concurrent stream. Caller-owned input and output buffers remain owned by
the caller and must remain valid for the duration of a call. Input and output
ranges must not overlap. No alignment beyond the natural alignment of
`float`/`uint8_t` is required by the public ABI; the executor copies into its
own aligned resident arena.

`run_preprocessed` accepts exactly `1x3x640x640` contiguous float32 RGB values
in `[0,1]` (`Y26_K1X_EXECUTOR_INPUT_ELEMENTS` elements). `run_rgb` accepts an
already-letterboxed contiguous 640x640 RGB uint8 image; `row_stride_bytes`
must cover at least 1,920 bytes. Both require an output capacity of
`Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS` floats and produce 300 rows of six float32
values:

```text
x1, y1, x2, y2, confidence, zero-based-class
```

Diagnostic tensor lookup/copy APIs are available only when prepare used
`Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES`. Boundary capture disables the optimized
resident core and is excluded from measured execution.

`y26_executor_options` must set `struct_size` and ABI version 1. The handoff
default is four workers starting at CPU0, controller CPU4, `safe`/SCHED_OTHER,
and no diagnostic flags. `rr20` is an optional privileged lab mode. The package
profile and trusted `asset_hashes.tsv` SHA-256 must match the prepared
`K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001` package; arbitrary profiles are
rejected.

The ABI returns `Y26_STATUS_INVALID_ARGUMENT`, `Y26_STATUS_PACKAGE_ERROR`,
`Y26_STATUS_RUNTIME_ERROR`, or `Y26_STATUS_UNSUPPORTED` without throwing C++
exceptions across the boundary. The string returned by `last_error` remains
owned by the handle and is replaced by a later call. `y26_executor_version()`
returns the profile and ABI identity without requiring a handle.

The installed C++ header `y26_k1x_full_executor.h` exposes the equivalent
move-only `y26::stage52::FullExecutor` API. Its source-level ABI is intended for
matching builds; the C ABI is the stable colleague-facing integration surface.

## Minimal C Lifecycle

```c
#include <stdio.h>
#include <stdlib.h>

#include "y26_k1x_executor.h"

int main(int argc, char** argv) {
    if (argc != 3) return 2;  /* PACKAGE_DIR TRUSTED_MANIFEST_SHA256 */

    y26_executor_options options = {
        .struct_size = sizeof(y26_executor_options),
        .abi_version = Y26_K1X_EXECUTOR_ABI_VERSION,
        .workers = 4,
        .worker_cpu_begin = 0,
        .controller_cpu = 4,
        .scheduler = Y26_SCHEDULER_SAFE,
        .flags = Y26_EXECUTOR_FLAG_NONE,
    };
    y26_executor* executor = y26_executor_create();
    float* input = calloc(Y26_K1X_EXECUTOR_INPUT_ELEMENTS, sizeof(float));
    float* output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    if (executor == NULL || input == NULL || output == NULL) {
        free(output);
        free(input);
        y26_executor_destroy(executor);
        return 3;
    }

    y26_status status = y26_executor_prepare(executor, argv[1], argv[2], &options);
    if (status == Y26_STATUS_OK) {
        y26_run_timing timing = {0};
        status = y26_executor_run_preprocessed(
            executor, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
            output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, &timing);
        if (status == Y26_STATUS_OK)
            printf("output_hash=%016llx total_us=%.3f\n",
                   (unsigned long long)timing.output_hash, timing.total_us);
    }
    if (status != Y26_STATUS_OK)
        fprintf(stderr, "executor error: %s\n", y26_executor_last_error(executor));

    free(output);
    free(input);
    y26_executor_destroy(executor);
    return status == Y26_STATUS_OK ? 0 : 1;
}
```

Link against `-ly26_k1x_int8_executor -lpthread`. The shared library and the
calling program must use the same ABI header version.

The release binary `y26_k1x_c_api_smoke` is compiled from
`custom_int8_engine/tools/stage52_c_api_smoke.c` and runs this lifecycle against
the frozen package and fixture during handoff validation.
