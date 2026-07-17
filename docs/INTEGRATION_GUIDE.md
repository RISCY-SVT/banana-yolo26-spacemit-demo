# Integration Guide

## ABI

The installed interface is C ABI version 1 in `include/y26_k1x_executor.h`.
Release 0.9.2 has shared-library SONAME `liby26_k1x_int8_executor.so.1`.
Only C ABI symbols are exported; C++ and research headers are not installed by
the default release build.

Existing ABI1 option structures remain accepted at their documented legacy
size. New callers must call `y26_executor_options_init()` before changing fields.
Unknown future tail fields are ignored according to `struct_size`; existing
fields may not be reordered or removed within ABI1.

## Lifecycle

1. Allocate one `y26_executor` per serialized input stream.
2. Initialize `y26_executor_options`.
3. Set the public wake policy if required.
4. Call `y26_executor_prepare()` once with the expected package manifest SHA.
5. Reuse `run_preprocessed()` or `run_rgb()` for every frame.
6. Read timing/output before the next run if needed.
7. Destroy the handle after all calls finish.

Preparing twice or running before prepare returns `Y26_STATUS_INVALID_STATE`.
Concurrent calls on one handle return `Y26_STATUS_BUSY`. Independent handles are
supported, although they compete for the same CPU0-4 resources and are not a
throughput recommendation.

## Complete C Example

```c
#include <y26_k1x_executor.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    float *input = calloc(Y26_K1X_EXECUTOR_INPUT_ELEMENTS, sizeof(float));
    float *output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    FILE *file = fopen(argv[2], "rb");
    if (!input || !output || !file ||
        fread(input, sizeof(float), Y26_K1X_EXECUTOR_INPUT_ELEMENTS, file) !=
            Y26_K1X_EXECUTOR_INPUT_ELEMENTS) return 1;
    fclose(file);

    y26_executor_options options;
    y26_executor_options_init(&options);
    options.wake_policy = Y26_WAKE_CONDITION_VARIABLE;
    y26_executor *executor = y26_executor_create();
    y26_status status = y26_executor_prepare(
        executor, argv[1],
        "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be",
        &options);
    y26_run_timing timing = {0};
    if (status == Y26_STATUS_OK)
        status = y26_executor_run_preprocessed(
            executor, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
            output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, &timing);
    if (status != Y26_STATUS_OK)
        fprintf(stderr, "%s: %s\n", y26_status_string(status),
                y26_executor_last_error(executor));
    else
        printf("hash=%016llx total_us=%.3f\n",
               (unsigned long long)timing.output_hash, timing.total_us);
    y26_executor_destroy(executor);
    free(output);
    free(input);
    return status == Y26_STATUS_OK ? 0 : 1;
}
```

## Build Through pkg-config

```bash
export PKG_CONFIG_PATH="$RELEASE/lib/pkgconfig"
riscv64-unknown-linux-gnu-gcc consumer.c \
  $(pkg-config --cflags --libs y26-k1x-int8-executor) -o consumer
```

For a static executor library, use the installed `.a` and include private
dependencies (`-lstdc++ -lpthread -lm`), or use the CMake static target.

## Build Through CMake

```cmake
find_package(y26K1xExecutor 0.9 CONFIG REQUIRED)
add_executable(consumer consumer.c)
target_link_libraries(consumer PRIVATE y26::executor_shared)
# Use y26::executor_static for the installed static executor.
```

Configure with `-DCMAKE_PREFIX_PATH="$RELEASE"`.

## Inputs

`y26_executor_run_preprocessed` accepts exactly 1,228,800 float values in NCHW
order. Values are semantic RGB in `[0,1]`; the caller owns decode, RGB ordering,
resize, and letterbox metadata.

`y26_executor_run_rgb` accepts exactly 640x640 interleaved RGB8 with a row stride
of at least 1920 bytes. It does not decode JPEG or perform letterbox resize.

Input and output buffers must not overlap. The executor validates null pointers,
element counts, dimensions, stride, and overlap.

## Outputs

The output is 1,800 float values interpreted as 300 rows:

```text
x1, y1, x2, y2, confidence, class_index
```

Coordinates are in the 640x640 letterboxed input coordinate system. Confidence
comes from the frozen Q24 score path. Class is an integral COCO class index stored
as float. Equal scores use the frozen deterministic point/class ordering.

## Package Identity

Always supply the expected manifest SHA. The loader verifies the manifest itself,
every listed asset size/hash, absence of unlisted files and symlinks, schema,
contract, graph profile, layout, and frozen model SHA. This prevents accidental
package mixing; it is not a signature or hostile security mechanism.

## Threading and CPU Ownership

The executor creates four IME workers pinned to CPU0-3 and uses CPU4 as controller.
CPU4-7 must never execute IME. Use one handle per serialized stream. Calls on the
same handle are intentionally rejected while another call is active.
