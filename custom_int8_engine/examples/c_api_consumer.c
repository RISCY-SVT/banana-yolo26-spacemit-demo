#include <y26_k1x_executor.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int read_input(const char* path, float* input) {
    FILE* stream = fopen(path, "rb");
    if (stream == NULL) return 0;
    const size_t count = fread(
        input, sizeof(float), Y26_K1X_EXECUTOR_INPUT_ELEMENTS, stream);
    const int complete = count == Y26_K1X_EXECUTOR_INPUT_ELEMENTS && fgetc(stream) == EOF;
    fclose(stream);
    return complete;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s PACKAGE MANIFEST_SHA256 INPUT_F32 EXPECTED_HASH\n", argv[0]);
        return 2;
    }
    float* input = calloc(Y26_K1X_EXECUTOR_INPUT_ELEMENTS, sizeof(float));
    float* output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    y26_executor* executor = y26_executor_create();
    if (input == NULL || output == NULL || executor == NULL || !read_input(argv[3], input)) {
        fprintf(stderr, "allocation or fixture read failed\n");
        y26_executor_destroy(executor);
        free(output);
        free(input);
        return 1;
    }

    y26_executor_options options;
    y26_executor_options_init(&options);
    options.wake_policy = Y26_WAKE_FRAME_GATED_SPIN;
    y26_status status = y26_executor_prepare(executor, argv[1], argv[2], &options);
    if (status == Y26_STATUS_OK) {
        y26_run_timing timing = {0};
        status = y26_executor_run_preprocessed(
            executor, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
            output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, &timing);
        const uint64_t expected = strtoull(argv[4], NULL, 0);
        if (status == Y26_STATUS_OK && timing.output_hash == expected) {
            printf("version=%s hash=0x%016" PRIx64 " total_us=%.3f\n",
                   y26_executor_version(), timing.output_hash, timing.total_us);
        } else if (status == Y26_STATUS_OK) {
            fprintf(stderr, "unexpected output hash\n");
            status = Y26_STATUS_RUNTIME_ERROR;
        }
    }
    if (status != Y26_STATUS_OK) {
        fprintf(stderr, "%s: %s\n", y26_status_string(status),
                y26_executor_last_error(executor));
    }
    y26_executor_destroy(executor);
    free(output);
    free(input);
    return status == Y26_STATUS_OK ? 0 : 1;
}
