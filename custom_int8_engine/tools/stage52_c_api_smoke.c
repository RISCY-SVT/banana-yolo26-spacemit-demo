#include "y26_k1x_executor.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(y26_executor* executor, const char* message) {
    fprintf(stderr, "%s: %s\n", message,
            executor == NULL ? "executor is null" : y26_executor_last_error(executor));
    return 1;
}

int main(int argc, char** argv) {
    y26_build_info build_info;
    y26_build_info_init(&build_info);
    if (y26_executor_get_build_info(&build_info) != Y26_STATUS_OK) {
        fprintf(stderr, "build-info query failed\n");
        return 1;
    }
    const uint32_t required_capabilities =
        Y26_CAPABILITY_IME | Y26_CAPABILITY_RVV |
        Y26_CAPABILITY_FROZEN_PROFILE | Y26_CAPABILITY_RGB_INPUT;
    if ((build_info.capability_flags & required_capabilities) != required_capabilities) {
        fprintf(stderr, "release capabilities missing: 0x%08" PRIx32 "\n",
                build_info.capability_flags);
        return 1;
    }
    if (argc == 2 && strcmp(argv[1], "--build-info") == 0) {
        printf("release=%s abi=%" PRIu32 " source=%s contract=%s profile=%s "
               "capabilities=0x%08" PRIx32 " manifest=%s\n",
               build_info.release_version, build_info.abi_version,
               build_info.source_commit, build_info.integer_contract_id,
               build_info.full_graph_profile_id, build_info.capability_flags,
               build_info.expected_package_manifest_sha256);
        return 0;
    }
    if (argc != 5) {
        fprintf(stderr, "usage: %s PACKAGE MANIFEST_SHA256 INPUT_F32 EXPECTED_HASH_HEX\n",
                argv[0]);
        return 2;
    }
    errno = 0;
    char* hash_end = NULL;
    const uint64_t expected_hash = strtoull(argv[4], &hash_end, 16);
    if (errno != 0 || hash_end == argv[4] || *hash_end != '\0') {
        fprintf(stderr, "invalid expected hash: %s\n", argv[4]);
        return 2;
    }

    float* input = calloc(Y26_K1X_EXECUTOR_INPUT_ELEMENTS, sizeof(float));
    float* output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    y26_executor* executor = y26_executor_create();
    if (input == NULL || output == NULL || executor == NULL) {
        free(output);
        free(input);
        y26_executor_destroy(executor);
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    FILE* stream = fopen(argv[3], "rb");
    if (stream == NULL ||
        fread(input, sizeof(float), Y26_K1X_EXECUTOR_INPUT_ELEMENTS, stream) !=
            Y26_K1X_EXECUTOR_INPUT_ELEMENTS ||
        fgetc(stream) != EOF) {
        if (stream != NULL) fclose(stream);
        free(output);
        free(input);
        y26_executor_destroy(executor);
        fprintf(stderr, "input read or size check failed: %s\n", argv[3]);
        return 1;
    }
    fclose(stream);

    y26_executor_options options = {
        .struct_size = sizeof(y26_executor_options),
        .abi_version = Y26_K1X_EXECUTOR_ABI_VERSION,
        .workers = 4,
        .worker_cpu_begin = 0,
        .controller_cpu = 4,
        .scheduler = Y26_SCHEDULER_SAFE,
        .flags = Y26_EXECUTOR_FLAG_NONE,
    };
    y26_status status = y26_executor_prepare(executor, argv[1], argv[2], &options);
    if (status != Y26_STATUS_OK) {
        const int result = fail(executor, "prepare failed");
        free(output);
        free(input);
        y26_executor_destroy(executor);
        return result;
    }

    y26_run_timing timing = {0};
    status = y26_executor_run_preprocessed(
        executor, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
        output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, &timing);
    if (status != Y26_STATUS_OK) {
        const int result = fail(executor, "run failed");
        free(output);
        free(input);
        y26_executor_destroy(executor);
        return result;
    }
    if (timing.output_hash != expected_hash || timing.affinity_ok != 1 ||
        timing.cpu4_7_ime_count != 0) {
        fprintf(stderr,
                "smoke mismatch: expected=%016" PRIx64 " actual=%016" PRIx64
                " affinity=%d cpu4_7_ime=%d\n",
                expected_hash, timing.output_hash, timing.affinity_ok,
                timing.cpu4_7_ime_count);
        free(output);
        free(input);
        y26_executor_destroy(executor);
        return 1;
    }

    printf("version=%s output_hash=%016" PRIx64 " total_us=%.3f\n",
           y26_executor_version(), timing.output_hash, timing.total_us);
    free(output);
    free(input);
    y26_executor_destroy(executor);
    return 0;
}
