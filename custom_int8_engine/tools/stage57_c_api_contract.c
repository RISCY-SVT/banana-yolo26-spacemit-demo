#define _GNU_SOURCE
#include "y26_k1x_executor.h"

#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct run_context {
    y26_executor* executor;
    const float* input;
    float* output;
    pthread_barrier_t* barrier;
    y26_status status;
    uint64_t hash;
} run_context;

static int failures = 0;

static void check(int condition, const char* name) {
    printf("%s\t%s\n", name, condition ? "pass" : "fail");
    if (!condition) ++failures;
}

static float* read_input(const char* path) {
    float* input = calloc(Y26_K1X_EXECUTOR_INPUT_ELEMENTS, sizeof(float));
    FILE* stream = input == NULL ? NULL : fopen(path, "rb");
    if (stream == NULL ||
        fread(input, sizeof(float), Y26_K1X_EXECUTOR_INPUT_ELEMENTS, stream) !=
            Y26_K1X_EXECUTOR_INPUT_ELEMENTS || fgetc(stream) != EOF) {
        if (stream != NULL) fclose(stream);
        free(input);
        return NULL;
    }
    fclose(stream);
    return input;
}

static y26_executor* prepare_handle(const char* package, const char* manifest,
                                    size_t struct_size, int wake_policy) {
    y26_executor* executor = y26_executor_create();
    y26_executor_options options;
    y26_executor_options_init(&options);
    options.struct_size = (uint32_t)struct_size;
    options.wake_policy = wake_policy;
    if (executor == NULL ||
        y26_executor_prepare(executor, package, manifest, &options) != Y26_STATUS_OK) {
        y26_executor_destroy(executor);
        return NULL;
    }
    return executor;
}

static void* run_thread(void* opaque) {
    run_context* context = (run_context*)opaque;
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(4, &cpus);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    if (context->barrier != NULL) (void)pthread_barrier_wait(context->barrier);
    y26_run_timing timing = {0};
    context->status = y26_executor_run_preprocessed(
        context->executor, context->input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
        context->output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, &timing);
    context->hash = timing.output_hash;
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s PACKAGE MANIFEST INPUT_F32 EXPECTED_HASH\n", argv[0]);
        return 2;
    }
    const uint64_t expected_hash = strtoull(argv[4], NULL, 0);
    float* input = read_input(argv[3]);
    float* output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    check(input != NULL && output != NULL, "fixture_allocation");
    if (input == NULL || output == NULL) return 1;

    y26_executor_options defaults;
    memset(&defaults, 0xff, sizeof(defaults));
    y26_executor_options_init(&defaults);
    check(defaults.struct_size == sizeof(defaults) &&
          defaults.abi_version == Y26_K1X_EXECUTOR_ABI_VERSION &&
          defaults.workers == 4 && defaults.controller_cpu == 4 &&
          defaults.scheduler == Y26_SCHEDULER_SAFE &&
          defaults.wake_policy == Y26_WAKE_CONDITION_VARIABLE,
          "options_init");
    check(strcmp(y26_status_string(Y26_STATUS_BUSY), "executor busy") == 0,
          "status_string");

    y26_executor* state = y26_executor_create();
    check(state != NULL, "create");
    check(y26_executor_run_preprocessed(
              state, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
              output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, NULL) == Y26_STATUS_INVALID_STATE,
          "run_before_prepare");
    y26_executor_options invalid = defaults;
    invalid.controller_cpu = 3;
    check(y26_executor_prepare(state, argv[1], argv[2], &invalid) ==
              Y26_STATUS_INVALID_ARGUMENT,
          "invalid_cpu_topology");
    check(y26_executor_prepare(
              state, argv[1],
              "0000000000000000000000000000000000000000000000000000000000000000",
              &defaults) == Y26_STATUS_PACKAGE_ERROR,
          "wrong_valid_manifest");
    check(y26_executor_prepare(state, argv[1], argv[2], &defaults) == Y26_STATUS_OK,
          "prepare");
    check(y26_executor_prepare(state, argv[1], argv[2], &defaults) == Y26_STATUS_INVALID_STATE,
          "prepare_twice");
    check(y26_executor_run_preprocessed(
              state, NULL, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
              output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, NULL) == Y26_STATUS_INVALID_ARGUMENT,
          "null_input");
    check(y26_executor_run_preprocessed(
              state, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS - 1,
              output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, NULL) == Y26_STATUS_INVALID_ARGUMENT,
          "undersized_input");
    check(y26_executor_run_preprocessed(
              state, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
              output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS - 1, NULL) == Y26_STATUS_INVALID_ARGUMENT,
          "undersized_output");
    check(y26_executor_run_preprocessed(
              state, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
              input, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, NULL) == Y26_STATUS_INVALID_ARGUMENT,
          "overlapping_buffers");
    y26_run_timing timing = {0};
    check(y26_executor_run_preprocessed(
              state, input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
              output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, &timing) == Y26_STATUS_OK &&
          timing.output_hash == expected_hash,
          "valid_run");
    y26_executor_destroy(state);

    y26_executor* legacy = prepare_handle(
        argv[1], argv[2], offsetof(y26_executor_options, wake_policy),
        Y26_WAKE_FRAME_GATED_SPIN);
    check(legacy != NULL, "legacy_abi1_options_size");
    y26_executor_destroy(legacy);

    y26_executor* first = prepare_handle(
        argv[1], argv[2], sizeof(y26_executor_options), Y26_WAKE_FRAME_GATED_SPIN);
    y26_executor* second = prepare_handle(
        argv[1], argv[2], sizeof(y26_executor_options), Y26_WAKE_FRAME_GATED_SPIN);
    float* output_first = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    float* output_second = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, 2);
    run_context first_context = {first, input, output_first, &barrier, Y26_STATUS_RUNTIME_ERROR, 0};
    run_context second_context = {second, input, output_second, &barrier, Y26_STATUS_RUNTIME_ERROR, 0};
    pthread_t first_thread;
    pthread_t second_thread;
    pthread_create(&first_thread, NULL, run_thread, &first_context);
    pthread_create(&second_thread, NULL, run_thread, &second_context);
    pthread_join(first_thread, NULL);
    pthread_join(second_thread, NULL);
    check(first_context.status == Y26_STATUS_OK && second_context.status == Y26_STATUS_OK &&
          first_context.hash == expected_hash && second_context.hash == expected_hash,
          "independent_handles_concurrent");
    pthread_barrier_destroy(&barrier);
    y26_executor_destroy(first);
    y26_executor_destroy(second);
    free(output_first);
    free(output_second);

    y26_executor* shared = prepare_handle(
        argv[1], argv[2], sizeof(y26_executor_options), Y26_WAKE_FRAME_GATED_SPIN);
    output_first = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    output_second = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    pthread_barrier_init(&barrier, NULL, 2);
    first_context = (run_context){shared, input, output_first, &barrier, Y26_STATUS_RUNTIME_ERROR, 0};
    second_context = (run_context){shared, input, output_second, &barrier, Y26_STATUS_RUNTIME_ERROR, 0};
    pthread_create(&first_thread, NULL, run_thread, &first_context);
    pthread_create(&second_thread, NULL, run_thread, &second_context);
    pthread_join(first_thread, NULL);
    pthread_join(second_thread, NULL);
    const int one_ok_one_busy =
        (first_context.status == Y26_STATUS_OK && second_context.status == Y26_STATUS_BUSY) ||
        (second_context.status == Y26_STATUS_OK && first_context.status == Y26_STATUS_BUSY);
    check(one_ok_one_busy, "same_handle_concurrent_busy");
    pthread_barrier_destroy(&barrier);
    y26_executor_destroy(shared);
    free(output_first);
    free(output_second);

    free(output);
    free(input);
    printf("failures\t%d\n", failures);
    return failures == 0 ? 0 : 1;
}
