#define _GNU_SOURCE
#include "y26_k1x_executor.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct run_context {
    y26_executor* executor;
    const float* input;
    float* output;
    atomic_int entered;
    atomic_int finished;
    y26_status status;
    uint64_t hash;
} run_context;

static int failures;

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

static void* run_once(void* opaque) {
    run_context* context = opaque;
    atomic_store_explicit(&context->entered, 1, memory_order_release);
    y26_run_timing timing = {0};
    context->status = y26_executor_run_preprocessed(
        context->executor, context->input, Y26_K1X_EXECUTOR_INPUT_ELEMENTS,
        context->output, Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, &timing);
    context->hash = timing.output_hash;
    atomic_store_explicit(&context->finished, 1, memory_order_release);
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr,
                "usage: %s PACKAGE MANIFEST INPUT_F32 EXPECTED_HASH TENSOR_NAME\n",
                argv[0]);
        return 2;
    }
    const uint64_t expected_hash = strtoull(argv[4], NULL, 0);
    float* input = read_input(argv[3]);
    float* run_output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    float* copied_output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    y26_executor* executor = y26_executor_create();
    check(input != NULL && run_output != NULL && copied_output != NULL && executor != NULL,
          "allocation_and_create");
    if (input == NULL || run_output == NULL || copied_output == NULL || executor == NULL) {
        y26_executor_destroy(executor);
        free(copied_output);
        free(run_output);
        free(input);
        return 1;
    }

    y26_executor_options options;
    y26_executor_options_init(&options);
    options.flags = Y26_EXECUTOR_FLAG_CAPTURE_BOUNDARIES;
    options.wake_policy = Y26_WAKE_FRAME_GATED_SPIN;
    check(y26_executor_prepare(executor, argv[1], argv[2], &options) == Y26_STATUS_OK,
          "prepare");
    const int tensor_id = y26_executor_tensor_id(executor, argv[5]);
    const size_t tensor_bytes = y26_executor_tensor_bytes(executor, tensor_id);
    uint8_t* boundary = calloc(tensor_bytes == 0 ? 1 : tensor_bytes, 1);
    check(tensor_id >= 0 && tensor_bytes > 0 && boundary != NULL, "boundary_fixture");

    run_context context = {
        .executor = executor,
        .input = input,
        .output = run_output,
        .entered = ATOMIC_VAR_INIT(0),
        .finished = ATOMIC_VAR_INIT(0),
        .status = Y26_STATUS_RUNTIME_ERROR,
        .hash = 0,
    };
    pthread_t thread;
    check(pthread_create(&thread, NULL, run_once, &context) == 0, "run_thread_create");
    while (!atomic_load_explicit(&context.entered, memory_order_acquire)) sched_yield();
    usleep(1000);

    int tensor_id_busy = 0;
    int tensor_bytes_busy = 0;
    int output_busy = 0;
    int boundary_busy = 0;
    while (!atomic_load_explicit(&context.finished, memory_order_acquire)) {
        if (y26_executor_tensor_id(executor, argv[5]) == -1) tensor_id_busy = 1;
        if (y26_executor_tensor_bytes(executor, tensor_id) == 0) tensor_bytes_busy = 1;
        if (y26_executor_get_output(executor, copied_output,
                                    Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS) == Y26_STATUS_BUSY)
            output_busy = 1;
        if (y26_executor_copy_boundary(executor, tensor_id, boundary, tensor_bytes) ==
            Y26_STATUS_BUSY)
            boundary_busy = 1;
    }
    pthread_join(thread, NULL);

    printf("run_status\t%d\nrun_hash\t0x%016llx\n",
           (int)context.status, (unsigned long long)context.hash);
    check(context.status == Y26_STATUS_OK && context.hash == expected_hash,
          "run_exact");
    check(tensor_id_busy, "run_vs_tensor_id_busy");
    check(tensor_bytes_busy, "run_vs_tensor_bytes_busy");
    check(output_busy, "run_vs_get_output_busy");
    check(boundary_busy, "run_vs_copy_boundary_busy");
    check(y26_executor_tensor_id(executor, argv[5]) == tensor_id,
          "tensor_id_after_run");
    check(y26_executor_tensor_bytes(executor, tensor_id) == tensor_bytes,
          "tensor_bytes_after_run");
    check(y26_executor_get_output(executor, copied_output,
                                  Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS) == Y26_STATUS_OK,
          "get_output_after_run");
    check(y26_executor_copy_boundary(executor, tensor_id, boundary, tensor_bytes) ==
              Y26_STATUS_OK,
          "copy_boundary_after_run");

    float* independent_output = calloc(Y26_K1X_EXECUTOR_OUTPUT_ELEMENTS, sizeof(float));
    y26_executor* independent_executor = y26_executor_create();
    check(independent_output != NULL && independent_executor != NULL,
          "independent_handle_create");
    check(independent_executor != NULL &&
              y26_executor_prepare(independent_executor, argv[1], argv[2], &options) ==
                  Y26_STATUS_OK,
          "independent_handle_prepare");
    if (independent_output != NULL && independent_executor != NULL) {
        run_context first = {
            .executor = executor,
            .input = input,
            .output = run_output,
            .entered = ATOMIC_VAR_INIT(0),
            .finished = ATOMIC_VAR_INIT(0),
            .status = Y26_STATUS_RUNTIME_ERROR,
            .hash = 0,
        };
        run_context second = {
            .executor = independent_executor,
            .input = input,
            .output = independent_output,
            .entered = ATOMIC_VAR_INIT(0),
            .finished = ATOMIC_VAR_INIT(0),
            .status = Y26_STATUS_RUNTIME_ERROR,
            .hash = 0,
        };
        pthread_t first_thread;
        pthread_t second_thread;
        const int first_status = pthread_create(&first_thread, NULL, run_once, &first);
        const int second_status = pthread_create(&second_thread, NULL, run_once, &second);
        check(first_status == 0 && second_status == 0,
              "independent_handles_thread_create");
        if (first_status == 0) pthread_join(first_thread, NULL);
        if (second_status == 0) pthread_join(second_thread, NULL);
        printf("independent_first_status\t%d\nindependent_first_hash\t0x%016llx\n"
               "independent_second_status\t%d\nindependent_second_hash\t0x%016llx\n",
               (int)first.status, (unsigned long long)first.hash,
               (int)second.status, (unsigned long long)second.hash);
        check(first_status == 0 && second_status == 0 &&
                  first.status == Y26_STATUS_OK && second.status == Y26_STATUS_OK &&
                  first.hash == expected_hash && second.hash == expected_hash,
              "independent_handles_parallel_exact");
    }

    y26_executor_destroy(independent_executor);
    y26_executor_destroy(executor);
    free(independent_output);
    free(boundary);
    free(copied_output);
    free(run_output);
    free(input);
    printf("failures\t%d\n", failures);
    return failures == 0 ? 0 : 1;
}
