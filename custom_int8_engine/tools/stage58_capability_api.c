#include "y26_k1x_executor.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char* name) {
    printf("%s\t%s\n", name, condition ? "pass" : "fail");
    if (!condition) ++failures;
}

int main(void) {
    y26_build_info_init(NULL);
    check(y26_executor_get_build_info(NULL) == Y26_STATUS_INVALID_ARGUMENT,
          "null_build_info");

    y26_build_info info;
    memset(&info, 0xff, sizeof(info));
    y26_build_info_init(&info);
    check(info.struct_size == sizeof(info) &&
          info.info_version == Y26_K1X_BUILD_INFO_VERSION,
          "build_info_init");
    check(y26_executor_get_build_info(&info) == Y26_STATUS_OK,
          "build_info_query");
    check(info.abi_version == Y26_K1X_EXECUTOR_ABI_VERSION,
          "abi_version");
    check(strcmp(info.release_version, "0.9.3") == 0,
          "release_version");
    check(strcmp(info.integer_contract_id, "K1X_INT8_V1") == 0,
          "integer_contract");
    check(strcmp(info.full_graph_profile_id,
                 "K1X_INT8_V1_YOLO26N_640_FULL_GRAPH_001") == 0,
          "full_graph_profile");
    check(strcmp(info.expected_package_manifest_sha256,
                 "fab4a72cf524ce0a205ceca0384144f2eee7bc79dff3f4db8b7208614e8407be") == 0,
          "package_manifest");
    check((info.capability_flags & Y26_CAPABILITY_RGB_INPUT) != 0,
          "rgb_capability");

    y26_build_info too_small;
    y26_build_info_init(&too_small);
    too_small.struct_size = sizeof(too_small) - 1;
    check(y26_executor_get_build_info(&too_small) == Y26_STATUS_INVALID_ARGUMENT,
          "undersized_build_info");
    y26_build_info wrong_version;
    y26_build_info_init(&wrong_version);
    wrong_version.info_version += 1;
    check(y26_executor_get_build_info(&wrong_version) == Y26_STATUS_INVALID_ARGUMENT,
          "wrong_info_version");

    printf("failures\t%d\n", failures);
    return failures == 0 ? 0 : 1;
}
