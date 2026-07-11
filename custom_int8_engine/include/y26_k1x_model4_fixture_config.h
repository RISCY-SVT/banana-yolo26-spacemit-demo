#pragma once

#include "y26_k1x_model4_c2f_runner.h"

#include <cstddef>
#include <cstdint>

extern "C" {

enum Y26Model4FixtureId {
    Y26_MODEL4_FIXTURE_SYNTHETIC_SEEDED = 0,
    Y26_MODEL4_FIXTURE_SYNTHETIC_GRADIENT = 1,
};

struct Y26Model4FixtureView {
    const char* label;
    const std::int8_t* input_nhwc_s8;
    const std::int32_t* expected_branch1_i32_nhwc;
    std::size_t expected_branch1_count;
    const std::int8_t* expected_concat_s8_nhwc;
    std::size_t expected_concat_count;
    const std::int32_t* expected_model4_cv2_i32_nhwc;
    std::size_t expected_model4_cv2_count;
};

int y26_model4_fixture_count();

int y26_model4_fixture_make(int fixture_id,
                            int activation_mode,
                            int merge_mode,
                            Y26Stage16Model4C2fConfig* cfg,
                            Y26Model4FixtureView* view);

}
