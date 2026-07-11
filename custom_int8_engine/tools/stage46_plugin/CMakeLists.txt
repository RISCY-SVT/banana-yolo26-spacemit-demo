cmake_minimum_required(VERSION 3.20)
project(stage46_u8_xor_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT SPACEMIT_ORT_ROOT)
    message(FATAL_ERROR "SPACEMIT_ORT_ROOT is required")
endif()

add_library(stage46_u8_xor_plugin SHARED stage46_u8_xor_plugin.cpp)
target_include_directories(stage46_u8_xor_plugin SYSTEM PRIVATE "${SPACEMIT_ORT_ROOT}/include")
target_link_libraries(stage46_u8_xor_plugin PRIVATE "${SPACEMIT_ORT_ROOT}/lib/libspacemit_ep.so")
target_compile_options(stage46_u8_xor_plugin PRIVATE -Wall -Wextra -Wpedantic -fvisibility=hidden)
set_target_properties(stage46_u8_xor_plugin PROPERTIES
    SKIP_BUILD_RPATH TRUE
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
