cmake_minimum_required(VERSION 3.20)

project(y26_k1x_custom_int8_engine VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Y26_K1X_ENABLE_TESTS "Build Stage 0 custom INT8 engine tests" ON)

add_library(y26_k1x_custom_int8_engine
    kernels/vmadot_stub.cpp
    src/engine_stub.cpp
    src/model_format.cpp
    src/quant_metadata.cpp
)

target_include_directories(y26_k1x_custom_int8_engine
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_options(y26_k1x_custom_int8_engine
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(Y26_K1X_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
