cmake_minimum_required(VERSION 3.20)

project(y26_k1x_custom_int8_engine VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Y26_K1X_ENABLE_TESTS "Build custom INT8 engine tests" ON)
option(Y26_K1X_ENABLE_IME "Build the guarded Stage 1 smt.vmadot IME path" OFF)
option(Y26_K1X_BUILD_TOOLS "Build custom INT8 engine helper tools" ON)

add_library(y26_k1x_custom_int8_engine
    kernels/vmadot_4x4x8_ime.cpp
    kernels/vmadot_4x4x8_scalar.cpp
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

if(Y26_K1X_ENABLE_IME)
    target_compile_definitions(y26_k1x_custom_int8_engine
        PRIVATE
            Y26_K1X_ENABLE_IME_ASM=1
    )
endif()

if(Y26_K1X_BUILD_TOOLS)
    add_executable(bench_vmadot_microkernel
        tools/bench_vmadot_microkernel.cpp
    )
    target_link_libraries(bench_vmadot_microkernel PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_vmadot_microkernel PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_vmadot_microkernel PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()
endif()

if(Y26_K1X_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
