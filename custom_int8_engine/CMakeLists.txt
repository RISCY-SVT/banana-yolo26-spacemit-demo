cmake_minimum_required(VERSION 3.20)

project(y26_k1x_custom_int8_engine VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Y26_K1X_ENABLE_TESTS "Build custom INT8 engine tests" ON)
option(Y26_K1X_ENABLE_IME "Build the guarded Stage 1 smt.vmadot IME path" OFF)
option(Y26_K1X_BUILD_TOOLS "Build custom INT8 engine helper tools" ON)

add_library(y26_k1x_custom_int8_engine
    kernels/conv1x1_i8s8s32_ime.cpp
    kernels/conv1x1_i8s8s32_scalar.cpp
    kernels/conv3x3_i8s8s32_ime.cpp
    kernels/conv3x3_i8s8s32_scalar.cpp
    kernels/conv_mmt4d_prepack.cpp
    kernels/pack_mmt4d.cpp
    kernels/vmadot_4x4x8_ime.cpp
    kernels/vmadot_4x4x8_scalar.cpp
    kernels/vmadot_stub.cpp
    src/engine_stub.cpp
    src/block_runner.cpp
    src/multiblock_runner.cpp
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

    add_executable(bench_vmadot_hotpath
        tools/bench_vmadot_hotpath.cpp
    )
    target_link_libraries(bench_vmadot_hotpath PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_vmadot_hotpath PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_vmadot_hotpath PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_conv_kernels
        tools/bench_conv_kernels.cpp
    )
    target_link_libraries(bench_conv_kernels PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_conv_kernels PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_conv_kernels PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage3_packing
        tools/bench_stage3_packing.cpp
    )
    target_link_libraries(bench_stage3_packing PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage3_packing PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage3_packing PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage3_packing PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage4_packing
        tools/bench_stage4_packing.cpp
    )
    target_link_libraries(bench_stage4_packing PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage4_packing PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage4_packing PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage4_packing PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage5_first_block
        tools/bench_stage5_first_block.cpp
    )
    target_link_libraries(bench_stage5_first_block PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage5_first_block PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage5_first_block PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage5_first_block PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage6_multiblock
        tools/bench_stage6_multiblock.cpp
    )
    target_link_libraries(bench_stage6_multiblock PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage6_multiblock PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage6_multiblock PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage6_multiblock PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()
endif()

if(Y26_K1X_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
