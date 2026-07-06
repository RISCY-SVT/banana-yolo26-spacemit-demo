cmake_minimum_required(VERSION 3.20)

project(y26_k1x_custom_int8_engine VERSION 0.0.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Y26_K1X_ENABLE_TESTS "Build custom INT8 engine tests" ON)
option(Y26_K1X_ENABLE_IME "Build the guarded Stage 1 smt.vmadot IME path" OFF)
option(Y26_K1X_BUILD_TOOLS "Build custom INT8 engine helper tools" ON)

add_library(y26_k1x_custom_int8_engine
    kernels/activation_requant.cpp
    kernels/conv1x1_i8s8s32_ime.cpp
    kernels/conv1x1_i8s8s32_scalar.cpp
    kernels/conv3x3_i8s8s32_ime.cpp
    kernels/conv3x3_i8s8s32_scalar.cpp
    kernels/conv_mmt4d_prepack.cpp
    kernels/conv_threaded.cpp
    kernels/pack_mmt4d.cpp
    kernels/vmadot_4x4x8_ime.cpp
    kernels/vmadot_4x4x8_scalar.cpp
    kernels/vmadot_stub.cpp
    src/branch_block_runner.cpp
    src/backbone_subset_runner.cpp
    src/backbone_stage10_runner.cpp
    src/c2f_block_runner.cpp
    src/engine_stub.cpp
    src/block_runner.cpp
    src/multiblock_runner.cpp
    src/model_format.cpp
    src/quant_metadata.cpp
    src/next_c2f_runner.cpp
    src/model4_branch_runner.cpp
    src/model4_c2f_runner.cpp
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

target_link_libraries(y26_k1x_custom_int8_engine
    PUBLIC
        Threads::Threads
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

    add_executable(bench_stage7_backbone_subset
        tools/bench_stage7_backbone_subset.cpp
    )
    target_link_libraries(bench_stage7_backbone_subset PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage7_backbone_subset PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage7_backbone_subset PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage7_backbone_subset PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage8_activation_requant
        tools/bench_stage8_activation_requant.cpp
    )
    target_link_libraries(bench_stage8_activation_requant PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage8_activation_requant PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage8_activation_requant PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage8_activation_requant PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage9_activation_fusion
        tools/bench_stage9_activation_fusion.cpp
    )
    target_link_libraries(bench_stage9_activation_fusion PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage9_activation_fusion PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage9_activation_fusion PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage9_activation_fusion PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage10_backbone_expansion
        tools/bench_stage10_backbone_expansion.cpp
    )
    target_link_libraries(bench_stage10_backbone_expansion PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage10_backbone_expansion PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage10_backbone_expansion PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage10_backbone_expansion PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage11_branch_block
        tools/bench_stage11_branch_block.cpp
    )
    target_link_libraries(bench_stage11_branch_block PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage11_branch_block PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage11_branch_block PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage11_branch_block PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage12_c2f_block
        tools/bench_stage12_c2f_block.cpp
    )
    target_link_libraries(bench_stage12_c2f_block PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage12_c2f_block PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage12_c2f_block PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage12_c2f_block PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage13_merge_dataflow
        tools/bench_stage13_merge_dataflow.cpp
    )
    target_link_libraries(bench_stage13_merge_dataflow PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage13_merge_dataflow PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage13_merge_dataflow PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage13_merge_dataflow PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage14_next_c2f
        tools/bench_stage14_next_c2f.cpp
    )
    target_link_libraries(bench_stage14_next_c2f PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage14_next_c2f PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage14_next_c2f PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage14_next_c2f PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage15_model4_branch
        tools/bench_stage15_model4_branch.cpp
    )
    target_link_libraries(bench_stage15_model4_branch PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage15_model4_branch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage15_model4_branch PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage15_model4_branch PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage16_fullshape_gate
        tools/bench_stage16_fullshape_gate.cpp
    )
    target_link_libraries(bench_stage16_fullshape_gate PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage16_fullshape_gate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage16_fullshape_gate PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage16_fullshape_gate PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage16_model4_c2f
        tools/bench_stage16_model4_c2f.cpp
    )
    target_link_libraries(bench_stage16_model4_c2f PRIVATE y26_k1x_custom_int8_engine)
    target_include_directories(bench_stage16_model4_c2f PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage16_model4_c2f PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage16_model4_c2f PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage17_roofline_threading
        tools/bench_stage17_roofline_threading.cpp
    )
    target_link_libraries(bench_stage17_roofline_threading PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage17_roofline_threading PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage17_roofline_threading PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage17_roofline_threading PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage18_threaded_conv_integration
        tools/bench_stage18_threaded_conv_integration.cpp
    )
    target_link_libraries(bench_stage18_threaded_conv_integration PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage18_threaded_conv_integration PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage18_threaded_conv_integration PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage18_threaded_conv_integration PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage19_threaded_c2f_integration
        tools/bench_stage19_threaded_c2f_integration.cpp
    )
    target_link_libraries(bench_stage19_threaded_c2f_integration PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage19_threaded_c2f_integration PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage19_threaded_c2f_integration PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage19_threaded_c2f_integration PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage20_model4_fullshape_c2f
        tools/bench_stage20_model4_fullshape_c2f.cpp
    )
    target_link_libraries(bench_stage20_model4_fullshape_c2f PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage20_model4_fullshape_c2f PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage20_model4_fullshape_c2f PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage20_model4_fullshape_c2f PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()
endif()

if(Y26_K1X_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
