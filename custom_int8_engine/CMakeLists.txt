cmake_minimum_required(VERSION 3.20)

project(y26_k1x_custom_int8_engine VERSION 0.0.0 LANGUAGES C CXX)

find_package(Threads REQUIRED)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(Y26_K1X_ENABLE_TESTS "Build custom INT8 engine tests" ON)
option(Y26_K1X_ENABLE_IME "Build the guarded Stage 1 smt.vmadot IME path" OFF)
option(Y26_K1X_BUILD_TOOLS "Build custom INT8 engine helper tools" ON)
set(Y26_K1X_ORT_INCLUDE_DIR "" CACHE PATH "Optional ONNX Runtime include directory for in-process scaffold tools")
set(Y26_K1X_ORT_LIBRARY "" CACHE FILEPATH "Optional ONNX Runtime library for in-process scaffold tools")
set(Y26_K1X_SPACEMIT_EP_LIBRARY "" CACHE FILEPATH "Optional SpacemiT EP library for Stage46 runtime probes")
set(Y26_K1X_OPENCV_DIR "" CACHE PATH "Optional OpenCV package directory for the Stage52 COCO predictor")

add_library(y26_k1x_custom_int8_engine
    kernels/activation_requant.cpp
    kernels/conv1x1_i8s8s32_ime.cpp
    kernels/conv1x1_i8s8s32_scalar.cpp
    kernels/conv3x3_i8s8s32_ime.cpp
    kernels/conv3x3_i8s8s32_scalar.cpp
    kernels/conv_mmt4d_prepack.cpp
    kernels/conv_threaded.cpp
    kernels/pack_mmt4d.cpp
    kernels/stage47_integrated_conv.cpp
    kernels/stage48_nchwc8_model5.cpp
    kernels/stage49_persistent_slice.cpp
    kernels/stage51_q62_epilogue.cpp
    kernels/vmadot_4x4x8_ime.cpp
    kernels/vmadot_4x4x8_scalar.cpp
    kernels/vmadot123_direct_conv.cpp
    kernels/vmadot123_probe.cpp
    kernels/vmadot_stub.cpp
    src/branch_block_runner.cpp
    src/backbone_subset_runner.cpp
    src/backbone_stage10_runner.cpp
    src/c2f_block_runner.cpp
    src/engine_stub.cpp
    src/int8_v1.cpp
    src/block_runner.cpp
    src/multiblock_runner.cpp
    src/model_format.cpp
    src/package_loader.cpp
    src/quant_metadata.cpp
    src/next_c2f_runner.cpp
    src/model4_branch_runner.cpp
    src/model4_c2f_runner.cpp
    src/model4_fixture_config.cpp
    src/model5_island.cpp
    src/stage42_runner_support.cpp
    src/stage47_aot_executor.cpp
    src/stage52_c_api.cpp
    src/stage52_full_executor.cpp
)

target_include_directories(y26_k1x_custom_int8_engine
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/tests"
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

set_target_properties(y26_k1x_custom_int8_engine PROPERTIES
    OUTPUT_NAME y26_k1x_int8_executor
)

install(TARGETS y26_k1x_custom_int8_engine
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES
    include/y26_k1x_executor.h
    include/y26_k1x_full_executor.h
    include/y26_k1x_int8_v1.h
    include/y26_k1x_package.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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

    add_executable(bench_stage21_model4_c2f_integrated
        tools/bench_stage21_model4_c2f_integrated.cpp
    )
    target_link_libraries(bench_stage21_model4_c2f_integrated PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage21_model4_c2f_integrated PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage21_model4_c2f_integrated PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage21_model4_c2f_integrated PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage22_model4_onnx_cut
        tools/bench_stage22_model4_onnx_cut.cpp
    )
    target_link_libraries(bench_stage22_model4_onnx_cut PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage22_model4_onnx_cut PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage22_model4_onnx_cut PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage22_model4_onnx_cut PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage23_model4_runner_cut
        tools/bench_stage23_model4_runner_cut.cpp
    )
    target_link_libraries(bench_stage23_model4_runner_cut PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage23_model4_runner_cut PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage23_model4_runner_cut PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage23_model4_runner_cut PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage30_vmadot123_probe
        tools/bench_stage30_vmadot123_probe.cpp
    )
    target_link_libraries(bench_stage30_vmadot123_probe PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_stage30_vmadot123_probe PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage30_vmadot123_probe PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage31_vmadot123_direct_conv
        tools/bench_stage31_vmadot123_direct_conv.cpp
    )
    target_link_libraries(bench_stage31_vmadot123_direct_conv PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage31_vmadot123_direct_conv PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage31_vmadot123_direct_conv PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage31_vmadot123_direct_conv PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage32_layout_decision
        tools/bench_stage32_layout_decision.cpp
    )
    target_link_libraries(bench_stage32_layout_decision PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_include_directories(bench_stage32_layout_decision PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_compile_options(bench_stage32_layout_decision PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage32_layout_decision PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage34_vmadot_throughput
        tools/bench_stage34_vmadot_throughput.cpp
    )
    target_link_libraries(bench_stage34_vmadot_throughput PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_stage34_vmadot_throughput PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage34_vmadot_throughput PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage35_vmadot_sigill
        tools/bench_stage35_vmadot_sigill.cpp
    )
    target_link_libraries(bench_stage35_vmadot_sigill PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_stage35_vmadot_sigill PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage35_vmadot_sigill PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage45_k1x_roofline
        tools/bench_stage45_k1x_roofline.cpp
    )
    target_link_libraries(bench_stage45_k1x_roofline PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_stage45_k1x_roofline PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage45_k1x_roofline PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage47_executor
        tools/bench_stage47_executor.cpp
    )
    target_link_libraries(bench_stage47_executor PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(bench_stage47_executor PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage47_executor PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(stage47_perf_event_probe
        tools/stage47_perf_event_probe.cpp
    )
    target_compile_options(stage47_perf_event_probe PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(bench_stage48_nchwc8
        tools/bench_stage48_nchwc8.cpp
    )
    target_link_libraries(bench_stage48_nchwc8 PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(bench_stage48_nchwc8 PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage48_nchwc8 PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage49_slice
        tools/bench_stage49_slice.cpp
    )
    target_link_libraries(bench_stage49_slice PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(bench_stage49_slice PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage49_slice PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage50_slice
        tools/bench_stage49_slice.cpp
    )
    target_link_libraries(bench_stage50_slice PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(bench_stage50_slice PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage50_slice PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(bench_stage50_layout_adapter
        tools/bench_stage50_layout_adapter.cpp
    )
    target_link_libraries(bench_stage50_layout_adapter PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(bench_stage50_layout_adapter PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(stage49_perf_exec
        tools/stage49_perf_exec.cpp
    )
    target_compile_options(stage49_perf_exec PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(stage49_q62_vectorization_probe
        tools/stage49_q62_vectorization_probe.cpp
    )
    target_compile_options(stage49_q62_vectorization_probe PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(stage50_integer_vector_sidecar
        tools/stage50_integer_vector_sidecar.cpp
    )
    target_compile_options(stage50_integer_vector_sidecar PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(stage51_q62_vsmul_probe
        tools/stage51_q62_vsmul_probe.cpp
    )
    target_link_libraries(stage51_q62_vsmul_probe PRIVATE y26_k1x_custom_int8_engine)
    target_compile_options(stage51_q62_vsmul_probe PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(yolo26_k1x_int8
        tools/yolo26_k1x_int8.cpp
    )
    target_link_libraries(yolo26_k1x_int8 PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(yolo26_k1x_int8 PRIVATE -Wall -Wextra -Wpedantic)
    install(TARGETS yolo26_k1x_int8 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(yolo26_k1x_int8 PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    add_executable(y26_k1x_c_api_smoke
        tools/stage52_c_api_smoke.c
    )
    target_link_libraries(y26_k1x_c_api_smoke PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(y26_k1x_c_api_smoke PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(y26_k1x_c_api_smoke PROPERTIES
        LINKER_LANGUAGE CXX
        BUILD_RPATH "$ORIGIN/../lib"
        INSTALL_RPATH "$ORIGIN/../lib"
    )
    install(TARGETS y26_k1x_c_api_smoke RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    if(Y26_K1X_OPENCV_DIR)
        set(OpenCV_DIR "${Y26_K1X_OPENCV_DIR}")
        find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
        if(OpenCV_FOUND)
            target_link_libraries(yolo26_k1x_int8 PRIVATE ${OpenCV_LIBS})
            target_include_directories(yolo26_k1x_int8 PRIVATE ${OpenCV_INCLUDE_DIRS})
            target_compile_definitions(yolo26_k1x_int8 PRIVATE Y26_K1X_HAVE_OPENCV=1)
            set_target_properties(yolo26_k1x_int8 PROPERTIES
                BUILD_RPATH "$ORIGIN/../lib;$ORIGIN/../opencv/lib"
                INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../opencv/lib"
            )
            add_executable(stage52_coco_predict
                tools/stage52_coco_predict.cpp
            )
            target_link_libraries(stage52_coco_predict
                PRIVATE
                    y26_k1x_custom_int8_engine
                    Threads::Threads
                    ${OpenCV_LIBS}
            )
            target_include_directories(stage52_coco_predict PRIVATE ${OpenCV_INCLUDE_DIRS})
            target_compile_options(stage52_coco_predict PRIVATE -Wall -Wextra -Wpedantic)
            set_target_properties(stage52_coco_predict PROPERTIES
                BUILD_RPATH "$ORIGIN/../opencv/lib"
                INSTALL_RPATH "$ORIGIN/../opencv/lib"
            )
            add_executable(stage52_image_pipeline_bench
                tools/stage52_image_pipeline_bench.cpp
            )
            target_link_libraries(stage52_image_pipeline_bench
                PRIVATE
                    y26_k1x_custom_int8_engine
                    Threads::Threads
                    ${OpenCV_LIBS}
            )
            target_include_directories(stage52_image_pipeline_bench PRIVATE ${OpenCV_INCLUDE_DIRS})
            target_compile_options(stage52_image_pipeline_bench PRIVATE -Wall -Wextra -Wpedantic)
            set_target_properties(stage52_image_pipeline_bench PROPERTIES
                BUILD_RPATH "$ORIGIN/../opencv/lib"
                INSTALL_RPATH "$ORIGIN/../opencv/lib"
            )
        endif()
    endif()

    add_executable(stage52_boundary_dump
        tools/stage52_boundary_dump.cpp
    )
    target_link_libraries(stage52_boundary_dump PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(stage52_boundary_dump PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(stage52_core_bridge_dump
        tools/stage52_core_bridge_dump.cpp
    )
    target_link_libraries(stage52_core_bridge_dump PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(stage52_core_bridge_dump PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(stage52_state_probe
        tools/stage52_state_probe.cpp
    )
    target_link_libraries(stage52_state_probe PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(stage52_state_probe PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(bench_stage43_model5_island
        tools/bench_stage43_model5_island.cpp
    )
    target_link_libraries(bench_stage43_model5_island PRIVATE y26_k1x_custom_int8_engine Threads::Threads)
    target_compile_options(bench_stage43_model5_island PRIVATE -Wall -Wextra -Wpedantic)
    if(Y26_K1X_ENABLE_IME)
        target_compile_definitions(bench_stage43_model5_island PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
    endif()

    if(Y26_K1X_ORT_INCLUDE_DIR AND Y26_K1X_ORT_LIBRARY)
        add_executable(bench_stage42_inprocess_runner
            tools/bench_stage41_inprocess_runner.cpp
        )
        target_link_libraries(bench_stage42_inprocess_runner
            PRIVATE
                y26_k1x_custom_int8_engine
                Threads::Threads
                "${Y26_K1X_ORT_LIBRARY}"
        )
        target_include_directories(bench_stage42_inprocess_runner
            PRIVATE
                "${Y26_K1X_ORT_INCLUDE_DIR}"
        )
        set_target_properties(bench_stage42_inprocess_runner PROPERTIES SKIP_BUILD_RPATH TRUE)
        target_compile_options(bench_stage42_inprocess_runner PRIVATE -Wall -Wextra -Wpedantic)
        if(Y26_K1X_ENABLE_IME)
            target_compile_definitions(bench_stage42_inprocess_runner PRIVATE Y26_K1X_ENABLE_IME_ASM=1)
        endif()

        if(Y26_K1X_SPACEMIT_EP_LIBRARY)
            add_executable(bench_stage46_ort_runtime
                tools/bench_stage46_ort_runtime.cpp
            )
            target_link_libraries(bench_stage46_ort_runtime
                PRIVATE
                    Threads::Threads
                    "${Y26_K1X_ORT_LIBRARY}"
                    "${Y26_K1X_SPACEMIT_EP_LIBRARY}"
                    ${CMAKE_DL_LIBS}
            )
            target_include_directories(bench_stage46_ort_runtime
                PRIVATE
                    "${Y26_K1X_ORT_INCLUDE_DIR}"
            )
            set_target_properties(bench_stage46_ort_runtime PROPERTIES SKIP_BUILD_RPATH TRUE)
            target_compile_options(bench_stage46_ort_runtime PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
endif()

if(Y26_K1X_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
