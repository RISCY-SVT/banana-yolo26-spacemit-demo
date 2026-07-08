#[[
@file CMakeLists.txt
@brief Top-level CMake entry point for the Banana YOLO11 SpaceMIT demo.
@details The build stays intentionally small: cross-toolchain setup lives in
`cmake/toolchains`, vendor runtime discovery lives in `cmake/modules`, and the
install layout mirrors the staged deploy tree used on the Banana board.
]]
cmake_minimum_required(VERSION 3.20)

project(banana_yolo11_spacemit_demo VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BANANA_DEMO_RUNTIME_SUBDIR "lib" CACHE STRING "Runtime subdirectory used for install RPATH")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(GNUInstallDirs)

option(BANANA_DEMO_ENABLE_ASAN "Enable address sanitizer" OFF)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn)
find_package(VendorSpacemitOrt REQUIRED)

add_library(banana_demo_core
    src/app/application.cpp
    src/app/options.cpp
    src/infer/yolo11_detector.cpp
    src/io/media_source.cpp
    src/render/renderer.cpp
    src/util/logger.cpp
    src/util/pinning.cpp
    src/util/sha256.cpp
)

target_include_directories(banana_demo_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(banana_demo_core
    PUBLIC
        Threads::Threads
        Vendor::onnxruntime
        Vendor::spacemit_ep
        ${OpenCV_LIBS}
)

target_link_options(banana_demo_core
    PUBLIC
        -Wl,--allow-shlib-undefined
)

target_compile_definitions(banana_demo_core
    PUBLIC
        __fp16=_Float16
)

target_compile_options(banana_demo_core
    PRIVATE
        -O3
        -DNDEBUG
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
)

if(BANANA_DEMO_ENABLE_ASAN)
    target_compile_options(banana_demo_core PRIVATE -fsanitize=address)
    target_link_options(banana_demo_core PRIVATE -fsanitize=address)
endif()

add_executable(banana_yolo11_demo src/main.cpp)
target_link_libraries(banana_yolo11_demo PRIVATE banana_demo_core)

add_executable(yolo26_coco_predict tools/yolo26_coco_predict.cpp)
target_link_libraries(yolo26_coco_predict PRIVATE banana_demo_core)
set_target_properties(banana_yolo11_demo PROPERTIES
    BUILD_RPATH "${VENDOR_SPACEMIT_ORT_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN/../runtime/${BANANA_DEMO_RUNTIME_SUBDIR};$ORIGIN/../../runtime/${BANANA_DEMO_RUNTIME_SUBDIR};$ORIGIN/../opencv/lib;$ORIGIN/../../opencv/lib"
)
set_target_properties(yolo26_coco_predict PROPERTIES
    BUILD_RPATH "${VENDOR_SPACEMIT_ORT_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN/../runtime/${BANANA_DEMO_RUNTIME_SUBDIR};$ORIGIN/../../runtime/${BANANA_DEMO_RUNTIME_SUBDIR};$ORIGIN/../opencv/lib;$ORIGIN/../../opencv/lib"
)

install(TARGETS banana_yolo11_demo yolo26_coco_predict
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(DIRECTORY assets DESTINATION .)
install(DIRECTORY scripts DESTINATION .)
install(DIRECTORY third_party_manifest DESTINATION .)
