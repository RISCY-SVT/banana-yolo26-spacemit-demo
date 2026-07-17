cmake_minimum_required(VERSION 3.20)

project(banana_yolo26_k1x_demo VERSION 0.9.2 LANGUAGES C CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(Y26_DEMO_USE_INSTALLED_EXECUTOR "Build against an installed y26 executor package" OFF)
option(Y26_DEMO_ENABLE_ASAN "Enable AddressSanitizer for the demo" OFF)
option(Y26_DEMO_BUILD_TESTS "Build demo geometry tests" ON)
option(Y26_DEMO_OFFICIAL_K1X_RELEASE
       "Build the embedded executor under the fail-closed K1X release contract" OFF)
set(Y26_DEMO_EXECUTOR_DIR "" CACHE PATH "Installed y26K1xExecutor CMake package directory")

find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

if(Y26_DEMO_USE_INSTALLED_EXECUTOR)
    if(Y26_DEMO_EXECUTOR_DIR)
        set(y26K1xExecutor_DIR "${Y26_DEMO_EXECUTOR_DIR}")
    endif()
    find_package(y26K1xExecutor 0.9.2 CONFIG REQUIRED)
else()
    set(Y26_K1X_BUILD_RELEASE ON CACHE BOOL "" FORCE)
    set(Y26_K1X_BUILD_RELEASE_TOOLS ON CACHE BOOL "" FORCE)
    set(Y26_K1X_BUILD_RESEARCH OFF CACHE BOOL "" FORCE)
    set(Y26_K1X_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(Y26_K1X_INSTALL_DEVELOPMENT_HEADERS OFF CACHE BOOL "" FORCE)
    set(Y26_K1X_OFFICIAL_RELEASE ${Y26_DEMO_OFFICIAL_K1X_RELEASE} CACHE BOOL "" FORCE)
    add_subdirectory(custom_int8_engine)
endif()

add_library(y26_demo_core STATIC
    src/app/application.cpp
    src/app/options.cpp
    src/infer/yolo26_executor_detector.cpp
    src/io/media_source.cpp
    src/render/renderer.cpp
    src/util/logger.cpp
)
target_include_directories(y26_demo_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(y26_demo_core
    PUBLIC
        y26::executor_shared
        Threads::Threads
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
        opencv_videoio
)
target_compile_options(y26_demo_core PRIVATE -Wall -Wextra -Wpedantic)

if(Y26_DEMO_ENABLE_ASAN)
    target_compile_options(y26_demo_core PRIVATE -fsanitize=address,undefined)
    target_link_options(y26_demo_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(y26_k1x_demo src/main.cpp)
target_link_libraries(y26_k1x_demo PRIVATE y26_demo_core)
target_link_options(y26_k1x_demo PRIVATE "LINKER:--allow-shlib-undefined")
set_target_properties(y26_k1x_demo PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR};$ORIGIN/../opencv/lib"
)

install(TARGETS y26_k1x_demo RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
add_executable(y26_v4l2_probe tools/stage59_v4l2_probe.cpp)
target_compile_options(y26_v4l2_probe PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS y26_v4l2_probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(FILES assets/coco80.txt DESTINATION "${CMAKE_INSTALL_DATADIR}/y26-k1x-demo/labels")

if(Y26_DEMO_BUILD_TESTS)
    enable_testing()
    add_executable(test_demo_geometry tests/test_demo_geometry.cpp)
    target_link_libraries(test_demo_geometry PRIVATE y26_demo_core)
    target_link_options(test_demo_geometry PRIVATE "LINKER:--allow-shlib-undefined")
    add_test(NAME demo_geometry COMMAND test_demo_geometry)
endif()
