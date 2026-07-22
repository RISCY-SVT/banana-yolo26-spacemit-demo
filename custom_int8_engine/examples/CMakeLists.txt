cmake_minimum_required(VERSION 3.20)
project(y26_k1x_c_consumer LANGUAGES C)

# The installed example is shared by the stable 0.9 and integrated 0.10
# deliveries. Applications should request their own required minor version.
find_package(y26K1xExecutor CONFIG REQUIRED)

option(Y26_CONSUMER_STATIC "Link the installed static executor library" OFF)
add_executable(y26_cmake_consumer c_api_consumer.c)
if(Y26_CONSUMER_STATIC)
    target_link_libraries(y26_cmake_consumer PRIVATE y26::executor_static)
else()
    target_link_libraries(y26_cmake_consumer PRIVATE y26::executor_shared)
endif()
