cmake_minimum_required(VERSION 3.24)
project(neuralnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/CompilerWarnings.cmake)

add_library(neuralnet STATIC
    src/activation.cpp
    src/layer.cpp
    src/network.cpp
    src/adapt.cpp
    src/serialization.cpp
    src/graph_network.cpp
    src/neural_neat_policy.cpp
)
target_include_directories(neuralnet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(neuralnet PUBLIC evolve PRIVATE project_warnings)

# Only build tests when this is the top-level project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(FetchContent)
    FetchContent_Declare(evolve
        GIT_REPOSITORY https://github.com/wldarden/evolve.git
        GIT_TAG        main
    )
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(evolve googletest)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
endif()
