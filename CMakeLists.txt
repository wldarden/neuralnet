add_library(neuralnet STATIC
    src/activation.cpp
    src/layer.cpp
    src/network.cpp
    src/serialization.cpp
    src/graph_network.cpp
    src/neural_neat_policy.cpp
)
target_include_directories(neuralnet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(neuralnet PUBLIC evolve PRIVATE project_warnings)

add_subdirectory(tests)
