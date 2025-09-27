cmake_minimum_required(VERSION 3.20)
project(MetalQwen3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FetchContent for dependencies
include(FetchContent)

# httplib for HTTP server
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.14.3
)
FetchContent_MakeAvailable(httplib)

# nlohmann_json for JSON parsing
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# fmt for better formatting
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1
)
FetchContent_MakeAvailable(fmt)

# Resolve source tree roots from the top-level project directory
set(SRC_ROOT ${CMAKE_SOURCE_DIR}/src)

# Helper macro to prefix source files with SRC_ROOT
macro(prefix_sources out_var)
    set(result "")
    foreach(path ${ARGN})
        list(APPEND result "${SRC_ROOT}/${path}")
    endforeach()
    set(${out_var} ${result})
endmacro()

prefix_sources(TEST_QWEN3_SRCS
    benchmark/test_qwen3_benchmark.cpp
    metal/MetalContext.cpp
    metal/MetalContext.h
    metal/MetalQwen3.cpp
    metal/MetalQwen3.h
    cpu/Qwen3Original.cpp
    cpu/Qwen3Original.h
    cpu/qwen3_original.c
    cpu/qwen3_c_api.h
    benchmark/Qwen3Benchmark.cpp
    benchmark/Qwen3Benchmark.h
    benchmark/Benchmark.cpp
    benchmark/Benchmark.h
    cpu/CPUMatrixMul.cpp
    cpu/CPUMatrixMul.h
)

prefix_sources(API_SERVER_SRCS
    api/qwen3_api_server.cpp
    metal/MetalContext.cpp
    metal/MetalContext.h
    metal/MetalQwen3.cpp
    metal/MetalQwen3.h
    api/Qwen3ApiHandler.cpp
    api/Qwen3ApiHandler.h
    tokenizer/Qwen3Tokenizer.cpp
    tokenizer/Qwen3Tokenizer.h
    benchmark/Benchmark.cpp
    benchmark/Benchmark.h
    cpu/CPUMatrixMul.cpp
    cpu/CPUMatrixMul.h
    cpu/Qwen3Original.cpp
    cpu/Qwen3Original.h
    cpu/qwen3_original.c
    cpu/qwen3_c_api.h
)

# Create executables with fully-qualified source paths
add_executable(TestQwen3Benchmark ${TEST_QWEN3_SRCS})
add_executable(Qwen3ApiServer ${API_SERVER_SRCS})

# Shader functionality test (minimal dependencies)
set(SHADER_TEST_SRCS
    ${SRC_ROOT}/test_shaders.cpp
    ${SRC_ROOT}/metal/MetalContext.cpp
)
add_executable(TestShaders ${SHADER_TEST_SRCS})

# Include directories
set(METAL_CPP_INCLUDE_DIRS
    ${METAL_CPP_ROOT}
    ${METAL_CPP_ROOT}/SingleHeader
    ${METAL_CPP_ROOT}/Metal
    ${METAL_CPP_ROOT}/Foundation
    ${METAL_CPP_ROOT}/QuartzCore
)

target_include_directories(TestQwen3Benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/metal
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/src/benchmark
    ${CMAKE_SOURCE_DIR}/src/api
    ${CMAKE_SOURCE_DIR}/src/tokenizer
    ${METAL_CPP_INCLUDE_DIRS}
)

target_include_directories(Qwen3ApiServer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/metal
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/src/benchmark
    ${CMAKE_SOURCE_DIR}/src/api
    ${CMAKE_SOURCE_DIR}/src/tokenizer
    ${METAL_CPP_INCLUDE_DIRS}
)

target_include_directories(TestShaders PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/metal
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${METAL_CPP_INCLUDE_DIRS}
)

# Apple Metal support
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)

    target_link_libraries(TestQwen3Benchmark
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        fmt::fmt
    )

    target_link_libraries(Qwen3ApiServer
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        httplib::httplib
        nlohmann_json::nlohmann_json
        fmt::fmt
    )

    target_link_libraries(TestShaders
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
    )

    # Compile Metal shaders
    set(SHADERS_ROOT ${CMAKE_SOURCE_DIR}/shaders)
    set(METAL_SHADER_SOURCES
        "${SHADERS_ROOT}/quantized_matmul.metal"
        "${SHADERS_ROOT}/rmsnorm.metal"
        "${SHADERS_ROOT}/attention.metal"
        "${SHADERS_ROOT}/rope.metal"
        "${SHADERS_ROOT}/swiglu.metal"
        "${SHADERS_ROOT}/softmax.metal"
        "${SHADERS_ROOT}/matrix_mul.metal"
    )

    foreach(SHADER_SOURCE ${METAL_SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
        set(AIR_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air")
        set(METALLIB_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.metallib")

        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND xcrun -sdk macosx metal -c ${SHADER_SOURCE} -o ${AIR_FILE}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling Metal shader ${SHADER_NAME}"
        )

        add_custom_command(
            OUTPUT ${METALLIB_FILE}
            COMMAND xcrun -sdk macosx metallib ${AIR_FILE} -o ${METALLIB_FILE}
            DEPENDS ${AIR_FILE}
            COMMENT "Creating Metal library ${SHADER_NAME}"
        )

        add_custom_target(CompileMetalShaders_${SHADER_NAME} DEPENDS ${METALLIB_FILE})
        add_dependencies(TestQwen3Benchmark CompileMetalShaders_${SHADER_NAME})
        add_dependencies(Qwen3ApiServer CompileMetalShaders_${SHADER_NAME})
        add_dependencies(TestShaders CompileMetalShaders_${SHADER_NAME})

        # Copy metal libraries to output directory
        add_custom_command(TARGET TestQwen3Benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${METALLIB_FILE} $<TARGET_FILE_DIR:TestQwen3Benchmark>
        )

        add_custom_command(TARGET Qwen3ApiServer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${METALLIB_FILE} $<TARGET_FILE_DIR:Qwen3ApiServer>
        )

        add_custom_command(TARGET TestShaders POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${METALLIB_FILE} $<TARGET_FILE_DIR:TestShaders>
        )
    endforeach()

    # Set C++ compilation for Metal integration using metal-cpp
    set_source_files_properties(
        ${SRC_ROOT}/metal/MetalContext.cpp
        ${SRC_ROOT}/metal/MetalQwen3.cpp
        ${SRC_ROOT}/test_shaders.cpp
        PROPERTIES COMPILE_FLAGS "-std=c++17"
    )
endif()

target_compile_definitions(TestQwen3Benchmark PRIVATE QWEN3_LIBRARY)
target_compile_definitions(Qwen3ApiServer PRIVATE QWEN3_LIBRARY)
target_compile_definitions(TestShaders PRIVATE QWEN3_LIBRARY)

if(WIN32)
    # Windows specific settings (though not recommended for Metal)
    set_property(TARGET TestQwen3Benchmark PROPERTY WIN32_EXECUTABLE FALSE)
endif()
