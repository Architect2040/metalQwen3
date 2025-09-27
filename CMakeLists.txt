cmake_minimum_required(VERSION 3.21)
project(MetalQwen3 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Expose metal-cpp headers location to subdirectories
set(METAL_CPP_ROOT "${CMAKE_SOURCE_DIR}/libs/metal-cpp" CACHE PATH "Path to the metal-cpp SDK")

# Auto-download metal-cpp v26 if not present
if(NOT EXISTS "${METAL_CPP_ROOT}")
    message(STATUS "metal-cpp SDK not found. Downloading metal-cpp v26...")

    # Create libs directory if it doesn't exist
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/libs")

    # Download metal-cpp v26
    set(METAL_CPP_URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS26_iOS26-beta2.zip")
    set(METAL_CPP_ZIP "${CMAKE_SOURCE_DIR}/libs/metal-cpp.zip")

    message(STATUS "Downloading ${METAL_CPP_URL}...")
    file(DOWNLOAD "${METAL_CPP_URL}" "${METAL_CPP_ZIP}"
         SHOW_PROGRESS
         STATUS DOWNLOAD_STATUS)

    # Check download status
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR)
    if(DOWNLOAD_ERROR)
        list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MSG)
        message(FATAL_ERROR "Failed to download metal-cpp: ${DOWNLOAD_ERROR_MSG}")
    endif()

    # Extract the zip file
    message(STATUS "Extracting metal-cpp...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf "${METAL_CPP_ZIP}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/libs"
        RESULT_VARIABLE EXTRACT_RESULT
    )

    if(EXTRACT_RESULT)
        message(FATAL_ERROR "Failed to extract metal-cpp zip file")
    endif()

    # Clean up zip file
    file(REMOVE "${METAL_CPP_ZIP}")

    # Verify extraction
    if(NOT EXISTS "${METAL_CPP_ROOT}")
        message(FATAL_ERROR "metal-cpp extraction failed - directory not found at ${METAL_CPP_ROOT}")
    endif()

    message(STATUS "metal-cpp v26 downloaded and extracted successfully")
endif()

add_subdirectory(scripts)
