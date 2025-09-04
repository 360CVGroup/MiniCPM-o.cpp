cmake_minimum_required(VERSION 3.15) # for add_link_options and implicit target directories.
project(minicpmo-cpp C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# set cuda arch for better compilation
# sm87 for jetson orin nano
set(CMAKE_CUDA_ARCHITECTURES "87;89")

#
# option list
#

# pybind11
option(USE_PYBINDING "enable python binding" OFF)

# openmp
option(USE_OPENMP "enable OpenMP" OFF)
# examples
option(BUILD_EXAMPLES "llama: build example programs" OFF)

# ggml
option(GGML_CUDA "enable cuda backend" ON)
option(GGML_METAL "enable metal backend" OFF)
# utils
option(LLAMA_BUILD_COMMON "llama: build common utils library" ON)

set(LLAMA_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp)

# llama.cpp and ggml
add_subdirectory(${LLAMA_ROOT_DIR})

set(LIBS_LIST)
list(APPEND LIBS_LIST llama common)
list(APPEND LIBS_LIST
    ggml
    ggml-base
    ggml-cpu
)
if(GGML_CUDA)
    list(APPEND LIBS_LIST ggml-cuda)
endif()

if(GGML_METAL)
    list(APPEND LIBS_LIST ggml-metal)
endif()

if(NOT LIBS_LIST STREQUAL "")
    set_target_properties(${LIBS_LIST} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )
endif()

# src
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

# examples
if (BUILD_EXAMPLES)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
else()
    message(STATUS "C++ examples are disabled")
endif()
