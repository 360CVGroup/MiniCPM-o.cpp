llama_add_compile_flags()

#
# libraries
#

# edge

add_library(edge
            ${LLAMA_PUBLIC_HEADERS}
            siglip.cpp
            utils.cpp
            whisper_encoder.cpp
            minicpmo.cpp
            outetts.cpp
            ${LLAMA_ROOT_DIR}/src/llama-model.h  # for tok_embd
            )

target_include_directories(edge PUBLIC ../include ${LLAMA_ROOT_DIR})
target_compile_features   (edge PUBLIC cxx_std_17) # don't bump


if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(edge PUBLIC -ffast-math)
endif()

if (USE_OPENMP)
    set(GGML_USE_OPENMP ON)
    target_compile_definitions(edge PUBLIC USE_OPENMP)
    target_compile_options(edge PUBLIC -fopenmp -march=native -O3)
endif()

target_link_libraries(edge PUBLIC ${LIBS_LIST})

if (BUILD_SHARED_LIBS)
    set_target_properties(edge PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(edge PRIVATE LLAMA_BUILD)
    target_compile_definitions(edge PUBLIC  LLAMA_SHARED)
endif()

if (NOT MSVC)
    target_compile_options(edge PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if (USE_PYBINDING)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(
        _C
        MODULE
        minicpmo_cpp/csrc/pybind.cpp
        minicpmo_cpp/csrc/minicpmo_iface.hpp
        WITH_SOABI
    )

    target_include_directories(_C PUBLIC ../include ${LLAMA_ROOT_DIR})
    target_link_libraries(_C PRIVATE edge ${LIBS_LIST} pybind11::headers)
    target_compile_features(_C PUBLIC cxx_std_17) # don't bump

    install(
        TARGETS ${LIBS_LIST} edge _C
        LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/minicpmo_cpp
        RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}/minicpmo_cpp
        ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}/minicpmo_cpp
        FRAMEWORK DESTINATION ${SKBUILD_PLATLIB_DIR}/minicpmo_cpp
        RESOURCE DESTINATION ${SKBUILD_PLATLIB_DIR}/minicpmo_cpp
    )
    set_target_properties(_C edge ${LIBS_LIST} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )

else()
    message(STATUS "pybind11 is disabled")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()
