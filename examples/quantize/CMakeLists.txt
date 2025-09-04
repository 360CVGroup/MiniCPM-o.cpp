set(TARGET quantize)

add_executable(${TARGET}
    quantize.cpp
    )

install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE .. ../../include)
target_link_libraries(${TARGET} PRIVATE edge ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
