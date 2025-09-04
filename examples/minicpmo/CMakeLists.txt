set(TARGET minicpmo-cli)

# from Kitware/VTK
set(FFMPEG_FIND_COMPONENTS "avcodec;avformat;swscale;swresample;avutil")
include(../../cmake/FindFFMPEG.cmake)

if(MACOS_STREAM)
    # 设置PortAudio的路径
    set(PORTAUDIO_ROOT "/opt/homebrew/Cellar/portaudio/19.7.0")
    set(PORTAUDIO_INCLUDE_DIR "${PORTAUDIO_ROOT}/include")
    set(PORTAUDIO_LIBRARY "${PORTAUDIO_ROOT}/lib/libportaudio.dylib")
    # 查找OpenCV包
    find_package(OpenCV REQUIRED)
    add_executable(${TARGET}
        cli_macos_stream.cpp
        ../common.cpp
        ../video_decoder.cpp
        ../macos_stream.cpp
        )
    # 确保MACOS_STREAM宏被传递给编译器
    if (MACOS_STREAM)
        target_compile_definitions(${TARGET} PRIVATE MACOS_STREAM)
    endif()
else()
    add_executable(${TARGET}
        cli.cpp
        ../common.cpp
        ../video_decoder.cpp
        )
endif()


install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE .. ../../include ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE edge ${CMAKE_THREAD_LIBS_INIT} ${FFMPEG_LIBRARIES})

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(MACOS_STREAM)
    target_include_directories(${TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS} ${PORTAUDIO_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${OpenCV_LIBS} ${PORTAUDIO_LIBRARY})
endif()

add_executable(bench
    bench.cpp
    ../common.cpp
    ../video_decoder.cpp
    )
target_include_directories(bench PRIVATE .. ../../include ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(bench PRIVATE edge ${CMAKE_THREAD_LIBS_INIT} ${FFMPEG_LIBRARIES})
target_compile_features(bench PRIVATE cxx_std_17)
