# libmentalese-voice: qwentts.cpp + its ggml fork linked statically into one shared library that
# exports only the mx_voice_* functions. Configure with the GPU backend you want, e.g.
#   cmake -S engines/voice -B build/voice-cuda -DGGML_CUDA=ON
cmake_minimum_required(VERSION 3.21)
project(mentalese-voice C CXX)

set(QWENTTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/qwentts.cpp" CACHE PATH "qwentts.cpp source tree")
set(GGML_SOURCE_DIR "${QWENTTS_DIR}/ggml" CACHE PATH "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${QWENTTS_DIR} qwentts EXCLUDE_FROM_ALL)
# qwentts.cpp's CMake assumes it's the top-level project (CMAKE_SOURCE_DIR / CMAKE_BINARY_DIR)
target_include_directories(qwen-core PRIVATE ${QWENTTS_DIR}/src ${QWENTTS_DIR} ${CMAKE_BINARY_DIR}/qwentts)

add_library(mentalese-voice SHARED mx_voice.cpp)
target_link_libraries(mentalese-voice PRIVATE qwen-core)
target_include_directories(mentalese-voice PRIVATE ${QWENTTS_DIR}/src)
set_target_properties(mentalese-voice PROPERTIES
    CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # keep the static ggml symbols private so llama.cpp's ggml in the same process can't interpose
    target_link_options(mentalese-voice PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,-rpath,$ORIGIN)
endif()
