# libmentalese-llm: llama.cpp + mtmd + ggml linked statically into one shared library that
# exports only the mx_llm_* functions. Configure with the GPU backend you want, e.g.
#   cmake -S engines/llm -B build/llm-cuda -DGGML_CUDA=ON
cmake_minimum_required(VERSION 3.21)
project(mentalese-llm C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/llama.cpp" CACHE PATH "llama.cpp source tree")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(opt LLAMA_BUILD_COMMON LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER
            LLAMA_BUILD_APP LLAMA_OPENSSL LLAMA_SUBPROCESS)
    set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()

add_subdirectory(${LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)
add_subdirectory(${LLAMA_DIR}/tools/mtmd mtmd EXCLUDE_FROM_ALL)

add_library(mentalese-llm SHARED mx_llm.cpp)
target_link_libraries(mentalese-llm PRIVATE llama mtmd)
target_include_directories(mentalese-llm PRIVATE ${LLAMA_DIR}/tools/mtmd)
set_target_properties(mentalese-llm PROPERTIES
    CXX_STANDARD 17
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # keep the static ggml/llama symbols private so another ggml in the process can't interpose
    target_link_options(mentalese-llm PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined -Wl,-rpath,$ORIGIN)
endif()
