cmake_minimum_required(VERSION 3.14)
project(ohseeare VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use ccache if available to speed up rebuilds
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/vendor/llama.cpp")

# Only build the libraries we link against; skip tools, tests, server, examples
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

option(GGML_CUDA   "Enable CUDA backend (slow to compile)"   ON)
option(GGML_VULKAN "Enable Vulkan backend"                   OFF)
option(GGML_METAL  "Enable Metal backend (macOS)"            OFF)

set(GGML_BLAS OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/llama.cpp EXCLUDE_FROM_ALL)
add_subdirectory(vendor/glfw)

# External packages and stuff
find_package(Threads REQUIRED)

set(OHSEEARE_MODEL_PATH  "${CMAKE_SOURCE_DIR}/models/GLM-OCR-f16.gguf"
    CACHE FILEPATH "Path to the ohseeare GGUF model")
set(OHSEEARE_MMPROJ_PATH "${CMAKE_SOURCE_DIR}/models/mmproj-GLM-OCR-Q8_0.gguf"
    CACHE FILEPATH "Path to the ohseeare vision-projector GGUF")

# ohseeare CLI OCR binary (links against llama.cpp)
add_executable(ohseeare src/ohseeare.cpp)
target_include_directories(ohseeare PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common)
target_link_libraries(ohseeare PRIVATE common mtmd Threads::Threads)
target_compile_features(ohseeare PRIVATE cxx_std_17)

# ohseeare-gui ImGui GUI with built-in screen capture
add_executable(ohseeare-gui
    src/ohseeare-gui.cpp
    src/tinyfiledialogs.c
    vendor/glad.c
    vendor/imgui/imgui.cpp
    vendor/imgui/imgui_draw.cpp
    vendor/imgui/imgui_tables.cpp
    vendor/imgui/imgui_widgets.cpp
    vendor/imgui/backends/imgui_impl_glfw.cpp
    vendor/imgui/backends/imgui_impl_opengl3.cpp)

target_include_directories(ohseeare-gui PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${CMAKE_SOURCE_DIR}/vendor/imgui
    ${CMAKE_SOURCE_DIR}/vendor/imgui/backends
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/fonts)

target_compile_definitions(ohseeare-gui PRIVATE
    OHSEEARE_MODEL="${OHSEEARE_MODEL_PATH}"
    OHSEEARE_MMPROJ="${OHSEEARE_MMPROJ_PATH}")

target_link_libraries(ohseeare-gui PRIVATE
    common
    mtmd
    Threads::Threads
    glfw
    ${CMAKE_DL_LIBS})

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(ohseeare-gui PRIVATE X11::X11)
endif()

target_compile_features(ohseeare-gui PRIVATE cxx_std_17)
