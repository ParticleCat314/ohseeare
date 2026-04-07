cmake_minimum_required(VERSION 3.14)
project(ohseeare VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# llama.cpp — required build dependency
#
# Point CMake to a llama.cpp source tree:
#   cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp
# Choose whether you want to build as build with cuda by passing -DLLAMA_CUBLAS=ON




set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp source tree")

if(NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp source tree not found.\n"
        "Please set LLAMA_CPP_DIR when configuring:\n"
        "  cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp\n")
endif()

# When llama.cpp is included as a subdirectory it sets LLAMA_STANDALONE=OFF,
# which disables the common/mtmd libraries we need.  Force them on.
set(LLAMA_BUILD_COMMON  ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)

# Only build the targets we actually link against.
add_subdirectory(${LLAMA_CPP_DIR} _deps/llama.cpp EXCLUDE_FROM_ALL)

# External packages and stuff
find_package(Threads REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# Default model paths (override with -DOHSEEARE_MODEL_PATH=… etc.)
set(OHSEEARE_MODEL_PATH  "${CMAKE_SOURCE_DIR}/models/glm-ocr/glm-ocr.gguf"
    CACHE FILEPATH "Path to the ohseeare GGUF model")
set(OHSEEARE_MMPROJ_PATH "${CMAKE_SOURCE_DIR}/models/glm-ocr/mmproj-glm-ocr.gguf"
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
    tiny-sco/src/sco.cpp
    tiny-sco/include/glad.c
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp
    imgui/backends/imgui_impl_opengl3.cpp)

target_include_directories(ohseeare-gui PRIVATE
    ${LLAMA_CPP_DIR}/include
    ${LLAMA_CPP_DIR}/common
    ${CMAKE_SOURCE_DIR}/imgui
    ${CMAKE_SOURCE_DIR}/imgui/backends
    ${CMAKE_SOURCE_DIR}/tiny-sco/src
    ${CMAKE_SOURCE_DIR}/tiny-sco/include)

target_compile_definitions(ohseeare-gui PRIVATE
    OHSEEARE_MODEL="${OHSEEARE_MODEL_PATH}"
    OHSEEARE_MMPROJ="${OHSEEARE_MMPROJ_PATH}"
    OHSEEARE_FONTS_DIR="${CMAKE_SOURCE_DIR}/fonts")

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
