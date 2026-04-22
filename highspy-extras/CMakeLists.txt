# CMakeLists.txt for highspy-extras wheel
# The wheel only wraps the standalone highs_extras shared library.
# Use the dedicated extras-library mode so the top-level build adds the
# highs_extras target and installs it into the highspy_extras package.

cmake_minimum_required(VERSION 3.15...3.27)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
include(vcpkg) # vcpkg must be configured BEFORE project()

project(highs_extras LANGUAGES C CXX)

add_subdirectory(
    "${CMAKE_CURRENT_LIST_DIR}/../extern"
    "${CMAKE_CURRENT_BINARY_DIR}/extern")

install(TARGETS highs_extras
    RUNTIME DESTINATION highspy_extras
    LIBRARY DESTINATION highspy_extras
)
