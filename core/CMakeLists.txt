cmake_minimum_required(VERSION 3.16)
project(p6_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Core static library ──────────────────────────────────────────
add_library(p6_core STATIC
    src/orderbook.cpp
    src/staircase.cpp
    src/cup_flip.cpp
    src/spectral_force.cpp
    src/spoof_detection.cpp
    src/aggregator.cpp
    src/pipeline.cpp
)
target_include_directories(p6_core PUBLIC include)

# ── pybind11 Python module ────────────────────────────────────────
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(_p6_core src/bindings.cpp)
    target_link_libraries(_p6_core PRIVATE p6_core)
    # Install into the p6 package directory
    install(TARGETS _p6_core DESTINATION .)
else()
    message(WARNING "pybind11 not found — Python bindings will not be built. Install with: pip install pybind11")
endif()

# ── Tests ─────────────────────────────────────────────────────────
option(BUILD_TESTS "Build C++ tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_orderbook tests/test_orderbook.cpp)
    target_link_libraries(test_orderbook PRIVATE p6_core)
    add_test(NAME test_orderbook COMMAND test_orderbook)

    add_executable(test_pipeline tests/test_pipeline.cpp)
    target_link_libraries(test_pipeline PRIVATE p6_core)
    add_test(NAME test_pipeline COMMAND test_pipeline)
endif()
