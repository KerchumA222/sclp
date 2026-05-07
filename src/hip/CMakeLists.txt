cmake_minimum_required(VERSION 3.20)
project(testmodule LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Aggressively strip LTO from all possible CMake flag variables globally
set(FLAGS_TO_STRIP "-flto" "-fno-fat-lto-objects")
set(FLAG_VARS 
    CMAKE_CXX_FLAGS 
    CMAKE_HIP_FLAGS 
    CMAKE_C_FLAGS 
    CMAKE_EXE_LINKER_FLAGS 
    CMAKE_SHARED_LINKER_FLAGS 
    CMAKE_MODULE_LINKER_FLAGS
)

foreach(flag_var ${FLAG_VARS})
    if(DEFINED ${flag_var})
        foreach(strip_flag ${FLAGS_TO_STRIP})
            string(REPLACE "${strip_flag}" "" ${flag_var} "${${flag_var}}")
        endforeach()
    endif()
endforeach()

find_package(pybind11 REQUIRED)
find_package(HIP REQUIRED)

include_directories(${HIP_INCLUDE_DIRS})

# Create a library for the HIP kernels
add_library(hip_kernels STATIC
    clipping.hip
    encoder.hip
    decoder.hip
    launcher.hip
)
set_target_properties(hip_kernels PROPERTIES HIP_STANDARD 17)

# Manually create the python module as a shared library
add_library(testmodule SHARED wrapper.cpp)

# Set the output name to be compatible with Python's import system
set_target_properties(testmodule PROPERTIES OUTPUT_NAME "testmodule.cpython-313-x86_64-linux-gnu" PREFIX "")

# Link all dependencies manually to ensure nothing is missed
target_link_libraries(testmodule PRIVATE 
    hip_kernels 
    hip::amdhip64 
    pybind11::module
)

# Post-creation cleanup for the target to kill LTO if pybind11 re-injects it
foreach(prop COMPILE_OPTIONS LINK_OPTIONS LINK_FLAGS)
    get_target_property(val testmodule ${prop})
    if(val)
        string(REPLACE "-flto" "" val "${val}")
        set_target_properties(testmodule PROPERTIES ${prop} "${val}")
    endif()
endforeach()

# Ensure the output goes to the correct directory
set_target_properties(testmodule PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python_pkg)