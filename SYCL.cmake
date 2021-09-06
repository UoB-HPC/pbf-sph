macro(setup_sycl SYCL_COMPILER SYCL_COMPILER_DIR COMPILE_DEF COMPILE_OPT LINK_OPT)

    if (${SYCL_COMPILER} STREQUAL "HIPSYCL")


        set(hipSYCL_DIR ${${SYCL_COMPILER_DIR}}/lib/cmake/hipSYCL)

        if (NOT EXISTS "${hipSYCL_DIR}")
            message(WARNING "Falling back to hipSYCL < 0.9.0 CMake structure")
            set(hipSYCL_DIR ${${SYCL_COMPILER_DIR}}/lib/cmake)
        endif ()
        if (NOT EXISTS "${hipSYCL_DIR}")
            message(FATAL_ERROR "Can't find the appropriate CMake definitions for hipSYCL")
        endif ()

        find_package(hipSYCL CONFIG REQUIRED)
    elseif (${SYCL_COMPILER} STREQUAL "COMPUTECPP")

        list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
        set(ComputeCpp_DIR ${${SYCL_COMPILER_DIR}})

        # don't point to the CL dir as the imports already have the CL prefix
        set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/CL")

        list(APPEND ${COMPILE_DEF} CL_TARGET_OPENCL_VERSION=220 _GLIBCXX_USE_CXX11_ABI=1)
        # ComputeCpp needs OpenCL
        set(SYCL_LANGUAGE_VERSION "2020")

        find_package(ComputeCpp REQUIRED)

        # this must come after FindComputeCpp (!)
        set(COMPUTECPP_USER_FLAGS -O3 -no-serial-memop)
        list(APPEND ${COMPILE_DEF} SYCL_LANGUAGE_VERSION=2020)

    elseif (${SYCL_COMPILER} STREQUAL "DPCPP")
        set(CMAKE_CXX_COMPILER ${${SYCL_COMPILER_DIR}}/bin/clang++)
        include_directories(${${SYCL_COMPILER_DIR}}/include/sycl)
        list(APPEND ${COMPILE_DEF} CL_TARGET_OPENCL_VERSION=220)
        list(APPEND ${COMPILE_OPT} -fsycl)
        list(APPEND ${LINK_OPT} -fsycl)
    elseif (${SYCL_COMPILER} STREQUAL "ONEAPI-DPCPP")
        # XXX the trick here is to set CMAKE_CXX_COMPILER *after* we got everything else setup
        # that way, libraries that don't work well with icpx will be compiled with the default compiler
        set(CMAKE_CXX_COMPILER icpx)
        list(APPEND ${COMPILE_DEF} CL_TARGET_OPENCL_VERSION=220)
        list(APPEND ${COMPILE_OPT} -fsycl -fiopenmp)
        list(APPEND ${LINK_OPT} -fsycl -fiopenmp)
    else ()
        message(FATAL_ERROR "SYCL_COMPILER=${${SYCL_COMPILER}} is unsupported")
    endif ()

endmacro()

macro(setup_sycl_target TARGET SYCL_COMPILER IMPL_SOURCES)
    if (
    (${SYCL_COMPILER} STREQUAL "COMPUTECPP") OR
    (${SYCL_COMPILER} STREQUAL "HIPSYCL"))
        # so ComputeCpp and hipSYCL has this weird CMake usage where they append their
        # own custom integration header flags AFTER the target has been specified
        # hence this macro here
        add_sycl_to_target(
                TARGET ${TARGET}
                SOURCES ${IMPL_SOURCES})
    endif ()
endmacro()