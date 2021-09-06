include("SYCL.cmake")
include("FetchContent")

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(pbf-sph VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SYCL_COMPILER "" CACHE STRING
        "Compile using the specified SYCL compiler implementation
        Supported values are
           ONEAPI-DPCPP - dpc++ that is part of an oneAPI Base Toolkit distribution (https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit.html)
           DPCPP        - dpc++ as a standalone compiler (https://github.com/intel/llvm)
           HIPSYCL      - hipSYCL compiler (https://github.com/illuhad/hipSYCL)
           COMPUTECPP   - ComputeCpp compiler (https://developer.codeplay.com/products/computecpp/ce/home)")

set(SYCL_COMPILER_DIR "" CACHE STRING
        "Absolute path to the selected SYCL compiler directory, most are packaged differently so set the path according to `SYCL_COMPILER`:
           ONEAPI-DPCPP             - not required but `dpcpp` must be on PATH, load oneAPI as per documentation (i.e `source /opt/intel/oneapi/setvars.sh` first)
           HIPSYCL|DPCPP|COMPUTECPP - set to the root of the binary distribution that contains at least `bin/`, `include/`, and `lib/`")


FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 0.9.9.8
)

FetchContent_Declare(
        polyscope
        GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
        GIT_TAG v1.1.1
)


FetchContent_MakeAvailable(glm)
FetchContent_MakeAvailable(polyscope)


find_package(OpenMP)
find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)


if (SYCL_COMPILER)
    setup_sycl(SYCL_COMPILER SYCL_COMPILER_DIR SYCL_COMPILE_DEF SYCL_COMPILE_OPT SYCL_LINK_OPT)
endif ()

include_directories(include)
include_directories(src src/sycl src/ocl src/omp)

set(SYCL_SOURCES src/specialisation.cpp)
set(CPP_SOURCES src/ocl/oclsph.cpp src/args.cpp)

set(ALL_SOURCES ${SYCL_SOURCES} ${CPP_SOURCES}) # XXX: SYCL_SOURCES must come BEFORE normals ones
add_executable(visualise ${ALL_SOURCES} src/polyscope_extra.cpp src/visualise.cpp)
add_executable(benchmark ${ALL_SOURCES} src/benchmark.cpp)

target_include_directories(visualise PRIVATE ${glm_SOURCE_DIR} ${polyscope_SOURCE_DIR})

target_link_libraries(visualise PRIVATE
        Threads::Threads
        OpenCL::OpenCL
        polyscope
        atomic)

target_include_directories(benchmark PRIVATE ${glm_SOURCE_DIR})
target_link_libraries(benchmark PRIVATE
        Threads::Threads
        OpenCL::OpenCL
        atomic)

if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(visualise PRIVATE
            OpenMP::OpenMP_CXX)
    target_link_libraries(benchmark PRIVATE
            OpenMP::OpenMP_CXX)
    else()

    message(WARNING "Cannot find OpenMP, the resulting binary will run *very* slowly")
endif ()


if (SYCL_COMPILER)
    target_link_options(visualise PRIVATE ${SYCL_LINK_OPT})
    target_compile_options(visualise PUBLIC ${SYCL_COMPILE_OPT})
    target_compile_definitions(visualise PUBLIC ${SYCL_COMPILE_DEF})
    target_compile_definitions(visualise PUBLIC USE_SYCL)
    #    add_sycl_to_target(TARGET visualise SOURCES src/specialisation.cpp)
    setup_sycl_target(visualise SYCL_COMPILER src/specialisation.cpp)

    target_link_options(benchmark PRIVATE ${SYCL_LINK_OPT})
    target_compile_options(benchmark PUBLIC ${SYCL_COMPILE_OPT})
    target_compile_definitions(benchmark PUBLIC ${SYCL_COMPILE_DEF})
    target_compile_definitions(benchmark PUBLIC USE_SYCL)
    #    add_sycl_to_target(TARGET benchmark SOURCES src/benchmark.cpp)
    setup_sycl_target(benchmark SYCL_COMPILER src/benchmark.cpp)
endif ()

target_compile_definitions(visualise PUBLIC CL_TARGET_OPENCL_VERSION=120)
target_compile_definitions(benchmark PUBLIC CL_TARGET_OPENCL_VERSION=120)

if (MSVC)
    set(COMPILE_OPTIONS
            /Wp64
            /W3
            /Zc:wchar_t
            )
    set(RELEASE_OPTIONS /MT /GF /Gy /O2 /arch:AVX2 /fp:fast)
    set(DEBUG_OPTIONS /MTd)
else ()
    set(COMPILE_OPTIONS

            -Wall
            -Wextra
            #            -Wshadow
            #            -Wold-style-cast
            -Wcast-align
            #-Wpedantic #  polyscope has extra ";"
            -Werror=return-type
            -Wno-volatile # GLM uses volatile to elide optimisation
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-unused-variable
            -Wno-ignored-attributes
            -Wno-missing-field-initializers

            -Wno-inline-namespace-reopened-noninline # oneAPI's CL namespace is inline

            )
    set(SANATISE_OPTIONS -fsanitize=address -fsanitize=undefined)

    set(RELEASE_OPTIONS -Ofast -march=native -g)
    set(DEBUG_OPTIONS -O2 -fno-omit-frame-pointer -march=native ${SANATISE_OPTIONS})

endif ()
target_compile_options(visualise PUBLIC ${COMPILE_OPTIONS})
target_compile_options(visualise PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")
target_compile_options(visualise PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_link_options(visualise PUBLIC "$<$<CONFIG:DEBUG>:${SANATISE_OPTIONS}>")

target_compile_options(benchmark PUBLIC ${COMPILE_OPTIONS})
target_compile_options(benchmark PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")
target_compile_options(benchmark PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_link_options(benchmark PUBLIC "$<$<CONFIG:DEBUG>:${SANATISE_OPTIONS}>")
