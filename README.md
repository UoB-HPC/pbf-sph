PBF-SPH
=======

This is 3D read-time SPH mini-app implemented in OpenCL, OpenMP, SYCL, and SYCL2020.

## Building

Prerequisites:

* A SYCL compiler that supports USM and SYCL2020's simplified `parallel_for`
* Any version of GCC/Clang that supports C++17
* Cmake >= 3.14

This mini-app has only been tested on Linux.

We got two targets: `visualise` and `benchmark`. Target `visualise` builds a GUI and requires
further development packages as required by [Polyscope](https://polyscope.run/). Target `benchmark`
has no special dependencies, it creates a CLI application for benchmarking use.

 * For DPCPP:
   `cmake -Bbuild -H. -DSYCL_COMPILER=COMPUTECPP`
 * For hipSYCL:
    `cmake -Bbuild -H. -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR=<path_to_hipsycl>`
 * For ComputeCpp:
   `cmake -Bbuild -H. -DSYCL_COMPILER=COMPUTECPP -DSYCL_COMPILER_DIR=<path_to_computecpp>`
