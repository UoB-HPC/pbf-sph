PBF-SPH
=======

This is a 3D real-time SPH mini-app implemented in OpenCL, OpenMP, SYCL, and SYCL2020.

## Building

Prerequisites:

* A SYCL compiler that supports USM and SYCL2020's simplified `parallel_for`
* Any version of GCC/Clang that supports C++17
* Cmake >= 3.14

This mini-app has only been tested on Linux.

We got two targets: `visualise` and `benchmark`. Target `visualise` builds a GUI and requires
further development packages as required by [Polyscope](https://polyscope.run/). Target `benchmark`
has no special dependencies, it creates a CLI application for benchmarking use.

First, configure your build based on the SYCL compiler you are using:

* For DPCPP:
  ```
  cmake -Bbuild -H. -DSYCL_COMPILER=DPCPP # dpcpp needs to be on path
  ```
* For hipSYCL:
  ```
  cmake -Bbuild -H. -DSYCL_COMPILER=HIPSYCL -DSYCL_COMPILER_DIR=<path_to_hipsycl>
  ```
* For ComputeCpp:
  ```
  cmake -Bbuild -H. -DSYCL_COMPILER=COMPUTECPP -DSYCL_COMPILER_DIR=<path_to_computecpp>
  ```

Proceed with building a target:

``` 
cmake --build build --target <visualise|benchmark> --config Release -j $(nproc)
```
