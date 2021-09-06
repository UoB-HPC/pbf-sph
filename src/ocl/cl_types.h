#pragma once

#ifndef __OPENCL_C_VERSION__

// not in kernel compiler, use host types
#include <cstddef>

// include CL host headers
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h>
#endif

// setup host struct alignments
#if defined(_MSC_VER)
#define ALIGNED_(x) __declspec(align(x))
#else
#define ALIGNED_(x) __attribute__((aligned(x)))
#endif

// name lookups
#ifndef __cplusplus
#define __device_constant const
#else
#define __device_constant constexpr
#endif

#define __device_uint cl_uint
#define __device_uchar cl_uchar
#define __device_float3 cl_float3
#define __device_double3 cl_float3
#define __device_uint3 cl_uint3

// non CL compiler
// typedef cl_float2 float2;
// typedef cl_float3 float3;
// typedef cl_float4 float4;
// typedef cl_int3 int3;
// typedef cl_int4 int4;
// typedef cl_uchar uchar;
// typedef cl_uchar4 uchar4;
// typedef cl_char3 char3;
// typedef cl_uint uint;
// typedef cl_uint2 uint2;
// typedef cl_uint3 uint3;
// typedef cl_uint4 uint4;
// typedef cl_ushort2 ushort2;

// OpenCL stubs
//#define M_PI_F 0f
//#define __global
//#define __local
//#define __kernel
//#define __constant
//#define __read_only
//#define __private

#else

// in kernel compiler, use device types

#define __device_constant constant
#define __device_uint uint
#define __device_uchar uchar
#define __device_float3 float3
#define __device_uint3 uint3


#define global global
#define local local

#define __private __private
#define kernel kernel
#define constant constant
#define read_only read_only

#define ALIGNED_(x) __attribute__((aligned(x)))

#endif
