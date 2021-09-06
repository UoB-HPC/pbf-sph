#pragma once
//#ifdef __SYCL_DEVICE_ONLY__
//#undef __SYCL_DEVICE_ONLY__
//#endif
#define __SYCL_DISABLE_NAMESPACE_INLINE__

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <type_traits>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/opencl.hpp"
#include "cl_types.h"
#include "glm/glm.hpp"
#include "utils.hpp"

namespace sph::ocl_impl::utils {

// static std::vector<cl::Device> findDeviceWithSignature(const std::vector<std::string> &needles) {
//   std::vector<cl::Platform> platforms;
//   cl::Platform::get(&platforms);
//   std::vector<cl::Device> matching;
//   for (auto &p : platforms) {
//     std::vector<cl::Device> devices;
//     try {
//       p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
//     } catch (const std::exception &e) {
//       std::cerr << "Enumeration failed at `" << p.getInfo<CL_PLATFORM_NAME>() << "` : " << e.what() << std::endl;
//     }
//     std::copy_if(devices.begin(), devices.end(), std::back_inserter(matching), [needles](const cl::Device &device) {
//       return std::any_of(needles.begin(), needles.end(), [&device](auto needle) {
//         return device.getInfo<CL_DEVICE_NAME>().find(needle) != std::string::npos;
//       });
//     });
//   }
//   return matching;
// }
//
// static cl::Device resolveDeviceVerbose(const std::vector<std::string> &signatures) {
//   const auto imploded = sph::utils::mk_string<std::string>(
//       signatures, [](auto x) { return x; }, ",");
//   auto found = sph::ocl_impl::utils::findDeviceWithSignature({
//       signatures,
//   });
//
//   if (found.empty()) {
//     throw std::runtime_error("No CL device found with signature:`" + imploded + "`");
//   }
//
//   std::cout << "Matching devices(" << found.size() << "):" << std::endl;
//   for (const auto &d : found)
//     std::cout << "\t" << d.getInfo<CL_DEVICE_NAME>() << std::endl;
//
//   if (found.size() > 1) {
//     std::cout << "Found more than one device signature:`" << imploded
//               << "`"
//                  ", using the first one."
//               << std::endl;
//   }
//
//   return found.front();
// }

static std::string showDevice(const cl::Device &device) { return std::string(device.getInfo<CL_DEVICE_NAME>()); }

static std::vector<std::pair<size_t, cl::Device>> listDevices() {
  std::vector<std::pair<size_t, cl::Device>> collected;
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (auto &p : platforms) {
    try {
      std::vector<cl::Device> devices;
      p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for (const auto &d : devices)
        collected.emplace_back(collected.size(), d);
    } catch (const std::exception &e) {
      std::cerr << "Enumeration failed at `" << p.getInfo<CL_PLATFORM_NAME>() << "` : " << e.what() << std::endl;
    }
  }
  return collected;
}

static void enumeratePlatform(std::ostream &out, std::ostream &err) {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  auto platform = cl::Platform::getDefault();
  for (auto &p : platforms) {
    try {
      out << "\t├─┬Platform" << (platform == p ? "(Default):" : ":") << p.getInfo<CL_PLATFORM_NAME>()
          << "\n\t│ ├Vendor     : " << p.getInfo<CL_PLATFORM_VENDOR>()     //
          << "\n\t│ ├Version    : " << p.getInfo<CL_PLATFORM_VERSION>()    //
          << "\n\t│ ├Profile    : " << p.getInfo<CL_PLATFORM_PROFILE>()    //
          << "\n\t│ ├Extensions : " << p.getInfo<CL_PLATFORM_EXTENSIONS>() //
          << "\n\t│ └Devices" << std::endl;
      std::vector<cl::Device> devices;
      p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for (auto &d : devices) {
        out << "\t│\t     └┬Name    : " << d.getInfo<CL_DEVICE_NAME>()                //
            << "\n\t│\t      ├Type    : " << d.getInfo<CL_DEVICE_TYPE>()              //
            << "\n\t│\t      ├Vendor  : " << d.getInfo<CL_DEVICE_VENDOR_ID>()         //
            << "\n\t│\t      ├Avail.  : " << d.getInfo<CL_DEVICE_AVAILABLE>()         //
            << "\n\t│\t      ├Max CU. : " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() //
            << "\n\t│\t      └Version : " << d.getInfo<CL_DEVICE_VERSION>() << std::endl;
      }
    } catch (const std::exception &e) {
      err << "Enumeration failed at `" << p.getInfo<CL_PLATFORM_NAME>() << "` : " << e.what() << std::endl;
    }
  }
}

static cl::Program loadProgramFromFile(const cl::Context &context, const std::string &file,
                                       const std::vector<std::string> &includes,
                                       const std::vector<std::string> &flags = {}) {
  std::cout << "Compiling CL kernel:`" << file << "` using " << std::endl;

  std::ifstream t(file);
  if (!t.good()) throw std::runtime_error("Unable to read file:`" + file + "`");

  for (const auto &incl : includes) {
    if (!std::filesystem::is_directory(incl)) throw std::runtime_error("Unable to stat include dir:`" + incl + "`");
  }

  std::stringstream source;
  source << t.rdbuf();
  cl::Program program = cl::Program(context, source.str());

  const auto printBuildInfo = [&program]() {
    auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
    std::cerr << "Compiler output(" << log.size() << "):\n" << std::endl;
    for (auto &pair : log) {
      std::cerr << ">" << pair.second << std::endl;
    }
  };
  const std::string clFlags = " -cl-std=CL1.2"
                              " -w"
                              " -cl-mad-enable"
                              " -cl-no-signed-zeros"
                              " -cl-unsafe-math-optimizations"
                              " -cl-finite-math-only"
                              " -g";

  auto includeLine = sph::utils::mk_string<std::string>(
      includes, [](const auto &f) { return "-I" + f; }, " ");
  auto flagLine = sph::utils::mk_string(flags, " ");

  const std::string build = clFlags + " " + includeLine + " " + flagLine;
  std::cout << "Using args:`" << build << "`" << std::endl;
  try {
    program.build(build.c_str());
  } catch (...) {
    std::cerr << "Program failed to compile, source:\n" << source.str() << std::endl;
    printBuildInfo();
    throw;
  }
  std::cout << "Program compiled" << std::endl;
  printBuildInfo();
  return program;
}

template <typename N, typename T> static inline T gen_type3(N x, N y, N z) {
  return {{static_cast<N>(x), static_cast<N>(y), static_cast<N>(z)}};
}

template <typename N> static inline cl_float3 float3(N x, N y, N z) {
  return {{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)}};
}

inline cl_float4 float4(float x, float y, float z, float w) { return {{x, y, z, w}}; }

template <typename N> static inline cl_float3 float3(N x) {
  return {{static_cast<float>(x), static_cast<float>(x), static_cast<float>(x)}};
}

inline cl_float4 float4(float x) { return {{x, x, x, x}}; }

inline cl_uchar4 uchar4(cl_uchar x, cl_uchar y, cl_uchar z, cl_uchar w) { return {{x, y, z, w}}; }

inline uint32_t packARGB(cl_uchar4 argb) {
  return ((argb.s[0] & 0xFF) << 24) | ((argb.s[1] & 0xFF) << 16) | ((argb.s[2] & 0xFF) << 8) |
         ((argb.s[3] & 0xFF) << 0);
}

template <typename N> inline uint32_t packARGB(glm::tvec4<N> argb) {
  return ((static_cast<uint8_t>(argb.x) & 0xFF) << 24) | ((static_cast<uint8_t>(argb.y) & 0xFF) << 16) |
         ((static_cast<uint8_t>(argb.z) & 0xFF) << 8) | ((static_cast<uint8_t>(argb.w) & 0xFF) << 0);
}

inline cl_uchar4 unpackARGB(uint32_t argb) {
  return uchar4((argb >> 24) & 0xFF, (argb >> 16) & 0xFF, (argb >> 8) & 0xFF, (argb >> 0) & 0xFF);
}

template <typename N> inline glm::tvec4<N> unpackARGB(uint32_t argb) {
  return glm::tvec4<N>((argb >> 24) & 0xFF, (argb >> 16) & 0xFF, (argb >> 8) & 0xFF, (argb >> 0) & 0xFF);
}

inline glm::tvec4<uint32_t> v4u32ToGlm(cl_uint4 v) { return glm::tvec4<uint32_t>(v.s[0], v.s[1], v.s[2], v.s[3]); }

template <typename N> inline glm::tvec3<N> clToVec3(cl_float3 v) { return glm::tvec3<float>(v.s[0], v.s[1], v.s[2]); }
template <typename N> inline glm::tvec4<N> clToVec4(cl_uchar4 v) {
  return glm::tvec4<uint8_t>(v.s[0], v.s[1], v.s[2], v.s[3]);
}
template <typename N> inline glm::tvec4<N> clToVec4(cl_float4 v) {
  return glm::tvec4<float>(v.s[0], v.s[1], v.s[2], v.s[3]);
}

inline cl_float3 vec3ToCl(glm::tvec3<float> v) { return float3(v.x, v.y, v.z); }
inline cl_float4 vec4ToCl(glm::tvec4<float> v) { return float4(v.x, v.y, v.z, v.w); }
inline cl_uchar4 vec4ToCl(glm::tvec4<uint8_t> v) { return cl_uchar4{{v.x, v.y, v.z, v.w}}; }
inline cl_uint3 uvec3ToCl(glm::tvec3<size_t> v) { return gen_type3<cl_uint, cl_uint3>(v.x, v.y, v.z); }

enum BufferType {
  RW = CL_MEM_READ_WRITE,
  RO = CL_MEM_READ_ONLY,
  WO = CL_MEM_WRITE_ONLY,
};

#define NEW_ERROR_TYPE(ERR)                                                                                            \
  { ERR, #ERR }

struct ClErrorType {
  cl_int code;
  const char *name;
};

const static ClErrorType CL_ERROR_LUT[63] = {
    NEW_ERROR_TYPE(CL_SUCCESS),
    NEW_ERROR_TYPE(CL_DEVICE_NOT_FOUND),
    NEW_ERROR_TYPE(CL_DEVICE_NOT_AVAILABLE),
    NEW_ERROR_TYPE(CL_COMPILER_NOT_AVAILABLE),
    NEW_ERROR_TYPE(CL_MEM_OBJECT_ALLOCATION_FAILURE),
    NEW_ERROR_TYPE(CL_OUT_OF_RESOURCES),
    NEW_ERROR_TYPE(CL_OUT_OF_HOST_MEMORY),
    NEW_ERROR_TYPE(CL_PROFILING_INFO_NOT_AVAILABLE),
    NEW_ERROR_TYPE(CL_MEM_COPY_OVERLAP),
    NEW_ERROR_TYPE(CL_IMAGE_FORMAT_MISMATCH),
    NEW_ERROR_TYPE(CL_IMAGE_FORMAT_NOT_SUPPORTED),
    NEW_ERROR_TYPE(CL_BUILD_PROGRAM_FAILURE),
    NEW_ERROR_TYPE(CL_MAP_FAILURE),
    NEW_ERROR_TYPE(CL_MISALIGNED_SUB_BUFFER_OFFSET),
    NEW_ERROR_TYPE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST),
    NEW_ERROR_TYPE(CL_COMPILE_PROGRAM_FAILURE),
    NEW_ERROR_TYPE(CL_LINKER_NOT_AVAILABLE),
    NEW_ERROR_TYPE(CL_LINK_PROGRAM_FAILURE),
    NEW_ERROR_TYPE(CL_DEVICE_PARTITION_FAILED),
    NEW_ERROR_TYPE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE),
    NEW_ERROR_TYPE(CL_INVALID_VALUE),
    NEW_ERROR_TYPE(CL_INVALID_DEVICE_TYPE),
    NEW_ERROR_TYPE(CL_INVALID_PLATFORM),
    NEW_ERROR_TYPE(CL_INVALID_DEVICE),
    NEW_ERROR_TYPE(CL_INVALID_CONTEXT),
    NEW_ERROR_TYPE(CL_INVALID_QUEUE_PROPERTIES),
    NEW_ERROR_TYPE(CL_INVALID_COMMAND_QUEUE),
    NEW_ERROR_TYPE(CL_INVALID_HOST_PTR),
    NEW_ERROR_TYPE(CL_INVALID_MEM_OBJECT),
    NEW_ERROR_TYPE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR),
    NEW_ERROR_TYPE(CL_INVALID_IMAGE_SIZE),
    NEW_ERROR_TYPE(CL_INVALID_SAMPLER),
    NEW_ERROR_TYPE(CL_INVALID_BINARY),
    NEW_ERROR_TYPE(CL_INVALID_BUILD_OPTIONS),
    NEW_ERROR_TYPE(CL_INVALID_PROGRAM),
    NEW_ERROR_TYPE(CL_INVALID_PROGRAM_EXECUTABLE),
    NEW_ERROR_TYPE(CL_INVALID_KERNEL_NAME),
    NEW_ERROR_TYPE(CL_INVALID_KERNEL_DEFINITION),
    NEW_ERROR_TYPE(CL_INVALID_KERNEL),
    NEW_ERROR_TYPE(CL_INVALID_ARG_INDEX),
    NEW_ERROR_TYPE(CL_INVALID_ARG_VALUE),
    NEW_ERROR_TYPE(CL_INVALID_ARG_SIZE),
    NEW_ERROR_TYPE(CL_INVALID_KERNEL_ARGS),
    NEW_ERROR_TYPE(CL_INVALID_WORK_DIMENSION),
    NEW_ERROR_TYPE(CL_INVALID_WORK_GROUP_SIZE),
    NEW_ERROR_TYPE(CL_INVALID_WORK_ITEM_SIZE),
    NEW_ERROR_TYPE(CL_INVALID_GLOBAL_OFFSET),
    NEW_ERROR_TYPE(CL_INVALID_EVENT_WAIT_LIST),
    NEW_ERROR_TYPE(CL_INVALID_EVENT),
    NEW_ERROR_TYPE(CL_INVALID_OPERATION),
    NEW_ERROR_TYPE(CL_INVALID_GL_OBJECT),
    NEW_ERROR_TYPE(CL_INVALID_BUFFER_SIZE),
    NEW_ERROR_TYPE(CL_INVALID_MIP_LEVEL),
    NEW_ERROR_TYPE(CL_INVALID_GLOBAL_WORK_SIZE),
    NEW_ERROR_TYPE(CL_INVALID_PROPERTY),
    NEW_ERROR_TYPE(CL_INVALID_IMAGE_DESCRIPTOR),
    NEW_ERROR_TYPE(CL_INVALID_COMPILER_OPTIONS),
    NEW_ERROR_TYPE(CL_INVALID_LINKER_OPTIONS),
    NEW_ERROR_TYPE(CL_INVALID_DEVICE_PARTITION_COUNT),
    // NEW_ERROR_TYPE(CL_INVALID_PIPE_SIZE),
    // NEW_ERROR_TYPE(CL_INVALID_DEVICE_QUEUE),
    // NEW_ERROR_TYPE(CL_INVALID_SPEC_ID),
    // NEW_ERROR_TYPE(CL_MAX_SIZE_RESTRICTION_EXCEEDED),
};

static const char *clResolveError(const cl_int error) {
  for (size_t i = 0; i < 63; ++i) {
    if (CL_ERROR_LUT[i].code == error) return CL_ERROR_LUT[i].name;
  }
  return "<Unknown>";
}

template <typename T, BufferType B> class TypedBuffer {

public:
  cl::Buffer actual;
  const size_t length;

private:
  // XXX last parameter is there to disambiguate cases where T = size_t
  TypedBuffer(const cl::Context &context, T &t, int)
      : actual(cl::Buffer(context, B | CL_MEM_COPY_HOST_PTR, sizeof(T), &t)), length(1) {}

public:
  TypedBuffer(const cl::Context &context, size_t count)
      : actual(cl::Buffer(context, B, sizeof(T) * count)), length(count) {}

  template <typename Iterable>
  TypedBuffer(const cl::CommandQueue &queue, Iterable &xs, bool useHostPtr = false)
      : actual(cl::Buffer(queue, xs.begin(), xs.end(), B == BufferType::RO, useHostPtr)), length(xs.size()) {
    static_assert(B != BufferType::WO, "BufferType must be RW or RO");
  }

  static TypedBuffer<T, B> ofStruct(const cl::Context &context, T &t) { return TypedBuffer<T, B>(context, t, 0); }

  //		template<typename IteratorType>
  //		void drainTo(const cl::CommandQueue &queue,
  //		             IteratorType startIterator, IteratorType endIterator) {
  //			cl::copy(queue, actual, startIterator, endIterator);
  //		}

  //		template<typename Iterable, typename = typename std::enable_if<
  //				std::is_same<
  //						typename std::iterator_traits<Iterable>::value_type,
  //						T>::value
  //		>>
  //		void drainTo(const cl::CommandQueue &queue, Iterable xs) {
  //			cl::copy(queue, actual, xs.begin(), xs.end());
  //		}

  template <typename Iterable> inline void drainTo(const cl::CommandQueue &queue, Iterable &xs) {
    cl::copy(queue, actual, xs.begin(), xs.end());
  }
};

} // namespace sph::ocl_impl::utils
