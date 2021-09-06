#pragma once

//#define GLM_ENABLE_EXPERIMENTAL
#define CURVE_UINT3_TYPE glm::tvec3<size_t>
#define CURVE_UINT3_CTOR(x, y, z) (glm::tvec3<size_t>((x), (y), (z)))

#include <vector>

#include "clutils.hpp"
#include "curves.h"
#include "mc_constants.h"
#include "oclsph_type.h"
#include "sph.hpp"

//#define DEBUG

namespace sph::ocl_impl {

template <size_t N, typename T> using V = glm::vec<N, T>;

using utils::RO;
using utils::RW;
using utils::TypedBuffer;
using utils::WO;

using ClSphConfigStruct = cl::Buffer;
using ClMcConfigStruct = cl::Buffer;

using SphDiffuseKernel =
    cl::KernelFunctor<ClSphConfigStruct &, cl::Buffer &, cl::Buffer &, cl_uint, // zIdx, grid, gridN
                      cl::Buffer &,                                             // type
                      cl::Buffer &,                                             // pstar
                      cl::Buffer &,                                             // original colour
                      cl::Buffer &                                              // colour
                      >;

using SphLambdaKernel =                                    //
    cl::KernelFunctor<ClSphConfigStruct &,                 //
                      cl::Buffer &, cl::Buffer &, cl_uint, // zIdx, grid, gridN
                      cl::Buffer &,                        // type
                      cl::Buffer &,                        // pstar
                      cl::Buffer &,                        // mass
                      cl::Buffer &                         // lambda
                      >;

using SphDeltaKernel = cl::KernelFunctor<ClSphConfigStruct &, cl::Buffer &, cl::Buffer &, uint, // zIdx, grid, gridN
                                         cl::Buffer &,                                          // type
                                         cl::Buffer &,                                          // pstar
                                         cl::Buffer &,                                          // lambda
                                         cl::Buffer &,                                          // pos
                                         cl::Buffer &,                                          // vel
                                         cl::Buffer &                                           // deltap
                                         >;

using SphFinaliseKernel = cl::KernelFunctor<ClSphConfigStruct &,
                                            cl::Buffer &, // type
                                            cl::Buffer &, // pstar
                                            cl::Buffer &, // pos
                                            cl::Buffer &  // vel
                                            >;

using McLatticeKernel =
    cl::KernelFunctor<ClSphConfigStruct &, ClMcConfigStruct &, cl::Buffer &, cl_uint,
                      cl::Buffer &, // type
                      cl_float3, cl_uint3, cl_uint3, cl::Buffer &, cl::Buffer &, cl::Buffer &, cl::Buffer &>;

using McSizeKernel = cl::KernelFunctor<ClMcConfigStruct &, cl_uint3, cl::Buffer &, cl::LocalSpaceArg, cl::Buffer &>;

using McEvalKernel = cl::KernelFunctor<ClSphConfigStruct &, ClMcConfigStruct &, cl_float3, cl_uint3, cl::Buffer &,
                                       cl::Buffer &, cl::Buffer &, cl_uint, cl::Buffer &, cl::Buffer &, cl::Buffer &>;

namespace details {
struct ClSphAtoms {
  size_t size;
  std::vector<cl_uint> zIndex;
  std::vector<ClSphType> type;
  std::vector<float> mass;
  std::vector<cl_float4> colour;
  std::vector<cl_float3> pStar;
  std::vector<cl_float3> position;
  std::vector<cl_float3> velocity;

  explicit ClSphAtoms(size_t size)
      : size(size), zIndex(size), type(size), mass(size), colour(size), pStar(size), position(size), velocity(size) {}
};

struct PartiallyAdvected {
  cl_uint zIndex{};
  cl_float3 pStar{};
  sph::Particle<size_t, float, V> particle;
  PartiallyAdvected() = default;
  explicit PartiallyAdvected(cl_uint zIndex, cl_float3 pStar, const sph::Particle<size_t, float, V> &particle)
      : zIndex(zIndex), pStar(pStar), particle(particle) {}
};
} // namespace details

using namespace details;
class Solver final : public sph::Solver<size_t, float, V> {

private:
  const float h;
  const cl::Device device;
  const cl::Context ctx;
  const cl::Program clsph;

  cl::CommandQueue queue;

  SphDiffuseKernel sphDiffuseKernel;
  SphLambdaKernel sphLambdaKernel;
  SphDeltaKernel sphDeltaKernel;
  SphFinaliseKernel sphFinaliseKernel;
  McLatticeKernel mcLatticeKernel;
  McSizeKernel mcSizeKernel;
  McEvalKernel mcEvalKernel;

public:
  static inline std::string to_string(V<3, size_t> v) {
    return "(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + ")";
  }

  constexpr ClSphType resolveType(sph::Type t) {
    switch (t) {
    case sph::Type::Fluid:
      return ClSphType::Fluid;
    case sph::Type::Obstacle:
      return ClSphType::Obstacle;
    default:
      throw std::logic_error("unhandled branch (fluid::Type->ClSphType)");
    }
  }

  constexpr sph::Type resolveType(ClSphType t) {
    switch (t) {
    case ClSphType::Fluid:
      return sph::Type::Fluid;
    case ClSphType::Obstacle:
      return sph::Type::Obstacle;
    default:
      throw std::logic_error("unhandled branch (fluid::ClSphType->Type)");
    }
  }

  inline void finishQueue() { queue.finish(); }

  explicit Solver(float h,
                  const std::string &kernelPath,            //
                  const std::vector<std::string> &includes, //
                  const cl::Device &device);

  void checkSize();
  std::vector<PartiallyAdvected> advectAndCopy(const sph::SphParams<size_t, float, V> &config,
                                               const sph::Scene<size_t, float, V> &scene,
                                               std::vector<sph::Particle<size_t, float, V>> &xs);
  [[nodiscard]] size_t zCurveGridIndexAtCoordAt(float x, float y, float z) const;

  std::tuple<ClSphConfig, V<3, float>, V<3, size_t>>
  computeBoundAndZindex(const sph::SphParams<size_t, float, V> &config,
                        std::vector<PartiallyAdvected> &advection) const;

  sph::ColouredMesh<float, V> runMcKernels(          //
      sph::utils::Stopwatch &watch,                  //
      V<3, size_t> sampleSize,                       //
      TypedBuffer<ClSphConfig, RO> &sphConfig,       //
      TypedBuffer<ClMcConfig, RO> &mcConfig,         //
      TypedBuffer<cl_uint, RO> &gridTable,           //
      TypedBuffer<ClSphType, RO> type,               //
      TypedBuffer<cl_float3, RW> &particlePositions, //
      TypedBuffer<cl_float4, RW> &particleColours,   //
      V<3, float> minExtent, V<3, size_t> extent);

  void runSphKernel(sph::utils::Stopwatch &watch,
                    size_t iterations,                       //
                    TypedBuffer<ClSphConfig, RO> &sphConfig, //
                    TypedBuffer<uint, RO> &gridTable,        //
                    TypedBuffer<uint, RO> &zIndex,           //
                    TypedBuffer<ClSphType, RO> type,         //
                    TypedBuffer<float, RO> &mass,            //
                    TypedBuffer<cl_float4, RO> &colour,      //
                    TypedBuffer<cl_float3, RW> &pStar,       //
                    TypedBuffer<cl_float3, RW> &deltaP,      //
                    TypedBuffer<float, RW> &lambda,          //
                    TypedBuffer<cl_float4, RW> &diffused,    //
                    TypedBuffer<cl_float3, RW> &position,    //
                    TypedBuffer<cl_float3, RW> &velocity);   //

  sph::Result<size_t, float, V> advance(              //
      const sph::SphParams<size_t, float, V> &config, //
      const sph::Scene<size_t, float, V> &scene,      //
      std::vector<sph::Particle<size_t, float, V>> &xs) final;
};

} // namespace sph::ocl_impl
