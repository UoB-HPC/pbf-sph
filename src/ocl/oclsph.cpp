#include "oclsph.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace sph::ocl_impl {

Solver::Solver(float h, const std::string &kernelPath, const std::vector<std::string> &includes,
               const cl::Device &device)
    : h(h), device(device), ctx(cl::Context(device)),
      clsph(utils::loadProgramFromFile(ctx,                                                      //
                                       kernelPath,                                               //
                                       includes,                                                 //
                                       std::vector{"-DSPH_H=((float)" + std::to_string(h) + ")"} //
                                       )),
      // TODO check capability
      queue(cl::CommandQueue(ctx, device, cl::QueueProperties::None)), sphDiffuseKernel(clsph, "sph_diffuse"),
      sphLambdaKernel(clsph, "sph_lambda"), sphDeltaKernel(clsph, "sph_delta"),
      sphFinaliseKernel(clsph, "sph_finalise"), mcLatticeKernel(clsph, "mc_lattice"), mcSizeKernel(clsph, "mc_size"),
      mcEvalKernel(clsph, "mc_eval") {
  checkSize();
}

void Solver::checkSize() {
  std::vector<size_t> expected(_SIZES, _SIZES + _SIZES_LENGTH);
  std::vector<size_t> actual(_SIZES_LENGTH, 0);

  try {
    TypedBuffer<size_t, WO> buffer(ctx, _SIZES_LENGTH);
    cl::KernelFunctor<cl::Buffer &>(clsph, "check_size")(cl::EnqueueArgs(queue, cl::NDRange(_SIZES_LENGTH)),
                                                         buffer.actual);
    buffer.drainTo(queue, actual);
    queue.finish();
  } catch (cl::Error &exc) {
    std::cerr << "Kernel failed to execute: " << exc.what() << " -> " << utils::clResolveError(exc.err()) << "("
              << exc.err() << ")" << std::endl;
    throw;
  }

#ifdef DEBUG
  std::cout << "Expected(" << _SIZES_LENGTH
            << ")=" << utils::mkString<size_t>(expected, [](auto x) { return std::to_string(x); }) << std::endl;
  std::cout << "Actual(" << _SIZES_LENGTH
            << ")  =" << utils::mkString<size_t>(actual, [](auto x) { return std::to_string(x); }) << std::endl;
#endif
  assert(expected == actual);
}

std::vector<PartiallyAdvected> Solver::advectAndCopy(const sph::SphParams<size_t, float, V> &config,
                                                     const sph::Scene<size_t, float, V> &scene,
                                                     std::vector<sph::Particle<size_t, float, V>> &xs) {
  std::vector<PartiallyAdvected> advected(xs.size());

  const float threshold = 200.f;
  const float thresholdSquared = threshold * threshold;

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
    sph::Particle<size_t, float, V> &p = xs[i];
    advected[i].particle = p;
    advected[i].pStar = utils::vec3ToCl(p.position / config.scale);

    if (p.type == sph::Type::Obstacle) continue;

    V<3, float> combinedForce = p.mass * config.constantForce;
    for (const sph::Well<size_t, float, V> &well : scene.wells) {
      const float dist = glm::distance(p.position, well.centre);

      if (dist < 75.f) {
        const V<3, float> rHat = glm::normalize(well.centre - p.position);
        const V<3, float> forceWell = glm::clamp((rHat * well.force * p.mass) / (dist * dist), -10.f, 10.f);
        combinedForce += forceWell;
      }
    }

    p.velocity = combinedForce * config.dt + p.velocity;
    advected[i].pStar = utils::vec3ToCl((p.velocity * config.dt) + (p.position / config.scale));
  }
  return advected;
}

[[nodiscard]] size_t Solver::zCurveGridIndexAtCoordAt(float x, float y, float z) const {
  return zCurveGridIndexAtCoord(static_cast<size_t>((x / h)), static_cast<size_t>((y / h)),
                                static_cast<size_t>((z / h)));
}

std::tuple<ClSphConfig, V<3, float>, V<3, size_t>>
Solver::computeBoundAndZindex(const sph::SphParams<size_t, float, V> &config,
                              std::vector<PartiallyAdvected> &advection) const {

  V<3, float> minExtent(std::numeric_limits<float>::max());
  V<3, float> maxExtent(std::numeric_limits<float>::min());
  const float padding = h * 2;
  minExtent = (config.minBound / config.scale) - padding;
  maxExtent = (config.maxBound / config.scale) + padding;

#pragma omp parallel for
  for (size_t i = 0; i < advection.size(); ++i) {
    const cl_float3 pStar = advection[i].pStar;
    advection[i].zIndex =
        Solver::zCurveGridIndexAtCoordAt(pStar.s[0] - minExtent.x, pStar.s[1] - minExtent.y, pStar.s[2] - minExtent.z);
  }

  ClSphConfig clConfig{
      .dt = config.dt,
      .scale = config.scale,
      .iteration = static_cast<size_t>(config.iteration),
      .minBound = utils::vec3ToCl(config.minBound),
      .maxBound = utils::vec3ToCl(config.maxBound),
  };
  return std::make_tuple(clConfig, minExtent, V<3, size_t>((maxExtent - minExtent) / h));
}

sph::ColouredMesh<float, V> Solver::runMcKernels(                      //
    sph::utils::Stopwatch &watch,                                      //
    const V<3, size_t> sampleSize,                                     //
    TypedBuffer<ClSphConfig, RO> &sphConfig,                           //
    TypedBuffer<ClMcConfig, RO> &mcConfig,                             //
    TypedBuffer<uint, RO> &gridTable, TypedBuffer<ClSphType, RO> type, //
    TypedBuffer<cl_float3, RW> &particlePositions,                     //
    TypedBuffer<cl_float4, RW> &particleColours,                       //
    const V<3, float> minExtent, const V<3, size_t> extent) {

  const uint gridTableN = static_cast<uint>(gridTable.length);

  const size_t latticeN = sampleSize.x * sampleSize.y * sampleSize.z;
  TypedBuffer<cl_float4, RW> latticePNs(ctx, latticeN);
  TypedBuffer<cl_float4, RW> latticeCs(ctx, latticeN);

  const size_t kernelWorkGroupSize = mcSizeKernel.getKernel().getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

  const V<3, size_t> marchRange = sampleSize - V<3, size_t>(1);
  const size_t marchVolume = marchRange.x * marchRange.y * marchRange.z;

  size_t workGroupSize = kernelWorkGroupSize;
  size_t numWorkGroup = std::ceil(static_cast<float>(marchVolume) / static_cast<float>(workGroupSize));

  std::cout << "[<>]Samples:" << sph::utils::to_string(marchRange) << " MarchVol=" << marchVolume
            << " WG:" << kernelWorkGroupSize << " nWG:" << numWorkGroup << "\n";

  auto create_field = watch.start("\t[GPU] mc-field");

  mcLatticeKernel(cl::EnqueueArgs(queue, cl::NDRange(sampleSize.x, sampleSize.y, sampleSize.z)),
                  sphConfig.actual,             //
                  mcConfig.actual,              //
                  gridTable.actual,             //
                  gridTableN,                   //
                  type.actual,                  //
                  utils::vec3ToCl(minExtent),   //
                  utils::uvec3ToCl(sampleSize), //
                  utils::uvec3ToCl(extent),     //
                  particlePositions.actual,     //
                  particleColours.actual,       //
                  latticePNs.actual,            //
                  latticeCs.actual);
  finishQueue();
  create_field();

  auto partial_trig_sum = watch.start("\t[GPU] mc_psum");

  TypedBuffer<uint, WO> partialTrigSum(ctx, numWorkGroup);
  mcSizeKernel(cl::EnqueueArgs(queue, cl::NDRange(numWorkGroup * workGroupSize), cl::NDRange(workGroupSize)),
               mcConfig.actual,                         //
               utils::uvec3ToCl(sampleSize),            //
               latticePNs.actual,                       //
               cl::Local(sizeof(uint) * workGroupSize), //
               partialTrigSum.actual);

  std::vector<uint> hostPartialTrigSum(numWorkGroup, 0);
  partialTrigSum.drainTo(queue, hostPartialTrigSum);

  uint numTrigs = 0;
  for (uint j = 0; j < numWorkGroup; ++j)
    numTrigs += hostPartialTrigSum[j];

  //  std::cout << "Vol=" << marchVolume << glm::to_string(marchRange) << "  " << numTrigs << "\n";

  finishQueue();
  partial_trig_sum();
#ifdef DEBUG
  std::cout << "[<>]Acc=" << numTrigs << std::endl;
#endif
  auto gpu_mc = watch.start("\t[GPU] gpu_mc");

  auto mesh = sph::ColouredMesh<float, V>(numTrigs * 3);
  if (numTrigs != 0) {
    std::vector<uint> zero{0};
    TypedBuffer<uint, RW> trigCounter(queue, zero);
    TypedBuffer<cl_float3, WO> outVxs(ctx, numTrigs * 3);
    TypedBuffer<cl_float3, WO> outNxs(ctx, numTrigs * 3);
    TypedBuffer<cl_float4, WO> outCxs(ctx, numTrigs * 3);
    mcEvalKernel(cl::EnqueueArgs(queue, cl::NDRange(marchVolume)), sphConfig.actual,
                 mcConfig.actual,              //
                 utils::vec3ToCl(minExtent),   //
                 utils::uvec3ToCl(sampleSize), //
                 latticePNs.actual,            //
                 latticeCs.actual,             //
                 trigCounter.actual,           //
                 numTrigs,                     //
                 outVxs.actual,                //
                 outNxs.actual,                //
                 outCxs.actual);

    finishQueue();
    gpu_mc();

    auto gpu_mc_drain = watch.start("\t[GPU] gpu_mc drain");
    std::vector<cl_float3> hostOutVxs(numTrigs * 3);
    std::vector<cl_float3> hostOutNxs(numTrigs * 3);
    std::vector<cl_float4> hostOutCxs(numTrigs * 3);
    outVxs.drainTo(queue, hostOutVxs);
    outNxs.drainTo(queue, hostOutNxs);
    outCxs.drainTo(queue, hostOutCxs);
    finishQueue();
    gpu_mc_drain();
    auto gpu_mc_assem = watch.start("\t[GPU] gpu_mc assem");
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(numTrigs * 3); ++i) {
      mesh.vs[i] = utils::clToVec3<float>(hostOutVxs[i]);
      mesh.ns[i] = utils::clToVec3<float>(hostOutNxs[i]);
      mesh.cs[i] = utils::clToVec4<float>(hostOutCxs[i]);
    }
    gpu_mc_assem();
#ifdef DEBUG
    std::cout << "[<>] LatticeDataN:" << (float)(sizeof(float4) * marchVolume) / 1000000.0 << "MB"
              << " MCGPuN:" << (float)(sizeof(float3) * numTrigs * 6) / 1000000.0 << "MB \n";
#endif
  }
  return mesh;
}

void Solver::runSphKernel(sph::utils::Stopwatch &watch, size_t iterations, TypedBuffer<ClSphConfig, RO> &sphConfig,
                          TypedBuffer<uint, RO> &gridTable, TypedBuffer<uint, RO> &zIndex,
                          TypedBuffer<ClSphType, RO> type, TypedBuffer<float, RO> &mass,
                          TypedBuffer<cl_float4, RO> &colour, TypedBuffer<cl_float3, RW> &pStar,
                          TypedBuffer<cl_float3, RW> &deltaP, TypedBuffer<float, RW> &lambda,
                          TypedBuffer<cl_float4, RW> &diffused, TypedBuffer<cl_float3, RW> &position,
                          TypedBuffer<cl_float3, RW> &velocity) {
  auto kernel_copy = watch.start("\t[GPU] kernel_copy");

  const uint gridTableN = static_cast<uint>(gridTable.length);

  finishQueue();
  kernel_copy();

  const auto localRange = cl::NDRange();
  const auto globalRange = cl::NDRange(position.length);

  //            std::cout << "r=" << position.length << std::endl;

  auto diffuse = watch.start("\t[GPU] sph-diffuse ");

  sphDiffuseKernel(cl::EnqueueArgs(queue, globalRange, localRange), sphConfig.actual, zIndex.actual, gridTable.actual,
                   gridTableN, type.actual, pStar.actual, colour.actual, diffused.actual);

  finishQueue();
  diffuse();

  //            exit(1);

  auto lambda_delta = watch.start("\t[GPU] sph-lambda/delta*" + std::to_string(iterations));

  for (size_t itr = 0; itr < iterations; ++itr) {
    sphLambdaKernel(cl::EnqueueArgs(queue, globalRange, localRange),
                    sphConfig.actual, //
                    zIndex.actual,    //
                    gridTable.actual, //
                    gridTableN,       //
                    type.actual,      //
                    pStar.actual,     //
                    mass.actual,      //
                    lambda.actual);
    sphDeltaKernel(cl::EnqueueArgs(queue, globalRange, localRange),
                   sphConfig.actual, //
                   zIndex.actual,    //
                   gridTable.actual, //
                   gridTableN,       //
                   type.actual,      //
                   pStar.actual,     //
                   lambda.actual,    //
                   position.actual,  //
                   velocity.actual,  //
                   deltaP.actual);
  }
  finishQueue();
  lambda_delta();

  auto finalise = watch.start("\t[GPU] sph-finalise");
  sphFinaliseKernel(cl::EnqueueArgs(queue, globalRange, localRange), sphConfig.actual, type.actual, pStar.actual,
                    position.actual, velocity.actual);
  finishQueue();
  finalise();
}

//		const std::array<V<3, float>, 27> NEIGHBOUR_OFFSETS = {
//				V<3, float>(-h, -h, -h), V<3, float>(+0, -h, -h), V<3, float>(+h, -h, -h),
//				V<3, float>(-h, +0, -h), V<3, float>(+0, +0, -h), V<3, float>(+h, +0, -h),
//				V<3, float>(-h, +h, -h), V<3, float>(+0, +h, -h), V<3, float>(+h, +h, -h),
//				V<3, float>(-h, -h, +0), V<3, float>(+0, -h, +0), V<3, float>(+h, -h, +0),
//				V<3, float>(-h, +0, +0), V<3, float>(+0, +0, +0), V<3, float>(+h, +0, +0),
//				V<3, float>(-h, +h, +0), V<3, float>(+0, +h, +0), V<3, float>(+h, +h, +0),
//				V<3, float>(-h, -h, +h), V<3, float>(+0, -h, +h), V<3, float>(+h, -h, +h),
//				V<3, float>(-h, +0, +h), V<3, float>(+0, +0, +h), V<3, float>(+h, +0, +h),
//				V<3, float>(-h, +h, +h), V<3, float>(+0, +h, +h), V<3, float>(+h, +h, +h)
//		};

static constexpr std::array<V<3, float>, 1> NEIGHBOUR_OFFSETS = {V<3, float>(0, 0, 0)};

sph::Result<size_t, float, V> Solver::advance(const sph::SphParams<size_t, float, V> &config,
                                              const sph::Scene<size_t, float, V> &scene,
                                              std::vector<sph::Particle<size_t, float, V>> &xs) {

  auto watch = sph::utils::Stopwatch("CPU advance");
  auto total = watch.start("Advance ===total===");

  auto sourceDrain = watch.start("CPU source+drain");

  const float spacing = (h * config.scale / 2);
  for (const sph::Source<size_t, float, V> &source : scene.sources) {
    const float size = std::sqrt(static_cast<float>(source.rate));
    const size_t width = std::floor(size);
    const size_t depth = std::ceil(size);
    const auto offset = source.centre - (V<3, float>(width, 0, depth) / 2.f * spacing);
    for (size_t x = 0; x < width; ++x) {
      for (size_t z = 0; z < depth; ++z) {
        auto pos = offset + V<3, float>(x, 0, z) * spacing;
        xs.emplace_back(source.tag, sph::Type::Fluid, 1, source.colour, pos, source.velocity);
      }
    }
  }

  xs.erase(std::remove_if(xs.begin(), xs.end(),
                          [&scene](const sph::Particle<size_t, float, V> &x) {
                            if (x.type == sph::Type::Obstacle) return false;

                            for (const sph::Drain<size_t, float, V> &drain : scene.drains) {
                              // FIXME needs to actually erase at surface, not shperically
                              if (glm::distance(drain.centre, x.position) < drain.width) {
                                return true;
                              }
                            }
                            return false;
                          }),
           xs.end());

  sourceDrain();

  if (xs.empty()) {
    std::cout << "Particles depleted" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return sph::Result<size_t, float, V>();
  }

  auto advect = watch.start("CPU advect+copy");
  std::vector<PartiallyAdvected> advected = advectAndCopy(config, scene, xs);
  const size_t atomsN = advected.size();
  advect();

  auto bound = watch.start("CPU bound+zindex");
  auto [sphConfig, minExtent, extent] = computeBoundAndZindex(config, advected);
  bound();

  auto sortz = watch.start("CPU sortz");

  std::sort(advected.begin(), advected.end(),
            [](const PartiallyAdvected &l, const PartiallyAdvected &r) { return l.zIndex < r.zIndex; });

  sortz();

  auto gridtable = watch.start("CPU gridtable");
  const size_t gridTableN = zCurveGridIndexAtCoord(extent.x, extent.y, extent.z);
#ifdef DEBUG
  std::cout << "Atoms = " << atomsN << " Extent = " << glm::to_string(minExtent) << " -> " << to_string(extent)
            << " GridTable = " << gridTableN << std::endl;
#endif
  std::vector<uint> hostGridTable(gridTableN);
  uint gridIndex = 0;
  for (size_t i = 0; i < gridTableN; ++i) {
    hostGridTable[i] = gridIndex;
    while (gridIndex != atomsN && advected[gridIndex].zIndex == i) {
      gridIndex++;
    }
  }
  gridtable();

  auto query = watch.start("CPU query(" + std::to_string(scene.queries.size()) + ")");
  std::vector<sph::QueryResult<size_t, float, V>> queries;
  for (const sph::Query<size_t, float, V> &q : scene.queries) {
    auto scaled = (q.point / config.scale) - minExtent;
    std::vector<size_t> neighbours;
    for (V<3, float> offset : NEIGHBOUR_OFFSETS) {
      auto r = offset + scaled;
      size_t zIdx = zCurveGridIndexAtCoordAt(r.x, r.y, r.z);
      if (zIdx < gridTableN && zIdx + 1 < gridTableN) {
        for (size_t a = hostGridTable[zIdx]; a < hostGridTable[zIdx + 1]; a++) {
          if (advected[a].particle.type != sph::Type::Fluid) continue;
          neighbours.push_back(advected[a].particle.id);
        }
      }
    }
    queries.push_back({q.id, q.point, neighbours});
  }
  query();

  auto kernel_alloc = watch.start("CPU host alloc+copy");

  ClSphAtoms atoms(advected.size());
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(advected.size()); ++i) {
    atoms.zIndex[i] = advected[i].zIndex;
    atoms.pStar[i] = advected[i].pStar;
    atoms.type[i] = resolveType(advected[i].particle.type);
    atoms.mass[i] = advected[i].particle.mass;
    atoms.colour[i] = utils::vec4ToCl(advected[i].particle.colour);
    atoms.position[i] = utils::vec3ToCl(advected[i].particle.position);
    atoms.velocity[i] = utils::vec3ToCl(advected[i].particle.velocity);
  }

  auto kernel_exec = watch.start("\t[GPU] ===total===");

  auto sphConfig_ = TypedBuffer<ClSphConfig, RO>::ofStruct(ctx, sphConfig);

  TypedBuffer<uint, RO> gridTable(queue, hostGridTable);
  TypedBuffer<uint, RO> zIndex(queue, atoms.zIndex);
  TypedBuffer<ClSphType, RO> type(queue, atoms.type);
  TypedBuffer<float, RO> mass(queue, atoms.mass);
  TypedBuffer<cl_float4, RO> colour(queue, atoms.colour);
  TypedBuffer<cl_float3, RW> pStar(queue, atoms.pStar);

  TypedBuffer<cl_float3, RW> deltaP(ctx, atoms.size);
  TypedBuffer<float, RW> lambda(ctx, atoms.size);
  TypedBuffer<cl_float4, RW> diffused(ctx, atoms.size);

  TypedBuffer<cl_float3, RW> position(queue, atoms.position);
  TypedBuffer<cl_float3, RW> velocity(queue, atoms.velocity);

  kernel_alloc();

  runSphKernel(watch, sphConfig.iteration, sphConfig_, gridTable, zIndex, type, mass, colour, pStar, deltaP, lambda,
               diffused, position, velocity);

  sph::ColouredMesh<float, V> surface;

  if (config.surface) {

    ClMcConfig mcConfig{
        .sampleResolution = config.surface->resolution,
        .particleSize = config.surface->particleSize,
        .particleInfluence = config.surface->particleInfluence,
        .isolevel = config.surface->isolevel,
    };

    auto mcConfig_ = TypedBuffer<ClMcConfig, RO>::ofStruct(ctx, mcConfig);

    const V<3, size_t> sampleSize =
        V<3, size_t>(glm::floor(V<3, float>(extent) * mcConfig.sampleResolution)) + V<3, size_t>(1);

    surface =
        runMcKernels(watch, sampleSize, sphConfig_, mcConfig_, gridTable, type, position, diffused, minExtent, extent);
  }

  std::vector<cl_float3> hostPosition(advected.size());
  std::vector<cl_float3> hostVelocity(advected.size());
  std::vector<cl_float4> hostDiffused(advected.size());
  position.drainTo(queue, hostPosition);
  velocity.drainTo(queue, hostVelocity);
  diffused.drainTo(queue, hostDiffused);

  auto write_back = watch.start("write_back");
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
    xs[i].id = advected[i].particle.id;
    xs[i].type = advected[i].particle.type;
    xs[i].mass = advected[i].particle.mass;

    if (advected[i].particle.type == sph::Type::Fluid) {
      xs[i].colour = utils::clToVec4<float>(hostDiffused[i]);
      xs[i].position = utils::clToVec3<float>(hostPosition[i]);
      xs[i].velocity = utils::clToVec3<float>(hostVelocity[i]);
    } else {
      xs[i].colour = advected[i].particle.colour;
      xs[i].position = advected[i].particle.position;
      xs[i].velocity = advected[i].particle.velocity;
    }
  }
  write_back();
  kernel_exec();

  total();

  //			std::vector<unsigned short> outIdx;
  //			std::vector<V<3, float>> outVert;
  //			hrc::time_point vbiStart = hrc::now();
  //			surface::indexVBO2<float>(triangles, outIdx, outVert);
  //			hrc::time_point vbiEnd = hrc::now();
  //			auto vbi = duration_cast<nanoseconds>(vbiEnd - vbiStart).count();
  //
  //			std::cout
  //					<< "\n\tTrigs = " << triangles.size()
  //					<< "\n\tIdx   = " << outIdx.size()
  //					<< "\n\tVert  = " << outVert.size()
  //					<< "\n\tVBI   = " << (vbi / 1000000.0) << "ms\n"
  //					<< std::endl;

  //    std::cout << watch << std::endl;
  return sph::Result<size_t, float, V>{surface, queries};
}

} // namespace sph::ocl_impl
