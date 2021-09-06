#pragma once

//#ifdef __SYCL_DEVICE_ONLY__
//#undef __SYCL_DEVICE_ONLY__
//#endif

#include "glm/glm.hpp"

#include "curves.h"

#include "sph_constants.h"
#include "utils.hpp"

#include "mc_constants.h"
#include "oclsph_type.h"
#include "sph.hpp"

#include "glm/gtc/constants.hpp"
#include "glm/gtx/fast_exponential.hpp"
#include "glm/gtx/fast_square_root.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/optimum_pow.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace sph::omp_impl {

namespace details {
template <size_t N, typename T> using V = glm::vec<N, T>;
} // namespace details

using namespace details;
using namespace sph;

template <typename N, typename F> void foreach_1d(const N size, const F &f) {
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(size); ++i) {
    f(i);
  }
}

template <typename N, typename L> void foreach_nd(const N groupSize, const N localSize, const L &l) {
#pragma omp parallel for
  for (int groupId = 0; groupId < static_cast<int>(groupSize); ++groupId) {
    for (int localId = 0; localId < static_cast<int>(localSize); ++localId) {
      auto globalId = (groupId * localSize) + localId;
      l(groupSize, localSize, globalId, groupId, localId);
    }
  }
}

template <typename N, typename F> void foreach_3d(const V<3, N> &size, const F &f) {
#pragma omp parallel for collapse(3)
  for (int x = 0; x < static_cast<int>(size.x); ++x) {
    for (int y = 0; y < static_cast<int>(size.y); ++y) {
      for (int z = 0; z < static_cast<int>(size.z); ++z) {
        f(x, y, z);
      }
    }
  }
}

template <typename N> constexpr N poly6Kernel(const N r, const N factor, const N h) {
  return r <= h ? factor * glm::pow3((h * h) - r * r) : 0.f;
}

template <typename N>
constexpr V<3, N> spikyKernelGradient(const V<3, N> x, const V<3, N> y, //
                                      const N r, const N h, const N factor) {
  return (r >= EPSILON && r <= h) ? (x - y) * (factor * (glm::pow2(h - r) / r)) : V<3, N>();
}

template <typename T, typename N> class Solver final : public sph::Solver<T, N, V> {

private:
  const N h;

public:
  explicit Solver(N h) : h(h) {}

  sph::Result<T, N, V> advance(const sph::SphParams<T, N, V> &config, //
                               const sph::Scene<T, N, V> &scene,      //
                               std::vector<sph::Particle<T, N, V>> &xs) final {

    auto watch = sph::utils::Stopwatch("advance");

    auto sourceDrain = watch.start("CPU source+drain");

    const N spacing = (h * config.scale / 2);
    for (const sph::Source<T, N, V> &source : scene.sources) {
      const N size = std::sqrt(static_cast<N>(source.rate));
      const size_t width = std::floor(size);
      const size_t depth = std::ceil(size);
      const auto offset = source.centre - (V<3, N>(width, 0, depth) * N(0.5) * spacing);
      for (size_t x = 0; x < width; ++x) {
        for (size_t z = 0; z < depth; ++z) {
          auto pos = offset + V<3, N>(x, 0, z) * spacing;
          xs.emplace_back(source.tag, sph::Type::Fluid, 1, source.colour, pos, source.velocity);
        }
      }
    }

    xs.erase(std::remove_if(xs.begin(), xs.end(),
                            [&scene](const sph::Particle<T, N, V> &x) {
                              if (x.type == sph::Type::Obstacle) return false;
                              for (const sph::Drain<T, N, V> &drain : scene.drains) {
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
      return sph::Result<T, N, V>();
    }

    std::vector<PartiallyAdvected<T, N, V>> advected(xs.size());

    auto advect = watch.start("CPU advect+copy");

    const N padding = h * 2;
    const auto minExtent = (config.minBound / config.scale) - padding;
    const auto maxExtent = (config.maxBound / config.scale) + padding;
    const auto extent = glm::tvec3<size_t>((maxExtent - minExtent) / h);

    foreach_1d(xs.size(), [&](size_t i) {
      const sph::Particle<T, N, V> &p = xs[i];
      if (p.type == sph::Type::Obstacle) return;
      V<3, N> combinedForce = p.mass * config.constantForce;
      for (const sph::Well<T, N, V> &well : scene.wells) {
        const N dist = glm::distance(p.position, well.centre);
        if (dist < N(75)) {
          const V<3, N> rHat = glm::normalize(well.centre - p.position);
          const V<3, N> forceWell = glm::clamp((rHat * well.force * p.mass) / (dist * dist), N(-10), N(10));
          combinedForce += forceWell;
        }
      }
      advected[i].particle = p;
      advected[i].particle.velocity = combinedForce * config.dt + advected[i].particle.velocity;
      advected[i].pStar = (advected[i].particle.velocity * config.dt) + (advected[i].particle.position / config.scale);
      advected[i].zIndex = zCurveGridIndexAtCoordAt(
          advected[i].pStar.x - minExtent.x, advected[i].pStar.y - minExtent.y, advected[i].pStar.z - minExtent.z, h);
    });
    advect();

    const auto sortz = watch.start("CPU sortz");
    std::sort(advected.begin(), advected.end(), [](auto &l, auto &r) { return l.zIndex < r.zIndex; });
    sortz();

    auto gridtable = watch.start("CPU gridtable");
    const std::vector<size_t> gridTable = makeGridTable(extent.x, extent.y, extent.z, advected.size(),
                                                        [&advected](auto i) { return advected[i].zIndex; });
    const size_t gridTableN = gridTable.size();
    gridtable();

    const auto query = watch.start("CPU query(" + std::to_string(scene.queries.size()) + ")");
    static const std::array<V<3, N>, 1> NEIGHBOUR_OFFSETS = {V<3, N>(0, 0, 0)};
    std::vector<sph::QueryResult<T, N, V>> queryResults(scene.queries.size());
    foreach_1d(scene.queries.size(), [&](size_t i) {
      const auto q = scene.queries[i];
      const auto scaled = (q.point / config.scale) - minExtent;
      std::vector<T> neighbours;
      for (V<3, N> offset : NEIGHBOUR_OFFSETS) {
        auto r = offset + scaled;
        size_t zIdx = zCurveGridIndexAtCoordAt(r.x, r.y, r.z, h);
        if (zIdx < gridTableN && zIdx + 1 < gridTableN) {
          for (size_t a = gridTable[zIdx]; a < gridTable[zIdx + 1]; a++) {
            if (advected[a].particle.type != sph::Type::Fluid) continue;
            neighbours.push_back(advected[a].particle.id);
          }
        }
      }
      queryResults[i] = {q.id, q.point, neighbours};
    });
    query();

    const auto diffuse = watch.start("\t[CPU] sph-diffuse ");
    foreach_1d(advected.size(), [&](size_t a) {
      if (advected[a].particle.type == sph::Type::Obstacle) return;
      int nNeighbours = 0;
      V<4, N> mixture{};
      foreach_grid(advected[a].zIndex, gridTable, gridTableN, [&](size_t b) {
        if (advected[b].particle.type != sph::Type::Obstacle) {
          mixture += advected[b].particle.colour;
          nNeighbours++;
        }
      });

      if (nNeighbours != 0) {
        auto out = glm::mix(advected[a].particle.colour,          //
                            (mixture / N(nNeighbours)) * N(1.33), //
                            config.dt / N(750.0));
        advected[a].particle.colour = glm::clamp(out, N(0.03), N(1.0));
      }
    });
    diffuse();

    auto lambda_delta = watch.start("\t[GPU] sph-lambda/delta*" + std::to_string(config.iteration));

    const N Poly6Factor = poly6Factor(h);
    const N SpikyKernelFactor = spikyKernelFactor(h);
    const N P6DeltaQ = poly6Kernel(CorrDeltaQ * h, Poly6Factor, h);

    for (size_t itr = 0; itr < config.iteration; ++itr) {
      // lambda
      foreach_1d(advected.size(), [&](size_t a) {
        if (advected[a].particle.type == sph::Type::Obstacle) {
          advected[a].lambda = 0;
          return;
        }
        V<3, N> norm2V{};
        N rho = 0;
        foreach_grid(advected[a].zIndex, gridTable, gridTableN, [&](size_t b) {
          const N r = glm::distance(advected[a].pStar, advected[b].pStar);
          norm2V += spikyKernelGradient(advected[a].pStar, advected[b].pStar, r, h, SpikyKernelFactor) * N(RHO_RECIP);
          rho += advected[a].particle.mass * poly6Kernel(r, Poly6Factor, h);
        });
        N norm2 = glm::length2(norm2V);
        N Ci = (rho / RHO - N(1));
        advected[a].lambda = -Ci / (norm2 + CFM_EPSILON);
      });

      // delta
      foreach_1d(advected.size(), [&](size_t a) {
        if (advected[a].particle.type == sph::Type::Obstacle) return;
        V<3, N> deltaPAcc{};
        foreach_grid(advected[a].zIndex, gridTable, gridTableN, [&](size_t b) {
          const N r = glm::distance(advected[a].pStar, advected[b].pStar);
          const N corr = N(-CorrK) * glm::pow(poly6Kernel(r, Poly6Factor, h) / P6DeltaQ, N(CorrN));
          const N factor = (advected[a].lambda + advected[b].lambda + corr) / N(RHO);
          deltaPAcc += spikyKernelGradient(advected[a].pStar, advected[b].pStar, r, h, SpikyKernelFactor) * factor;
        });
        advected[a].deltaP = deltaPAcc;
        V<3, N> pos = (advected[a].pStar + advected[a].deltaP) * config.scale;
        pos = min(config.maxBound, max(config.minBound, pos)); // clamp to extent
        advected[a].pStar = pos / config.scale;
      });
    }
    lambda_delta();

    auto finalise = watch.start("\t[GPU] sph-finalise");



    foreach_1d(advected.size(), [&](size_t a) {
      if (advected[a].particle.type == sph::Type::Obstacle) return;
      const auto deltaX = advected[a].pStar -
                          advected[a].particle.position / config.scale;
      advected[a].particle.position = advected[a].pStar * config.scale;
      advected[a].particle.velocity = (deltaX * (N(1) / config.dt) +
                                       advected[a].particle.velocity) *
                                      N(VD);
    });






    finalise();

    std::vector<V<3, N>> outVxs;
    std::vector<V<3, N>> outNxs;
    std::vector<V<4, N>> outCxs;

    if (config.surface) {

      const auto create_field = watch.start("\t[GPU] mc-field");

      const N particleSize = config.surface->particleSize;
      const N particleInfluence = config.surface->particleInfluence;
      const V<3, size_t> sampleSize =
          V<3, size_t>(glm::floor(V<3, N>(extent) * config.surface->resolution)) + V<3, size_t>(1);
      const size_t latticeN = sampleSize.x * sampleSize.y * sampleSize.z;
      std::vector<V<4, N>> latticePNs(latticeN);
      std::vector<V<4, N>> latticeCs(latticeN);
      foreach_3d(sampleSize, [&](size_t x, size_t y, size_t z) {
        const V<3, N> pos(x, y, z);
        const N step = h / config.surface->resolution;
        const V<3, N> a = (minExtent + (pos * step)) * config.scale;
        const N threshold = h * config.scale * 1;

        const size_t zIndex = zCurveGridIndexAtCoord((size_t)(pos.x / config.surface->resolution),
                                                     (size_t)(pos.y / config.surface->resolution),
                                                     (size_t)(pos.z / config.surface->resolution));
        const size_t zX = coordAtZCurveGridIndex0(zIndex);
        const size_t zY = coordAtZCurveGridIndex1(zIndex);
        const size_t zZ = coordAtZCurveGridIndex2(zIndex);

        if (zX == extent.x && zY == extent.y && zZ == extent.z) {
          // XXX there is exactly one case where this may happen: the last element of the z-curve
          return;
        }

        const size_t x_l = glm::clamp(((int)zX) - 1, 0, (int)extent.x - 1);
        const size_t x_r = glm::clamp(((int)zX) + 1, 0, (int)extent.x - 1);
        const size_t y_l = glm::clamp(((int)zY) - 1, 0, (int)extent.y - 1);
        const size_t y_r = glm::clamp(((int)zY) + 1, 0, (int)extent.y - 1);
        const size_t z_l = glm::clamp(((int)zZ) - 1, 0, (int)extent.z - 1);
        const size_t z_r = glm::clamp(((int)zZ) + 1, 0, (int)extent.z - 1);

        std::array<size_t, 27> offsets = {zCurveGridIndexAtCoord(x_l, y_l, z_l), zCurveGridIndexAtCoord(zX, y_l, z_l),
                                          zCurveGridIndexAtCoord(x_r, y_l, z_l), zCurveGridIndexAtCoord(x_l, zY, z_l),
                                          zCurveGridIndexAtCoord(zX, zY, z_l),   zCurveGridIndexAtCoord(x_r, zY, z_l),
                                          zCurveGridIndexAtCoord(x_l, y_r, z_l), zCurveGridIndexAtCoord(zX, y_r, z_l),
                                          zCurveGridIndexAtCoord(x_r, y_r, z_l), zCurveGridIndexAtCoord(x_l, y_l, zZ),
                                          zCurveGridIndexAtCoord(zX, y_l, zZ),   zCurveGridIndexAtCoord(x_r, y_l, zZ),
                                          zCurveGridIndexAtCoord(x_l, zY, zZ),   zCurveGridIndexAtCoord(zX, zY, zZ),
                                          zCurveGridIndexAtCoord(x_r, zY, zZ),   zCurveGridIndexAtCoord(x_l, y_r, zZ),
                                          zCurveGridIndexAtCoord(zX, y_r, zZ),   zCurveGridIndexAtCoord(x_r, y_r, zZ),
                                          zCurveGridIndexAtCoord(x_l, y_l, z_r), zCurveGridIndexAtCoord(zX, y_l, z_r),
                                          zCurveGridIndexAtCoord(x_r, y_l, z_r), zCurveGridIndexAtCoord(x_l, zY, z_r),
                                          zCurveGridIndexAtCoord(zX, zY, z_r),   zCurveGridIndexAtCoord(x_r, zY, z_r),
                                          zCurveGridIndexAtCoord(x_l, y_r, z_r), zCurveGridIndexAtCoord(zX, y_r, z_r),
                                          zCurveGridIndexAtCoord(x_r, y_r, z_r)};

        N v = 0.f;
        V<3, N> normal{};
        V<4, N> colour{};
        size_t nNeighbours = 0;


        //
        foreach_grid(gridTable, gridTableN, offsets, [&](size_t b) {
          if (advected[b].particle.type != sph::Type::Obstacle &&
              glm::fastDistance(advected[b].particle.position, a) < threshold) {
            const auto l = advected[b].particle.position - a;
            const N len = glm::fastLength(l);
            const N denominator = glm::pow(len, particleInfluence);
            v += (particleSize / denominator);
            normal += (-particleInfluence) * particleSize * (l / denominator);
            colour += advected[b].particle.colour;
            nNeighbours++;
          }
        });

        normal = glm::fastNormalize(normal);

        const size_t idx = index3d(x, y, z, sampleSize.x, sampleSize.y, sampleSize.z);
        latticePNs[idx].x = v;
        latticePNs[idx].y = normal.x;
        latticePNs[idx].z = normal.y;
        latticePNs[idx].w = normal.z;
        latticeCs[idx] = colour / N(nNeighbours);
      });
      create_field();

      const auto partial_trig_sum = watch.start("\t[GPU] mc_psum");

      const static std::vector<V<3, size_t>> CUBE_OFFSETS = {
          V<3, size_t>(0, 0, 0), V<3, size_t>(1, 0, 0), V<3, size_t>(1, 1, 0), V<3, size_t>(0, 1, 0),
          V<3, size_t>(0, 0, 1), V<3, size_t>(1, 0, 1), V<3, size_t>(1, 1, 1), V<3, size_t>(0, 1, 1)};

      const V<3, size_t> marchRange = sampleSize - V<3, size_t>(1);
      const size_t marchVolume = marchRange.x * marchRange.y * marchRange.z;

      size_t numGroup = 24;
      size_t localSize = std::ceil(static_cast<float>(marchVolume) / static_cast<float>(numGroup));

      std::vector<size_t> partialSums(numGroup);
      foreach_nd(
          numGroup, localSize, //
          [&](size_t groupSize, size_t localSize, size_t globalId, size_t groupId, size_t localId) {
            // because global size needs to be divisible by local group size (CL1.2), we discard the padding
            size_t nVert = 0;
            if (globalId >= marchVolume) {
              // NOOP
            } else {
              const auto pos = utils::to3d<V>(globalId, marchRange.x, marchRange.y, marchRange.z);
              size_t ci = 0;
              for (int i = 0; i < 8; ++i) {
                const auto offset = CUBE_OFFSETS[i] + pos;
                const N v =
                    latticePNs[index3d(offset.x, offset.y, offset.z, sampleSize.x, sampleSize.y, sampleSize.z)].x;
                ci = v < config.surface->isolevel ? ci | (1 << i) : ci;
              }
              nVert = EdgeTable[ci] == 0 ? 0u : (size_t)NumVertsTable[ci] / 3;
            }
            partialSums[groupId] += nVert;
          });

      size_t numTrigs = std::accumulate(partialSums.begin(), partialSums.end(), 0);
      std::cout << "Vol=" << marchVolume << " " << utils::to_string(marchRange) << "  " << numTrigs << " "
                << "\n";

      partial_trig_sum();

      const auto gpu_mc = watch.start("\t[GPU] gpu_mc");

      outVxs.resize(numTrigs * 3);
      outNxs.resize(numTrigs * 3);
      outCxs.resize(numTrigs * 3);

      std::atomic_size_t trigCounter(0);

      foreach_1d(marchVolume, [&](size_t i) {
        const auto pos = utils::to3d<V>(i, sampleSize.x - 1, sampleSize.y - 1, sampleSize.z - 1);
        const auto isolevel = config.surface->isolevel;
        const auto step = h / config.surface->resolution;

        std::array<N, 8> values;
        std::array<V<3, N>, 8> offsets;
        std::array<V<3, N>, 8> normals;
        std::array<V<4, N>, 8> colours;

        size_t ci = 0;
        for (int i = 0; i < 8; ++i) {
          const auto offset = CUBE_OFFSETS[i] + pos;
          const size_t idx = index3d(offset.x, offset.y, offset.z, sampleSize.x, sampleSize.y, sampleSize.z);
          const auto point = latticePNs[idx];

          values[i] = point.x;
          offsets[i] = (minExtent + (V<3, N>(offset) * step)) * config.scale;
          normals[i] = V<3, N>(point.y, point.z, point.w);
          colours[i] = latticeCs[idx];

          ci = values[i] < isolevel ? ci | (1 << i) : ci;
        }

        std::array<V<3, N>, 12> ts;
        std::array<V<3, N>, 12> ns;
        std::array<V<4, N>, 12> cs;

        const auto lerpAll = [&](size_t index, size_t from, size_t to, N v0, N v1) {
          const N t = sph::utils::scale(isolevel, v0, v1);
          ts[index] = glm::mix(offsets[from], offsets[to], t);
          ns[index] = glm::mix(normals[from], normals[to], t);
          cs[index] = glm::mix(colours[from], colours[to], t);
        };

        const size_t edge = EdgeTable[ci];
        if (edge & 1 << 0) lerpAll(0, 0, 1, values[0], values[1]);
        if (edge & 1 << 1) lerpAll(1, 1, 2, values[1], values[2]);
        if (edge & 1 << 2) lerpAll(2, 2, 3, values[2], values[3]);
        if (edge & 1 << 3) lerpAll(3, 3, 0, values[3], values[0]);
        if (edge & 1 << 4) lerpAll(4, 4, 5, values[4], values[5]);
        if (edge & 1 << 5) lerpAll(5, 5, 6, values[5], values[6]);
        if (edge & 1 << 6) lerpAll(6, 6, 7, values[6], values[7]);
        if (edge & 1 << 7) lerpAll(7, 7, 4, values[7], values[4]);
        if (edge & 1 << 8) lerpAll(8, 0, 4, values[0], values[4]);
        if (edge & 1 << 9) lerpAll(9, 1, 5, values[1], values[5]);
        if (edge & 1 << 10) lerpAll(10, 2, 6, values[2], values[6]);
        if (edge & 1 << 11) lerpAll(11, 3, 7, values[3], values[7]);

        for (size_t i = 0; TriTable[ci][i] != 255; i += 3) {
          const size_t trigIndex = trigCounter++;
          const size_t x = TriTable[ci][i + 0];
          const size_t y = TriTable[ci][i + 1];
          const size_t z = TriTable[ci][i + 2];

          outVxs[trigIndex * 3 + 0] = ts[x];
          outVxs[trigIndex * 3 + 1] = ts[y];
          outVxs[trigIndex * 3 + 2] = ts[z];

          outNxs[trigIndex * 3 + 0] = ns[x];
          outNxs[trigIndex * 3 + 1] = ns[y];
          outNxs[trigIndex * 3 + 2] = ns[z];

          outCxs[trigIndex * 3 + 0] = cs[x];
          outCxs[trigIndex * 3 + 1] = cs[y];
          outCxs[trigIndex * 3 + 2] = cs[z];
        }
      });

      gpu_mc();
    }

    auto writeback = watch.start("\t[GPU] write back");
    foreach_1d(advected.size(), [&](size_t a) { xs[a] = advected[a].particle; });
    writeback();
    std::cout << watch << std::endl;

    return sph::Result<T, N, V>{sph::ColouredMesh<N, V>{outVxs, outNxs, outCxs}, queryResults};
  }
};
} // namespace sph::omp_impl
