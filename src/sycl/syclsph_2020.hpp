#pragma once

#ifdef USE_SYCL

#include "mc_constants.h"
#include "sph_constants.h"

#include "sph.hpp"
#include "utils.hpp"

#include "syclutils.h"
#include <CL/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

namespace sph::sycl2020_impl {

namespace details {

using namespace cl;

template <size_t N, typename T> using V = sycl::vec<T, N>;

template <typename T, typename N> class kernel_diffuse;
template <typename T, typename N> class kernel_lambda;
template <typename T, typename N> class kernel_delta;
template <typename T, typename N> class kernel_finalise;
template <typename T, typename N> class kernel_mc_lattice;
template <typename T, typename N> class kernel_mc_psum;
template <typename T, typename N> class kernel_mc;
template <typename T, typename N> class kernel_write_back;
} // namespace details

using namespace details;
using namespace sph;

template <typename N, typename F> void foreach_1d(const N size, const F &f) {
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(size); ++i) {
    f(i);
  }
}

template <typename N> constexpr N poly6Kernel(const N r, const N factor, const N h) {
  return r <= h ? factor * sycl::pown((h * h) - r * r, 3) : 0.f;
}

template <typename N>
constexpr V<3, N> spikyKernelGradient(const V<3, N> x, const V<3, N> y, //
                                      const N r, const N h, const N factor) {
  return (r >= EPSILON && r <= h) ? (x - y) * (factor * (sycl::pown(h - r, 2) / r)) : V<3, N>();
}

template <typename T, typename N> class Solver final : public sph::Solver<T, N, V> {

private:
  const N h;
  sycl::queue queue;

public:
  explicit Solver(N h, const sycl::device &device) : h(h), queue(device) {}

  sph::Result<T, N, V> advance(const sph::SphParams<T, N, V> &config, //
                               const sph::Scene<T, N, V> &scene,      //
                               std::vector<sph::Particle<T, N, V>> &xs) final {

    auto watch = sph::utils::Stopwatch("advance");

    const auto h = this->h;
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
                                // FIXME needs to actually erase at surface, not spherically
                                if (sycl::distance(drain.centre, x.position) < drain.width) {
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

    auto advected = sycl::malloc_shared<PartiallyAdvected<T, N, V>>(xs.size(), queue);

    auto advect = watch.start("CPU advect+copy");

    const N padding = h * 2;
    const auto minExtent = (config.minBound / config.scale) - padding;
    const auto maxExtent = (config.maxBound / config.scale) + padding;
    const auto extent = ((maxExtent - minExtent) / h).template convert<size_t, sycl::rounding_mode::automatic>();

    foreach_1d(xs.size(), [&](size_t i) {
      const sph::Particle<T, N, V> &p = xs[i];
      if (p.type == sph::Type::Obstacle) return;
      V<3, N> combinedForce = p.mass * config.constantForce;
      for (const sph::Well<T, N, V> &well : scene.wells) {
        const N dist = sycl::distance(p.position, well.centre);
        if (dist < N(75)) {
          const V<3, N> rHat = sycl::normalize(well.centre - p.position);
          const V<3, N> forceWell = sycl::clamp((rHat * well.force * p.mass) / (dist * dist), N(-10), N(10));
          combinedForce += forceWell;
        }
      }
      advected[i].particle = p;
      advected[i].particle.velocity = combinedForce * config.dt + advected[i].particle.velocity;

      auto pStar = (advected[i].particle.velocity * config.dt) + (advected[i].particle.position / config.scale);
      advected[i].pStar = pStar;
      advected[i].zIndex = zCurveGridIndexAtCoordAt(N(advected[i].pStar.x() - minExtent.x()), //
                                                    N(advected[i].pStar.y() - minExtent.y()), //
                                                    N(advected[i].pStar.z() - minExtent.z()), //
                                                    h);
    });
    advect();

    const auto sortz = watch.start("CPU sortz");
    std::sort(advected, advected + xs.size(), [](auto &l, auto &r) { return l.zIndex < r.zIndex; });
    sortz();

    const auto wait = [this, &config]() {
      if (config.wait) queue.wait_and_throw();
    };

    auto gridtable = watch.start("CPU gridtable");
    const auto gridTable = makeGridTable(extent.x(), extent.y(), extent.z(), xs.size(),
                                         [&advected](auto i) { return advected[i].zIndex; });
    const auto gridTableN = gridTable.size();
    auto hostGridTable = sycl::malloc_shared<size_t>(gridTableN, queue);
    queue.memcpy(hostGridTable, gridTable.data(), gridTableN * sizeof(size_t)).wait();
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

        size_t zIdx = zCurveGridIndexAtCoordAt<N>(r.x(), r.y(), r.z(), h);
        if (zIdx < gridTableN && zIdx + 1 < gridTableN) {
          for (size_t a = hostGridTable[zIdx]; a < hostGridTable[zIdx + 1]; a++) {
            if (advected[a].particle.type != sph::Type::Fluid) continue;
            neighbours.push_back(advected[a].particle.id);
          }
        }
      }
      queryResults[i] = {q.id, q.point, neighbours};
    });

    query();

    const auto diffuse = watch.start("\t[CPU] sph-diffuse ");
    queue.parallel_for<kernel_diffuse<T, N>>(sycl::range<1>(xs.size()), [=](sycl::id<1> id) {
      if (advected[id[0]].particle.type == sph::Type::Obstacle) return;
      int nNeighbours = 0;
      V<4, N> mixture{};
      foreach_grid(advected[id[0]].zIndex, hostGridTable, gridTableN, [&](size_t b) {
        if (advected[b].particle.type != sph::Type::Obstacle) {
          mixture += advected[b].particle.colour;
          nNeighbours++;
        }
      });
      if (nNeighbours != 0) {
        auto a = advected[id[0]].particle.colour;
        auto b = (mixture / N(nNeighbours)) * N(1.33);
        auto out = sycl::mix(a, b, config.dt / N(750.0));
        advected[id[0]].particle.colour = sycl::clamp(out, N(0.03), N(1.0));
      }
    });
    wait();
    diffuse();

    auto lambda_delta = watch.start("\t[GPU] sph-lambda/delta*" + std::to_string(config.iteration));

    const N Poly6Factor = poly6Factor(h);
    const N SpikyKernelFactor = spikyKernelFactor(h);
    const N P6DeltaQ = poly6Kernel(CorrDeltaQ * h, Poly6Factor, h);

    for (size_t itr = 0; itr < config.iteration; ++itr) {
      // lambda

      queue.parallel_for<kernel_lambda<T, N>>(sycl::range<1>(xs.size()), [=](sycl::id<1> id) {
        if (advected[id[0]].particle.type == sph::Type::Obstacle) {
          advected[id[0]].lambda = 0;
          return;
        }
        V<3, N> norm2V{};
        N rho = 0;
        foreach_grid(advected[id[0]].zIndex, hostGridTable, gridTableN, [&](size_t b) {
          const N r = sycl::distance(advected[id[0]].pStar, advected[b].pStar);
          norm2V +=
              spikyKernelGradient(advected[id[0]].pStar, advected[b].pStar, r, h, SpikyKernelFactor) * N(RHO_RECIP);
          rho += advected[id[0]].particle.mass * poly6Kernel(r, Poly6Factor, h);
        });
        N norm2 = sycl::dot(norm2V, norm2V);
        N Ci = (rho / RHO - N(1));
        advected[id[0]].lambda = -Ci / (norm2 + CFM_EPSILON);
      });
      wait();

      // delta

      queue.parallel_for<kernel_delta<T, N>>(sycl::range<1>(xs.size()), [=](sycl::id<1> id) {
        if (advected[id[0]].particle.type == sph::Type::Obstacle) return;
        V<3, N> deltaPAcc{};
        foreach_grid(advected[id[0]].zIndex, hostGridTable, gridTableN, [&](size_t b) {
          const N r = sycl::distance(advected[id[0]].pStar, advected[b].pStar);
          const N corr = -CorrK * sycl::pow(poly6Kernel(r, Poly6Factor, h) / P6DeltaQ, N(CorrN));
          const N factor = (advected[id[0]].lambda + advected[b].lambda + corr) / RHO;
          deltaPAcc += spikyKernelGradient(advected[id[0]].pStar, advected[b].pStar, r, h, SpikyKernelFactor) * factor;
        });
        advected[id[0]].deltaP = deltaPAcc;
        V<3, N> pos = (advected[id[0]].pStar + advected[id[0]].deltaP) * config.scale;
        pos = min(config.maxBound, max(config.minBound, pos)); // clamp to extent
        advected[id[0]].pStar = pos / config.scale;
      });
      wait();
    }
    lambda_delta();

    auto finalise = watch.start("\t[GPU] sph-finalise");



    queue.parallel_for<kernel_finalise<T, N>>(sycl::range<1>(xs.size()), [=](sycl::id<1> id) {
      if (advected[id[0]].particle.type == sph::Type::Obstacle) return;
      const auto deltaX = advected[id[0]].pStar -
                          advected[id[0]].particle.position / config.scale;
      advected[id[0]].particle.position = advected[id[0]].pStar * config.scale;
      advected[id[0]].particle.velocity = (deltaX * (N(1) / config.dt) +
                                       advected[id[0]].particle.velocity) *
                                      N(VD);
    });





wait();
    finalise();

    std::vector<V<3, N>> _outVxs;
    std::vector<V<3, N>> _outNxs;
    std::vector<V<4, N>> _outCxs;

    if (!config.surface) {

      const auto create_field = watch.start("\t[GPU] mc-field");

      const N particleSize = 25;
      const N particleInfluence = 0.5;
      const V<3, size_t> sampleSize =
          (sycl::floor(extent.template convert<N, sycl::rounding_mode::rtz>() * config.surface->resolution))
              .template convert<size_t, sycl::rounding_mode::rtz>() +
          V<3, size_t>(1, 1, 1);
      const size_t latticeN = sampleSize.x() * sampleSize.y() * sampleSize.z();

      auto latticePNs = sycl::malloc_shared<V<4, N>>(latticeN, queue);
      auto latticeCs = sycl::malloc_shared<V<4, N>>(latticeN, queue);

      queue.parallel_for<kernel_mc_lattice<T, N>>(
          sycl::range<3>(sampleSize.x(), sampleSize.y(), sampleSize.z()), [=](sycl::id<3> id) {
            const V<3, N> pos(id[0], id[1], id[2]);
            const N step = h / config.surface->resolution;
            const V<3, N> a = (minExtent + (pos * step)) * config.scale;
            const N threshold = h * config.scale * 1;

            const size_t zIndex = zCurveGridIndexAtCoord((size_t)(N(pos.x()) / config.surface->resolution),
                                                         (size_t)(N(pos.y()) / config.surface->resolution),
                                                         (size_t)(N(pos.z()) / config.surface->resolution));
            const size_t zX = coordAtZCurveGridIndex0(zIndex);
            const size_t zY = coordAtZCurveGridIndex1(zIndex);
            const size_t zZ = coordAtZCurveGridIndex2(zIndex);

            if (zX == extent.x() && zY == extent.y() && zZ == extent.z()) {
              // XXX there is exactly one case where this may happen: the last element of the z-curve
              return;
            }

            const size_t x_l = sycl::clamp(((int)zX) - 1, 0, (int)extent.x() - 1);
            const size_t x_r = sycl::clamp(((int)zX) + 1, 0, (int)extent.x() - 1);
            const size_t y_l = sycl::clamp(((int)zY) - 1, 0, (int)extent.y() - 1);
            const size_t y_r = sycl::clamp(((int)zY) + 1, 0, (int)extent.y() - 1);
            const size_t z_l = sycl::clamp(((int)zZ) - 1, 0, (int)extent.z() - 1);
            const size_t z_r = sycl::clamp(((int)zZ) + 1, 0, (int)extent.z() - 1);

            std::array<size_t, 27> offsets = {
                zCurveGridIndexAtCoord(x_l, y_l, z_l), zCurveGridIndexAtCoord(zX, y_l, z_l),
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

            N v{};
            V<3, N> normal{};
            V<4, N> colour{};
            size_t nNeighbours = 0;
            foreach_grid(hostGridTable, gridTableN, offsets, [&](size_t b) {
              if (advected[b].particle.type != sph::Type::Obstacle &&
                  sycl::distance(advected[b].particle.position, a) < threshold) {
                const auto l = advected[b].particle.position - a;
                const N len = sycl::length(l);
                const N denominator = sycl::pow(len, particleInfluence);
                v += (particleSize / denominator);
                normal += (-particleInfluence) * particleSize * (l / denominator);
                colour += advected[b].particle.colour;
                nNeighbours++;
              }
            });

            normal = sycl::normalize(normal);

            const size_t idx =
                index3d(id[0], id[1], id[2], size_t(sampleSize.x()), size_t(sampleSize.y()), size_t(sampleSize.z()));

            latticePNs[idx] = V<4, N>(v, normal.x(), normal.y(), normal.z());

            latticeCs[idx] = colour / N(nNeighbours);
          });
      wait();

      create_field();

      const auto partial_trig_sum = watch.start("\t[GPU] mc_psum");

      const V<3, size_t> marchRange = sampleSize - V<3, size_t>(1, 1, 1);
      const size_t marchVolume = marchRange.x() * marchRange.y() * marchRange.z();

      auto group_size =
          std::max(size_t(32), std::min(queue.get_device().template get_info<sycl::info::device::max_work_group_size>(),
                                        size_t(256)));
      auto num_group = std::ceil(N(marchVolume) / group_size);

      std::cout << "G=" << group_size << " n=" << num_group << "\n";

      auto partialSums = sycl::malloc_shared<size_t>(num_group, queue);

      const std::array<V<3, size_t>, 8> CUBE_OFFSETS = {
          V<3, size_t>(0, 0, 0), V<3, size_t>(1, 0, 0), V<3, size_t>(1, 1, 0), V<3, size_t>(0, 1, 0),
          V<3, size_t>(0, 0, 1), V<3, size_t>(1, 0, 1), V<3, size_t>(1, 1, 1), V<3, size_t>(0, 1, 1)};

      queue
          .submit([&](sycl::handler &H) {
            auto localSums = sycl::accessor<size_t, 1,                      //
                                            sycl::access::mode::read_write, //
                                            sycl::access::target::local     //
                                            >(group_size, H);

            H.parallel_for<kernel_mc_psum<T, N>>(
                sycl::nd_range<1>(num_group * group_size, group_size), [=](sycl::nd_item<1> item) {
                  uint nVert = 0;
                  // because global size needs to be divisible by local group size (CL1.2), we discard the padding
                  if (item.get_global_id(0) >= marchVolume) {
                    // NOOP
                  } else {
                    const auto pos =
                        sph::utils::to3d<V>(item.get_global_id(0), marchRange.x(), marchRange.y(), marchRange.z());
                    size_t ci = 0;
                    for (int i = 0; i < 8; ++i) {
                      const auto offset = CUBE_OFFSETS[i] + pos;
                      const N v = latticePNs[index3d(offset.x(), offset.y(), offset.z(), sampleSize.x(), sampleSize.y(),
                                                     sampleSize.z())]
                                      .x();
                      ci = v < config.surface->isolevel ? ci | (1 << i) : ci;
                    }
                    nVert = EdgeTable[ci] == 0 ? 0u : (size_t)NumVertsTable[ci] / 3;
                  }

                  const size_t localId = item.get_local_id(0);
                  const size_t groupSize = item.get_local_range()[0];

                  // zero out local memory first, this is needed because workgroup size might not divide
                  // group size perfectly; we need to zero out trailing cells
                  if (localId == 0) {
                    for (size_t i = 0; i < groupSize; ++i)
                      localSums[i] = 0;
                  }
                  item.barrier(sycl::access::fence_space::local_space);

                  localSums[localId] = nVert;

                  for (size_t stride = groupSize / 2; stride > 0; stride >>= 1u) {
                    item.barrier(sycl::access::fence_space::local_space);
                    if (localId < stride) localSums[localId] += localSums[localId + stride];
                  }

                  if (localId == 0) {
                    partialSums[item.get_group(0)] = localSums[0];
                  }
                });
          })
          .wait();
      wait();

      size_t numTrigs = 0;
      for (size_t i = 0; i < num_group; ++i)
        numTrigs += partialSums[i];

      std::cout << "Vol=" << marchVolume << "  trigs=" << numTrigs << " "
                << "\n";

      partial_trig_sum();
      //
      const auto gpu_mc = watch.start("\t[GPU] gpu_mc");

      _outVxs.resize(numTrigs * 3);
      _outNxs.resize(numTrigs * 3);
      _outCxs.resize(numTrigs * 3);

      if (numTrigs != 0) {

        auto outVxs = sycl::malloc_shared<V<3, N>>(_outVxs.size(), queue);
        auto outNxs = sycl::malloc_shared<V<3, N>>(_outNxs.size(), queue);
        auto outCxs = sycl::malloc_shared<V<4, N>>(_outCxs.size(), queue);

        size_t counter = 0;
        sycl::buffer<size_t> trigCounterBuffer(&counter, 1);

        queue
            .submit([&](sycl::handler &H) {
              auto trigCounter = trigCounterBuffer.template get_access<sycl::access::mode::atomic>(H);

              H.parallel_for<kernel_mc<T, N>>(sycl::range<1>(marchVolume), [=](sycl::id<1> id) {
                const auto pos = sph::utils::to3d<V>(id[0], size_t(sampleSize.x()) - 1, size_t(sampleSize.y()) - 1,
                                                     size_t(sampleSize.z()) - 1);
                const auto isolevel = config.surface->isolevel;
                const auto step = h / config.surface->resolution;

                std::array<N, 8> values;
                std::array<V<3, N>, 8> offsets;
                std::array<V<3, N>, 8> normals;
                std::array<V<4, N>, 8> colours;

                size_t ci = 0;
                for (int i = 0; i < 8; ++i) {
                  const auto offset = CUBE_OFFSETS[i] + pos;
                  const size_t idx =
                      index3d(offset.x(), offset.y(), offset.z(), sampleSize.x(), sampleSize.y(), sampleSize.z());
                  const auto point = latticePNs[idx];

                  values[i] = point.x();
                  offsets[i] = (minExtent + ((offset).template convert<N, sycl::rounding_mode::automatic>() * step)) *
                               config.scale;
                  normals[i] = V<3, N>(point.y(), point.z(), point.w());
                  colours[i] = latticeCs[idx];
                  ci = values[i] < isolevel ? ci | (1 << i) : ci;
                }

                std::array<V<3, N>, 12> ts;
                std::array<V<3, N>, 12> ns;
                std::array<V<4, N>, 12> cs;

                const auto lerpAll = [&](size_t index, size_t from, size_t to, N v0, N v1) {
                  const N t = sph::utils::scale(isolevel, v0, v1);
                  ts[index] = sycl::mix(offsets[from], offsets[to], t);
                  ns[index] = sycl::mix(normals[from], normals[to], t);
                  cs[index] = sycl::mix(colours[from], colours[to], t);
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

                  const auto trigIndex = trigCounter[0].fetch_add(1);

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
            })
            .wait();

        wait();

        queue.memcpy(_outVxs.data(), outVxs, sizeof(V<3, N>) * _outVxs.size());
        queue.memcpy(_outNxs.data(), outNxs, sizeof(V<3, N>) * _outNxs.size());
        queue.memcpy(_outCxs.data(), outCxs, sizeof(V<4, N>) * _outCxs.size());
        queue.wait();

        wait();

        sycl::free(outVxs, queue);
        sycl::free(outNxs, queue);
        sycl::free(outCxs, queue);
      }

      sycl::free(latticeCs, queue);
      sycl::free(latticePNs, queue);
      wait();

      gpu_mc();
    }

    auto writeback = watch.start("\t[GPU] write back");
    //
    auto source = sycl::malloc_shared<sph::Particle<T, N, V>>(xs.size(), queue);
    queue
        .parallel_for<kernel_write_back<T, N>>(sycl::range<1>(xs.size()),
                                               [=](sycl::id<1> id) { source[id[0]] = advected[id[0]].particle; })
        .wait();
    wait();

    std::copy(source, source + xs.size(), xs.begin());

    sycl::free(source, queue);
    sycl::free(advected, queue);
    sycl::free(hostGridTable, queue);

    wait();

    writeback();
    std::cout << watch << std::endl;
    return sph::Result<T, N, V>{sph::ColouredMesh<N, V>{_outVxs, _outNxs, _outCxs}, queryResults};
  }
};

} // namespace sph::sycl2020_impl
#endif
