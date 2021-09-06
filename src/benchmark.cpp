#include "sph.hpp"

#include "ocl/oclsph.hpp"
#include "omp/ompsph.hpp"
#include "sycl/syclsph.hpp"
#include "sycl/syclsph_2020.hpp"

#include <mutex>
#include <optional>
#include <thread>

#include "args.hpp"

using hrc_timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using duration_millis = std::chrono::duration<double, std::milli>;

template <typename T, typename N, template <size_t, typename C = N> typename V>
std::tuple<duration_millis,              //
           std::vector<duration_millis>, //
           sph::Result<T, N, V>,         //
           std::vector<sph::Particle<T, N, V>>>
runN(sph::Solver<T, N, V> &solver, size_t iterations, size_t warmup) {
  const size_t pcount = 20 * 1000;
  const size_t solverIter = 6;
  const float scaling = 500; // less = less space between particle
  auto [initialMcParam, initialParam, particles] = sph::simpleConfigWith2Cubes<T, N, V>(pcount, solverIter, scaling);
  sph::Result<T, N, V> result;
  auto param = initialParam;
  param.surface = initialMcParam;

  for (size_t frame = 0; frame < warmup; ++frame) {
    try {
      result = solver.advance(sph::applyMotionSinXCosZ(param, frame), {}, particles);
    } catch (std::exception const &e) {
      std::cout << "Caught asynchronous exception at warmup frame" << frame << ":\n" << e.what() << "\n";
      throw e;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();

  std::vector<duration_millis> frameTime;
  for (size_t frame = 0; frame < iterations; ++frame) {
    auto frameStart = std::chrono::high_resolution_clock::now();
    try {

      result = solver.advance(sph::applyMotionSinXCosZ(param, frame), {}, particles);
    } catch (std::exception const &e) {
      std::cout << "Caught asynchronous exception at benchmark frame" << frame << ":\n" << e.what() << "\n";
      throw e;
    }
    auto frameEnd = std::chrono::high_resolution_clock::now();
    frameTime.emplace_back(frameEnd - frameStart);
  }
  auto end = std::chrono::high_resolution_clock::now();

  return {duration_millis(end - start), frameTime, result, particles};
}

template <typename N, typename T>
std::tuple<N, N, N, N, N> summaryStats(const std::vector<T> &xs, const std::function<N(const T &)> f) {

  std::vector<N> ys;
  std::transform(xs.begin(), xs.end(), std::back_inserter(ys), f);

  N sum = std::accumulate(ys.begin(), ys.end(), N(0));
  N mean = sum / ys.size();
  N variance =
      std::accumulate(ys.begin(), ys.end(), N(0), [&](auto acc, auto t) { return acc + std::pow(t - mean, 2); }) /
      xs.size();
  N stdDev = std::sqrt(variance);
  auto [min, max] = std::minmax_element(ys.begin(), ys.end());

  return {*min, *max, mean, variance, stdDev};
}

int main(int argc, char *argv[]) {
  sph::driver::Args args(200, "./out_{impl}_{type}_{iter}");
  if (args.parse(argc, argv)) {
    auto output = args.renderedOutputName();
    auto run = [&](auto &&x) {
      std::cout << "Using " << output << " for output" << std::endl;
      auto [elapsed, frameTime, result, particles] = runN(x, args.iterations.Get(), args.warmup.Get());
      auto frames = args.iterations.Get();
      auto seconds = elapsed.count() / 1000.0;
      auto fps = frames / seconds;

      auto [min, max, mean, _, stdDev] =
          summaryStats<double, duration_millis>(frameTime, [](const auto &t) { return t.count(); });

      std::cout << "Benchmark completed after " << frames << " frames:\n"
                << std::setprecision(4) //
                << "Runtime              : " << seconds << " s\n"
                << "Framerate            : " << fps << " fps\n"
                << "Frame-time min       : " << min << " ms\n"
                << "Frame-time max       : " << max << " ms\n"
                << "Frame-time mean       : " << mean << " ms\n"
                << "Frame-time stdDev     : " << stdDev << " ms\n"
                << "Final Vertex count   : " << result.mesh.vs.size() << "\n"
                << "Final Particle count : " << particles.size() << " \n"
                << std::endl;
      save(result, output);
      std::cout << "Results flushed." << std::endl;
    };
    switch (args.impl.Get()) {
    case sph::driver::Impl::SYCL2020:
    case sph::driver::Impl::SYCL: {
#ifdef USE_SYCL
      using namespace sph::sycl_impl::utils;
      if (auto device = sph::utils::findDevice<sycl::device>(args.list, args.verbose, args.devices.Get(),
                                                             &enumeratePlatform, &listDevices, &showDevice)) {
        auto _2020 = args.impl.Get() == sph::driver::Impl::SYCL2020;
        if (args.fp64) {
          if (_2020) {
            auto s = sph::sycl2020_impl::Solver<size_t, double>(0.1, *device);
            run(s);
          } else {
            auto s = sph::sycl_impl::Solver<size_t, double>(0.1, *device);
            run(s);
          }
        } else {
          if (_2020) {
            auto s = sph::sycl2020_impl::Solver<size_t, double>(0.1, *device);
            run(s);
          } else {
            auto s = sph::sycl_impl::Solver<size_t, double>(0.1, *device);
            run(s);
          }
        }
      }
#else
      std::cerr << "SYCL unavailable: program not compiled with USE_SYCL" << std::endl;
#endif
      break;
    }
    case sph::driver::Impl::OCL: {
      using namespace sph::ocl_impl::utils;
      if (auto device = sph::utils::findDevice<cl::Device>(args.list, args.verbose, args.devices.Get(),
                                                           &enumeratePlatform, &listDevices, &showDevice)) {
        if (args.fp64) {
          std::cerr << "FP64 is not supported for OCL!" << std::endl;
        } else {
          const std::string kernelRoot = "/home/tom/pbf-sph/src/ocl/";
          const std::string kernelIncl = "/home/tom/pbf-sph/src/";
          auto s = sph::ocl_impl::Solver(0.1, kernelRoot + "oclsph_kernel.h", {kernelRoot, kernelIncl}, *device);
          run(s);
        }
      }
      break;
    }
    case sph::driver::Impl::OMP: {
      if (auto device = sph::utils::findDevice<std::string>(
              args.list, args.verbose, args.devices.Get(), [](auto &, auto &) {}, //
              []() {
                return std::vector<std::pair<size_t, std::string>>{{0, "CPU"}};
              },                        //
              [](auto d) { return d; }) //
      ) {
        if (args.fp64) {
          auto s = sph::omp_impl::Solver<size_t, double>(0.1);
          run(s);
        } else {
          auto s = sph::omp_impl::Solver<size_t, float>(0.1);
          run(s);
        }
      }
      break;
    }
    default:
      throw std::runtime_error("Unexpected implementation type: " +
                               std::to_string(static_cast<uint8_t>(args.impl.Get())));
    }
  }
  return EXIT_SUCCESS;
}
