//#define GLM_ENABLE_EXPERIMENTAL

#include "glm/glm.hpp"
#include "sph.hpp"

#include "ocl/oclsph.hpp"
#include "omp/ompsph.hpp"
#include "sycl/syclsph.hpp"
#include "sycl/syclsph_2020.hpp"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/render/shaders.h"
#include "polyscope_extra.hpp"

#include <mutex>
#include <thread>

#include "args.hpp"

#ifdef USE_SYCL
using sph::sycl_impl::utils::to_glm3f;
using sph::sycl_impl::utils::to_glm4f;
#endif

using sph::utils::to_glm3f;
using sph::utils::to_glm4f;

template <typename T, typename N, template <size_t, typename C = N> typename V>
sph::Result<T, N, V> render(sph::Solver<T, N, V> &solver, size_t iterations) {
  polyscope::options::programName = "PBF-SPH";
  polyscope::options::maxFPS = 300;
  polyscope::view::windowWidth = 1280;
  polyscope::view::windowHeight = 720;
  polyscope::view::upDir = polyscope::view::UpDir::NegYUp;
  polyscope::init();

  auto *mesh = new polyscope::extra::SimpleMesh("");
  polyscope::registerStructure(mesh);
  auto *cloud = polyscope::registerPointCloud("", std::vector<glm::tvec3<float>>{});
  cloud->setPointRadius(0.005f);
  //  auto globalC = std::vector<glm::tvec3<float>>{};

  const size_t pcount = 20 * 1000;
  const float scaling = 500; // less = less space between particle

  int solverIter = 3;
  //  auto [initMcParams, initParams, xs] = sph::simpleConfigWith2Cubes<T, N, V>(pcount, solverIter, scaling);

  sph::SphParams<T, N, V> paramsN;
  float dtScalar;
  float scaleF;
  bool surface;
  glm::tvec3<float> cff;
  std::array<float, 3> constantForceF;
  sph::McParams<float> mcParamsF;
  std::vector<sph::Particle<T, N, V>> particles;

  auto bind = [&]() {
    auto [initMcParams, initParams, xs] = sph::simpleConfigWith2Cubes<T, N, V>(pcount, solverIter, scaling);
    paramsN = initParams;
    dtScalar = 1;
    scaleF = initParams.scale;
    glm::tvec3<float> cff = to_glm3f(initParams.constantForce);
    constantForceF = {cff.x, cff.y, cff.z};
    mcParamsF = initMcParams.template as<float>();
    surface = initParams.surface.has_value();
    particles = xs;
  };

  bind();

  //  auto initParamsN = initParams;
  //  sph::McParams<float> mcParamsF = initMcParams.template as<float>();
  //  bool surface = initParams.surface.has_value();

  sph::Result<T, N, V> result;

  std::mutex m;
  std::atomic_bool update(true);
  std::atomic_bool exit(false);

  std::chrono::duration<double, std::milli> frameTime;

  auto t = std::thread([&]() {
    size_t frame = 0;
    while (!exit && ((frame < iterations && iterations > 0) || iterations == 0)) {
      try {
        auto c = sph::applyMotionSinXCosZ(paramsN, frame);
        c.dt = paramsN.dt * dtScalar;
        c.scale = scaleF;
        c.iteration = solverIter;
        c.constantForce = V<3>(N(constantForceF[0]), N(constantForceF[1]), N(constantForceF[2]));
        c.surface = surface ? std::make_optional(mcParamsF.template as<N>()) : std::nullopt;
        auto start = std::chrono::high_resolution_clock::now();
        auto rr = solver.advance(c, {}, particles);
        auto end = std::chrono::high_resolution_clock::now();
        frameTime = end - start;

        //        std::scoped_lock lock(m);
        result = rr;
        update = true;
      } catch (std::exception const &e) {
        std::cout << "Caught asynchronous exception:\n" << e.what() << "\n";
        throw e;
      }
      frame++;
    }
  });

  auto vgt = [](const auto &l, const auto &r) { return l.x > r.x && l.y > r.y && l.z > r.z; };

  glm::tvec3<float> minExtent = {0, 0, 0};
  glm::tvec3<float> maxExtent = {1000, 1000, 1000};

  polyscope::state::userCallback = [&]() {
    std::scoped_lock lock(m);

    ImGui::PushItemWidth(200);

    ImGui::Text("%.2f FPS", 1000.0 / frameTime.count());

    //    ImGui::Begin("MC");
    ImGui::Checkbox("Enable", &surface);
    ImGui::SliderFloat("Resolution", &mcParamsF.resolution, 0.5, 5.0);
    ImGui::SliderFloat("Isolevel", &mcParamsF.isolevel, 10.0, 500.0);
    ImGui::SliderFloat("Size", &mcParamsF.particleSize, 1, 50);
    ImGui::SliderFloat("Influence", &mcParamsF.particleInfluence, 0.1, 1);
    //    ImGui::End();

    //    ImGui::Begin("SPH");
    ImGui::SliderInt("Iteration", &solverIter, 1, 100);
    ImGui::SliderFloat("DeltaT", &dtScalar, 0.1, 10.0);
    ImGui::SliderFloat("Scale", &scaleF, 10.0, 500.0);
    ImGui::SliderFloat3("Constant Force", constantForceF.data(), 0, 30);

    if (ImGui::Button("Reset")) {
      bind();
      return;
    }
    ImGui::PopItemWidth();

    //    ImGui::End();

    //    ImGui::InputInt("num points", &params.wait);             // set a int variable
    //    ImGui::InputFloat("param value", &anotherParam);  // set a float variable

    if (!update) return;

    //        auto result = solver->advance(config, particles);

    std::vector<glm::tvec3<float>> pos;
    std::transform(particles.begin(), particles.end(), std::back_inserter(pos),
                   [](const auto x) { return to_glm3f(x.position); });

    std::cout
        //            << "Min=" << sph::utils::to_string((*min)) << "Max= " << sph::utils::to_string((*max)) << "\n"
        << "MM=" << sph::utils::to_string((minExtent)) << "MM= " << sph::utils::to_string((maxExtent)) << "\n"
        << "\n";

    std::vector<glm::tvec3<float>> col;
    std::transform(particles.begin(), particles.end(), std::back_inserter(col),
                   [](const auto x) { return to_glm4f(x.colour); });

    cloud->updatePointPositions(pos);
    auto cc = cloud->addColorQuantity("colour", col);
    cc->setEnabled(true);

    // GL is float only
    const auto size = result.mesh.vs.size();
    std::vector<glm::tvec3<float>> vs123(size);
    std::vector<glm::tvec3<float>> ns123(size);
    std::vector<glm::tvec3<float>> cs123(size);
    for (size_t i = 0; i < size; ++i) {
      vs123[i] = to_glm3f(result.mesh.vs[i]);
      ns123[i] = to_glm3f(result.mesh.ns[i]);
      cs123[i] = to_glm4f(result.mesh.cs[i]);
    }
    mesh->update(vs123, ns123, cs123);

    polyscope::requestRedraw();

    auto [min, max] = std::minmax_element(pos.begin(), pos.end(), vgt);
    auto minBound = glm::any(glm::isnan(*min)) ? glm::zero<glm::tvec3<float>>() : *min;
    auto maxBound = glm::any(glm::isnan(*max)) ? glm::zero<glm::tvec3<float>>() : *max;
    if (vgt(minExtent, minBound)) minExtent = glm::clamp(minBound, {0, 0, 0}, {2000, 2000, 2000});
    if (vgt(maxBound, maxExtent)) maxExtent = glm::clamp(maxBound, {0, 0, 0}, {2000, 2000, 2000});
    polyscope::state::boundingBox = std::make_tuple(minExtent, maxExtent);
    polyscope::state::lengthScale = glm::length(maxExtent - minExtent);
    polyscope::state::center = 0.5f * (minExtent + maxExtent);
    update = false;
  };

  polyscope::show();
  exit = true;
  return result;
}

int main(int argc, char *argv[]) {


//  impl::Solver<size_t, double> solver(...); // create solver
//  vector<Particle<size_t, double, impl::V> particles = ...; // initial particles
//  sph::SphParams<size_t, double, impl::V> config = { // setup
//      .dt = 0.0083 * 1.5,
//      .scale = 500.0,
//      .iteration = 4,
//      .constantForce = impl::V<3>(0, 9.8 , 0),
//      .minBound = impl::V<3>(0, 0, 0),
//      .maxBound = impl::V<3>(1000, 1000, 1000)
//  };
//
//  while(true){ // run the solver
//    auto mesh = solver.advance(config, particles);
//    // draw mesh and particles
//  }



  sph::driver::Args args(0, "");
  if (args.parse(argc, argv)) {
    auto output = args.renderedOutputName();
    auto run = [&](auto &&x) {
      std::cout << "Using " << output << " for output" << std::endl;
      auto r = render(x, args.iterations.Get());
      std::cout << "Benchmark completed after " << args.iterations.Get() << " frames" << std::endl;
      save(r, output);
      std::cout << "Results flushed." << std::endl;
    };
    std::cout << "Backend: " << args.implName() << std::endl;

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