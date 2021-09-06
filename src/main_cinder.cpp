//#define GLM_ENABLE_EXPERIMENTAL

#include <iostream>

#include <cinder/CameraUi.h>
#include <cinder/TriMesh.h>
#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>

#include <cinder/CinderImGui.h>

#include "sph.hpp"
////#include "glm/glm.hpp"
//
#include "ocl/oclsph.hpp"
#include "omp/ompsph.hpp"
#include "sycl/syclsph.hpp"

//#include "polyscope/point_cloud.h"
//#include "polyscope/polyscope.h"
//#include "polyscope/render/shaders.h"
//#include "polyscope_extra.hpp"

//#ifdef __SYCL_DEVICE_ONLY__
//#undef __SYCL_DEVICE_ONLY__
//#endif

#include <mutex>
#include <thread>

template <typename T, typename N, template <size_t, typename C = N> typename V>
T makeCube(T offset,
           N spacing,          //
           const size_t count, //
           V<3> origin,        //
           std::vector<fluid::Particle<T, N, V>> &xs) {
  auto len = static_cast<size_t>(std::cbrt(count));
  for (size_t x = 0; x < len; ++x) {
    for (size_t y = 0; y < len; ++y) {
      for (size_t z = 0; z < len; ++z) {
        auto pos = (V<3>(x, y, z) * spacing) + origin;
        //				uint32_t colour = i > half ? 0xFFFF0000 : 0xFF00FF00;
        xs.emplace_back(offset++, fluid::Type::Fluid, 1.0, V<4>(0, 0, 1, 1), pos, V<3>(0));
      }
    }
  }
  return offset;
}

const std::vector<std::string> signatures = { //	        "710",
    "Intel", "gfx1012", "Oclgrind", "gfx906", "Ellesmere", "Quadro", "ATI", "1050", "1080", "980", "NEO", "Tesla"};

// struct MyTag {
//   int a = 0;
//
//   MyTag() {}
//   MyTag(int a) : a(a) {}
//   MyTag &operator++() {
//     a++;
//     return *this;
//   }
//
//   // Define postfix increment operator.
//   MyTag operator++(int) {
//     MyTag temp = *this;
//     ++*this;
//     return temp;
//   }
// };
//

template <typename N> constexpr glm::tvec3<float> cvt3(const cl::sycl::vec<N, 3> &x) {
  return glm::tvec3<float>(x.s0(), x.s1(), x.s2());
}
template <typename N> constexpr glm::tvec4<float> cvt4(const cl::sycl::vec<N, 4> &x) {
  return glm::tvec4<float>(x.s0(), x.s1(), x.s2(), x.s3());
}

template <typename N> constexpr glm::tvec3<float> to_glm3(const glm::tvec3<N> &x) { return glm::tvec3<float>(x); }
template <typename N> constexpr glm::tvec4<float> to_glm4(const glm::tvec4<N> &x) { return glm::tvec4<float>(x); }


struct Output{
  std::vector<glm::tvec3<float>>  pos;
  std::vector<glm::tvec3<float>>  col;
  std::vector<glm::tvec3<float>>  vs123;
  std::vector<glm::tvec3<float>>  ns123;
  std::vector<glm::tvec3<float>>  cs123;
};

template <typename T, typename N, template <size_t, typename C = N> typename V>
std::function<Output()> render(std::atomic_bool &exit,                                //
            const std::unique_ptr<fluid::Solver<T, N, V>> &solver, //
            std::vector<glm::tvec3<float>> &pos,                   //
            std::vector<glm::tvec3<float>> &col,                   //
            std::vector<glm::tvec3<float>> &vs123,                 //
            std::vector<glm::tvec3<float>> &ns123,                 //
            std::vector<glm::tvec3<float>> &cs123) {

  const size_t pcount = 32 * 1000;
  const size_t solverIter = 5;
  const float scaling = 450; // less = less space between particle

  std::vector<fluid::Particle<T, N, V>> prepared;
  T tag = T{};
  tag = makeCube<T, N, V>(tag, 22.f, pcount / 2, V<3>(+100, +1500 * 0, +100), prepared);
  tag = makeCube<T, N, V>(tag, 22.f, pcount / 2, V<3>(+1000, +1500 * 0, 1000), prepared);

  fluid::SphParams<T, N, V> config = {.dt = 0.0083 * 1.f, // var
                                      .scale = scaling,   // var
                                      .resolution = 2.f,  // var
                                      .isolevel = 100,
                                      .iteration = solverIter,
                                      .constantForce = V<3>(0, 9.8 * 3.f, 0),
                                      .minBound = V<3>(0, 0, 0),
                                      .maxBound = V<3>(2000, 2000, 2000)};

  fluid::Scene<T, N, V> scene = {.wells = {}, .sources = {}, .drains = {}, .queries = {}};



   return ([&]() {
     Output o;
      try {
        std::vector<fluid::Particle<T, N, V>> particles = prepared;

        auto rr = solver->advance(config, scene, particles);

        o.pos.resize(particles.size());
        o.col.resize(particles.size());

        std::transform(particles.begin(), particles.end(), std::back_inserter(pos),
                       [](const auto x) { return cvt3(x.position); });
        std::transform(particles.begin(), particles.end(), std::back_inserter(col),
                       [](const auto x) { return cvt4(x.colour); });

        const auto size = rr.mesh.vs.size();
        o.vs123.resize(size);
        o.ns123.resize(size);
        o.cs123.resize(size);
        for (size_t i = 0; i < size; ++i) {
          o.vs123[i] = cvt3(rr.mesh.vs[i]);
          o.ns123[i] = cvt3(rr.mesh.ns[i]);
          o.cs123[i] = cvt4(rr.mesh.cs[i]);
        }

      } catch (sycl::exception const &e) {
        std::cout << "Caught asynchronous exception:\n"
                  << e.what()
                  << "\n"
                  //                  << e.get_file_name() << ":" << e.get_line_number()
                  << std::endl;
      }
      return o;
  });


  //  polyscope::state::userCallback = [&]() {
  //    //        auto result = solver->advance(config, particles);
  //    std::vector<glm::tvec3<float>> pos;
  //    std::vector<glm::tvec3<float>> col;

  //    std::transform(particles.begin(), particles.end(), std::back_inserter(pos),
  //                   [](const auto x) { return cvt3(x.position); });
  //
  //    std::transform(particles.begin(), particles.end(), std::back_inserter(col),
  //                   [](const auto x) { return cvt4(x.colour); });
  //
  //    cloud->updatePointPositions(pos);
  //    auto cc = cloud->addColorQuantity("red", col);
  //
  //    std::scoped_lock lock(m);
  //
  //    // GL is float only
  //    const auto size = result.mesh.vs.size();
  //    std::vector<glm::tvec3<float>> vs123(size);
  //    std::vector<glm::tvec3<float>> ns123(size);
  //    std::vector<glm::tvec3<float>> cs123(size);
  //    for (size_t i = 0; i < size; ++i) {
  //      vs123[i] = cvt3(result.mesh.vs[i]);
  //      ns123[i] = cvt3(result.mesh.ns[i]);
  //      cs123[i] = cvt4(result.mesh.cs[i]);
  //    }
  //    mesh->update(vs123, ns123, cs123);
  //
  ////    polyscope::requestRedraw();
  ////    polyscope::updateStructureExtents();
  //  };

  //  polyscope::show();
}


class Main : public ci::app::App {

  ci::CameraPersp m_cam;
  ci::CameraUi m_cam_ui;
  ci::gl::VboRef m_vbo;
  ci::gl::Texture2dRef m_star_tex;
  ci::gl::GlslProgRef m_shader;
  ci::gl::BatchRef m_batch;

  std::atomic_bool exit{false};

  std::vector<glm::tvec3<float>> pos;
  std::vector<glm::tvec3<float>> col;
  std::vector<glm::tvec3<float>> vs123;
  std::vector<glm::tvec3<float>> ns123;
  std::vector<glm::tvec3<float>> cs123;
  std::function<Output()> update;

public:
  void setup() final;
  void draw() final;
};

void Main::setup() {

  ci::gl::enableDepthWrite();
  ci::gl::enableDepthRead();

  ImGui::Initialize();
  m_cam.lookAt(ci::vec3(200), ci::vec3(0));
  m_cam.setPerspective(60.0f, getWindowAspectRatio(), 0.01f, 50000.0f);
  m_cam_ui = ci::CameraUi(&m_cam, getWindow());

  using Numeric = float;
  using Tag = size_t;

  const auto kernelPaths = "/home/tom/pbs-sph/src/ocl/";
  const auto kernelPaths2 = "/home/tom/pbs-sph/src/";

//  const std::vector<std::string> signatures = {//	        "710",
//                                               "Intel", "gfx1012", "Oclgrind", "gfx906", "Ellesmere", "Quadro",
//                                               "ATI",   "1050",    "1080",     "980",    "NEO",       "Tesla"};
//  sph::ocl::utils::enumeratePlatformToCout();
//
//  const auto device = sph::ocl::utils::resolveDeviceVerbose(signatures);
//
//  std::unique_ptr<fluid::Solver<Tag, Numeric, sph::sycl_impl::V>> solver_sycl = //
//      std::make_unique<sph::sycl_impl::Solver<Tag, Numeric>>(0.1);

//  std::unique_ptr<fluid::Solver<size_t, float, sph::ocl::V>> solver_ocl = //
//      std::make_unique<sph::ocl::Solver>(0.1, kernelPaths, std::vector<std::string>{kernelPaths, kernelPaths2}, device);

  std::unique_ptr<fluid::Solver<Tag, Numeric, sph::omp_impl::V>> solver_omp = //
      std::make_unique<sph::omp_impl::Solver<Tag, Numeric>>(0.1);




 update = render(exit, solver_omp, pos, col, vs123, ns123, cs123);
}

void Main::draw() {
  ci::gl::clear(ci::Color::gray(0.2));
  // Set transform to the camera view
  ci::gl::setMatrices(m_cam);

  // Draw coordinate system arrows
  ci::gl::color(ci::Color(1.0f, 0.0f, 0.0f));
  ci::gl::drawVector(ci::vec3(90, 0, 0), ci::vec3(100, 0, 0), 2, 2);
  ci::gl::color(ci::Color(0.0f, 1.0f, 0.0f));
  ci::gl::drawVector(ci::vec3(0, 90, 0), ci::vec3(0, 100, 0), 2, 2);
  ci::gl::color(ci::Color(0.0f, 0.0f, 1.0f));
  ci::gl::drawVector(ci::vec3(0, 0, 90), ci::vec3(0, 0, 100), 2, 2);

  Output o = update();
//  for (const auto &p : pos){
//    ci::gl::drawSphere(p, 5);
//  }



}

CINDER_APP(Main, ci::app::RendererGl(ci::app::RendererGl::Options{}))
