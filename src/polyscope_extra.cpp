#include <tuple>

#include "polyscope/polyscope.h"
#include "polyscope/render/shaders.h"
#include "polyscope_extra.hpp"

using glm::tvec3;
using glm::tvec4;

void polyscope::extra::SimpleMesh::update(const std::vector<tvec3<float>> &vs, //
                                          const std::vector<tvec3<float>> &ns, //
                                          const std::vector<tvec3<float>> &cs) {
  const auto set = [](auto &xs, const auto &ys) {
    if (xs.size() < ys.size()) xs.assign(ys.begin(), ys.end());
    else
      std::fill(std::copy(ys.begin(), ys.end(), xs.begin()), xs.end(), tvec3<float>{});
  };
  size_t size = vertices.size();
  set(vertices, vs);
  set(normals, ns);
  set(colours, cs);
  resetVBO = size != vertices.size();
}

polyscope::extra::SimpleMesh::SimpleMesh(std::string name) : Structure(name, typeName()) {}

void polyscope::extra::SimpleMesh::draw() {
  if (!isEnabled() || vertices.empty()) return;
  if (program == nullptr) {
    program = polyscope::render::engine->generateShaderProgram(
        {polyscope::render::VERTCOLOR3_SURFACE_VERT_SHADER, polyscope::render::VERTCOLOR3_SURFACE_FRAG_SHADER},
        polyscope::DrawMode::Triangles);
  }

  if (resetVBO) {
    program->setAttribute("a_position", vertices);
    program->setAttribute("a_normal", normals);
    program->setAttribute("a_colorval", colours);
    resetVBO = false;
  } else {
    program->setAttribute("a_position", vertices, true, 0, -1);
    program->setAttribute("a_normal", normals, true, 0, -1);
    program->setAttribute("a_colorval", colours, true, 0, -1);
  }
  polyscope::render::engine->setMaterial(*program, "clay");
  setTransformUniforms(*program);
  program->draw();
}
void polyscope::extra::SimpleMesh::drawPick() {}
void polyscope::extra::SimpleMesh::buildCustomUI() {}
void polyscope::extra::SimpleMesh::buildCustomOptionsUI() { Structure::buildCustomOptionsUI(); }
void polyscope::extra::SimpleMesh::buildQuantitiesUI() { Structure::buildQuantitiesUI(); }
void polyscope::extra::SimpleMesh::buildSharedStructureUI() { Structure::buildSharedStructureUI(); }
void polyscope::extra::SimpleMesh::buildPickUI(unsigned long localPickID) {}

std::tuple<tvec3<float>, tvec3<float>> polyscope::extra::SimpleMesh::boundingBox() {

  tvec3<float> min = tvec3<float>{1, 1, 1} * std::numeric_limits<float>::infinity();
  tvec3<float> max = -tvec3<float>{1, 1, 1} * std::numeric_limits<float>::infinity();
  for (tvec3<float> pOrig : vertices) {
    tvec3<float> p = tvec3<float>(objectTransform * tvec4<float>(pOrig, 1.0));
    min = componentwiseMin(min, p);
    max = componentwiseMax(max, p);
  }
  return std::make_tuple(min, max);
}
double polyscope::extra::SimpleMesh::lengthScale() {
  // Measure length scale as twice the radius from the center of the bounding box
  auto bound = boundingBox();
  tvec3<float> center = 0.5f * (std::get<0>(bound) + std::get<1>(bound));

  double lengthScale = 0.0;
  for (tvec3<float> p : vertices) {
    tvec3<float> transPos = tvec3<float>(objectTransform * tvec4<float>(p.x, p.y, p.z, 1.0));
    lengthScale = std::max(lengthScale, (double)glm::length2(transPos - center));
  }

  return 2 * std::sqrt(lengthScale);
}
std::string polyscope::extra::SimpleMesh::typeName() { return "Simple Mesh"; }

// class SimpleMesh : public polyscope::Structure {
// private:
//   std::shared_ptr<polyscope::render::ShaderProgram> program;
//   bool resetVBO = false;
//
//   inline vec<3, float, defaultp> componentwiseMin(const vec<3, float, defaultp> &vA, const vec<3, float, defaultp>
//   &vB) {
//     return vec<3, float, defaultp>{std::min(vA.x, vB.x), std::min(vA.y, vB.y), std::min(vA.z, vB.z)};
//   }
//   inline vec<3, float, defaultp> componentwiseMax(const vec<3, float, defaultp> &vA, const vec<3, float, defaultp>
//   &vB) {
//     return vec<3, float, defaultp>{std::max(vA.x, vB.x), std::max(vA.y, vB.y), std::max(vA.z, vB.z)};
//   }
//
// public:
//   std::vector<tvec3<float>> vertices;
//   std::vector<tvec3<float>> normals;
//   std::vector<tvec3<float>> colours;
//
//   void update(const std::vector<tvec3<float>> &vs, const std::vector<tvec3<float>> &ns,
//               const std::vector<tvec3<float>> &cs) {
//     const auto set = [](auto xs, const auto ys) {
//       if (xs.size() < ys.size()) xs = ys;
//       else
//         std::fill(std::copy(ys.begin(), ys.end(), xs.begin()), xs.end(), tvec3<float>{});
//     };
//     unsigned long size = vertices.size();
//     set(vertices, vs);
//     set(normals, ns);
//     set(colours, cs);
//     resetVBO = size != vertices.size();
//   }
//
//   explicit SimpleMesh(basic_string<char> name) : Structure(name, typeName()) {}
//   ~SimpleMesh() override = default;
//   void draw() override {
//     if (!isEnabled()) return;
//
//     if (program == nullptr) {
//       program = polyscope::render::engine->generateShaderProgram(
//           {polyscope::render::VERTCOLOR3_SURFACE_VERT_SHADER, polyscope::render::VERTCOLOR3_SURFACE_FRAG_SHADER},
//           polyscope::DrawMode::Triangles);
//     }
//
//     if (resetVBO) {
//       program->setAttribute("a_position", vertices);
//       program->setAttribute("a_normal", normals);
//       program->setAttribute("a_colorval", colours);
//     } else {
//       program->setAttribute("a_position", vertices, true, 0, -1);
//       program->setAttribute("a_normal", normals, true, 0, -1);
//       program->setAttribute("a_colorval", colours, true, 0, -1);
//     }
//     polyscope::render::engine->setMaterial(*program, "clay");
//     setTransformUniforms(*program);
//     program->draw();
//   }
//   void drawPick() override {}
//   void buildCustomUI() override {}
//   void buildCustomOptionsUI() override { Structure::buildCustomOptionsUI(); }
//   void buildQuantitiesUI() override { Structure::buildQuantitiesUI(); }
//   void buildSharedStructureUI() override { Structure::buildSharedStructureUI(); }
//   void buildPickUI(unsigned long localPickID) override {}
//   std::tuple<vec<3, float, defaultp>, vec<3, float, defaultp>> boundingBox() override {
//
//     vec<3, float, defaultp> min = vec<3, float, defaultp>{1, 1, 1} * std::numeric_limits<float>::infinity();
//     vec<3, float, defaultp> max = -vec<3, float, defaultp>{1, 1, 1} * std::numeric_limits<float>::infinity();
//     for (vec<3, float, defaultp> pOrig : vertices) {
//       vec<3, float, defaultp> p = vec<3, float, defaultp>(objectTransform * vec<4, float, defaultp>(pOrig, 1.0));
//       min = componentwiseMin(min, p);
//       max = componentwiseMax(max, p);
//     }
//     return std::make_tuple(min, max);
//   }
//   double lengthScale() override {
//     // Measure length scale as twice the radius from the center of the bounding box
//     auto bound = boundingBox();
//     vec<3, float, defaultp> center = 0.5f * (std::get<0>(bound) + std::get<1>(bound));
//
//     double lengthScale = 0.0;
//     for (vec<3, float, defaultp> p : vertices) {
//       vec<3, float, defaultp> transPos =
//           vec<3, float, defaultp>(objectTransform * vec<4, float, defaultp>(p.x, p.y, p.z, 1.0));
//       lengthScale = std::max(lengthScale, (double)glm::length2(transPos - center));
//     }
//
//     return 2 * std::sqrt(lengthScale);
//   }
//   basic_string<char> typeName() override { return "Simple Mesh"; }
// };