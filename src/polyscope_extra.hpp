#pragma once

#include "glm/glm.hpp"
#include "polyscope/structure.h"

namespace polyscope::extra {

using namespace polyscope;

using glm::tvec3;
using glm::tvec4;

class SimpleMesh : public polyscope::Structure {
private:
  std::shared_ptr<polyscope::render::ShaderProgram> program;
  bool resetVBO = true;
  std::vector<tvec3<float>> vertices;
  std::vector<tvec3<float>> normals;
  std::vector<tvec3<float>> colours;

  inline tvec3<float> componentwiseMin(const tvec3<float> &vA, const tvec3<float> &vB) {
    return tvec3<float>{std::min(vA.x, vB.x), std::min(vA.y, vB.y), std::min(vA.z, vB.z)};
  }
  inline tvec3<float> componentwiseMax(const tvec3<float> &vA, const tvec3<float> &vB) {
    return tvec3<float>{std::max(vA.x, vB.x), std::max(vA.y, vB.y), std::max(vA.z, vB.z)};
  }

public:
  void update(const std::vector<tvec3<float>> &vs, //
              const std::vector<tvec3<float>> &ns, //
              const std::vector<tvec3<float>> &cs);

  explicit SimpleMesh(std::string name);
  ~SimpleMesh() override = default;
  void draw() override;
  void drawPick() override;
  void buildCustomUI() override;
  void buildCustomOptionsUI() override;
  void buildQuantitiesUI() override;
  void buildSharedStructureUI() override;
  void buildPickUI(unsigned long localPickID) override;
  std::tuple<tvec3<float>, tvec3<float>> boundingBox() override;
  double lengthScale() override;
  std::string typeName() override;
};
} // namespace polyscope::extra
