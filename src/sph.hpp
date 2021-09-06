#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

#include "curves.h"

namespace sph {

enum class Type : uint8_t { Fluid = 0, Obstacle = 1 };

//template <typename N>
//struct Particle {
//  int id;
//  int mass;
//  glm::tvec3<N> position, velocity;
//  glm::tvec4<N>V<4> colour;
//}

template <typename T, typename N, template <size_t, typename C = N> typename V> struct Query {
  T id;
  V<3> point;
};

template <typename T, typename N, template <size_t, typename C = N> typename V> struct QueryResult {
  T id{};
  V<3> point{};
  std::vector<T> neighbours{};
};

template <typename T, typename N, template <size_t, typename C = N> typename V> struct Particle {
  T id;
  Type type;
  N mass;
  V<3> position, velocity;
  V<4> colour;

  constexpr Particle() : type(Type::Fluid) {}

  constexpr explicit Particle(T t, Type type, N mass, const V<4> &colour, const V<3> &position, const V<3> &velocity)
      : id(t), type(type), mass(mass), position(position), velocity(velocity), colour(colour) {}

  bool operator==(const Particle &rhs) const {
    return id == rhs.id && type == rhs.type && mass == rhs.mass && colour == rhs.colour && position == rhs.position &&
           velocity == rhs.velocity;
  }

  bool operator!=(const Particle &rhs) const { return rhs != *this; }
};

template <typename T, typename N, template <size_t, typename C = N> typename V> struct Well {
  T tag;
  V<3> centre;
  N force;
};

template <typename T, typename N, template <size_t, typename C = N> typename V> struct Source {
  T tag;
  V<3> centre, velocity;
  V<4> colour;
  N rate;
};

template <typename T, typename N, template <size_t, typename C = N> typename V> struct Drain {
  T tag;
  V<3> centre;
  N width, depth;
};

template <typename T, typename N, template <size_t S, typename C = N> typename V> struct Scene {
  std::vector<sph::Well<T, N, V>> wells;
  std::vector<sph::Source<T, N, V>> sources;
  std::vector<sph::Drain<T, N, V>> drains;
  std::vector<sph::Query<T, N, V>> queries;
};

template <typename N> struct McParams {
  N resolution;
  N isolevel;
  N particleSize;
  N particleInfluence;
  template <typename O> McParams<O> as() {
    return {
        .resolution = O(resolution),
        .isolevel = O(isolevel),
        .particleSize = O(particleSize),
        .particleInfluence = O(particleInfluence),
    };
  }
};

template <typename T, typename N, template <size_t S, typename C = N> typename V> struct SphParams {
  N h, dt, scale;
  size_t iteration;
  V<3> constantForce, minBound, maxBound;
  bool wait;
  std::optional<McParams<N>> surface;
};

template <typename N, template <size_t S, typename C = N> typename V> struct ColouredMesh {
  std::vector<V<3>> vs{};
  std::vector<V<3>> ns{};
  std::vector<V<4>> cs{};
  ColouredMesh(const decltype(vs) &vs, const decltype(ns) &ns, const decltype(cs) &cs) : vs(vs), ns(ns), cs(cs) {}
  ColouredMesh(size_t size) : vs(size), ns(size), cs(size) {}
  ColouredMesh() = default;
};

template <typename T, typename N, template <size_t, typename C = N> typename V> struct Result {
  ColouredMesh<N, V> mesh{};
  std::vector<QueryResult<T, N, V>> queries{};
};

template <typename T, typename N, template <size_t, typename _ = N> typename V> class Solver {
public:
  virtual ~Solver() = default;
  virtual Result<T, N, V> advance(const SphParams<T, N, V> &config, //
                                  const Scene<T, N, V> &scene,      //
                                  std::vector<Particle<T, N, V>> &xs) = 0;
};

template <typename T, typename N, template <size_t, typename C = N> typename V>
T makeCube(T offset,
           N spacing,          //
           const size_t count, //
           V<3> origin,        //
           V<4> colour,        //
           std::vector<sph::Particle<T, N, V>> &xs) {
  auto len = static_cast<size_t>(std::cbrt(count));
  for (size_t x = 0; x < len; ++x) {
    for (size_t y = 0; y < len; ++y) {
      for (size_t z = 0; z < len; ++z) {
        auto pos = (V<3>(x, y, z) * spacing) + origin;
        //				uint32_t colour = i > half ? 0xFFFF0000 : 0xFF00FF00;
        xs.emplace_back(offset++, sph::Type::Fluid, 1.0, colour, pos, V<3>(0, 0, 0));
      }
    }
  }
  return offset;
}

template <typename T, typename N, template <size_t, typename C = N> typename V>
SphParams<T, N, V> applyMotionSinXCosZ(const SphParams<T, N, V> &config, size_t frame) {
  static constexpr float offsetScale = 300.f;
  static constexpr float offsetRate = 20.f;
  const N offsetX = N(std::sin(float(frame) / offsetRate) * offsetScale);
  const N offsetZ = N(std::cos(float(frame) / offsetRate) * offsetScale * 0.3);
  auto waveConfig = config;
  auto offset = V<3>(offsetX, N{}, offsetZ);
  waveConfig.minBound += offset;
  waveConfig.maxBound += offset;
  return waveConfig;
}

template <typename T, typename N, template <size_t, typename C = N> typename V>
std::tuple<sph::McParams<N>, sph::SphParams<T, N, V>, std::vector<sph::Particle<T, N, V>>>
simpleConfigWith2Cubes(size_t count, size_t solverIter, N scaling) {
  std::vector<sph::Particle<T, N, V>> prepared;
  T tag = {};
  tag = sph::makeCube<T, N, V>(tag, 22.f, count / 2, V<3>(+100, +600 * 0, +100), V<4>(0, 0.1, 0.8, 1), prepared);
  tag = sph::makeCube<T, N, V>(tag, 22.f, count / 2, V<3>(+600, +600 * 0, 600), V<4>(0.1, 0.8, 0.1, 1), prepared);

  sph::SphParams<T, N, V> config = {.dt = 0.0083 * 1.5f, // var
                                    .scale = scaling,    // var
                                    .iteration = solverIter,
                                    .constantForce = V<3>(0, 9.8  , 0),
                                    .minBound = V<3>(0, 0, 0),
                                    .maxBound = V<3>(1000, 1000, 1000),
                                    .wait = true,
                                    .surface = {}

  };

  return {McParams<N>{
              .resolution = 2.0f, // var
              .isolevel = 100,
              .particleSize = 25,
              .particleInfluence = 0.5,
          },
          config, prepared};
}

template <typename T, typename N, template <size_t, typename C = N> typename V>
void save(sph::Result<T, N, V> &solver, std::string dir) {
  std::filesystem::path p(dir);
  if (!std::filesystem::create_directories(dir)) {
    std::cerr << "Can't create directory `" << dir << "`";
    return;
  }
  // TODO impl
}

template <typename N> [[nodiscard]] size_t constexpr zCurveGridIndexAtCoordAt(N x, N y, N z, N h) {
  return zCurveGridIndexAtCoord(static_cast<size_t>((x / h)), static_cast<size_t>((y / h)),
                                static_cast<size_t>((z / h)));
}

template <typename F, typename IC, typename C>
constexpr void foreach_grid(const C &hostGridTable, size_t hostGridTableN, const IC &offsets, const F &f) {
  for (size_t offset : offsets) {
    if (offset >= hostGridTableN) continue;
    const size_t start = hostGridTable[offset];
    const size_t end = (offset + 1) < hostGridTableN ? hostGridTable[offset + 1] : start;
    for (size_t b = start; b < end; ++b) {
      f(b);
    }
  }
}

template <typename F, typename C>
constexpr void foreach_grid(const size_t zIndex, const C &hostGridTable, size_t hostGridTableN, const F &f) {
  const size_t x = coordAtZCurveGridIndex0(zIndex);
  const size_t y = coordAtZCurveGridIndex1(zIndex);
  const size_t z = coordAtZCurveGridIndex2(zIndex);
  const std::array<size_t, 27> offsets = {
      zCurveGridIndexAtCoord(x - 1, y - 1, z - 1), zCurveGridIndexAtCoord(x + 0, y - 1, z - 1),
      zCurveGridIndexAtCoord(x + 1, y - 1, z - 1), zCurveGridIndexAtCoord(x - 1, y + 0, z - 1),
      zCurveGridIndexAtCoord(x + 0, y + 0, z - 1), zCurveGridIndexAtCoord(x + 1, y + 0, z - 1),
      zCurveGridIndexAtCoord(x - 1, y + 1, z - 1), zCurveGridIndexAtCoord(x + 0, y + 1, z - 1),
      zCurveGridIndexAtCoord(x + 1, y + 1, z - 1), zCurveGridIndexAtCoord(x - 1, y - 1, z + 0),
      zCurveGridIndexAtCoord(x + 0, y - 1, z + 0), zCurveGridIndexAtCoord(x + 1, y - 1, z + 0),
      zCurveGridIndexAtCoord(x - 1, y + 0, z + 0), zCurveGridIndexAtCoord(x + 0, y + 0, z + 0),
      zCurveGridIndexAtCoord(x + 1, y + 0, z + 0), zCurveGridIndexAtCoord(x - 1, y + 1, z + 0),
      zCurveGridIndexAtCoord(x + 0, y + 1, z + 0), zCurveGridIndexAtCoord(x + 1, y + 1, z + 0),
      zCurveGridIndexAtCoord(x - 1, y - 1, z + 1), zCurveGridIndexAtCoord(x + 0, y - 1, z + 1),
      zCurveGridIndexAtCoord(x + 1, y - 1, z + 1), zCurveGridIndexAtCoord(x - 1, y + 0, z + 1),
      zCurveGridIndexAtCoord(x + 0, y + 0, z + 1), zCurveGridIndexAtCoord(x + 1, y + 0, z + 1),
      zCurveGridIndexAtCoord(x - 1, y + 1, z + 1), zCurveGridIndexAtCoord(x + 0, y + 1, z + 1),
      zCurveGridIndexAtCoord(x + 1, y + 1, z + 1)};
  foreach_grid(hostGridTable, hostGridTableN, offsets, f);
}

template <typename Index> std::vector<size_t> makeGridTable(size_t x, size_t y, size_t z, size_t size, Index get) {
  static_assert(std::is_invocable_r<size_t, Index, size_t>::value, "");
  const size_t maxZIndex = zCurveGridIndexAtCoord(x, y, z);
  std::vector<size_t> hostGridTable(maxZIndex);
  size_t gridIndex = 0;
  for (size_t zIndex = 0; zIndex < maxZIndex; ++zIndex) {
    hostGridTable[zIndex] = gridIndex;
    while (gridIndex != size && get(gridIndex) == zIndex) {
      gridIndex++;
    }
  }
  return hostGridTable;
}
template <typename T> T const pi = std::acos(-T(1));
template <typename N> N constexpr poly6Factor(N h) { return N(315.0) / (N(64.0) * pi<N> * std::pow(h, 9)); }
template <typename N> N constexpr spikyKernelFactor(N h) { return -(N(45.0) / (pi<N> * std::pow(h, 6))); }

template <typename T, typename N, template <size_t, typename C = N> typename V> struct PartiallyAdvected {
  size_t zIndex{};
  V<3, N> pStar{};
  V<3, N> deltaP{};
  N lambda;
  Particle<T, N, V> particle;
};

} // namespace sph
