#pragma once

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <numeric>
#include <optional>
#include <type_traits>

namespace sph::utils {

class Stopwatch {

  using hrc_timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

  struct Entry {
    const std::string name;
    const hrc_timepoint begin;
    hrc_timepoint end;

    Entry(std::string name, const hrc_timepoint &begin) : name(std::move(name)), begin(begin) {}
  };

  std::string name;
  std::vector<Entry> entries;

public:
  explicit Stopwatch(std::string name) : name(std::move(name)) {}

public:
  std::function<void(void)> start(const std::string &entry) {
    const size_t size = entries.size();
    entries.emplace_back(entry, std::chrono::high_resolution_clock::now());
    return [size, this]() { entries[size].end = std::chrono::high_resolution_clock::now(); };
  }

  friend std::ostream &operator<<(std::ostream &os, const Stopwatch &stopwatch) {
    os << "Stopwatch[ " << stopwatch.name << "]:\n";

    size_t maxLen = std::max_element(stopwatch.entries.begin(), stopwatch.entries.end(),
                                     [](auto &l, auto &r) { return l.name.size() < r.name.size(); })
                        ->name.size() +
                    3;

    for (const Entry &e : stopwatch.entries) {
      os << "    ->"
         << "`" << e.name << "` " << std::setw(static_cast<int>(maxLen - e.name.size())) << " : "
         << (static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(e.end - e.begin).count()) /
             1000'000.0)
         << "ms" << std::endl;
    }
    return os;
  }
};

template <typename T>
inline std::string mk_string(const std::vector<T> &xs,
                             const std::function<std::string(const T &)> &f, //
                             const std::string &delim) {
  return std::accumulate(xs.begin(), xs.end(), std::string(), [&f, &delim](const std::string &acc, const T &x) {
    return acc + (acc.length() > 0 ? delim : "") + f(x);
  });
}

inline std::string mk_string(const std::vector<std::string> &xs, const std::string &sep = ",") {
  return mk_string<std::string>(
      xs, [](const auto &x) { return x; }, sep);
}

template <template <size_t, typename C> typename V>
inline V<3, size_t> to3d(size_t index, size_t xMax, size_t yMax, size_t zMax) {
  size_t x = index / (yMax * zMax);
  size_t y = (index - x * yMax * zMax) / zMax;
  size_t z = index - x * yMax * zMax - y * zMax;
  return V<3, size_t>(x, y, z);
}

constexpr size_t index3d(size_t x, size_t y, size_t z, size_t xMax, size_t yMax, size_t zMax) {
  return x * yMax * zMax + y * zMax + z;
}

template <typename N> constexpr N scale(N x, N y, N a) { return (x - y) / (a - y); }

template <typename T>
std::vector<T> findWithSignature(const std::vector<T> &devices,
                                 const std::function<std::pair<size_t, std::string>(const T &)> &f,
                                 const std::vector<std::string> &needles) {

  std::vector<T> matching;
  std::copy_if(devices.begin(), devices.end(), std::back_inserter(matching), [&](const T &device) {
    return std::any_of(needles.begin(), needles.end(), [&](auto needle) {
      auto [idx, name] = f(device);
      try {
        return size_t(std::stoi(needle)) == idx;
      } catch (...) {
        return name.find(needle) != std::string::npos;
      }
    });
  });

  return matching;
}

template <typename N> constexpr glm::tvec3<float> to_glm3f(const glm::tvec3<N> &x) { return glm::tvec3<float>(x); }
template <typename N> constexpr glm::tvec4<float> to_glm4f(const glm::tvec4<N> &x) { return glm::tvec4<float>(x); }

// https://stackoverflow.com/a/14678946/896997
inline std::string replace(std::string subject, const std::string &search, const std::string &replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
}

template <typename N> constexpr auto to_string(glm::tvec3<N> v) {
  return "(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + ")";
}
template <typename N> constexpr auto to_string(glm::tvec4<N> v) {
  return "(" + std::to_string(v.x) + "," + std::to_string(v.y) + "," + std::to_string(v.z) + "," + std::to_string(v.w) +
         ")";
}

template <typename D>
std::optional<D> findDevice(bool list,                                                             //
                            bool verbose,                                                          //
                            const std::vector<std::string> &needle,                                //
                            const std::function<void(std::ostream &, std::ostream &)> &enumerate,  //
                            const std::function<std::vector<std::pair<size_t, D>>()> &listDevices, //
                            const std::function<std::string(const D &)> &showDevice) {
  if (verbose) {
    enumerate(std::cout, std::cerr);
  }
  if (list) {
    for (auto [idx, dev] : listDevices())
      std::cout << "[" << idx << "] " << showDevice(dev) << std::endl;
    return {};
  } else {
    auto found = sph::utils::findWithSignature<std::pair<size_t, D>>(
        listDevices(), [&](const auto &d) { return std::make_pair(d.first, showDevice(d.second)); }, needle);

    if (found.empty()) {
      std::cerr << "No device found with the following signatures: " << sph::utils::mk_string(needle) << std::endl;
      std::cerr << "Available devices: " << sph::utils::mk_string(needle) << std::endl;
      for (auto [idx, dev] : listDevices())
        std::cout << "[" << idx << "] " << showDevice(dev) << std::endl;
      return {};
    } else {
      auto [idx, device] = found[0];
      if (found.size() != 1) std::cout << "(ignoring " << found.size() - 1 << " more matches)" << std::endl;
      std::cout << "Using device: [" << idx << "] " << showDevice(device) << std::endl;
      return device;
    }
  }
}

} // namespace sph::utils
