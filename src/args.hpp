#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "args.hxx"

#include "utils.hpp"

namespace sph::driver {

template <typename K, typename V, template <typename, typename, typename...> class M>
static constexpr M<V, K> reverse_map(const M<K, V> &m) {
  M<V, K> r;
  for (const auto &[k, v] : m)
    r[v] = k;
  return r;
}
enum class Impl : uint8_t { OMP, OCL, SYCL, SYCL2020 };

static const inline std::unordered_map<std::string, Impl> IMPLS{
    {"omp", Impl::OMP},
    {"ocl", Impl::OCL},
    {"sycl", Impl::SYCL},
    {"sycl2020", Impl::SYCL2020},

};

static const inline std::unordered_map<Impl, std::string> IMPLS_R = reverse_map(IMPLS);

static constexpr auto DEFAULT_IMPL = "omp";
static constexpr auto DEFAULT_DEVICE = "0";
static constexpr auto DEFAULT_WARMUP = 200;

struct Args {

  args::ArgumentParser parser;
  args::HelpFlag help;
  args::CompletionFlag completion;
  args::MapFlag<std::string, Impl> impl;
  args::Flag list;
  args::Flag verbose;
  args::ValueFlagList<std::string> devices;
  args::ValueFlag<size_t> iterations;
  args::ValueFlag<size_t> warmup;
  args::Flag fp64;
  args::ValueFlag<std::string> output;

  Args(size_t defaultIterations, const std::string &defaultOutput);
  bool parse(int argc, char *argv[]);
  std::string implName();
  std::string renderedOutputName();
};
} // namespace sph::driver
