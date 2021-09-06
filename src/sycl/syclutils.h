#pragma once

#include <iostream>
#include "utils.hpp"
#include <CL/sycl.hpp>

namespace sph::sycl_impl::utils {

using namespace cl;

static std::string showDevice(const sycl::device &device) {
  return std::string(device.get_info<sycl::info::device::name>());
}

static std::vector<std::pair<size_t, sycl::device>> listDevices() {
  std::vector<std::pair<size_t, sycl::device>> collected;
  for (const auto &p : sycl::platform().get_platforms()) {
    try {
      for (const auto &d : p.get_devices())
        collected.emplace_back(collected.size(), d);
    } catch (const std::exception &e) {
      std::cerr << "Enumeration failed at `" << p.get_info<sycl::info::platform::name>() << "` : " << e.what()
                << std::endl;
    }
  }
  return collected;
}

static void enumeratePlatform(std::ostream &out, std::ostream &err) {
  std::vector<sycl::platform> platforms = sycl::platform().get_platforms();
  const auto type_name = [](const sycl::info::device_type &type) -> std::string {
    switch (type) {
    case sycl::info::device_type::cpu:
      return "cpu";
    case sycl::info::device_type::gpu:
      return "gpu";
    case sycl::info::device_type::accelerator:
      return "accelerator";
    case sycl::info::device_type::custom:
      return "custom";
    case sycl::info::device_type::automatic:
      return "automatic";
    case sycl::info::device_type::host:
      return "host";
    case sycl::info::device_type::all:
      return "all";
    default:
      return "(unknown: " + std::to_string(static_cast<unsigned int>(type)) + ")";
    }
  };
  for (auto &p : platforms) {
    try {
      auto exts = p.get_info<sycl::info::platform::extensions>();
      std::ostringstream extensions;
      std::copy(exts.begin(), exts.end(), std::ostream_iterator<std::string>(extensions, ","));

      out << "\t├─┬Platform:" << p.get_info<sycl::info::platform::name>()
          << "\n\t│ ├Vendor     : " << p.get_info<sycl::info::platform::vendor>()                            //
          << "\n\t│ ├Version    : " << p.get_info<sycl::info::platform::version>()                           //
          << "\n\t│ ├Profile    : " << p.get_info<sycl::info::platform::profile>()                           //
          << "\n\t│ ├Extensions : " << sph::utils::mk_string(p.get_info<sycl::info::platform::extensions>()) //
          << "\n\t│ └Devices" << std::endl;
      std::vector<sycl::device> devices = p.get_devices();
      for (auto &d : devices) {
        out << "\t│\t     └┬Name    : " << d.get_info<sycl::info::device::name>()                     //
            << "\n\t│\t      ├Type    : " << type_name(d.get_info<sycl::info::device::device_type>()) //
            << "\n\t│\t      ├Vendor  : " << d.get_info<sycl::info::device::vendor_id>()              //
            << "\n\t│\t      ├Avail.  : " << d.get_info<sycl::info::device::is_available>()           //
            << "\n\t│\t      ├Max CU. : " << d.get_info<sycl::info::device::max_compute_units>()      //
            << "\n\t│\t      └Version : " << d.get_info<sycl::info::device::version>() << std::endl;
      }
    } catch (const std::exception &e) {
      err << "Enumeration failed at `" << p.get_info<sycl::info::platform::name>() << "` : " << e.what() << std::endl;
    }
  }
}

template <typename N> constexpr glm::tvec3<N> to_glm3f(const sycl::vec<N, 3> &x) {
  return glm::tvec3<N>(x.s0(), x.s1(), x.s2());
}
template <typename N> constexpr glm::tvec4<N> to_glm4f(const sycl::vec<N, 4> &x) {
  return glm::tvec4<N>(x.s0(), x.s1(), x.s2(), x.s3());
}

} // namespace sph::sycl_impl::utils