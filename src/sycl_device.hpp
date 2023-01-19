#pragma once

#include <cstring>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;
#endif

namespace Librett {

  auto sycl_asynchandler = [] (sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const& ex) {
        std::cout << "Caught asynchronous SYCL exception:" << std::endl
                  << ex.what() << ", SYCL code: " << ex.code() << std::endl;
      }
    }
  };

  class DeviceProp_t {
  public:
    int get_major_version() const { return _major; }
    int get_max_clock_frequency() const { return _clockRate; }
    int get_max_compute_units() const { return _max_compute_units; }
    int get_max_work_group_size() const { return _max_work_group_size; }
    int get_min_sub_group_size() const { return _warpSize; }
    size_t get_local_mem_size() const { return _local_mem_size; }
    // set interface
    void set_major_version(int major) { _major = major; }
    void set_max_clock_frequency(int frequency) { _clockRate = frequency; }
    void set_max_compute_units(int max_compute_units) {
      _max_compute_units = max_compute_units;
    }
    void set_max_work_group_size(int max_work_group_size) {
      _max_work_group_size = max_work_group_size;
    }
    void set_min_sub_group_size(int min_sub_group_size) {
      //_warpSize = min_sub_group_size;
      _warpSize = 32;
    }
    void set_local_mem_size(size_t local_mem_size) {
      _local_mem_size = local_mem_size;
    }
  private:
    int _warpSize;
    int _clockRate;
    int _major;
    int _max_compute_units;
    int _max_work_group_size;
    size_t _local_mem_size;
  };

/// Util function to get number of GPU devices (default: explicit scaling)
  static inline void syclGetDeviceCount(int* id) {
    int gpuCount=0;
    std::vector<sycl::device> sycl_all_devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    for(auto& dev: sycl_all_devs) {
      if(dev.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
        auto subdevices = dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
        for(auto& tile: subdevices) {
          gpuCount++;
        }
      }
      else {
        gpuCount++;
      }
    }

    *id = gpuCount;
  }

/// Util function to get device properties for the deviceID
  static inline void syclGetDeviceProperties(DeviceProp_t* prop, sycl::queue* stream) {
    sycl::device dev = stream->get_device();

    std::vector<size_t> sub_group_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    prop->set_min_sub_group_size( *std::min_element(sub_group_sizes.begin(), sub_group_sizes.end()) );

    prop->set_max_clock_frequency( dev.get_info<sycl::info::device::max_clock_frequency>() );

    prop->set_max_compute_units( dev.get_info<sycl::info::device::max_compute_units>() );

    prop->set_max_work_group_size( dev.get_info<sycl::info::device::max_work_group_size>() );

    prop->set_local_mem_size( dev.get_info<sycl::info::device::local_mem_size>() );

    int major;
    // Version string has the following format:
    // a. OpenCL<space><major.minor><space><vendor-specific-information>
    // b. <major.minor>
    std::string ver = dev.get_info<sycl::info::device::version>();
    std::string::size_type i = 0;
    while (i < ver.size()) {
      if (isdigit(ver[i]))
        break;
      i++;
    }
    major = std::stoi(&(ver[i]));
    prop->set_major_version(major);
  }

} // namespace Librett
