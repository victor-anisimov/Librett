#pragma once

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

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
      _warpSize = min_sub_group_size;
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

  class device_ext: public sycl::device {
  public:
    device_ext(): sycl::device(), _ctx(*this) {}
    ~device_ext() { std::lock_guard<std::mutex> lock(m_mutex); }
    device_ext(const sycl::device& base): sycl::device(base), _ctx(*this) {}

  private:
    sycl::context      _ctx;
    mutable std::mutex m_mutex;
  };

  static inline int get_tid() { return syscall(SYS_gettid); }

  class dev_mgr {
  public:
    int current_device() {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto                        it = _thread2dev_map.find(get_tid());
      if(it != _thread2dev_map.end()) {
        check_id(it->second);
        return it->second;
      }
      printf("WARNING: no SYCL device found in the map, returning DEFAULT_DEVICE_ID\n");
      return DEFAULT_DEVICE_ID;
    }
    device_ext* get_sycl_device(int id) const {
      std::lock_guard<std::mutex> lock(m_mutex);
      check_id(id);
      return (_devs[id].first).get();
    }
    sycl::queue* get_sycl_queue(int id) const {
      std::lock_guard<std::mutex> lock(m_mutex);
      check_id(id);
      return (_devs[id].second).get();
    }
    void get_device_prop(DeviceProp_t* prop, int id) const {
      std::lock_guard<std::mutex> lock(m_mutex);
      check_id(id);
      *prop = *((_deviceProps[id]).get());
    }

    void select_device(int id) {
      std::lock_guard<std::mutex> lock(m_mutex);
      check_id(id);
      _thread2dev_map[get_tid()] = id;
    }
    int device_count() { return _devs.size(); }

    /// Returns the instance of device manager singleton.
    static dev_mgr& instance() {
      static dev_mgr d_m;
      return d_m;
    }
    dev_mgr(const dev_mgr&)            = delete;
    dev_mgr& operator=(const dev_mgr&) = delete;
    dev_mgr(dev_mgr&&)                 = delete;
    dev_mgr& operator=(dev_mgr&&)      = delete;

  private:
    mutable std::mutex m_mutex;

    dev_mgr() {
      std::vector<sycl::device> sycl_all_devs = sycl::device::get_devices(sycl::info::device_type::gpu);
      for(auto& dev: sycl_all_devs) {
        if(dev.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
          auto subdevices = dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);
          for(auto& tile: subdevices) {
            _devs.push_back(std::make_pair(std::make_shared<device_ext>(tile),
                                           std::make_shared<sycl::queue>(tile,
                                                                         sycl_asynchandler,
                                                                         sycl::property_list{sycl::property::queue::in_order{}})));
            DeviceProp_t devProp;
            populate_deviceprop(&devProp, tile);
            _deviceProps.push_back( std::make_shared<DeviceProp_t>(devProp) );
          }
        }
        else {
          _devs.push_back(
            std::make_pair(std::make_shared<device_ext>(dev), std::make_shared<sycl::queue>(dev, sycl_asynchandler,
                             sycl::property_list{sycl::property::queue::in_order{}})));
          DeviceProp_t devProp;
          populate_deviceprop(&devProp, dev);
          _deviceProps.push_back( std::make_shared<DeviceProp_t>(devProp) );
        }
      }
    }

    void check_id(int id) const {
      if(id >= _devs.size()) { throw std::runtime_error("invalid device id"); }
    }

    void populate_deviceprop(DeviceProp_t* prop, sycl::device& dev) {

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

    std::vector<std::shared_ptr<DeviceProp_t>> _deviceProps;

    std::vector<std::pair<std::shared_ptr<device_ext>, std::shared_ptr<sycl::queue>>> _devs;
    /// DEFAULT_DEVICE_ID is used, if current_device() can not find current
    /// thread id in _thread2dev_map, which means default device should be used
    /// for the current thread.
    const int DEFAULT_DEVICE_ID = 0;
    /// thread-id to device-id map.
    std::map<int, int> _thread2dev_map;
  };

/// Util function to get the current device (in int).
  static inline void syclGetDevice(int* id) { *id = dev_mgr::instance().current_device(); }

/// Util function to get the current sycl::device by id.
  static inline device_ext* sycl_get_device(int id) {
    return dev_mgr::instance().get_sycl_device(id);
  }

/// Util function to set a device by id. (to _thread2dev_map)
  static inline void syclSetDevice(int id) { dev_mgr::instance().select_device(id); }

/// Util function to get number of GPU devices (default: explicit scaling)
  static inline void syclGetDeviceCount(int* id) { *id = dev_mgr::instance().device_count(); }

/// Util function to get default queue. (i.e., returns the queue active on the current device)
  static inline sycl::queue* sycl_default_queue() {
    return dev_mgr::instance().get_sycl_queue( dev_mgr::instance().current_device() );
  }

/// Util function to get device properties for the deviceID
  static inline void syclGetDeviceProperties(DeviceProp_t* prop, int id) {
    dev_mgr::instance().get_device_prop( prop, id );
  }

} // namespace Librett
