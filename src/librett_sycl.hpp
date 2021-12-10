#ifndef __LIBRETT_SYCL_HPP__
#define __LIBRETT_SYCL_HPP__

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <sstream>
#include <map>
#include <vector>

#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>

namespace librett {

/// DPC++ default exception handler
auto exception_handler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                << e.what() << std::endl
                << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
    }
  }
};

class syclDeviceProp {
public:
  // get interface
  char *get_name() { return _name; }
  sycl::id<3> get_max_work_item_sizes() { return _max_work_item_sizes; }
  bool get_host_unified_memory() { return _host_unified_memory; }
  int get_max_compute_units() { return _max_compute_units; }
  int get_max_work_group_size() { return _max_work_group_size; }
  int get_max_sub_group_size() { return _max_sub_group_size; }
  int get_max_work_items_per_compute_unit() {
    return _max_work_items_per_compute_unit;
  }
  size_t *get_max_nd_range_size() { return _max_nd_range_size; }
  size_t get_global_mem_size() { return _global_mem_size; }
  size_t get_local_mem_size() { return _local_mem_size; }
  // set interface
  void set_name(const char *name) { std::strncpy(_name, name, 256); }
  void set_max_work_item_sizes(const sycl::id<3> max_work_item_sizes) {
    _max_work_item_sizes = max_work_item_sizes;
  }
  void set_host_unified_memory(bool host_unified_memory) {
    _host_unified_memory = host_unified_memory;
  }
  void set_max_compute_units(int max_compute_units) {
    _max_compute_units = max_compute_units;
  }
  void set_global_mem_size(size_t global_mem_size) {
    _global_mem_size = global_mem_size;
  }
  void set_local_mem_size(size_t local_mem_size) {
    _local_mem_size = local_mem_size;
  }
  void set_max_work_group_size(int max_work_group_size) {
    _max_work_group_size = max_work_group_size;
  }
  void set_max_sub_group_size(int max_sub_group_size) {
    _max_sub_group_size = max_sub_group_size;
  }
  void
  set_max_work_items_per_compute_unit(int max_work_items_per_compute_unit) {
    _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
  }
  void set_max_nd_range_size(int max_nd_range_size[]) {
    for (int i = 0; i < 3; i++)
      _max_nd_range_size[i] = max_nd_range_size[i];
  }

private:
  char _name[256];
  sycl::id<3> _max_work_item_sizes;
  bool _host_unified_memory = false;
  int _max_compute_units;
  int _max_work_group_size;
  int _max_sub_group_size;
  int _max_work_items_per_compute_unit;
  size_t _global_mem_size;
  size_t _local_mem_size;
  size_t _max_nd_range_size[3];
};

/// device extension
class device_ext : public sycl::device {
public:
  device_ext() : sycl::device(), _ctx(*this) {}
  ~device_ext() {
    std::lock_guard<std::mutex> lock(m_mutex);
    _queues.clear();
  }
  device_ext(const sycl::device &base)
      : sycl::device(base), _ctx(*this) {
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, base, exception_handler, sycl::property::queue::in_order()));
    _default_queue = _queues[0].get();
  }

  int is_native_atomic_supported() { return 0; }
  int get_major_version() {
    int major, minor;
    get_version(major, minor);
    return major;
  }

  int get_minor_version() {
    int major, minor;
    get_version(major, minor);
    return minor;
  }
  
  int get_max_compute_units() {
    return get_device_info().get_max_compute_units();
  }

  void get_device_info(syclDeviceProp &out) {
    syclDeviceProp prop;
    prop.set_name(get_info<sycl::info::device::name>().c_str());

    prop.set_max_work_item_sizes(
        get_info<sycl::info::device::max_work_item_sizes>());
    prop.set_host_unified_memory(
        get_info<sycl::info::device::host_unified_memory>());

    prop.set_max_compute_units(
        get_info<sycl::info::device::max_compute_units>());
    prop.set_max_work_group_size(
        get_info<sycl::info::device::max_work_group_size>());
    prop.set_global_mem_size(
        get_info<sycl::info::device::global_mem_size>());
    prop.set_local_mem_size(get_info<sycl::info::device::local_mem_size>());

    size_t max_sub_group_size = 1;
    std::vector<size_t> sub_group_sizes =
        get_info<sycl::info::device::sub_group_sizes>();

    for (const auto &sub_group_size : sub_group_sizes) {
      if (max_sub_group_size < sub_group_size)
        max_sub_group_size = sub_group_size;
    }

    prop.set_max_sub_group_size(max_sub_group_size);

    prop.set_max_work_items_per_compute_unit(
        get_info<sycl::info::device::max_work_group_size>());
    int max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
    prop.set_max_nd_range_size(max_nd_range_size);

    out = prop;
  }

  syclDeviceProp get_device_info() {
    syclDeviceProp prop;
    get_device_info(prop);
    return prop;
  }

  void reset() {
    // The queues are shared_ptrs and the ref counts of the shared_ptrs increase
    // only in wait_and_throw(). If there is no other thread calling
    // wait_and_throw(), the queues will be destructed. The destructor waits for
    // all commands executing on the queue to complete. It isn't possible to
    // destroy a queue immediately. This is a synchronization point in SYCL.
    _queues.clear();
    // create new default queue.
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, exception_handler, sycl::property::queue::in_order()));
    _default_queue = _queues.front().get();
  }

  sycl::queue &default_queue() { return *_default_queue; }

  void queues_wait_and_throw() {
    std::vector<std::shared_ptr<sycl::queue>> current_queues(
        _queues);
    lock.unlock();
    for (const auto &q : current_queues) {
      q->wait_and_throw();
    }
    // Guard the destruct of current_queues to make sure the ref count is safe.
    lock.lock();
  }
  sycl::queue *create_queue(bool enable_exception_handler = false) {
    std::lock_guard<std::mutex> lock(m_mutex);    
    _queues.push_back(std::make_shared<sycl::queue>(
        _ctx, *this, exception_handler,
        sycl::property::queue::in_order()));
    return _queues.back().get();
  }
  void destroy_queue(sycl::queue *&queue) {
    std::lock_guard<std::mutex> lock(m_mutex);    
    _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                  [=](const std::shared_ptr<sycl::queue> &q) -> bool {
                                    return q.get() == queue;
                                  }),
                   _queues.end());
    queue = nullptr;
  }
  sycl::context get_context() { return _ctx; }

private:
  void get_version(int &major, int &minor) {
    // Version string has the following format:
    // a. OpenCL<space><major.minor><space><vendor-specific-information>
    // b. <major.minor>
    std::string ver;
    ver = get_info<sycl::info::device::version>();
    std::string::size_type i = 0;
    while (i < ver.size()) {
      if (isdigit(ver[i]))
        break;
      i++;
    }
    major = std::stoi(&(ver[i]));
    while (i < ver.size()) {
      if (ver[i] == '.')
        break;
      i++;
    }
    i++;
    minor = std::stoi(&(ver[i]));
  }  
  sycl::queue *_default_queue;
  sycl::context _ctx;
  std::vector<std::shared_ptr<sycl::queue>> _queues;
  mutable std::mutex m_mutex;  
};


static inline unsigned int get_tid(){
  return syscall(SYS_gettid);
}
  
/// device manager
class dev_mgr {
public:
  device_ext &current_device() {
    unsigned int dev_id=current_device_id();
    check_id(dev_id);
    return *_devs[dev_id];
  }
  device_ext &get_device(unsigned int id) const {
    check_id(id);
    return *_devs[id];
  }
  unsigned int current_device_id() const {
   auto it=_thread2dev_map.find(get_tid());
   if(it != _thread2dev_map.end())
      return it->second;
    return DEFAULT_DEVICE_ID;
  }
  void select_device(unsigned int id) {
    std::lock_guard<std::mutex> lock(m_mutex);    
    check_id(id);
    _thread2dev_map[get_tid()]=id;
  }
  unsigned int device_count() { return _devs.size(); }

  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

private:
  mutable std::mutex m_mutex;  
  dev_mgr() {
    sycl::device default_device =
        sycl::device(sycl::default_selector{});
    _devs.push_back(std::make_shared<device_ext>(default_device));

    std::vector<sycl::device> sycl_all_devs =
        sycl::device::get_devices(sycl::info::device_type::all);
    // Collect other devices except for the default device.
    for (auto &dev : sycl_all_devs) {
      if (dev == default_device) {
        continue;
      }
      _devs.push_back(std::make_shared<device_ext>(dev));
    }
  }
  void check_id(unsigned int id) const {
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }
  std::vector<std::shared_ptr<device_ext>> _devs;
  /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const unsigned int DEFAULT_DEVICE_ID = 0;
};

/// Util function to get the defualt queue of current device in
/// dpct device manager.
static inline sycl::queue &get_default_queue() {
  return dev_mgr::instance().current_device().default_queue();
}

/// Util function to get the current device.
static inline device_ext &get_current_device() {
  return dev_mgr::instance().current_device();
}

/// Util function to get a device by id.
static inline device_ext &get_device(unsigned int id) {
  return dev_mgr::instance().get_device(id);
}

/// Util function to get the context of the default queue of current
/// device in dpct device manager.
static inline sycl::context get_default_context() {
  return dpct::get_current_device().get_context();
}

/// Util function to set the default device
static inline void select_device(unsigned int id) {
  dev_mgr::instance().select_device(id);
}
  
} // namespace librett

#endif // __LIRETT_SYCL_HPP__
