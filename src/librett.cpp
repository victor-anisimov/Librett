/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/
#ifdef SYCL
  #include <CL/sycl.hpp>
  #include "dpct/dpct.hpp"
#elif HIP
  #include <hip/hip_runtime.h>
#else // CUDA
  #include <cuda.h>
#endif
#include <list>
#include <unordered_map>
#include "GpuUtils.h"
#include "GpuMem.h"
#include "plan.h"
#include "kernel.h"
#include "Timer.h"
#include "librett.h"
#include <atomic>
#include <mutex>
#include <cstdlib>
// #include <chrono>
#include "uniapi.h"

// global Umpire allocator
#ifdef LIBRETT_HAS_UMPIRE
umpire::Allocator librett_umpire_allocator;
#endif

// Hash table to store the plans
static std::unordered_map<librettHandle, librettPlan_t* > planStorage;
static std::mutex planStorageMutex;

// Current handle
static std::atomic<librettHandle> curHandle(0);

// Table of devices that have been initialized
static std::unordered_map<int, gpuDeviceProp_t> deviceProps;
static std::mutex devicePropsMutex;

// Checks prepares device if it's not ready yet and returns device properties
// Also sets shared memory configuration
void getDeviceProp(int& deviceID, gpuDeviceProp_t &prop) {
  #if SYCL
    deviceID = dpct::dev_mgr::instance().current_device_id();
  #elif HIP
    hipCheck(hipGetDevice(&deviceID));
  #else // CUDA
    cudaCheck(cudaGetDevice(&deviceID));
  #endif

  // need to lock this function
  std::lock_guard<std::mutex> lock(devicePropsMutex);

  auto it = deviceProps.find(deviceID);
  if (it == deviceProps.end()) {
    // Get device properties and store it for later use
    #if SYCL
      dpct::dev_mgr::instance().get_device(deviceID).get_device_info(prop);
    #elif HIP
      hipCheck(hipGetDeviceProperties(&prop, deviceID));
      librettKernelSetSharedMemConfig();
    #else // CUDA
      cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
      librettKernelSetSharedMemConfig();
    #endif
    deviceProps.insert({deviceID, prop});
  } else {
    prop = it->second;
  }
}

librettResult librettPlanCheckInput(int rank, int* dim, int* permutation, size_t sizeofType) {
  // Check sizeofType
  if (sizeofType != 4 && sizeofType != 8) return LIBRETT_INVALID_PARAMETER;
  // Check rank
  if (rank <= 1) return LIBRETT_INVALID_PARAMETER;
  // Check dim[]
  for (int i=0;i < rank;i++) {
    if (dim[i] <= 1) return LIBRETT_INVALID_PARAMETER;
  }
  // Check permutation
  bool permutation_fail = false;
  int* check = new int[rank];
  for (int i=0;i < rank;i++) check[i] = 0;
  for (int i=0;i < rank;i++) {
    if (permutation[i] < 0 || permutation[i] >= rank || check[permutation[i]]++) {
      permutation_fail = true;
      break;
    }
  }
  delete [] check;
  if (permutation_fail) return LIBRETT_INVALID_PARAMETER;  

  return LIBRETT_SUCCESS;
}

librettResult librettPlan(librettHandle *handle, int rank, int *dim, int *permutation, size_t sizeofType, 
  gpuStream_t stream) {

#ifdef ENABLE_NVTOOLS
  gpuRangeStart("init");
#endif

  // Check that input parameters are valid
  librettResult inpCheck = librettPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != LIBRETT_SUCCESS) return inpCheck;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return LIBRETT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
  gpuDeviceProp_t prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<librettPlan_t> plans;
  // if (rank != redDim.size()) {
  //   if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return LIBRETT_INTERNAL_ERROR;
  // }

  // // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return LIBRETT_INTERNAL_ERROR;

#if 0
  if (!librettKernelDatabase(deviceID, prop)) return LIBRETT_INTERNAL_ERROR;
#endif

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("createPlans");
#endif

  // std::chrono::high_resolution_clock::time_point plan_start;
  // plan_start = std::chrono::high_resolution_clock::now();

  if (!librettPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return LIBRETT_INTERNAL_ERROR;

  // std::chrono::high_resolution_clock::time_point plan_end;
  // plan_end = std::chrono::high_resolution_clock::now();
  // double plan_duration = std::chrono::duration_cast< std::chrono::duration<double> >(plan_end - plan_start).count();
  // printf("createPlans took %lf ms\n", plan_duration*1000.0);

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("countCycles");
#endif

  // Count cycles
  for (auto it=plans.begin();it != plans.end();it++) {
    if (!it->countCycles(prop, 10)) return LIBRETT_INTERNAL_ERROR;
  }

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("rest");
#endif

  // Choose the plan
  std::list<librettPlan_t>::iterator bestPlan = choosePlanHeuristic(plans);
  if (bestPlan == plans.end()) return LIBRETT_INTERNAL_ERROR;

  // bestPlan->print();

  // Create copy of the plan outside the list
  librettPlan_t* plan = new librettPlan_t();
  // NOTE: No deep copy needed here since device memory hasn't been allocated yet
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  // Set stream
  plan->setStream(stream);

  // Activate plan
  plan->activate();

  // Insert plan into storage
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    planStorage.insert( {*handle, plan} );
  }

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
#endif

  return LIBRETT_SUCCESS;
}

librettResult librettPlanMeasure(librettHandle *handle, int rank, int *dim, int *permutation, size_t sizeofType,
  gpuStream_t stream, void* idata, void* odata)
{

  // Check that input parameters are valid
  librettResult inpCheck = librettPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != LIBRETT_SUCCESS) return inpCheck;

  if (idata == odata) return LIBRETT_INVALID_PARAMETER;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return LIBRETT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
  gpuDeviceProp_t prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<librettPlan_t> plans;
#if 0
  // if (rank != redDim.size()) {
    if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return LIBRETT_INTERNAL_ERROR;
  // }

  // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return LIBRETT_INTERNAL_ERROR;
#else
  if (!librettPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return LIBRETT_INTERNAL_ERROR;
#endif

  // // Count cycles
  // for (auto it=plans.begin();it != plans.end();it++) {
  //   if (!it->countCycles(prop, 10)) return LIBRETT_INTERNAL_ERROR;
  // }

  // // Count the number of elements
  size_t numBytes = sizeofType;
  for (int i=0;i < rank;i++) numBytes *= dim[i];

  // Choose the plan
  double bestTime = 1.0e40;
  auto bestPlan = plans.end();
  Timer timer;
  std::vector<double> times;
  for (auto it=plans.begin();it != plans.end();it++) {
    // Activate plan
    it->activate();
    // Clear output data to invalidate caches
#if SYCL
    set_device_array<char>((char *)odata, -1, numBytes, stream);
    dpct::get_current_device().queues_wait_and_throw();
#elif HIP
    set_device_array<char>((char *)odata, -1, numBytes);
    hipCheck(hipDeviceSynchronize());
#else // CUDA
    set_device_array<char>((char *)odata, -1, numBytes);
    cudaCheck(cudaDeviceSynchronize());
#endif
    timer.start();
    // Execute plan
    if (!librettKernel(*it, idata, odata)) return LIBRETT_INTERNAL_ERROR;
    timer.stop();
    double curTime = timer.seconds();
    // it->print();
    // printf("curTime %1.2lf\n", curTime*1000.0);
    times.push_back(curTime);
    if (curTime < bestTime) {
      bestTime = curTime;
      bestPlan = it;
    }
  }
  if (bestPlan == plans.end()) return LIBRETT_INTERNAL_ERROR;

  // bestPlan = plans.begin();

  // printMatlab(prop, plans, times);
  // findMispredictionBest(plans, times, bestPlan, bestTime);
  // bestPlan->print();

  // Create copy of the plan outside the list
  librettPlan_t* plan = new librettPlan_t();
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  // Set stream
  plan->setStream(stream);

  // Activate plan
  plan->activate();

  // Insert plan into storage
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    planStorage.insert( {*handle, plan} );
  }

  return LIBRETT_SUCCESS;
}

//void CUDART_CB librettDestroy_callback(gpuStream_t stream, gpuError_t status,
void librettDestroy_callback(gpuStream_t stream, gpuError_t status,
  void *userData) {
  librettPlan_t* plan = (librettPlan_t*) userData;
  delete plan;
}

librettResult librettDestroy(librettHandle handle) {
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return LIBRETT_INVALID_PLAN;
#ifdef LIBRETT_HAS_UMPIRE
  // get the pointer librettPlan_t
  librettPlan_t* plan = it->second;
  cudaStream_t stream = plan->stream;
  // Delete entry from plan storage
  planStorage.erase(it);
  // register callback to deallocate plan
  cudaStreamAddCallback(stream, librettDestroy_callback, plan, 0);
#else
  // Delete instance of librettPlan_t	 
  delete it->second;	  
  // Delete entry from plan storage	  
  planStorage.erase(it);
#endif
  return LIBRETT_SUCCESS;
}

librettResult librettExecute(librettHandle handle, void *idata, void *odata) 
#ifdef SYCL 
try 
#endif
{
  // prevent modification when find
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return LIBRETT_INVALID_PLAN;

  if (idata == odata) return LIBRETT_INVALID_PARAMETER;

  librettPlan_t& plan = *(it->second);

  int deviceID;
#if SYCL
  deviceID = dpct::dev_mgr::instance().current_device_id();
#elif HIP
  hipCheck(hipGetDevice(&deviceID));
#else // CUDA
  cudaCheck(cudaGetDevice(&deviceID));
#endif
  if (deviceID != plan.deviceID) return LIBRETT_INVALID_DEVICE;

  if (!librettKernel(plan, idata, odata)) return LIBRETT_INTERNAL_ERROR;
  return LIBRETT_SUCCESS;
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

void librettInitialize() {
#ifdef LIBRETT_HAS_UMPIRE
  const char* alloc_env_var = std::getenv("LIBRETT_USES_THIS_UMPIRE_ALLOCATOR");
#define __LIBRETT_STRINGIZE(x) #x
#define __LIBRETT_XSTRINGIZE(x) __LIBRETT_STRINGIZE(x)
  const char* alloc_cstr = alloc_env_var ? alloc_env_var : __LIBRETT_XSTRINGIZE(LIBRETT_USES_THIS_UMPIRE_ALLOCATOR);
  librett_umpire_allocator = umpire::ResourceManager::getInstance().getAllocator(alloc_cstr);
#endif
}

void librettFinalize() {
}

#if SYCL
sycl::vec<unsigned, 4> ballot(sycl::sub_group sg, bool predicate = true) __attribute__((convergent)) {
  #ifdef __SYCL_DEVICE_ONLY__
    return __spirv_GroupNonUniformBallot(__spv::Scope::Subgroup, predicate);
  #else
    throw cl::sycl::runtime_error("Sub-groups are not supported on host device.", PI_INVALID_DEVICE);
  #endif
}
#endif
