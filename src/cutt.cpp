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
#else
#include <cuda.h>
#endif
#include <list>
#include <unordered_map>
#include "CudaUtils.h"
#include "CudaMem.h"
#include "cuttplan.h"
#include "cuttkernel.h"
#include "cuttTimer.h"
#include "librett.h"
#include <atomic>
#include <mutex>
#include <cstdlib>
// #include <chrono>

// global Umpire allocator
#ifdef CUTT_HAS_UMPIRE
umpire::Allocator cutt_umpire_allocator;
#endif

// Hash table to store the plans
static std::unordered_map<cuttHandle, cuttPlan_t* > planStorage;
static std::mutex planStorageMutex;

// Current handle
static std::atomic<cuttHandle> curHandle(0);

// Table of devices that have been initialized
#ifdef SYCL
static std::unordered_map<int, dpct::device_info> deviceProps;
#else  // CUDA
static std::unordered_map<int, cudaDeviceProp> deviceProps;
#endif
static std::mutex devicePropsMutex;

// Checks prepares device if it's not ready yet and returns device properties
// Also sets shared memory configuration
#ifdef SYCL
void getDeviceProp(int &deviceID, dpct::device_info &prop) try {
  cudaCheck(deviceID = dpct::dev_mgr::instance().current_device_id());

  // need to lock this function	
  std::lock_guard<std::mutex> lock(devicePropsMutex);

  auto it = deviceProps.find(deviceID);
  if (it == deviceProps.end()) {
    // Get device properties and store it for later use
    /*
    DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    cudaCheck( (dpct::dev_mgr::instance().get_device(deviceID).get_device_info(prop), 0));
    //cuttKernelSetSharedMemConfig();
    deviceProps.insert({deviceID, prop});
  } else {
    prop = it->second;
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#else // CUDA
void getDeviceProp(int& deviceID, cudaDeviceProp &prop) {
  cudaCheck(cudaGetDevice(&deviceID));

  // need to lock this function
  std::lock_guard<std::mutex> lock(devicePropsMutex);

  auto it = deviceProps.find(deviceID);
  if (it == deviceProps.end()) {
    // Get device properties and store it for later use
    cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
    cuttKernelSetSharedMemConfig();
    deviceProps.insert({deviceID, prop});
  } else {
    prop = it->second;
  }
}
#endif

cuttResult cuttPlanCheckInput(int rank, int* dim, int* permutation, size_t sizeofType) {
  // Check sizeofType
  if (sizeofType != 4 && sizeofType != 8) return CUTT_INVALID_PARAMETER;
  // Check rank
  if (rank <= 1) return CUTT_INVALID_PARAMETER;
  // Check dim[]
  for (int i=0;i < rank;i++) {
    if (dim[i] <= 1) return CUTT_INVALID_PARAMETER;
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
  if (permutation_fail) return CUTT_INVALID_PARAMETER;  

  return CUTT_SUCCESS;
}

cuttResult cuttPlan(cuttHandle *handle, int rank, int *dim, int *permutation, size_t sizeofType, 
#ifdef SYCL
  sycl::queue *stream
#else // CUDA
  cudaStream_t stream
#endif
  ) {

#ifdef ENABLE_NVTOOLS
  gpuRangeStart("init");
#endif

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
#ifdef SYCL
  dpct::device_info prop;
#else // CUDA
  cudaDeviceProp prop;
#endif
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<cuttPlan_t> plans;
  // if (rank != redDim.size()) {
  //   if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;

#if 0
  if (!cuttKernelDatabase(deviceID, prop)) return CUTT_INTERNAL_ERROR;
#endif

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("createPlans");
#endif

  // std::chrono::high_resolution_clock::time_point plan_start;
  // plan_start = std::chrono::high_resolution_clock::now();

  if (!cuttPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return CUTT_INTERNAL_ERROR;

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
    if (!it->countCycles(prop, 10)) return CUTT_INTERNAL_ERROR;
  }

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("rest");
#endif

  // Choose the plan
  std::list<cuttPlan_t>::iterator bestPlan = choosePlanHeuristic(plans);
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan->print();

  // Create copy of the plan outside the list
  cuttPlan_t* plan = new cuttPlan_t();
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

  return CUTT_SUCCESS;
}

cuttResult cuttPlanMeasure(cuttHandle *handle, int rank, int *dim, int *permutation, size_t sizeofType,
#ifdef SYCL
  sycl::queue *stream, void *idata, void *odata) try 
#else // CUDA
  cudaStream_t stream, void* idata, void* odata)
#endif
{

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  // Create new handle
  *handle = curHandle;
  curHandle++;

  // Check that the current handle is available (it better be!)
  {
    std::lock_guard<std::mutex> lock(planStorageMutex);
    if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;
  }

  // Prepare device
  int deviceID;
#ifdef SYCL
  dpct::device_info prop;
#else // CUDA
  cudaDeviceProp prop;
#endif
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<cuttPlan_t> plans;
#if 0
  // if (rank != redDim.size()) {
    if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
#else
  if (!cuttPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return CUTT_INTERNAL_ERROR;
#endif

  // // Count cycles
  // for (auto it=plans.begin();it != plans.end();it++) {
  //   if (!it->countCycles(prop, 10)) return CUTT_INTERNAL_ERROR;
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
#ifdef SYCL
    set_device_array<char>((char *)odata, -1, numBytes, stream);
    cudaCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
#else // CUDA
    set_device_array<char>((char *)odata, -1, numBytes);
    cudaCheck(cudaDeviceSynchronize());
#endif
    timer.start();
    // Execute plan
    if (!cuttKernel(*it, idata, odata)) return CUTT_INTERNAL_ERROR;
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
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan = plans.begin();

  // printMatlab(prop, plans, times);
  // findMispredictionBest(plans, times, bestPlan, bestTime);
  // bestPlan->print();

  // Create copy of the plan outside the list
  cuttPlan_t* plan = new cuttPlan_t();
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

  return CUTT_SUCCESS;
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

#ifdef SYCL
void cuttDestroy_callback(sycl::queue *stream, int status, 
#else // CUDA
void CUDART_CB cuttDestroy_callback(cudaStream_t stream, cudaError_t status,
#endif
  void *userData) {
  cuttPlan_t* plan = (cuttPlan_t*) userData;
  delete plan;
}

cuttResult cuttDestroy(cuttHandle handle) {
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;
#ifdef CUTT_HAS_UMPIRE
  // get the pointer cuttPlan_t
  cuttPlan_t* plan = it->second;
  cudaStream_t stream = plan->stream;
  // Delete entry from plan storage
  planStorage.erase(it);
  // register callback to deallocate plan
  cudaStreamAddCallback(stream, cuttDestroy_callback, plan, 0);
#else
  // Delete instance of cuttPlan_t	 
  delete it->second;	  
  // Delete entry from plan storage	  
  planStorage.erase(it);
#endif
  return CUTT_SUCCESS;
}

cuttResult cuttExecute(cuttHandle handle, void *idata, void *odata) 
#ifdef SYCL 
try 
#endif
{
  // prevent modification when find
  std::lock_guard<std::mutex> lock(planStorageMutex);
  auto it = planStorage.find(handle);
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  cuttPlan_t& plan = *(it->second);

  int deviceID;
#ifdef SYCL
  cudaCheck(deviceID = dpct::dev_mgr::instance().current_device_id());
#else // CUDA
  cudaCheck(cudaGetDevice(&deviceID));
#endif
  if (deviceID != plan.deviceID) return CUTT_INVALID_DEVICE;

  if (!cuttKernel(plan, idata, odata)) return CUTT_INTERNAL_ERROR;
  return CUTT_SUCCESS;
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

void cuttInitialize() {
#ifdef CUTT_HAS_UMPIRE
  const char* alloc_env_var = std::getenv("CUTT_USES_THIS_UMPIRE_ALLOCATOR");
#define __CUTT_STRINGIZE(x) #x
#define __CUTT_XSTRINGIZE(x) __CUTT_STRINGIZE(x)
  const char* alloc_cstr = alloc_env_var ? alloc_env_var : __CUTT_XSTRINGIZE(CUTT_USES_THIS_UMPIRE_ALLOCATOR);
  cutt_umpire_allocator = umpire::ResourceManager::getInstance().getAllocator(alloc_cstr);
#endif
}

void cuttFinalize() {
}

#ifdef SYCL
sycl::vec<unsigned, 4> ballot(ONEAPI::sub_group sg, bool predicate = true) __attribute__((convergent)) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_GroupNonUniformBallot(__spv::Scope::Subgroup, predicate);
#else
  throw cl::sycl::runtime_error("Sub-groups are not supported on host device.", PI_INVALID_DEVICE);
#endif
}
#endif
