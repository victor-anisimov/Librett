/******************************************************************************
MIT License

Copyright (c) 2022 Victor Anisimov
Copyright (c) 2022 Argonne National Laboratory

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

#ifndef UNIAPI_H
#define UNIAPI_H

#ifdef SYCL
  #include <CL/sycl.hpp>
  #include "dpct/dpct.hpp"
#elif HIP
  #include <hip/hip_runtime.h>
#else // CUDA
  #include <cuda.h>
#endif

// Work units
#ifdef SYCL
  #define threadIdx_x   item_ct1.get_local_id(2)
  #define blockIdx_x    item_ct1.get_group(2)
  #define blockDim_x    item_ct1.get_local_range().get(2)
  #define gridDim_x     item_ct1.get_group_range(2)
  #define threadIdx_y   item_ct1.get_local_id(1)
  #define blockIdx_y    item_ct1.get_group(1)
  #define blockDim_y    item_ct1.get_local_range().get(1)
  #define gridDim_y     item_ct1.get_group_range(1)
  #define threadIdx_z   item_ct1.get_local_id(0)
  #define blockIdx_z    item_ct1.get_group(0)
  #define blockDim_z    item_ct1.get_local_range().get(0)
  #define gridDim_z     item_ct1.get_group_range(0)
  #define syncthreads() item_ct1.barrier()
  #define subgroup      item_ct1.get_sub_group()
  #define maxThreadsPerBlock  get_max_work_group_size()
  #define sharedMemPerBlock   get_local_mem_size()
#else // CUDA & HIP
  #define threadIdx_x   threadIdx.x
  #define blockIdx_x    blockIdx.x
  #define blockDim_x    blockDim.x
  #define gridDim_x     gridDim.x
  #define threadIdx_y   threadIdx.y
  #define blockIdx_y    blockIdx.y
  #define blockDim_y    blockDim.y
  #define gridDim_y     gridDim.y
  #define threadIdx_z   threadIdx.z
  #define blockIdx_z    blockIdx.z
  #define blockDim_z    blockDim.z
  #define gridDim_z     gridDim.z
  #define syncthreads() __syncthreads()
#endif

// Data types
#ifdef SYCL
  using ndItem3_t       = sycl::nd_item<3>;
  using int2_t          = sycl::int2;
  using int4_t          = sycl::int4;
  using float4_t        = sycl::float4;
  using gpuStream_t     = sycl::queue*;
  using gpuDeviceProp_t = dpct::device_info;
#elif HIP
  using int2_t          = int2;
  using int4_t          = int4;
  using float4_t        = float4;
  using gpuStream_t     = hipStream_t;
  using gpuDeviceProp_t = hipDeviceProp_t;
#else // CUDA
  using int2_t          = int2;
  using int4_t          = int4;
  using float4_t        = float4;
  using gpuStream_t     = cudaStream_t;
  using gpuDeviceProp_t = cudaDeviceProp;
#endif

// Error checking wrappers
#ifdef SYCL
  #define cudaCheck(stmt) do { int err = stmt; } while (0)

#elif HIP
  #define hipCheck(stmt) do {                                                       \
    hipError_t err = stmt;                                                          \
    if (err != hipSuccess) {                                                        \
      fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
      fprintf(stderr, "Error String: %s\n", hipGetErrorString(err));                \
      exit(1);                                                                      \
    }                                                                               \
  } while(0)

#else // CUDA
  #define cudaCheck(stmt) do {                                                      \
    cudaError_t err = stmt;                                                         \
    if (err != cudaSuccess) {                                                       \
      fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
      fprintf(stderr, "Error String: %s\n", cudaGetErrorString(err));               \
      exit(1);                                                                      \
    }                                                                               \
  } while(0)
#endif

#endif // UNIAPI_H
