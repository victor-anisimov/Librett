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
  #include <complex>
  typedef std::complex<double> librett_complex;  
#elif HIP
  #include <hip/hip_runtime.h>
  #include <hip/hip_complex.h>
  typedef hipDoubleComplex librett_complex;  
#else // CUDA
  #include <cuda.h>
  #include <cuComplex.h>
  typedef cuDoubleComplex librett_complex;
#endif

#if !defined(SYCL) && !defined(HIP)
  #define LIBRETT_USES_CUDA 1
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
  #define numthread_x   numthread[2]
  #define numthread_y   numthread[1]
  #define numthread_z   numthread[0]
  #define numblock_x    numblock[2]
  #define numblock_y    numblock[1]
  #define numblock_z    numblock[0]
  #define gpuMultiProcessorCount   prop.get_max_compute_units()
  #define gpuMaxGridSize           prop.get_max_nd_range_size()
  #define clockRate     get_max_clock_frequency()
  #define tiledVol_x    tiledVol.x()
  #define tiledVol_y    tiledVol.y()
  #define __gpu_inline__     __dpct_inline__
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
  #define numthread_x   numthread.x
  #define numthread_y   numthread.y
  #define numthread_z   numthread.z
  #define numblock_x    numblock.x 
  #define numblock_y    numblock.y 
  #define numblock_z    numblock.z 
  #define gpuMultiProcessorCount   prop.multiProcessorCount
  #define gpuMaxGridSize           prop.maxGridSize
  #define tiledVol_x    tiledVol.x
  #define tiledVol_y    tiledVol.y
  #define __gpu_inline__     __device__ __forceinline__
#endif

// Data types
#ifdef SYCL
  using ndItem3_t       = sycl::nd_item<3>;
  using int2_t          = sycl::int2;
  using int4_t          = sycl::int4;
  using float4_t        = sycl::float4;
  using gpuStream_t     = sycl::queue*;
  using gpuDeviceProp_t = dpct::device_info;
  using gpuError_t      = int;
#elif HIP
  using int2_t          = int2;
  using int4_t          = int4;
  using float4_t        = float4;
  using gpuStream_t     = hipStream_t;
  using gpuDeviceProp_t = hipDeviceProp_t;
  using gpuError_t      = hipError_t;
#else // CUDA
  using int2_t          = int2;
  using int4_t          = int4;
  using float4_t        = float4;
  using gpuStream_t     = cudaStream_t;
  using gpuDeviceProp_t = cudaDeviceProp;
  using gpuError_t      = cudaError_t;
#endif

// Functions
#ifdef SYCL
  #define gpu_shfl_xor(a,b)   item_ct1.get_sub_group().shuffle_xor(a,b)
  #define gpu_shuffle(a,b)    item_ct1.get_sub_group().shuffle(a,b)
  #define gpu_shfl_down(a,b)  item_ct1.get_sub_group().shuffle_down(a,b)
  #define gpu_atomicAdd(a,b)  sycl::atomic<int>(sycl::global_ptr<int>(&(a))).fetch_add(b)
  #define DeviceSynchronize() dpct::get_current_device().queues_wait_and_throw()
#elif HIP
  #define gpu_shfl_xor(a,b)   __shfl_xor(a,b)
  #define gpu_shuffle(a,b)    __shfl(a,b)
  #define gpu_shfl_down(a,b)  __shfl_down(a,b)
  #define gpu_atomicAdd(a,b)  atomicAdd(&(a), b)
  #define DeviceSynchronize() hipCheck(hipDeviceSynchronize())
#else // CUDA
  #define gpu_shfl_xor(a,b)   __shfl_xor_sync(0xffffffff,a,b)
  #define gpu_shuffle(a,b)    __shfl_sync(0xffffffff,a,b)
  #define gpu_shfl_down(a,b)  __shfl_down_sync(0xffffffff,a,b)
  #define gpu_atomicAdd(a,b)  atomicAdd(&(a), b)
  #define DeviceSynchronize() cudaCheck(cudaDeviceSynchronize())
#endif

#endif // UNIAPI_H
