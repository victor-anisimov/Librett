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

#ifdef LIBRETT_USES_SYCL
  #include "sycl_device.hpp"
  #include <complex>
  typedef std::complex<double> librett_complex;
#elif LIBRETT_USES_HIP
  #include <hip/hip_runtime.h>
  #include <hip/hip_complex.h>
  typedef hipDoubleComplex librett_complex;
#elif LIBRETT_USES_CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
  #include <cuComplex.h>
  typedef cuDoubleComplex librett_complex;
#endif

#if !defined(LIBRETT_USES_SYCL) && !defined(LIBRETT_USES_HIP)
  #define LIBRETT_USES_CUDA 1
#endif

// Work units
#ifdef LIBRETT_USES_SYCL
  #define threadIdx_x   item.get_local_id(2)
  #define blockIdx_x    item.get_group(2)
  #define blockDim_x    item.get_local_range().get(2)
  #define gridDim_x     item.get_group_range(2)
  #define threadIdx_y   item.get_local_id(1)
  #define blockIdx_y    item.get_group(1)
  #define blockDim_y    item.get_local_range().get(1)
  #define gridDim_y     item.get_group_range(1)
  #define threadIdx_z   item.get_local_id(0)
  #define blockIdx_z    item.get_group(0)
  #define blockDim_z    item.get_local_range().get(0)
  #define gridDim_z     item.get_group_range(0)
  #define sharedMemPerBlock
  #define numthread_x   numthread[2]
  #define numthread_y   numthread[1]
  #define numthread_z   numthread[0]
  #define numblock_x    numblock[2]
  #define numblock_y    numblock[1]
  #define numblock_z    numblock[0]
  #define gpuMaxThreadsPerBlock    (prop.get_max_work_group_size())
  #define gpuMultiProcessorCount   (prop.get_max_compute_units())
  #define gpuWarpSize              (prop.get_min_sub_group_size())
  #define gpuClockRate             (prop.get_max_clock_frequency())
  #define gpuMajor                 (prop.get_major_version())
  #define gpuSharedMemPerBlock     (prop.get_local_mem_size())
  #define tiledVol_x    tiledVol.x()
  #define tiledVol_y    tiledVol.y()
  #define __gpu_inline__     __attribute__((always_inline))
  #define __global__
  #define __device__

#ifdef __SYCL_DEVICE_ONLY__
extern SYCL_EXTERNAL sycl::detail::ConvertToOpenCLType_t<sycl::vec<unsigned, 4>> __spirv_GroupNonUniformBallot(int, bool) __attribute__((convergent));
#endif

extern SYCL_EXTERNAL sycl::vec<unsigned, 4> ballot(sycl::sub_group, bool);

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
  #define gpuMaxThreadsPerBlock    (prop.maxThreadsPerBlock)
  #define gpuMultiProcessorCount   (prop.multiProcessorCount)
  #define gpuWarpSize              (prop.warpSize)
  #define gpuClockRate             (prop.clockRate / 1000.0)
  #define gpuMajor                 (prop.major)
  #define gpuSharedMemPerBlock     (prop.sharedMemPerBlock)
  #define tiledVol_x    tiledVol.x
  #define tiledVol_y    tiledVol.y
  #define __gpu_inline__     __device__ __forceinline__
#endif

// Data types
#ifdef LIBRETT_USES_SYCL
  using int2_t          = sycl::int2;
  using int4_t          = sycl::int4;
  using float4_t        = sycl::float4;
  using gpuStream_t     = sycl::queue*;
  using gpuDeviceProp_t = Librett::DeviceProp_t;
  using gpuError_t      = int;
#elif LIBRETT_USES_HIP
  using int2_t          = int2;
  using int4_t          = int4;
  using float4_t        = float4;
  using gpuStream_t     = hipStream_t;
  using gpuDeviceProp_t = hipDeviceProp_t;
  using gpuError_t      = hipError_t;
#elif LIBRETT_USES_CUDA
  using int2_t          = int2;
  using int4_t          = int4;
  using float4_t        = float4;
  using gpuStream_t     = cudaStream_t;
  using gpuDeviceProp_t = cudaDeviceProp;
  using gpuError_t      = cudaError_t;
#endif

// Functions
#ifdef LIBRETT_USES_SYCL
  #define gpu_shfl_xor(a,b)   sycl::ext::oneapi::experimental::this_sub_group().shuffle_xor(a,b)
  #define gpu_shuffle(a,b)    sycl::ext::oneapi::experimental::this_sub_group().shuffle(a,b)
  #define gpu_shfl_down(a,b)  sycl::ext::oneapi::experimental::this_sub_group().shuffle_down(a,b)
#define gpu_atomicAdd(a,b)    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>((a)).fetch_add(b)
#elif LIBRETT_USES_HIP
  #define gpu_shfl_xor(a,b)   __shfl_xor(a,b)
  #define gpu_shuffle(a,b)    __shfl(a,b)
  #define gpu_shfl_down(a,b)  __shfl_down(a,b)
  #define gpu_atomicAdd(a,b)  atomicAdd(&(a), b)
  #define gpuOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#elif LIBRETT_USES_CUDA
  #define gpu_shfl_xor(a,b)   __shfl_xor_sync(0xffffffff,a,b)
  #define gpu_shuffle(a,b)    __shfl_sync(0xffffffff,a,b)
  #define gpu_shfl_down(a,b)  __shfl_down_sync(0xffffffff,a,b)
  #define gpu_atomicAdd(a,b)  atomicAdd(&(a), b)
  #define gpuOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
#endif

#endif // UNIAPI_H
