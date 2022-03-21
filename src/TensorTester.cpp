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

//
// Testing utilities
//
#if SYCL
  #include <CL/sycl.hpp>
  #include "dpct/dpct.hpp"
  #define warpSize      item_ct1.get_sub_group().get_local_range().get(0)
#elif HIP
  #include <hip/hip_runtime.h>
#else // CUDA
  #include <cuda.h>
#endif
#include "GpuUtils.h"
#include "GpuMem.h"
#include "TensorTester.h"
#include "uniapi.h"

#if SYCL
void setTensorCheckPatternKernel(unsigned int* data, unsigned int ndata, sycl::nd_item<3> item_ct1) 
#else
__global__ void setTensorCheckPatternKernel(unsigned int* data, unsigned int ndata) 
#endif
{
  for (unsigned int i = threadIdx_x + blockIdx_x*blockDim_x; i < ndata; i += blockDim_x*gridDim_x) {
    data[i] = i;
  }
}

template<typename T>
#if SYCL
void checkTransposeKernel(T* data, unsigned int ndata, int rank, TensorConv* glTensorConv,
  TensorError_t* glError, int* glFail, sycl::nd_item<3> item_ct1, uint8_t *dpct_local) 
#else
__global__ void checkTransposeKernel(T* data, unsigned int ndata, int rank, TensorConv* glTensorConv,
  TensorError_t* glError, int* glFail)
#endif
{

#if SYCL
  auto shPos = (unsigned int *)dpct_local;
#elif HIP
  HIP_DYNAMIC_SHARED( unsigned int, shPos)
#else // CUDA
  extern __shared__ unsigned int shPos[];
#endif

  const int warpLane = threadIdx_x & (warpSize - 1);
  TensorConv tc;
  if (warpLane < rank) {
    tc = glTensorConv[warpLane];
  }

  TensorError_t error;
  error.pos = 0xffffffff;
  error.refVal = 0;
  error.dataVal = 0;

  for (int base = blockIdx_x * blockDim_x; base < ndata; base += blockDim_x * gridDim_x) {
    int i = base + threadIdx_x;
    T dataValT = (i < ndata) ? data[i] : -1;
    int refVal = 0;
    for (int j=0; j < rank; j++) {
#if SYCL
      /* DPCT1023:117: The DPC++ sub-group does not support mask options for shuffle.  */
      refVal += ((i / item_ct1.get_sub_group().shuffle(tc.c, j)) %
                 item_ct1.get_sub_group().shuffle(tc.d, j)) *
                item_ct1.get_sub_group().shuffle(tc.ct, j);
#elif HIP
      refVal += ((i/__shfl(tc.c,j)) % __shfl(tc.d,j))*__shfl(tc.ct,j);
#else // CUDA
      refVal += ((i/__shfl_sync(0xffffffff,tc.c,j)) % __shfl_sync(0xffffffff,tc.d,j))*__shfl_sync(0xffffffff,tc.ct,j);
#endif
    }

    int dataVal = (dataValT & 0xffffffff)/(sizeof(T)/4);

    if (i < ndata && refVal != dataVal && i < error.pos) {
      error.pos = i;
      error.refVal = refVal;
      error.dataVal = dataVal;
    }
  }

  // Set FAIL flag
  if (error.pos != 0xffffffff) {
    // printf("error %d %d %d\n", error.pos, error.refVal, error.dataVal);
    *glFail = 1;
  }

  shPos[threadIdx_x] = error.pos;
  /*
  DPCT1065:116: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  syncthreads();
  for (int d = 1; d < blockDim_x; d *= 2) {
    int t = threadIdx_x + d;
    unsigned int posval = (t < blockDim_x) ? shPos[t] : 0xffffffff;
    /*
    DPCT1065:118: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    syncthreads();
#if SYCL
    shPos[threadIdx_x] = sycl::min(posval, shPos[threadIdx_x]);
#else // CUDA or HIP
    shPos[threadIdx_x] = min(posval, shPos[threadIdx_x]);
#endif
  /*
  DPCT1065:119: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  syncthreads();
  }
  // Minimum error.pos is in shPos[0] (or 0xffffffff in case of no error)

  if (shPos[0] != 0xffffffff && shPos[0] == error.pos) {
    // Error has occured and this thread has the minimum error.pos
    // printf("BOO error %d %d %d | %d\n", error.pos, error.refVal, error.dataVal, blockIdx.x);
    glError[blockIdx_x] = error;
  }

}

// ################################################################################
// ################################################################################
// ################################################################################

//
// Class constructor
//
TensorTester::TensorTester() : maxRank(32), maxNumblock(256) {
  h_tensorConv = new TensorConv[maxRank];
  h_error      = new TensorError_t[maxNumblock];
  allocate_device<TensorConv>(&d_tensorConv, maxRank);
  allocate_device<TensorError_t>(&d_error, maxNumblock);
  allocate_device<int>(&d_fail, 1);
}

//
// Class destructor
//
TensorTester::~TensorTester() {
  delete [] h_tensorConv;
  delete [] h_error;
  deallocate_device<TensorConv>(&d_tensorConv);
  deallocate_device<TensorError_t>(&d_error);
  deallocate_device<int>(&d_fail);
}

void TensorTester::setTensorCheckPattern(unsigned int* data, unsigned int ndata) {
  int numthread = 512;
#if SYCL
  int numblock = std::min<unsigned int>(65535, (ndata - 1) / numthread + 1);
  /*
  DPCT1049:120: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numblock) *
                     sycl::range<3>(1, 1, numthread), sycl::range<3>(1, 1, numthread)),
                     [=](sycl::nd_item<3> item_ct1) {
                       setTensorCheckPatternKernel(data, ndata, item_ct1);
                     });
  });
#elif HIP
  int numblock = min(65535, (ndata - 1)/numthread + 1 );
  hipLaunchKernelGGL(setTensorCheckPatternKernel, dim3(numblock), dim3(numthread ), 0, 0, data, ndata);
  hipCheck(hipGetLastError());
#else // CUDA
  int numblock = min(65535, (ndata - 1)/numthread + 1 );
  setTensorCheckPatternKernel<<< numblock, numthread >>>(data, ndata);
  cudaCheck(cudaGetLastError());
#endif
}

// void calcTensorConv(const int rank, const int* dim, const int* permutation,
//   TensorConv* tensorConv) {

//   tensorConv[0].c = 1;
//   tensorConv[0].d = dim[0];
//   tensorConv[permutation[0]].ct = 1;
//   int ct_prev = 1;
//   for (int i=1;i < rank;i++) {
//     tensorConv[i].c = tensorConv[i-1].c*dim[i-1];
//     tensorConv[i].d = dim[i];
//     int ct = ct_prev*dim[permutation[i-1]];
//     tensorConv[permutation[i]].ct = ct;
//     ct_prev = ct;
//   }

// }

//
// Calculates tensor conversion constants. Returns total volume of tensor
//
int TensorTester::calcTensorConv(const int rank, const int* dim, const int* permutation,
  TensorConv* tensorConv) {

  int vol = dim[0];

  tensorConv[permutation[0]].c  = 1;
  tensorConv[0].ct = 1;
  tensorConv[0].d  = dim[0];
  for (int i=1;i < rank;i++) {
    vol *= dim[i];

    tensorConv[permutation[i]].c = tensorConv[permutation[i-1]].c*dim[permutation[i-1]];

    tensorConv[i].d  = dim[i];
    tensorConv[i].ct = tensorConv[i-1].ct*dim[i-1];

  }

  return vol;
}

template <typename T>
bool TensorTester::checkTranspose(int rank, int *dim, int *permutation, T *data) 
#if SYCL
try 
#endif
{
  if (rank > 32) {
    return false;
  }

  int ndata = calcTensorConv(rank, dim, permutation, h_tensorConv);
#if SYCL
  sycl::queue q = dpct::get_default_queue();
  copy_HtoD<TensorConv>(h_tensorConv, d_tensorConv, rank, &q);
  q.wait();
#else
  copy_HtoD<TensorConv>(h_tensorConv, d_tensorConv, rank);
#endif

  // printf("tensorConv\n");
  // for (int i=0;i < rank;i++) {
  //   printf("%d %d %d\n", h_tensorConv[i].c, h_tensorConv[i].d, h_tensorConv[i].ct);
  // }

#if SYCL
  set_device_array<TensorError_t>(d_error, 0, maxNumblock, &q);
  q.wait();
  set_device_array<int>(d_fail, 0, 1, &q);
#else
  set_device_array<TensorError_t>(d_error, 0, maxNumblock);
  set_device_array<int>(d_fail, 0, 1);
#endif

  int numthread = 512;
  int numblock = std::min(maxNumblock, (ndata - 1) / numthread + 1);
  int shmemsize = numthread*sizeof(unsigned int);
  /*
  DPCT1049:122: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
#if SYCL
  q.wait();
  q.submit([&](sycl::handler &cgh) {
  //dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        dpct_local_acc_ct1(sycl::range<1>(shmemsize), cgh);

    auto d_tensorConv_ct3 = d_tensorConv;
    auto d_error_ct4 = d_error;
    auto d_fail_ct5 = d_fail;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numblock) *
            sycl::range<3>(1, 1, numthread), sycl::range<3>(1, 1, numthread)),
            [=](sycl::nd_item<3> item_ct1) {
               checkTransposeKernel(data, ndata, rank, d_tensorConv_ct3,
               d_error_ct4, d_fail_ct5, item_ct1, dpct_local_acc_ct1.get_pointer());
            });
  });
  q.wait();

  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1, &q);
  dpct::get_current_device().queues_wait_and_throw();
#elif HIP
  hipLaunchKernelGGL(checkTransposeKernel, dim3(numblock), dim3(numthread), shmemsize, 
		     0, data, ndata, rank, d_tensorConv, d_error, d_fail);
  hipCheck(hipGetLastError());

  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1);
  hipCheck(hipDeviceSynchronize());
#else
  checkTransposeKernel<<< numblock, numthread, shmemsize >>>(data, ndata, rank, d_tensorConv, d_error, d_fail);
  cudaCheck(cudaGetLastError());

  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1);
  cudaCheck(cudaDeviceSynchronize());
#endif

  if (h_fail) {
    copy_DtoH_sync<TensorError_t>(d_error, h_error, maxNumblock);
    TensorError_t error;
    error.pos = 0x0fffffff;
    for (int i=0; i < numblock; i++) {
      // printf("%d %d %d\n", error.pos, error.refVal, error.dataVal);
      if (h_error[i].refVal != h_error[i].dataVal && error.pos > h_error[i].pos) {
        error = h_error[i];
      }
    }
    printf("TensorTester::checkTranspose FAIL at %d ref %d data %d\n", error.pos, error.refVal, error.dataVal);
    return false;
  }

  return true;
}
#if SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

// Explicit instances
template bool TensorTester::checkTranspose<int>(int rank, int* dim, int* permutation, int* data);
template bool TensorTester::checkTranspose<long long int>(int rank, int* dim, int* permutation, long long int* data);
