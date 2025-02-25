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
#include "TensorTester.h"

#if LIBRETT_USES_SYCL
void setTensorCheckPatternKernel(unsigned int* data, unsigned int ndata, sycl::nd_item<3>& item)
#else
__global__ void setTensorCheckPatternKernel(unsigned int* data, unsigned int ndata)
#endif
{
  for (unsigned int i = threadIdx_x + blockIdx_x*blockDim_x; i < ndata; i += blockDim_x*gridDim_x) {
    data[i] = i;
  }
}

template<typename T>
#if LIBRETT_USES_SYCL
void checkTransposeKernel(T* data, unsigned int ndata, int rank, TensorConv* glTensorConv,
  TensorError_t* glError, int* glFail, sycl::nd_item<3>& item, uint8_t *dpct_local)
#else
__global__ void checkTransposeKernel(T* data, unsigned int ndata, int rank, TensorConv* glTensorConv,
  TensorError_t* glError, int* glFail)
#endif
{

#if LIBRETT_USES_SYCL
  sycl::group wrk_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();
  int warpSize = sg.get_local_range().get(0);
  auto shPos = (unsigned int *)dpct_local;
#elif LIBRETT_USES_HIP
  HIP_DYNAMIC_SHARED( unsigned int, shPos)
#elif LIBRETT_USES_CUDA
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
      refVal += ((i/gpu_shuffle(tc.c,j)) % gpu_shuffle(tc.d,j)) * gpu_shuffle(tc.ct,j);
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
  #if LIBRETT_USES_SYCL
  sycl::group_barrier( wrk_grp );
  #else
  syncthreads();
  #endif

  for (int d = 1; d < blockDim_x; d *= 2) {
    int t = threadIdx_x + d;
    unsigned int posval = (t < blockDim_x) ? shPos[t] : 0xffffffff;
#if LIBRETT_USES_SYCL
    sycl::group_barrier( wrk_grp );
    shPos[threadIdx_x] = sycl::min(posval, shPos[threadIdx_x]);
    sycl::group_barrier( wrk_grp );
#elif LIBRETT_USES_CUDA
    syncthreads();
    shPos[threadIdx_x] = min(posval, shPos[threadIdx_x]);
    syncthreads();
#endif

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
TensorTester::TensorTester(gpuStream_t& stream) : maxRank(32), maxNumblock(256) {
  h_tensorConv = new TensorConv[maxRank];
  h_error      = new TensorError_t[maxNumblock];

  this->tt_gpustream = stream;

  allocate_device<TensorConv>(&d_tensorConv, maxRank, tt_gpustream);
  allocate_device<TensorError_t>(&d_error, maxNumblock, tt_gpustream);
  allocate_device<int>(&d_fail, 1, tt_gpustream);
}

//
// Class destructor
//
TensorTester::~TensorTester() {
  delete [] h_tensorConv;
  delete [] h_error;

  deallocate_device<TensorConv>(&d_tensorConv, tt_gpustream);
  deallocate_device<TensorError_t>(&d_error, tt_gpustream);
  deallocate_device<int>(&d_fail, tt_gpustream);
}

void TensorTester::setTensorCheckPattern(unsigned int* data, unsigned int ndata) {
  int numthread = 512;
#if LIBRETT_USES_SYCL
  int numblock = std::min<unsigned int>(65535, (ndata - 1) / numthread + 1);
  this->tt_gpustream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(
                       sycl::range<3>(1, 1, numblock) *
                       sycl::range<3>(1, 1, numthread), sycl::range<3>(1, 1, numthread)),
                     [=](sycl::nd_item<3> item) {
                       setTensorCheckPatternKernel(data, ndata, item);
                     });
  });
#elif LIBRETT_USES_HIP
  int numblock = min(65535, (ndata - 1)/numthread + 1 );
  hipLaunchKernelGGL(setTensorCheckPatternKernel, dim3(numblock), dim3(numthread ), 0, this->tt_gpustream, data, ndata);
  hipCheck(hipGetLastError());
#elif LIBRETT_USES_CUDA
  int numblock = min(65535, (ndata - 1)/numthread + 1 );
  setTensorCheckPatternKernel<<< numblock, numthread, 0, this->tt_gpustream >>>(data, ndata);
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
{
  if (rank > 32) {
    return false;
  }

  int ndata = calcTensorConv(rank, dim, permutation, h_tensorConv);
  copy_HtoD<TensorConv>(h_tensorConv, d_tensorConv, rank, this->tt_gpustream);

  // printf("tensorConv\n");
  // for (int i=0;i < rank;i++) {
  //   printf("%d %d %d\n", h_tensorConv[i].c, h_tensorConv[i].d, h_tensorConv[i].ct);
  // }

  set_device_array<TensorError_t>(d_error, 0, maxNumblock, this->tt_gpustream);
  set_device_array<int>(d_fail, 0, 1, this->tt_gpustream);

  int numthread = 512;
  int numblock = std::min(maxNumblock, (ndata - 1) / numthread + 1);
  int shmemsize = numthread*sizeof(unsigned int);
#if LIBRETT_USES_SYCL
  this->tt_gpustream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1{sycl::range<1>(shmemsize), cgh};

    auto d_tensorConv_ct3 = d_tensorConv;
    auto d_error_ct4 = d_error;
    auto d_fail_ct5 = d_fail;

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1,1,numblock) * sycl::range<3>(1,1,numthread),
                                       sycl::range<3>(1,1,numthread)),
            [=](sycl::nd_item<3> item) {
               checkTransposeKernel(data, ndata, rank, d_tensorConv_ct3,
               d_error_ct4, d_fail_ct5, item, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
  });

  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1, this->tt_gpustream);
  this->tt_gpustream->wait_and_throw();
#elif LIBRETT_USES_HIP
  hipLaunchKernelGGL(checkTransposeKernel, dim3(numblock), dim3(numthread), shmemsize,
		     this->tt_gpustream, data, ndata, rank, d_tensorConv, d_error, d_fail);
  hipCheck(hipGetLastError());

  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1, this->tt_gpustream);
  hipCheck(hipDeviceSynchronize());
#else
  checkTransposeKernel<<< numblock, numthread, shmemsize, this->tt_gpustream >>>(data, ndata, rank, d_tensorConv, d_error, d_fail);
  cudaCheck(cudaGetLastError());

  int h_fail;
  copy_DtoH<int>(d_fail, &h_fail, 1, this->tt_gpustream);
  cudaCheck(cudaDeviceSynchronize());
#endif


  if (h_fail) {
    copy_DtoH_sync<TensorError_t>(d_error, h_error, maxNumblock, this->tt_gpustream);
    TensorError_t error;
    error.pos = 0x0fffffff;
    for (int i=0; i < numblock; i++) {
      // printf("%d %d %d\n", error.pos, error.refVal, error.dataVal);
      if (h_error[i].refVal != h_error[i].dataVal && error.pos > h_error[i].pos) {
        error = h_error[i];
      }
    }
    printf("TensorTester::checkTranspose FAIL at %llu ref %d data %d\n", error.pos, error.refVal, error.dataVal);
    return false;
  }

  return true;
}

// Explicit instances
template bool TensorTester::checkTranspose<int>(int rank, int* dim, int* permutation, int* data);
template bool TensorTester::checkTranspose<long long int>(int rank, int* dim, int* permutation, long long int* data);
