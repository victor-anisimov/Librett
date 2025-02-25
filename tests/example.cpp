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
#include <vector>
#include <algorithm>
#include <ctime>           // std::time
#include <cstring>         // strcmp
#include <cmath>
#include <complex>
#include "librett.h"
#include "GpuUtils.h"
#include "GpuMem.hpp"
#include "TensorTester.h"
#include "Timer.h"
#include "GpuModel.h"      // testCounters
#include "GpuUtils.h"

std::complex<double> * dataIn  = NULL;
std::complex<double> * dataOut = NULL;
int nData = 200000000;

#define DEBUG_PRINT 0   // set to 1 to print all tensor index values

template <typename T> bool tensor_transpose(std::vector<int>& dim, std::vector<int>& permutation, gpuStream_t& gpustream);
void printVec(std::vector<int>& vec);

void CreateGpuStream(gpuStream_t& master_gpustream) {
  #if LIBRETT_USES_SYCL
  sycl::device dev(sycl::gpu_selector_v);
  sycl::context ctxt(dev, Librett::sycl_asynchandler, sycl::property_list{sycl::property::queue::in_order{}});
  master_gpustream = new sycl::queue(ctxt, dev, Librett::sycl_asynchandler, sycl::property_list{sycl::property::queue::in_order{}});
  #elif LIBRETT_USES_HIP
  hipCheck(hipStreamCreate(&master_gpustream));
  #elif LIBRETT_USES_CUDA
  cudaCheck(cudaStreamCreate(&master_gpustream));
  #endif
}

void DestroyGpuStream(gpuStream_t& master_gpustream) {
  #if LIBRETT_USES_SYCL
  delete master_gpustream;
  #elif LIBRETT_USES_HIP
  hipCheck(hipStreamDestroy(master_gpustream));
  #elif LIBRETT_USES_CUDA
  cudaCheck(cudaStreamDestroy(master_gpustream));
  #endif
}

void gpuDeviceSynchronize(gpuStream_t& master_gpustream) {
  #if LIBRETT_USES_SYCL
  master_gpustream->wait_and_throw();
  #elif LIBRETT_USES_HIP
  hipCheck(hipStreamSynchronize(master_gpustream));
  #elif LIBRETT_USES_CUDA
  cudaCheck(cudaStreamSynchronize(master_gpustream));
  #endif
}

int main(int argc, char *argv[])
{
  DeviceReset();
  // create a master gpu stream
  gpuStream_t gpumasterstream;
  CreateGpuStream(gpumasterstream);

  bool passed = true;
  std::vector<int> dim{3,3,3,2,4};
  std::vector<int> permutation{1,3,0,4,2};

  // Allocate device data, 200M elements
  allocate_device<std::complex<double>>(&dataIn, nData, gpumasterstream);
  allocate_device<std::complex<double>>(&dataOut, nData, gpumasterstream);

  printf("\n4-byte test\n");
  if (!tensor_transpose<int>(dim, permutation, gpumasterstream)) passed = false;

  printf("\n8-byte test\n");
  if (!tensor_transpose<long long int>(dim, permutation, gpumasterstream)) passed = false;

  printf("\n16-byte test\n");
  if (!tensor_transpose<std::complex<double>>(dim, permutation, gpumasterstream)) passed = false;

  if(passed)
    printf("\nTest OK\n");
  else
    printf("\nTest failed\n");

  deallocate_device<std::complex<double>>(&dataIn, gpumasterstream);
  deallocate_device<std::complex<double>>(&dataOut, gpumasterstream);

  DestroyGpuStream(gpumasterstream);
  DeviceReset();

  if(passed)
    return 0;
  else
    return 1;
}


template <typename T>
bool tensor_transpose(std::vector<int> &dim, std::vector<int> &permutation, gpuStream_t& gpustream)
{
  int rank = dim.size();

  int vol = 1;
  for (int r=0; r < rank; r++) {
    vol *= dim[r];
  }

  printf("Number of elements %d\n",vol);
  printf("Dimensions\n");
  printVec(dim);
  printf("Permutation\n");
  printVec(permutation);

  size_t volmem = vol * sizeof(T);
  size_t datamem = nData * sizeof(long long int);
  if (volmem > datamem) {
    printf("#ERROR(test_tensor): Data size exceeded: %zu %zu\n",volmem,datamem);
    return false;
  }

  // allocate memory on the host
  T *inp = (T *) malloc(vol * sizeof(T));   // original tensor
  T *out = (T *) malloc(vol * sizeof(T));   // transposed tensor

  // Initialize the input tensor with sequence of numbers (0, 1, 2, ...)
  for(int i=0; i<vol; i++)
    inp[i] = i;

  // copy the data to GPU
  copy_HtoD_sync((T *)inp, (T *)dataIn, vol, gpustream);

  // create plan
  librettHandle plan;
  librettCheck(librettPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), gpustream));
  gpuDeviceSynchronize(gpustream);

  // execute plan
  librettCheck(librettExecute(plan, dataIn, dataOut));

  // delete plan
  librettCheck(librettDestroy(plan));

  // copy transposed tensor back to the host
  copy_DtoH_sync((T *)dataOut, (T *)out, vol, gpustream);

  // for original tensor, global linear index = i + a*j + a*b*k + a*b*c*l + ...
  // a is size of index i
  // b is size of index j
  // c is size of index l
  // etc.
  std::vector<int> abc(rank);      // product of dimensions 1,a,ab,abc,... of original tensor
  std::vector<int> cba(rank);      // product of dimensions of transposed tensor
  abc[0] = 1;
  cba[0] = 1;
  for(int i=1; i<rank; i++) {
    abc[i] = abc[i-1] * dim[i-1];
    cba[i] = cba[i-1] * dim[permutation[i-1]];
  }

  // perform validation of tensor transpose
  bool passed = true;
  std::vector<int> ijk(rank);
  for(int index=0; index<vol; index++) {

    // determine tensor indices from the global index of the original tensor
    int remainder = index;
    for(int r=rank-1; r>=0; r--) {
      ijk[r] = remainder / abc[r];
      remainder -= abc[r] * ijk[r];
    }

    // determine global index, tindex of the transposed tensor
    int tindex = 0;
    for(int r=0; r<rank; r++)
      tindex += cba[r] * ijk[permutation[r]];

    if(DEBUG_PRINT || index < 10) {
      printf("index: %5d   ijk:", index);
      for(int r=0; r<rank; r++)
        printf(" %3d", ijk[r]);
      printf("    out[%5d]: %5d  out[%5d]: %5d  tindex: %5d\n", index, (int) std::real(out[index]), tindex, (int) std::real(out[tindex]), tindex);
    }

    // check correctness of the transposed matrix
    if(out[tindex] != inp[index]) {
      passed = false;
      break;
    }
  }

  free(inp);
  free(out);

  return passed;
}

void printVec(std::vector<int>& vec) {
  for(int i=0;i < vec.size();i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}
