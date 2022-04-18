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
#include "librett.h"
#include "GpuUtils.h"
#include "GpuMem.h"
#include "TensorTester.h"
#include "Timer.h"
#include "GpuModel.h"      // testCounters

#ifdef SYCL
sycl::queue q = dpct::get_default_queue();
#endif

librettTimer* timerFloat;
librettTimer* timerDouble;

long long int* dataIn  = NULL;
long long int* dataOut = NULL;
int dataSize  = 200000000;
TensorTester* tester = NULL;

bool test();
template <typename T> bool test_tensor(std::vector<int>& dim, std::vector<int>& permutation);
void printVec(std::vector<int>& vec);

int main(int argc, char *argv[]) 
#ifdef SYCL
try 
#endif
{

  int gpuid = -1;
  bool arg_ok = true;
  if (argc >= 3) {
    if (strcmp(argv[1], "-device") == 0) {
      sscanf(argv[2], "%d", &gpuid);
    } else {
      arg_ok = false;
    }
  } else if (argc > 1) {
    arg_ok = false;
  }

  if (!arg_ok) {
    printf("librett_test [options]\n");
    printf("Options:\n");
    printf("-device gpuid : use GPU with ID gpuid\n");
    return 1;
  }

  if (gpuid >= 0) {
#if SYCL
    dpct::dev_mgr::instance().select_device(gpuid);
#elif HIP
    hipCheck(hipSetDevice(gpuid));
#else // CUDA
    cudaCheck(cudaSetDevice(gpuid));
#endif
  }

#if SYCL
  dpct::get_current_device().reset();
#elif HIP
  hipCheck(hipDeviceReset());
  // hipCheck(hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte));
#else // CUDA
  cudaCheck(cudaDeviceReset());
  // cudaCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif

  timerFloat = new librettTimer(4);
  timerDouble = new librettTimer(8);

  // Allocate device data, 100M elements
  allocate_device<long long int>(&dataIn, dataSize);
  allocate_device<long long int>(&dataOut, dataSize);

  // Create tester
  tester = new TensorTester();
  tester->setTensorCheckPattern((unsigned int *)dataIn, dataSize*2);

  bool passed = true;
  if(passed){passed = test(); if(!passed) printf("Test failed\n");}

  if(passed){
    std::vector<int> worstDim;
    std::vector<int> worstPermutation;
    double worstBW = timerDouble->getWorst(worstDim, worstPermutation);
    printf("worstBW %4.2lf GB/s\n", worstBW);
    printf("dim\n");
    printVec(worstDim);
    printf("permutation\n");
    printVec(worstPermutation);
    printf("test OK\n");
  }

  deallocate_device<long long int>(&dataIn);
  deallocate_device<long long int>(&dataOut);
  delete tester;

  delete timerFloat;
  delete timerDouble;

#if SYCL
  dpct::get_current_device().reset();
#elif HIP
  hipCheck(hipDeviceReset());
#else // CUDA
  cudaCheck(cudaDeviceReset());
#endif

  if(passed)
    return 0;
  else
    return 1;
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

//
// Test: hand picked examples
//
bool test() {

  {
    int rank = 3;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 1305;
    dim[1] = 599;
    dim[2] = 88;
    permutation[0] = 0;
    permutation[1] = 2;
    permutation[2] = 1;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;
  }
  {
    std::vector<int> dim(10);
    std::vector<int> permutation(10);
    dim[0] = 5;
    dim[1] = 4;
    dim[2] = 7;
    dim[3] = 8;
    dim[4] = 6;
    dim[5] = 5;
    dim[6] = 4;
    dim[7] = 7;
    dim[8] = 8;
    dim[9] = 5;
    permutation[0] = 2 - 1;
    permutation[1] = 4 - 1;
    permutation[2] = 5 - 1;
    permutation[3] = 3 - 1;
    permutation[4] = 1 - 1;
    permutation[5] = 7 - 1;
    permutation[6] = 9 - 1;
    permutation[7] = 10 - 1;
    permutation[8] = 8 - 1;
    permutation[9] = 6 - 1;
    if (!test_tensor<long long int>(dim, permutation)) return false;
    if (!test_tensor<int>(dim, permutation)) return false;
  }

  return true;
}


template <typename T>
bool test_tensor(std::vector<int> &dim, std::vector<int> &permutation)
#ifdef SYCL
try
#endif
{
  int rank = dim.size();

  int vol = 1;
  for (int r=0;r < rank;r++) {
    vol *= dim[r];
  }

  printf("Number of elements %d\n",vol);
  printf("Dimensions\n");
  printVec(dim);
  printf("Permutation\n");
  printVec(permutation);

  size_t volmem = vol*sizeof(T);
  size_t datamem = dataSize*sizeof(long long int);
  if (volmem > datamem) {
    printf("#ERROR(test_tensor): Data size exceeded: %llu %llu\n",volmem,datamem);
    return false;
  }

  librettTimer* timer;
  if (sizeof(T) == 4) {
    timer = timerFloat;
  } else {
    timer = timerDouble;
  }

  librettHandle plan;
#if SYCL
  sycl::queue q = dpct::get_default_queue();
  librettCheck(librettPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), &q));
  set_device_array<T>((T *)dataOut, -1, vol, &q);
  dpct::get_current_device().queues_wait_and_throw();
#elif HIP
  librettCheck(librettPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), 0));
  set_device_array<T>((T *)dataOut, -1, vol);
  hipCheck(hipDeviceSynchronize());
#else // CUDA
  librettCheck(librettPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), 0));
  set_device_array<T>((T *)dataOut, -1, vol);
  cudaCheck(cudaDeviceSynchronize());
#endif

  if (vol > 1000000) timer->start(dim, permutation);
  librettCheck(librettExecute(plan, dataIn, dataOut));
  if (vol > 1000000) timer->stop();
  //q.wait();

  librettCheck(librettDestroy(plan));

  return tester->checkTranspose<T>(rank, dim.data(), permutation.data(), (T *)dataOut);
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

void printVec(std::vector<int>& vec) {
  for (int i=0;i < vec.size();i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}

