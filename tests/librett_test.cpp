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

Modifications Copyright (c) 2022 Advanced Micro Devices, Inc.
All rights reserved.
*******************************************************************************/
#include <vector>
#include <algorithm>
#include <ctime>           // std::time
#include <cstring>         // strcmp
#include <cmath>
#include <random>
#include "librett.h"
#include "GpuUtils.h"
#include "GpuMem.hpp"
#include "TensorTester.h"
#include "Timer.h"
#include "GpuModel.h"      // testCounters
#include "GpuUtils.h"

#ifdef LIBRETT_USES_SYCL
auto sycl_asynchandler = [] (sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const& ex) {
      std::cout << "Caught asynchronous SYCL exception:" << std::endl
                << ex.what() << ", SYCL code: " << ex.code() << std::endl;
    }
  }
};
#endif

librettTimer* timerFloat;
librettTimer* timerDouble;

long long int* dataIn  = NULL;
long long int* dataOut = NULL;
#if PERFTEST
  int dataSize  = 20000000;
#else
  int dataSize  = 200000000;
#endif
TensorTester* tester = NULL;

bool test1(gpuStream_t&);
bool test2(gpuStream_t&);
bool test3(gpuStream_t&);
bool test4();
bool test5();
template <typename T> bool test_tensor(std::vector<int>& dim, std::vector<int>& permutation, gpuStream_t& stream);
void printVec(std::vector<int>& vec);

void gpuDeviceSynchronize(gpuStream_t& master_gpustream) {
  #if LIBRETT_USES_SYCL
  master_gpustream->wait_and_throw();
  #elif LIBRETT_USES_HIP
  hipCheck(hipDeviceSynchronize());
  #elif LIBRETT_USES_CUDA
  cudaCheck(cudaDeviceSynchronize());
  #endif
}

void CreateGpuStream(gpuStream_t& master_gpustream) {
  #if LIBRETT_USES_SYCL
  sycl::device dev(sycl::gpu_selector_v);
  sycl::context ctxt(dev, sycl_asynchandler, sycl::property_list{sycl::property::queue::in_order{}});
  master_gpustream = new sycl::queue(ctxt, dev, sycl_asynchandler, sycl::property_list{sycl::property::queue::in_order{}});
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

int main(int argc, char *argv[])
{
  DeviceReset();
  // create a master gpu stream
  gpuStream_t gpumasterstream;
  CreateGpuStream(gpumasterstream);


  timerFloat = new librettTimer(4);
  timerDouble = new librettTimer(8);

  // Allocate device data, 100M elements
  allocate_device<long long int>(&dataIn, dataSize, gpumasterstream);
  allocate_device<long long int>(&dataOut, dataSize, gpumasterstream);

  // Create tester
  tester = new TensorTester(gpumasterstream);
  tester->setTensorCheckPattern((unsigned int *)dataIn, dataSize*2);

  bool passed = true;
  if(passed){passed = test1(gpumasterstream); if(!passed) printf("Test 1 failed\n");}
  if(passed){passed = test2(gpumasterstream); if(!passed) printf("Test 2 failed\n");}
  if(passed){passed = test3(gpumasterstream); if(!passed) printf("Test 3 failed\n");}
#ifndef PERFTEST
  if(passed){passed = test4(); if(!passed) printf("Test 4 failed\n");}
#ifndef LIBRETT_USES_HIP
  if(passed){passed = test5(); if(!passed) printf("Test 5 failed\n");}
#endif
#endif

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

  deallocate_device<long long int>(&dataIn, gpumasterstream);
  deallocate_device<long long int>(&dataOut, gpumasterstream);
  delete tester;

  delete timerFloat;
  delete timerDouble;

  DestroyGpuStream(gpumasterstream);
  DeviceReset();

  if(passed)
    return 0;
  else
    return 1;
}

//
// Test 1: Test all permutations up to rank 7 on smallish tensors
//
bool test1(gpuStream_t& master_gpustream) {
  const int minDim = 2;
  const int maxDim = 16;
  for (int rank = 2;rank <= 7;rank++) {

    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    for (int r=0;r < rank;r++) {
      permutation[r] = r;
      dim[r] = minDim + r*(maxDim - minDim)/rank;
    }

    do {
      if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
      if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
    } while (std::next_permutation(permutation.begin(), permutation.begin() + rank));

  }

  return true;
}

//
// Test 2: Test ranks 2-15, random volume, random permutation, random dimensions
//         100 samples each rank
//
bool test2(gpuStream_t& master_gpustream) {
  double minDim = 2.0;

  std::srand(unsigned (std::time(0)));

  for (int rank = 2;rank <= 15;rank++) {
    double volmin = pow(minDim + 1, rank);
    double volmax = (double)dataSize;

    for (int isample=0;isample < 100;isample++) {

      std::vector<int> dim(rank);
      std::vector<int> permutation(rank);
      for (int r=0;r < rank;r++) permutation[r] = r;
      double vol = 1.0;
      double curvol = 1.0;
      int iter = 0;
      do {
        vol = (volmin + (volmax - volmin)*((double)rand())/((double)RAND_MAX) );

        int subiter = 0;
        do {
          for (int r=0;r < rank;r++) {
            double vol_left = vol/(curvol*pow(minDim, (double)(rank-r)));
            double aveDim = pow(vol, 1.0/(double)rank);
            double dimSpread = (aveDim - minDim);
            // rn = -1 ... 1
            double rn = 2.0*(((double)rand())/((double)RAND_MAX) - 0.5);
            dim[r] = (int)(aveDim + dimSpread*rn);
            curvol *= (double)dim[r];
          }

          // printf("vol %lf curvol %lf\n", vol, curvol);
          // printf("dim");
          // for (int r=0;r < rank;r++) printf(" %d", dim[r]);
          // printf("\n");

          double vol_scale = pow(vol/curvol, 1.0/(double)rank);
          // printf("vol_scale %lf\n", vol_scale);
          curvol = 1.0;
          for (int r=0;r < rank;r++) {
            dim[r] = std::max(2, (int)round((double)dim[r]*vol_scale));
            curvol *= dim[r];
          }

          // printf("vol %lf curvol %lf\n", vol, curvol);
          // printf("dim");
          // for (int r=0;r < rank;r++) printf(" %d", dim[r]);
          // printf("\n");
          // return false;

          subiter++;
        } while (subiter < 50 && (curvol > volmax || fabs(curvol-vol)/(double)vol > 2.3));

        // printf("vol %lf curvol %lf volmin %lf volmax %lf\n", vol, curvol, volmin, volmax);
        // printf("dim");
        // for (int r=0;r < rank;r++) printf(" %d", dim[r]);
        // printf("\n");

        iter++;
        if (iter == 1000) {
          printf("vol %lf\n", vol);
          printf("Unable to determine dimensions in 1000 iterations\n");
          return false;
        }
      } while (curvol > volmax || fabs(curvol-vol)/(double)vol > 2.3);

      std::random_device rd;  // Obtain a random number from hardware
      std::mt19937 eng(rd()); // Seed the generator
      std::shuffle(permutation.begin(), permutation.end(), eng);

      if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
      if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
    }

  }

  return true;
}

//
// Test 3: hand picked examples
//
bool test3(gpuStream_t& master_gpustream) {

  {
    int rank = 2;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 43;
    dim[1] = 67;
    permutation[0] = 1;
    permutation[1] = 0;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
    dim[0] = 65536*32;
    dim[1] = 2;
    permutation[0] = 1;
    permutation[1] = 0;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
  }

  {
    int rank = 3;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
#if PERFTEST
    dim[0] = 651;
    dim[1] = 299;
    dim[2] = 44;
#else
    dim[0] = 1305;
    dim[1] = 599;
    dim[2] = 88;
#endif
    permutation[0] = 0;
    permutation[1] = 2;
    permutation[2] = 1;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
  }

  {
    int rank = 4;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 24;
#if PERFTEST
    dim[1] = 170;
    dim[2] = 32;
    dim[3] = 97;
#else
    dim[1] = 330;
    dim[2] = 64;
    dim[3] = 147;
#endif
    permutation[0] = 1;
    permutation[1] = 0;
    permutation[2] = 2;
    permutation[3] = 3;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
  }

  {
    int rank = 4;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 2;
    dim[1] = 5;
    dim[2] = 9;
    dim[3] = 12;
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 2;
    permutation[3] = 3;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
  }

  {
    int rank = 6;
    std::vector<int> dim(rank);
    std::vector<int> permutation(rank);
    dim[0] = 2;
    dim[1] = 4;
    dim[2] = 6;
    dim[3] = 9;
    dim[4] = 11;
    dim[5] = 13;
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 2;
    permutation[3] = 3;
    permutation[4] = 4;
    permutation[5] = 5;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
  }

  {
    std::vector<int> dim(5);
    std::vector<int> permutation(5);
    dim[0] = 5;
#if PERFTEST
    dim[1] = 32;
    dim[2] = 45;
    dim[3] = 63;
    dim[4] = 37;
#else
    dim[1] = 42;
    dim[2] = 75;
    dim[3] = 86;
    dim[4] = 57;
#endif
    permutation[0] = 2 - 1;
    permutation[1] = 4 - 1;
    permutation[2] = 5 - 1;
    permutation[3] = 3 - 1;
    permutation[4] = 1 - 1;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
  }

  {
    std::vector<int> dim(5);
    std::vector<int> permutation(5);
    dim[0] = 5;
    dim[1] = 3;
    dim[2] = 2;
    dim[3] = 9;
    dim[4] = 14;
    permutation[0] = 0;
    permutation[1] = 1;
    permutation[2] = 3;
    permutation[3] = 2;
    permutation[4] = 4;
    if (!test_tensor<long long int>(dim, permutation, master_gpustream)) return false;
    if (!test_tensor<int>(dim, permutation, master_gpustream)) return false;
  }

  return true;
}

//
// Test 4: streaming
//
bool test4()
{
  std::vector<int> dim = {24, 32, 16, 36, 43, 9};
  std::vector<int> permutation = {5, 1, 4, 2, 3, 0};

  const int numStream = 10;
  gpuStream_t streams[numStream];

#if LIBRETT_USES_SYCL
  sycl::device dev(sycl::gpu_selector_v);
  sycl::context ctxt(dev, sycl_asynchandler, sycl::property_list{sycl::property::queue::in_order{}});
  for (int i=0;i < numStream;i++) {
    streams[i] = new sycl::queue(ctxt, dev, sycl_asynchandler, sycl::property_list{sycl::property::queue::in_order{}});
  }
#elif LIBRETT_USES_HIP
  for (int i=0;i < numStream;i++) {
    hipCheck(hipStreamCreate(&streams[i]));
  }
#elif LIBRETT_USES_CUDA
  for (int i=0;i < numStream;i++) {
    cudaCheck(cudaStreamCreate(&streams[i]));
  }
#endif

  librettHandle plans[numStream];

  for (int i=0;i < numStream;i++) {
    librettCheck(librettPlan(&plans[i], dim.size(), dim.data(), permutation.data(), sizeof(double), streams[i]));
    librettCheck(librettExecute(plans[i], dataIn, dataOut));
  }

#if LIBRETT_USES_SYCL
  for (int i=0;i < numStream;i++) {
    streams[i]->wait_and_throw();
  }
#elif LIBRETT_USES_HIP
  hipCheck(hipDeviceSynchronize());
#elif LIBRETT_USES_CUDA
  cudaCheck(cudaDeviceSynchronize());
#endif

  bool run_ok = tester->checkTranspose(dim.size(), dim.data(), permutation.data(), (long long int *)dataOut);

#if LIBRETT_USES_SYCL
  for (int i=0;i < numStream;i++) {
    streams[i]->wait_and_throw();
  }
#elif LIBRETT_USES_HIP
  hipCheck(hipDeviceSynchronize());
#elif LIBRETT_USES_CUDA
  cudaCheck(cudaDeviceSynchronize());
#endif

  for (int i=0;i < numStream;i++) {
    librettCheck(librettDestroy(plans[i]));

#if LIBRETT_USES_SYCL
    delete streams[i];
#elif LIBRETT_USES_HIP
    hipCheck(hipStreamDestroy(streams[i]));
#elif LIBRETT_USES_CUDA
    cudaCheck(cudaStreamDestroy(streams[i]));
#endif
  }

  return run_ok;
}

//
// Test 5: Transaction and cache line counters
//
bool test5() {

  {
    // Number of elements that are loaded per memory transaction:
    // 128 bytes per transaction
    const  int accWidth = 128/sizeof(double);
    // L2 cache line width is 32 bytes
    const int cacheWidth = 32/sizeof(double);
    if (!testCounters(32, accWidth, cacheWidth)) return false;
  }

  {
    // Number of elements that are loaded per memory transaction:
    // 128 bytes per transaction
    const  int accWidth = 128/sizeof(float);
    // L2 cache line width is 32 bytes
    const int cacheWidth = 32/sizeof(float);
    if (!testCounters(32, accWidth, cacheWidth)) return false;
  }

  return true;
}

template <typename T>
bool test_tensor(std::vector<int> &dim, std::vector<int> &permutation, gpuStream_t& gpustream)
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
    printf("#ERROR(test_tensor): Data size exceeded: %zu %zu\n",volmem,datamem);
    return false;
  }

  librettTimer* timer;
  if (sizeof(T) == 4) {
    timer = timerFloat;
  } else {
    timer = timerDouble;
  }

  librettHandle plan;
  librettCheck(librettPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), gpustream));
  set_device_array<T>((T *)dataOut, -1, vol, gpustream);

#if LIBRETT_USES_SYCL
  gpustream->wait_and_throw();
#elif LIBRETT_USES_HIP
  hipCheck(hipDeviceSynchronize());
#elif LIBRETT_USES_CUDA
  cudaCheck(cudaDeviceSynchronize());
#endif

  if (vol > 1000000) timer->start(dim, permutation);
  librettCheck(librettExecute(plan, dataIn, dataOut));
  if (vol > 1000000) timer->stop();
  //q.wait();

  librettCheck(librettDestroy(plan));

  return tester->checkTranspose<T>(rank, dim.data(), permutation.data(), (T *)dataOut);
}

void printVec(std::vector<int>& vec) {
  for (int i=0;i < vec.size();i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}
