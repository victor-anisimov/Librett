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
#endif
#include <vector>
#include <algorithm>
#include <cstring>         // strcmp
#include <ctime>           // std::time
#include <chrono>
#include <cmath>
#include <cctype>
#include <random>
#include "librett.h"
#include "Utils.h"
#include "Mem.h"
#include "TensorTester.h"
#include "Timer.h"
#include "Memcpy.h"
#include "int_vector.h"

#define MILLION 1000000
#define BILLION 1000000000

//
// Error checking wrapper for librett
//
#define librettCheck(stmt) do {                                 \
  librettResult err = stmt;                            \
  if (err != LIBRETT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

char* dataIn  = NULL;
char* dataOut = NULL;
size_t dataSize = 0;
TensorTester* tester = NULL;

librettTimer* timer;
bool use_librettPlanMeasure;
bool use_plantimer;

std::default_random_engine generator;

bool bench1(int numElem);
bool bench2(int numElem);
bool bench3(int numElem);
template <typename T> bool bench5(int numElem, int ratio);
bool bench6();
template <typename T> bool bench7();
template <typename T> bool bench_input(std::vector<int>& dim, std::vector<int>& permutation);
template <typename T> bool bench_memcpy(int numElem);

bool isTrivial(std::vector<int>& permutation);
void getRandomDim(double vol, std::vector<int>& dim);
template <typename T> bool bench_tensor(std::vector<int>& dim, std::vector<int>& permutation);
void printVec(std::vector<int>& vec);
void printDeviceInfo();

int main(int argc, char *argv[]) 
#ifdef SYCL
try 
#endif
{

  int gpuid = -1;
  unsigned seed = unsigned (std::time(0));
  bool arg_ok = true;
  int benchID = 0;
  use_librettPlanMeasure = false;
  use_plantimer = false;
  int elemsize = 8;
  std::vector<int> dimIn;
  std::vector<int> permutationIn;
  if (argc >= 2) {
    int i = 1;
    while (i < argc) {
      if (strcmp(argv[i], "-device") == 0) {
        sscanf(argv[i+1], "%d", &gpuid);
        i += 2;
      } else if (strcmp(argv[i], "-bench") == 0) {
        sscanf(argv[i+1], "%d", &benchID);
        i += 2;
      } else if (strcmp(argv[i], "-measure") == 0) {
        use_librettPlanMeasure = true;
        i++;
      } else if (strcmp(argv[i], "-seed") == 0) {
        sscanf(argv[i+1], "%u", &seed);
        i += 2;
      } else if (strcmp(argv[i], "-plantimer") == 0) {
        use_plantimer = true;
        i++;
      } else if (strcmp(argv[i], "-elemsize") == 0) {
        sscanf(argv[i+1], "%u", &elemsize);
        i += 2;
      } else if (strcmp(argv[i], "-dim") == 0) {
        i++;
        while (i < argc && isdigit(*argv[i])) {
          int val;
          sscanf(argv[i++], "%d", &val);
          dimIn.push_back(val);
        }
      } else if (strcmp(argv[i], "-permutation") == 0) {
        i++;
        while (i < argc && isdigit(*argv[i])) {
          int val;
          sscanf(argv[i++], "%d", &val);
          permutationIn.push_back(val);
        }
      } else {
        arg_ok = false;
        break;
      }
    }
  } else if (argc > 1) {
    arg_ok = false;
  }

  if (elemsize != 4 && elemsize != 8) {
    arg_ok = false;
  }

  if (!arg_ok) {
    printf("librett_bench [options]\n");
    printf("Options:\n");
    printf("-device [int]    : GPU ID (default is 0)\n");
    printf("-measure         : use librettPlanMeasure (default is librettPlan)\n");
    printf("-plantimer       : planning is timed (default is no)\n");
    printf("-seed [int]      : seed value for random number generator (default is system timer)\n");
    printf("-elemsize [int]  : size of elements in bytes, 4 or 8. (default is 8)\n");
    printf("-dim ...         : space-separated list of dimensions\n");
    printf("-permutation ... : space-separated list of permutations\n");
    printf("-bench benchID   : benchmark to run\n");
    return 1;
  }

#ifdef SYCL
  if (gpuid >= 0) {
    gpuCheck((dpct::dev_mgr::instance().select_device(gpuid), 0));
  }

  gpuCheck((dpct::get_current_device().reset(), 0));
  /*
  DPCT1027:7: The call to cudaDeviceSetSharedMemConfig was replaced with 0,
  because DPC++ currently does not support configuring shared memory on
  devices.
  */
#else // CUDA
  if (gpuid >= 0) {
    gpuCheck(cudaSetDevice(gpuid));
  }

  gpuCheck(cudaDeviceReset());
  if (elemsize == 4) {
    gpuCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
  } else {
    gpuCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  }
#endif

  printDeviceInfo();
  printf("CPU using vector type %s of length %d\n", INT_VECTOR_TYPE, INT_VECTOR_LEN);

  timer = new librettTimer(elemsize);

  //dataSize = (elemsize == 4) ? 420*MILLION : 370*MILLION;
  dataSize = (elemsize == 4) ? 420*MILLION : 530*MILLION;

  // Allocate device data, 100M elements
  allocate_device<char>(&dataIn, dataSize*(size_t)elemsize);
  allocate_device<char>(&dataOut, dataSize*(size_t)elemsize);

  // Create tester
  tester = new TensorTester();
  tester->setTensorCheckPattern((unsigned int *)dataIn, dataSize*(size_t)elemsize/sizeof(unsigned int));

  std::vector<int> worstDim;
  std::vector<int> worstPermutation;

  std::srand(seed);
  generator.seed(seed);

  // if (!bench1(40*MILLION, bandwidths)) goto fail;
  // printf("bench1:\n");
  // for (int i=0;i < bandwidths.size();i++) {
  //   printf("%lf\n", bandwidths[i]);
  // }

  // if (!bench2(40*MILLION, bandwidths)) goto fail;
  // printf("bench2:\n");
  // for (int i=0;i < bandwidths.size();i++) {
  //   printf("%lf\n", bandwidths[i]);
  // }

  if (dimIn.size() > 0) {
    bool ok = (elemsize == 4) ? bench_input<int>(dimIn, permutationIn) : bench_input<long long int>(dimIn, permutationIn);
    if (ok) goto benchOK;
    goto fail;
  }

  if (benchID == 3) {
    if (elemsize == 4) {
      printf("bench 3 not implemented for elemsize = 4\n");
      goto fail;
    }
    if (bench3(200*MILLION)) {
      printf("bench3:\n");
      printf("rank best worst average median\n");
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        double worstBW = timer->getWorst(*it);
        double bestBW = timer->getBest(*it);
        double aveBW = timer->getAverage(*it);
        double medBW = timer->getMedian(*it);
        printf("%d %6.2lf %6.2lf %6.2lf %6.2lf\n", *it, bestBW, worstBW, aveBW, medBW);
      }
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        std::vector<int> dim;
        std::vector<int> permutation;
        double worstBW = timer->getWorst(*it, dim, permutation);
        printf("rank %d BW %4.2lf\n", *it, worstBW);
        printf("dimensions\n");
        printVec(dim);
        printf("permutation\n");
        printVec(permutation);
      }
      goto benchOK;
    } else {
      goto fail;
    }
  }

  if (benchID/100 == 5) {
    bool ok = (elemsize == 4) ? bench5<int>(200*MILLION, benchID % 100) : bench5<long long int>(200*MILLION, benchID % 100);
    if (ok) {
      printf("bench5:\n");
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        std::vector<double> v = timer->getData(*it);
        printf("RANK%d", *it);
        for (int i=0;i < v.size();i++) {
          printf(" %1.2lf", v[i]);
        }
        printf("\n");
      }
      printf("rank best worst average median\n");
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        double worstBW = timer->getWorst(*it);
        double bestBW = timer->getBest(*it);
        double aveBW = timer->getAverage(*it);
        double medBW = timer->getMedian(*it);
        printf("%d %6.2lf %6.2lf %6.2lf %6.2lf\n", *it, bestBW, worstBW, aveBW, medBW);
      }
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        std::vector<int> dim;
        std::vector<int> permutation;
        double worstBW = timer->getWorst(*it, dim, permutation);
        printf("rank %d BW %4.2lf\n", *it, worstBW);
        printf("dimensions\n");
        printVec(dim);
        printf("permutation\n");
        printVec(permutation);
      }
      goto benchOK;
    } else {
      goto fail;
    }
  }

  if (benchID == 6) {
    if (elemsize == 4) {
      printf("bench 6 not implemented for elemsize = 4\n");
      goto fail;
    }
    if (bench6()) {
      printf("bench6:\n");
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        std::vector<double> v = timer->getData(*it);
        printf("RANK%d", *it);
        for (int i=0;i < v.size();i++) {
          printf(" %1.2lf", v[i]);
        }
        printf("\n");
      }
      printf("rank best worst average\n");
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        double worstBW = timer->getWorst(*it);
        double bestBW = timer->getBest(*it);
        double aveBW = timer->getAverage(*it);
        printf("%d %6.2lf %6.2lf %6.2lf\n", *it, bestBW, worstBW, aveBW);
      }
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        std::vector<int> dim;
        std::vector<int> permutation;
        double worstBW = timer->getWorst(*it, dim, permutation);
        printf("rank %d BW %4.2lf\n", *it, worstBW);
        printf("dimensions\n");
        printVec(dim);
        printf("permutation\n");
        printVec(permutation);
      }
      goto benchOK;
    } else {
      goto fail;
    }
  }

  if (benchID == 7) {
    bool ok = (elemsize == 4) ? bench7<int>() : bench7<long long int>();
    if (ok) {
      printf("bench7:\n");
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        std::vector<double> v = timer->getData(*it);
        printf("RANK%d", *it);
        for (int i=0;i < v.size();i++) {
          printf(" %1.2lf", v[i]);
        }
        printf("\n");
      }
      printf("rank best worst average median\n");
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        double worstBW = timer->getWorst(*it);
        double bestBW = timer->getBest(*it);
        double aveBW = timer->getAverage(*it);
        double medBW = timer->getMedian(*it);
        printf("%d %6.2lf %6.2lf %6.2lf %6.2lf\n", *it, bestBW, worstBW, aveBW, medBW);
      }
      for (auto it=timer->ranksBegin();it != timer->ranksEnd();it++) {
        std::vector<int> dim;
        std::vector<int> permutation;
        double worstBW = timer->getWorst(*it, dim, permutation);
        printf("rank %d BW %4.2lf\n", *it, worstBW);
        printf("dimensions\n");
        printVec(dim);
        printf("permutation\n");
        printVec(permutation);
      }
      goto benchOK;
    } else {
      goto fail;
    }
  }

  // Otherwise, do memcopy benchmark
  {
    bool ok = (elemsize == 4) ? bench_memcpy<int>(benchID) : bench_memcpy<long long int>(benchID);
    if (ok) goto benchOK;
    goto fail;
  }

benchOK:
  printf("bench OK\n");

  goto end;
fail:
  printf("bench FAIL\n");
end:
  deallocate_device<char>(&dataIn);
  deallocate_device<char>(&dataOut);
  delete tester;

  printf("seed %u\n", seed);

  delete timer;

#ifdef SYCL
  gpuCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
  gpuCheck((dpct::get_current_device().reset(), 0));
#else // CUDA
  gpuCheck(cudaDeviceSynchronize());
  gpuCheck(cudaDeviceReset());
#endif

  return 0;
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

//
// Benchmark 1: ranks 2-8,15 in inverse permutation. 32 start and end dimension
//
bool bench1(int numElem) {
  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};
  for (int i=0;i <= 7;i++) {
    std::vector<int> dim(ranks[i]);
    std::vector<int> permutation(ranks[i]);
    //int dimave = (int)pow<double>(numElem, 1.0 / (double)ranks[i]);
    int dimave = (int)pow(numElem, 1.0 / (double)ranks[i]);

    if (dimave < 100.0) {
      dim[0]            = 32;
      dim[ranks[i] - 1] = 32;
    } else {
      dim[0]            = dimave;
      dim[ranks[i] - 1] = dimave;
    }
    // Distribute remaining volume to the middle ranks
    int ranks_left = ranks[i] - 2;
    double numElem_left = numElem/(double)(dim[0]*dim[ranks[i] - 1]);
    for (int r=1;r < ranks[i] - 1;r++) {
      dim[r] = (int)pow(numElem_left, 1.0/(double)ranks_left);
      numElem_left /= (double)dim[r];
      ranks_left--;
    }

    // Inverse order
    for (int r=0;r < ranks[i];r++) {
      permutation[r] = ranks[i] - 1 - r;
    }

    if (!bench_tensor<long long int>(dim, permutation)) return false;
  }

  return true;
}

//
// Benchmark 2: ranks 2-8,15 in inverse permutation. Even spread of dimensions.
//
bool bench2(int numElem) {
  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};
  for (int i=0;i <= 7;i++) {
    std::vector<int> dim(ranks[i]);
    std::vector<int> permutation(ranks[i]);
    //int dimave = (int)pow<double>(numElem, 1.0 / (double)ranks[i]);
    int dimave = (int)pow(numElem, 1.0 / (double)ranks[i]);

    double numElem_left = numElem;
    for (int r=0;r < ranks[i];r++) {
      dim[r] = (int)pow(numElem_left, 1.0/(double)(ranks[i] - r));
      numElem_left /= (double)dim[r];
    }

    // Inverse order
    for (int r=0;r < ranks[i];r++) {
      permutation[r] = ranks[i] - 1 - r;
    }

    if (!bench_tensor<long long int>(dim, permutation)) return false;
  }

  return true;
}

//
// Benchmark 3: ranks 2-8,15 in random permutation and dimensions.
//
bool bench3(int numElem) {

  int ranks[8] = {2, 3, 4, 5, 6, 7, 8, 15};

  for (int i=0;i <= 7;i++) {
    std::vector<int> dim(ranks[i]);
    std::vector<int> permutation(ranks[i]);
    for (int r=0;r < ranks[i];r++) permutation[r] = r;
    for (int nsample=0;nsample < 50;nsample++) {
      std::random_shuffle(permutation.begin(), permutation.end());
      getRandomDim((double)numElem, dim);
      if (!bench_tensor<long long int>(dim, permutation)) return false;
    }
  }

  return true;
}

template <typename T>
bool bench_input(std::vector<int>& dim, std::vector<int>& permutation) {
  if (!bench_tensor<T>(dim, permutation)) return false;
  printf("dimensions\n");
  printVec(dim);
  printf("permutation\n");
  printVec(permutation);
  printf("bandwidth %4.2lf GB/s\n", timer->GBs());
  return true;  
}

//
// Benchmark 5: All permutations for ranks 2-4, limited permutations for ranks 5-7
//
template <typename T>
bool bench5(int numElemAvg, int ratio) {

  std::normal_distribution<double> numElem_dist((double)numElemAvg, (double)numElemAvg*0.2);

  const int minDim = 2;
  const int maxDim = 16;
  for (int rank=2;rank <= 7;rank++) {

    for (int iter=0;iter < 500;iter++) {

      int numElem = (int)numElem_dist(generator);

      std::vector<int> dim(rank);
      std::vector<int> permutation(rank);
      std::vector<double> dimf(rank);
      double volf = 1.0;
      for (int r=0;r < rank;r++) {
        permutation[r] = r;
        dimf[r] = 1.0 + (double)r*(ratio - 1.0)/(double)(rank - 1);
        volf *= dimf[r];
      }
      // fprintf(stderr, "volf %lf\n", volf);
      double scale = pow((double)numElem/volf, 1.0/(double)rank);
      // fprintf(stderr, "scale %lf\n", scale);
      int vol = 1;
      for (int r=0;r < rank;r++) {
        if (r == rank - 1) {
          dim[r] = ratio*dim[0];
        } else {
          dim[r] = (int)round(dimf[r]*scale);
        }
        dim[r] = std::max(2, dim[r]);
        vol *= dim[r];
      }
      // fprintf(stderr, "dim[0] %lf\n", dim[0]);
      double cur_ratio = (double)dim[rank-1]/(double)dim[0];
      double vol_re = fabs((double)(vol - numElem)/(double)numElem);
      // fprintf(stderr, "cur_ratio %lf vol_re %lf\n", cur_ratio, vol_re);
      // Fix dimensions if volume is off by more than 5%
      if (vol_re > 0.05) {
        int d = (vol < numElem) ? 1 : -1;
        int r = 1;
        while (vol_re > 0.05 && r < rank) {
          int dim_plus_d = std::max(2, dim[r] + d);
          // fprintf(stderr, "r %d vol %lf dim[r] %d dim_plus_d %d\n", vol, dim[r], dim_plus_d);
          vol = (vol/dim[r])*dim_plus_d;
          dim[r] = dim_plus_d;
          vol_re = fabs((double)(vol - numElem)/(double)numElem);
          r++;
        }
      }
      int minDim = *(std::min_element(dim.begin(), dim.end()));
      int maxDim = *(std::max_element(dim.begin(), dim.end()));
      // fprintf(stderr, "minDim %lf maxDim\n", minDim, maxDim);
      cur_ratio = (double)maxDim/(double)minDim;
      printf("vol %d cur_ratio %lf | %lf\n", vol, cur_ratio, vol_re);
      printVec(dim);

      std::random_shuffle(dim.begin(), dim.end());
      while (isTrivial(permutation)) {
        std::random_shuffle(permutation.begin(), permutation.end());
      }
      if (!bench_tensor<T>(dim, permutation)) return false;
    }
  }

  return true;
}

//
// Benchmark 6: from "TTC: A Tensor Transposition Compiler for Multiple Architectures"
//
bool bench6() {

  std::vector< std::vector<int> > dims = {
    std::vector<int>{7248,7248},
    std::vector<int>{43408,1216},
    std::vector<int>{1216,43408},
    std::vector<int>{368,384,384},
    std::vector<int>{2144,64,384},
    std::vector<int>{368,64,2307},
    std::vector<int>{384,384,355},
    std::vector<int>{2320,384,59},
    std::vector<int>{384,2320,59},
    std::vector<int>{384,355,384},
    std::vector<int>{2320,59,384},
    std::vector<int>{384,59,2320},
    std::vector<int>{80,96,75,96},
    std::vector<int>{464,16,75,96},
    std::vector<int>{80,16,75,582},
    std::vector<int>{96,75,96,75},
    std::vector<int>{608,12,96,75},
    std::vector<int>{96,12,608,75},
    std::vector<int>{96,75,96,75},
    std::vector<int>{608,12,96,75},
    std::vector<int>{96,12,608,75},
    std::vector<int>{96,96,75,75},
    std::vector<int>{608,96,12,75},
    std::vector<int>{96,608,12,75},
    std::vector<int>{96,75,75,96},
    std::vector<int>{608,12,75,96},
    std::vector<int>{96,12,75,608},
    std::vector<int>{32,48,28,28,48},
    std::vector<int>{176,8,28,28,48},
    std::vector<int>{32,8,28,28,298},
    std::vector<int>{48,28,28,48,28},
    std::vector<int>{352,4,28,48,28},
    std::vector<int>{48,4,28,352,28},
    std::vector<int>{48,28,48,28,28},
    std::vector<int>{352,4,48,28,28},
    std::vector<int>{48,4,352,28,28},
    std::vector<int>{48,48,28,28,28},
    std::vector<int>{352,48,4,28,28},
    std::vector<int>{48,352,4,28,28},
    std::vector<int>{48,28,28,28,48},
    std::vector<int>{352,4,28,28,48},
    std::vector<int>{48,4,28,28,352},
    std::vector<int>{16,32,15,32,15,15},
    std::vector<int>{48,10,15,32,15,15},
    std::vector<int>{16,10,15,103,15,15},
    std::vector<int>{32,15,15,32,15,15},
    std::vector<int>{112,5,15,32,15,15},
    std::vector<int>{32,5,15,112,15,15},
    std::vector<int>{32,15,32,15,15,15},
    std::vector<int>{112,5,32,15,15,15},
    std::vector<int>{32,5,112,15,15,15},
    std::vector<int>{32,15,15,32,15,15},
    std::vector<int>{112,5,15,32,15,15},
    std::vector<int>{32,5,15,112,15,15},
    std::vector<int>{32,15,15,15,15,32},
    std::vector<int>{112,5,15,15,15,32},
    std::vector<int>{32,5,15,15,15,112}
  };

  std::vector< std::vector<int> > permutations = {
    std::vector<int>{1,0},
    std::vector<int>{1,0},
    std::vector<int>{1,0},
    std::vector<int>{0,2,1},
    std::vector<int>{0,2,1},
    std::vector<int>{0,2,1},
    std::vector<int>{1,0,2},
    std::vector<int>{1,0,2},
    std::vector<int>{1,0,2},
    std::vector<int>{2,1,0},
    std::vector<int>{2,1,0},
    std::vector<int>{2,1,0},
    std::vector<int>{0,3,2,1},
    std::vector<int>{0,3,2,1},
    std::vector<int>{0,3,2,1},
    std::vector<int>{2,1,3,0},
    std::vector<int>{2,1,3,0},
    std::vector<int>{2,1,3,0},
    std::vector<int>{2,0,3,1},
    std::vector<int>{2,0,3,1},
    std::vector<int>{2,0,3,1},
    std::vector<int>{1,0,3,2},
    std::vector<int>{1,0,3,2},
    std::vector<int>{1,0,3,2},
    std::vector<int>{3,2,1,0},
    std::vector<int>{3,2,1,0},
    std::vector<int>{3,2,1,0},
    std::vector<int>{0,4,2,1,3},
    std::vector<int>{0,4,2,1,3},
    std::vector<int>{0,4,2,1,3},
    std::vector<int>{3,2,1,4,0},
    std::vector<int>{3,2,1,4,0},
    std::vector<int>{3,2,1,4,0},
    std::vector<int>{2,0,4,1,3},
    std::vector<int>{2,0,4,1,3},
    std::vector<int>{2,0,4,1,3},
    std::vector<int>{1,3,0,4,2},
    std::vector<int>{1,3,0,4,2},
    std::vector<int>{1,3,0,4,2},
    std::vector<int>{4,3,2,1,0},
    std::vector<int>{4,3,2,1,0},
    std::vector<int>{4,3,2,1,0},
    std::vector<int>{0,3,2,5,4,1},
    std::vector<int>{0,3,2,5,4,1},
    std::vector<int>{0,3,2,5,4,1},
    std::vector<int>{3,2,0,5,1,4},
    std::vector<int>{3,2,0,5,1,4},
    std::vector<int>{3,2,0,5,1,4},
    std::vector<int>{2,0,4,1,5,3},
    std::vector<int>{2,0,4,1,5,3},
    std::vector<int>{2,0,4,1,5,3},
    std::vector<int>{3,2,5,1,0,4},
    std::vector<int>{3,2,5,1,0,4},
    std::vector<int>{3,2,5,1,0,4},
    std::vector<int>{5,4,3,2,1,0},
    std::vector<int>{5,4,3,2,1,0},
    std::vector<int>{5,4,3,2,1,0}
  };

  for (int i=0;i < dims.size();i++) {
    if (!bench_tensor<long long int>(dims[i], permutations[i])) return false;
    printf("dimensions\n");
    printVec(dims[i]);
    printf("permutation\n");
    printVec(permutations[i]);
    printf("bandwidth %4.2lf GiB/s\n", timer->GiBs());
  }

  return true;
}

//
// Benchmark 7: ranks 8 and 12 with 4 large dimensions and rest small dimensions
//
template <typename T>
bool bench7() {

  // 199584000 elements
  {
    std::vector<int> dim = {5, 3, 2, 4, 35, 33, 37, 40};
    std::vector<int> permutation(8);
    // Inverse
    for (int r=0;r < dim.size();r++) permutation[r] = dim.size() - 1 - r;
    if (!bench_tensor<T>(dim, permutation)) return false;
    // Random
    for (int r=0;r < dim.size();r++) permutation[r] = r;
    for (int nsample=0;nsample < 500;nsample++) {
      std::random_shuffle(dim.begin(), dim.end());
      std::random_shuffle(permutation.begin(), permutation.end());
      if (!isTrivial(permutation)) {
        if (!bench_tensor<T>(dim, permutation)) return false;
      }
    }
  }

  // 328458240 elements
  {
    std::vector<int> dim = {2, 3, 4, 3, 2, 2, 3, 2, 20, 18, 22, 24};
    std::vector<int> permutation(12);
    // Inverse
    for (int r=0;r < dim.size();r++) permutation[r] = dim.size() - 1 - r;
    if (!bench_tensor<T>(dim, permutation)) return false;
    // Random
    for (int r=0;r < dim.size();r++) permutation[r] = r;
    for (int nsample=0;nsample < 500;nsample++) {
      std::random_shuffle(dim.begin(), dim.end());
      std::random_shuffle(permutation.begin(), permutation.end());
      if (!isTrivial(permutation)) {
        if (!bench_tensor<T>(dim, permutation)) return false;
      }
    }
  }

  return true;
}

//
// Returns true for trivial permutation
//
bool isTrivial(std::vector<int>& permutation) {
  for (int i=0;i < permutation.size();i++) {
    if (permutation[i] != i) return false;
  }
  return true;
}

//
// Get random dimensions for a fixed volume tensor
//
void getRandomDim(double vol, std::vector<int>& dim) {
  double dimave = floor(pow(vol, 1.0/(double)dim.size()));
  double curvol = 1.0;
  int iter = 0;
  do {
    curvol = 1.0;
    for (int r=0;r < dim.size();r++) {
      // p is -1 ... 1
      double p = (((double)rand()/(double)RAND_MAX) - 0.5)*2.0;
      dim[r] = round(dimave + p*(dimave - 2.0));
      curvol *= (double)dim[r];
    }

    double vol_scale = pow(vol/curvol, 1.0/(double)dim.size());
    curvol = 1.0;
    for (int r=0;r < dim.size();r++) {
      dim[r] = std::max(2, (int)(dim[r]*vol_scale));
      curvol *= dim[r];
    }
    // printf("curvol %lf\n", curvol/MILLION);
    iter++;
  } while (iter < 5000 && (fabs(curvol-vol)/(double)vol > 0.3));

  if (iter == 5000) {
    printf("getRandomDim: Unable to determine dimensions in 5000 iterations\n");
    exit(1);
  }
}

template <typename T>
bool bench_tensor(std::vector<int> &dim, std::vector<int> &permutation) 
#ifdef SYCL
try 
#endif
{
#ifdef SYCL
sycl::queue q = dpct::get_default_queue();
#endif

  int rank = dim.size();

  int vol = 1;
  for (int r=0;r < rank;r++) {
    vol *= dim[r];
  }

  size_t volmem = vol*sizeof(T);
  size_t datamem = dataSize*sizeof(long long int);
  if (volmem > datamem) {
    printf("test_tensor, data size exceeded\n");
    return false;
  }

  std::vector<int> dimp(rank);
  for (int r=0;r < rank;r++) {
    dimp[r] = dim[permutation[r]];
  }

  printf("number of elements %d\n", vol);
  printf("dimensions\n");
  printVec(dim);
  printVec(dimp);
  printf("permutation\n");
  printVec(permutation);

  librettHandle plan;
  std::chrono::high_resolution_clock::time_point plan_start;
  if (use_plantimer) {
    plan_start = std::chrono::high_resolution_clock::now();
  }
  if (use_librettPlanMeasure) {
#ifdef SYCL
    librettCheck(librettPlanMeasure(&plan, rank, dim.data(), permutation.data(), sizeof(T), &q, dataIn, dataOut));
#else // CUDA
    librettCheck(librettPlanMeasure(&plan, rank, dim.data(), permutation.data(), sizeof(T), 0, dataIn, dataOut));
#endif
  } else {
#ifdef SYCL
    librettCheck(librettPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), &q));
#else // CUDA
    librettCheck(librettPlan(&plan, rank, dim.data(), permutation.data(), sizeof(T), 0));
#endif
  }
  if (use_plantimer) {
    std::chrono::high_resolution_clock::time_point plan_end;
    plan_end = std::chrono::high_resolution_clock::now();
    double plan_duration = std::chrono::duration_cast< std::chrono::duration<double> >(plan_end - plan_start).count();
    printf("plan took %lf ms\n", plan_duration*1000.0);
  }

  for (int i=0;i < 4;i++) {
#ifdef SYCL
    set_device_array<T>((T *)dataOut, -1, vol, &q);
    gpuCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
#else // CUDA
    set_device_array<T>((T *)dataOut, -1, vol);
    gpuCheck(cudaDeviceSynchronize());
#endif

    timer->start(dim, permutation);
    librettCheck(librettExecute(plan, dataIn, dataOut));
    timer->stop();

    printf("wall time %lf ms %lf GB/s\n", timer->seconds()*1000.0, timer->GBs());
  }

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

//
// Benchmarks memory copy. Returns bandwidth in GB/s
//
template <typename T> bool bench_memcpy(int numElem) 
#ifdef SYCL
try 
#endif
{
#ifdef SYCL
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue q = dpct::get_default_queue();
#endif

  std::vector<int> dim(1, numElem);
  std::vector<int> permutation(1, 0);

  {
    librettTimer timer(sizeof(T));
    for (int i=0;i < 4;i++) {
#ifdef SYCL
      set_device_array<T>((T *)dataOut, -1, numElem, &q);
      gpuCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
      timer.start(dim, permutation);
      scalarCopy<T>(numElem, (T *)dataIn, (T *)dataOut, &q);
#else // CUDA
      set_device_array<T>((T *)dataOut, -1, numElem);
      gpuCheck(cudaDeviceSynchronize());
      timer.start(dim, permutation);
      scalarCopy<T>(numElem, (T *)dataIn, (T *)dataOut, 0);
#endif
      timer.stop();
      printf("%4.2lf GB/s\n", timer.GBs());
    }
    if (!tester->checkTranspose<T>(1, dim.data(), permutation.data(), (T *)dataOut)) return false;
    printf("scalarCopy %lf GB/s\n", timer.getAverage(1));
  }

  {
    librettTimer timer(sizeof(T));
    for (int i=0;i < 4;i++) {
#ifdef SYCL
      set_device_array<T>((T *)dataOut, -1, numElem, &q);
      gpuCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
      timer.start(dim, permutation);
      vectorCopy<T>(numElem, (T *)dataIn, (T *)dataOut, &q);
#else // CUDA
      set_device_array<T>((T *)dataOut, -1, numElem);
      gpuCheck(cudaDeviceSynchronize());
      timer.start(dim, permutation);
      vectorCopy<T>(numElem, (T *)dataIn, (T *)dataOut, 0);
#endif
      timer.stop();
      printf("%4.2lf GB/s\n", timer.GBs());
    }
    if (!tester->checkTranspose<T>(1, dim.data(), permutation.data(), (T *)dataOut)) return false;
    printf("vectorCopy %lf GB/s\n", timer.getAverage(1));
  }

  {
    librettTimer timer(sizeof(T));
    for (int i=0;i < 4;i++) {
#ifdef SYCL
      set_device_array<T>((T *)dataOut, -1, numElem, &q);
      gpuCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
      timer.start(dim, permutation);
      memcpyFloat(numElem*sizeof(T)/sizeof(float), (float *)dataIn, (float *)dataOut, &q);
#else // CUDA
      set_device_array<T>((T *)dataOut, -1, numElem);
      gpuCheck(cudaDeviceSynchronize());
      timer.start(dim, permutation);
      memcpyFloat(numElem*sizeof(T)/sizeof(float), (float *)dataIn, (float *)dataOut, 0);
#endif
      timer.stop();
      printf("%4.2lf GB/s\n", timer.GBs());
    }
    if (!tester->checkTranspose<T>(1, dim.data(), permutation.data(), (T *)dataOut)) return false;
    printf("memcpyFloat %lf GB/s\n", timer.getAverage(1));
  }

  return true;
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

#ifdef SYCL
void printDeviceInfo() try {
  int deviceID;
  gpuCheck(deviceID = dpct::dev_mgr::instance().current_device_id());
  dpct::device_info prop;
  gpuCheck(( dpct::dev_mgr::instance().get_device(deviceID).get_device_info(prop), 0));
  int pConfig = 0;
  /*
  DPCT1007:12: Migration of this CUDA API is not supported by the Intel(R) DPC++
  Compatibility Tool.
  */
  //gpuCheck(cudaDeviceGetSharedMemConfig(&pConfig));
  int shMemBankSize = 4;
  if (pConfig == 2) shMemBankSize = 8;
  //double mem_BW = (double)(prop.memoryClockRate*2*(prop.memoryBusWidth/8))/1.0e6;
  double mem_BW = 0.0;
  /*
  DPCT1005:13: The device version is different. You need to rewrite this code.
  */
  printf("Using %s SM version %d.%d\n", prop.get_name(),
         prop.get_major_version(), prop.get_minor_version());
  int ECCEnabled = 0;
  double l2CacheSize = 0.0;
  printf(
      "Clock %1.3lfGhz numSM %d ECC %d mem BW %1.2lfGB/s shMemBankSize %dB\n",
      (double)prop.get_max_clock_frequency() / 1e6,
      prop.get_max_compute_units(), ECCEnabled, mem_BW, shMemBankSize);
  printf("L2 %1.2lfMB\n", (double)l2CacheSize/(double)(1024*1024));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#else // CUDA
void printDeviceInfo() {
  int deviceID;
  gpuCheck(cudaGetDevice(&deviceID));
  cudaDeviceProp prop;
  gpuCheck(cudaGetDeviceProperties(&prop, deviceID));
  cudaSharedMemConfig pConfig;
  gpuCheck(cudaDeviceGetSharedMemConfig(&pConfig));
  int shMemBankSize = 4;
  if (pConfig == cudaSharedMemBankSizeEightByte) shMemBankSize = 8;
  double mem_BW = (double)(prop.memoryClockRate*2*(prop.memoryBusWidth/8))/1.0e6;
  printf("Using %s SM version %d.%d\n", prop.name, prop.major, prop.minor);
  printf("Clock %1.3lfGhz numSM %d ECC %d mem BW %1.2lfGB/s shMemBankSize %dB\n", (double)prop.clockRate/1e6,
    prop.multiProcessorCount, prop.ECCEnabled, mem_BW, shMemBankSize);
  printf("L2 %1.2lfMB\n", (double)prop.l2CacheSize/(double)(1024*1024));

}
#endif
