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
#include <CL/sycl.hpp>
#include "Utils.h"
#include "LRUCache.h"
#include "kernel.h"

#define RESTRICT __restrict__

// suppress Clang warning about it being unable to unroll a loop
#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wpass-failed"
#endif

//
// Transpose when Mm and Mk don't overlap and contain only single rank
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMk-1)/TILEDIM+1), 1, plan.volMbar);
//
template <typename T>
void transposeTiled(const int numMm, const int volMbar, const int sizeMbar,
                    const sycl::int2 tiledVol, const int cuDimMk,
                    const int cuDimMm, const TensorConvInOut *RESTRICT glMbar,
                    const T *RESTRICT dataIn, T *RESTRICT dataOut,
                    sycl::nd_item<3>& item,
                    localAcc<T, 2> shTile) {

  sycl::group work_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();

  // Shared memory

  const int warpLane = item.get_local_id(2) &
                       (sg.get_local_range().get(0) - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  const int bx = (item.get_group(2) % numMm) * TILEDIM;
  const int by = (item.get_group(2) / numMm) * TILEDIM;

  const int xin = bx + item.get_local_id(2);
  const int yin = by + item.get_local_id(1);

  const int xout = bx + item.get_local_id(1);
  const int yout = by + item.get_local_id(2);

  const unsigned int maskIny =
      ballot(sg, (yin + warpLane < tiledVol.y()))[0] * (xin < tiledVol.x());
  const unsigned int maskOutx =
      ballot(sg, (xout + warpLane < tiledVol.x()))[0] * (yout < tiledVol.y());

  const int posMinorIn = xin + yin*cuDimMk;
  const int posMinorOut = yout + xout*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int posMbar = item.get_group(0); posMbar < volMbar;
       posMbar += item.get_group_range(0))
  {

    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMajorIn += sg.shuffle_xor(posMajorIn, i);
      posMajorOut += sg.shuffle_xor(posMajorOut, i);
    }
    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

    // Read from global memory
    sycl::group_barrier(work_grp);

    // Read data into shared memory tile
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // int pos = posIn + j*cuDimMk;
      // if (xin < readVol.x && yin + j < readVol.y) {
      if ((maskIny & (1 << j)) != 0) {
        shTile[item.get_local_id(1) + j][item.get_local_id(2)] =
            dataIn[posIn];
      }
      posIn += posInAdd;
    }

    // Write to global memory
    sycl::group_barrier(work_grp);

#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // int pos = posOut + j*cuDimMm;
      // if (xout + j < readVol.x && yout < readVol.y) {
      if ((maskOutx & (1 << j)) != 0 ) {
        dataOut[posOut] =
            shTile[item.get_local_id(2)][item.get_local_id(1) + j];
      }
      posOut += posOutAdd;
    }

  }

}

//
// Packed transpose. Thread block loads plan.volMmk number of elements
//
template <typename T, int numRegStorage>
void transposePacked(
  const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* RESTRICT gl_Mmk,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const TensorConv* RESTRICT gl_Msh,
  const T* RESTRICT dataIn, T* RESTRICT dataOut, sycl::nd_item<3>& item,
  uint8_t *dpct_local) {

  sycl::group work_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();

  // Shared memory. volMmk elements
  auto shBuffer_char = (char *)dpct_local;
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = item.get_local_id(2) &
                       (sg.get_local_range().get(0) - 1);

  TensorConvInOut Mmk;
  Mmk.c_in = 1;
  Mmk.d_in = 1;
  Mmk.c_out = 1;
  Mmk.d_out = 1;
  if (warpLane < sizeMmk) {
    Mmk = gl_Mmk[warpLane];
  }
  TensorConv Msh;
  Msh.c = 1;
  Msh.d = 1;
  if (warpLane < sizeMmk) {
    Msh = gl_Msh[warpLane];
  }

  // Pre-compute tensor positions in Mmk
  // 3*numRegStorage registers
  int posMmkIn[numRegStorage];
  int posMmkOut[numRegStorage];
  int posSh[numRegStorage];
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
    posSh[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item.get_local_id(2) + j * item.get_local_range().get(2);

      posMmkIn[j] += ((posMmk / sg.shuffle(Mmk.c_in, i)) %
                      sg.shuffle(Mmk.d_in, i)) *
                     sg.shuffle(Mmk.ct_in, i);

      posMmkOut[j] +=
          ((posMmk / sg.shuffle(Mmk.c_out, i)) %
           sg.shuffle(Mmk.d_out, i)) *
          sg.shuffle(Mmk.ct_out, i);

      posSh[j] += ((posMmk / sg.shuffle(Msh.c, i)) %
                   sg.shuffle(Msh.d, i)) *
                  sg.shuffle(Msh.ct, i);
    }
  }

  // 6 registers
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  for (int posMbar = item.get_group(2); posMbar < volMbar;
       posMbar += item.get_group_range(2))
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarOut += sg.shuffle_xor(posMbarOut, i);
    }

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarIn += sg.shuffle_xor(posMbarIn, i);
    }

    sycl::group_barrier(work_grp);

    // Read from global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item.get_local_id(2) + j * item.get_local_range().get(2);
      int posIn = posMbarIn + posMmkIn[j];
      if (posMmk < volMmk) shBuffer[posMmk] = dataIn[posIn];
    }

    sycl::group_barrier(work_grp);

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item.get_local_id(2) + j * item.get_local_range().get(2);
      int posOut = posMbarOut + posMmkOut[j];
      if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j]];
    }


  }
}

//
// Packed method with a split rank
//
// dim nthread(((volMmkWithSplit - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize, 1, 1)
// dim nblock(ts.numSplit, min(256, max(1, ts.volMbar)), 1)
//
template <typename T, int numRegStorage>
void transposePackedSplit(
  const int splitDim, const int volMmkUnsplit, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const int cMmSplit, const int cMkSplit,
  const TensorConvInOut* RESTRICT glMmk,
  const TensorConvInOut* RESTRICT glMbar,
  const TensorConv* RESTRICT glMsh,
  const T* RESTRICT dataIn, T* RESTRICT dataOut, sycl::nd_item<3>& item,
  uint8_t *dpct_local) {

  sycl::group work_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();

  // Shared memory. max(volSplit)*volMmkUnsplit T elements
  auto shBuffer_char = (char *)dpct_local;
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = item.get_local_id(2) &
                       (sg.get_local_range().get(0) - 1);

  // const int plusone = (blockIdx.x < (splitDim % gridDim.x));
  const int p0 = item.get_group(2) * splitDim / item.get_group_range(2);
  const int volSplit =
      (item.get_group(2) + 1) * splitDim / item.get_group_range(2) - p0;
  const int plusone = volSplit - splitDim / item.get_group_range(2);

  TensorConvInOut Mmk;
  Mmk.c_in = 1;
  Mmk.d_in = 1;
  Mmk.c_out = 1;
  Mmk.d_out = 1;
  if (warpLane < sizeMmk) {
    Mmk = glMmk[warpLane + plusone*sizeMmk];
  }
  TensorConv Msh;
  Msh.c = 1;
  Msh.d = 1;
  if (warpLane < sizeMmk) {
    Msh = glMsh[warpLane + plusone*sizeMmk];
  }

  // gridDim.x = number of splits
  // blockIdx.x = {0 ... gridDim.x - 1} is the split-index
  // Volume of this split
  // const int volSplit = (splitDim/gridDim.x) + plusone;
  // Start position in this split
  // const int p0 = (splitDim/gridDim.x)*blockIdx.x + min(blockIdx.x, (splitDim % gridDim.x));
  const int posMmkIn0  = p0*cMmSplit;
  const int posMmkOut0 = p0*cMkSplit;
  // Volume of split Mmk
  const int volMmkSplit = volSplit*volMmkUnsplit;

  // Pre-compute tensor positions in Mmk
  // 3*numRegStorage registers
  int posMmkIn[numRegStorage];
  int posMmkOut[numRegStorage];
  int posSh[numRegStorage];
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j]  = posMmkIn0;
    posMmkOut[j] = posMmkOut0;
    posSh[j] = 0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int t = item.get_local_id(2) + j * item.get_local_range().get(2);
                      sg.shuffle(Mmk.d_in, i)) *
                     sg.shuffle(Mmk.ct_in, i);
      posMmkOut[j] += ((t / sg.shuffle(Mmk.c_out, i)) %
                       sg.shuffle(Mmk.d_out, i)) *
                      sg.shuffle(Mmk.ct_out, i);
      posSh[j] += ((t / sg.shuffle(Msh.c, i)) %
                   sg.shuffle(Msh.d, i)) *
                  sg.shuffle(Msh.ct, i);
    }
  }

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  const int posMbar0 =
      item.get_group(1) * volMbar / item.get_group_range(1);
  const int posMbar1 =
      (item.get_group(1) + 1) * volMbar / item.get_group_range(1);
  for (int posMbar=posMbar0;posMbar < posMbar1;posMbar++)
  // for (int posMbar=blockIdx.y;posMbar < volMbar;posMbar+=gridDim.y)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarOut += sg.shuffle_xor(posMbarOut, i);
    }

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMbarIn += sg.shuffle_xor(posMbarIn, i);
    }

    // Read from global memory
    sycl::group_barrier(work_grp);

#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item.get_local_id(2) + j * item.get_local_range().get(2);
      int posIn = posMbarIn + posMmkIn[j];
      if (posMmk < volMmkSplit) shBuffer[posMmk] = dataIn[posIn];
    }

    // Write to global memory
    sycl::group_barrier(work_grp);

#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item.get_local_id(2) + j * item.get_local_range().get(2);
      int posOut = posMbarOut + posMmkOut[j];
      if (posMmk < volMmkSplit) dataOut[posOut] = shBuffer[posSh[j]];
    }

  }

}


#if 1
//
// Transpose when the lead dimension is the same, e.g. (1, 2, 3) -> (1, 3, 2)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMkBar-1)/TILEDIM+1), 1, plan.volMbar);
//
template <typename T>
void transposeTiledCopy(const int numMm, const int volMbar, const int sizeMbar,
                        const int cuDimMk, const int cuDimMm,
                        const sycl::int2 tiledVol,
                        const TensorConvInOut *RESTRICT gl_Mbar,
                        const T *RESTRICT dataIn, T *RESTRICT dataOut,
                        sycl::nd_item<3>& item) {

  sycl::sub_group sg = item.get_sub_group();

  const int warpLane = item.get_local_id(2) &
                       (sg.get_local_range().get(0) - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  const int bx = (item.get_group(2) % numMm) * TILEDIM;
  const int by = (item.get_group(2) / numMm) * TILEDIM;

  const int x = bx + item.get_local_id(2);
  const int y = by + item.get_local_id(1);

  const unsigned int mask =
      ballot(sg, (y + warpLane < tiledVol.y()))[0] * (x < tiledVol.x());

  const int posMinorIn = x + y*cuDimMk;
  const int posMinorOut = x + y*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int posMbar = item.get_group(0); posMbar < volMbar;
       posMbar += item.get_group_range(0))
  {

    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      posMajorIn += sg.shuffle_xor(posMajorIn, i);
      posMajorOut += sg.shuffle_xor(posMajorOut, i);
    }
    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

    // Variables where values are stored
    T val[TILEDIM/TILEROWS];

    // Read global memory
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
      if ((mask & (1 << j)) != 0) {
        val[j/TILEROWS] = dataIn[posIn];
      }
      posIn += posInAdd;
    }

    // Write global memory
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      // if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
      if ((mask & (1 << j)) != 0) {
        dataOut[posOut] = val[j/TILEROWS];
      }
      posOut += posOutAdd;
    }

  }

}
#else

//
// Returns scalar tensor position. Each lane has the same p
// NOTE: c and d on inactive warps must be 1 !!
//
__device__ __forceinline__
int tensorPos(
  const int p, const int rank, const int c, const int d, const int ct,
  const int numLane=warpSize
  ) {

  int r = ((p/c) % d)*ct;
#pragma unroll
  for (int i=numLane/2;i >= 1;i/=2) {
    r += __shfl_xor_sync(0xffffffff,r,i);
  }
  return r;

}

//
// Transpose when the lead dimension is the same, e.g. (1, 2, 3) -> (1, 3, 2)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMkBar-1)/TILEDIM+1), 1, plan.volMbar);
//
template <typename T>
__global__ void transposeTiledCopy(
  const int numMm, const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const int2 tiledVol,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut) {

  const int warpLane = threadIdx.x & (warpSize - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  const int bx = (blockIdx.x % numMm)*TILEDIM;
  const int by = (blockIdx.x / numMm)*TILEDIM;

  const int x = bx + threadIdx.x;
  const int y = by + threadIdx.y;

  for (int posMbar=blockIdx.z;posMbar < volMbar;posMbar += gridDim.z)
  {

    // Variables where values are stored
    T val[TILEDIM/TILEROWS];

    // Read global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in);
      pos0 += x + y*cuDimMk;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos  = pos0  + j*cuDimMk;
        if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
          val[j/TILEROWS] = dataIn[pos];
        }
      }
    }

    // Write global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out, Mbar.ct_out);
      pos0 += x + y*cuDimMm;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
        if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
          dataOut[pos] = val[j/TILEROWS];
        }
      }
    }

  }

}
#endif

//######################################################################################
//######################################################################################
//######################################################################################

//
// Sets shared memory bank configuration for all kernels. Needs to be called once per device.
//
//void librettKernelSetSharedMemConfig() try {
//#define CALL(NREG) gpuCheck(0)
//#include "calls.h"
//#undef CALL
//
//#define CALL(NREG) gpuCheck(0)
//#include "calls.h"
//#undef CALL
//
//#define CALL(NREG) gpuCheck(0)
//#include "calls.h"
//#undef CALL
//
//#define CALL(NREG) gpuCheck(0)
//#include "calls.h"
//#undef CALL
//
//  /*
//  DPCT1027:92: The call to cudaFuncSetSharedMemConfig was replaced with 0,
//  because DPC++ currently does not support configuring shared memory on devices.
//  */
//  gpuCheck(0);
//  /*
//  DPCT1027:93: The call to cudaFuncSetSharedMemConfig was replaced with 0,
//  because DPC++ currently does not support configuring shared memory on devices.
//  */
//  gpuCheck(0);
//
//  /*
//  DPCT1027:94: The call to cudaFuncSetSharedMemConfig was replaced with 0,
//  because DPC++ currently does not support configuring shared memory on devices.
//  */
//  gpuCheck(0);
//  /*
//  DPCT1027:95: The call to cudaFuncSetSharedMemConfig was replaced with 0,
//  because DPC++ currently does not support configuring shared memory on devices.
//  */
//  gpuCheck(0);
//}
//catch (sycl::exception const &exc) {
//  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
//            << ", line:" << __LINE__ << std::endl;
//  std::exit(1);
//}

// Caches for PackedSplit kernels. One cache for all devices
// NOTE: Not thread safe
const int CACHE_SIZE = 100000;
const int MAX_NUMWARP = (1024/32);
const int MAX_NUMTYPE = 2;
static int numDevices = -1;
LRUCache<unsigned long long int, int> nabCache(CACHE_SIZE, -1);

//
// Returns the maximum number of active blocks per SM
//
int getNumActiveBlock(const int method, const int sizeofType,
                      const LaunchConfig &lc, const int deviceID,
                      const dpct::device_info &prop) try {

  int numActiveBlock = 1;
  int numthread = lc.numthread[2] * lc.numthread[1] * lc.numthread[0];
  switch(method) {
    case Trivial:
    {
      // This value does not matter, but should be > 0
      numActiveBlock = 1;
    }
    break;

    case Packed:
    {
/*
DPCT1007:96: Migration of this CUDA API is not supported by the Intel(R) DPC++
Compatibility Tool.
*/
//#define CALL0(TYPE, NREG)                                                      \
//  cudaOccupancyMaxActiveBlocksPerMultiprocessor(                               \
//      &numActiveBlock, transposePacked<TYPE, NREG>, numthread, lc.shmemsize)
//      switch(lc.numRegStorage) {
//#define CALL(ICASE) case ICASE: if (sizeofType == 4) CALL0(float,  ICASE); if (sizeofType == 8) CALL0(double, ICASE); break
//#include "calls.h"
//      }
//#undef CALL
//#undef CALL0
    }
    break;

    case PackedSplit:
    {
      // Allocate cache structure if needed
      if (numDevices == -1) {
        /*
        DPCT1003:97: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        gpuCheck((numDevices = dpct::dev_mgr::instance().device_count(), 0));
      }
      // Build unique key for cache
      int key_warp = (numthread / prop.get_max_sub_group_size() - 1);
      if (key_warp >= MAX_NUMWARP) {
        printf("getNumActiveBlock maximum number of warps exceeded\n");
        exit(1);
      }
      int key_reg = (lc.numRegStorage - 1);
      int key_type = (sizeofType == 4);
      unsigned long long int key =
      (unsigned long long int)(lc.shmemsize/sizeofType)*MAX_NUMWARP*MAX_REG_STORAGE*MAX_NUMTYPE*numDevices +
      (unsigned long long int)deviceID*MAX_NUMWARP*MAX_REG_STORAGE*MAX_NUMTYPE +
      (unsigned long long int)key_type*MAX_NUMWARP*MAX_REG_STORAGE +
      (unsigned long long int)key_reg*MAX_NUMWARP +
      (unsigned long long int)key_warp;

      numActiveBlock = nabCache.get(key);
      if (numActiveBlock == -1) {
        // key not found in cache, determine value and add it to cache
/*
DPCT1007:98: Migration of this CUDA API is not supported by the Intel(R) DPC++
Compatibility Tool.
*/
//#define CALL0(TYPE, NREG)                                                      \
//  cudaOccupancyMaxActiveBlocksPerMultiprocessor(                               \
//      &numActiveBlock, transposePackedSplit<TYPE, NREG>, numthread,            \
//      lc.shmemsize)
//      switch(lc.numRegStorage) {
//#define CALL(ICASE) case ICASE: if (sizeofType == 4) CALL0(float,  ICASE); if (sizeofType == 8) CALL0(double, ICASE); break
//#include "calls.h"
//      }
//#undef CALL
//#undef CALL0
        nabCache.set(key, numActiveBlock);
      }
    }
    break;

    case Tiled:
    {
      if (sizeofType == 4) {
        /*
        DPCT1007:99: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        //cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //    &numActiveBlock, transposeTiled<float>, numthread, lc.shmemsize);
      } else {
        /*
        DPCT1007:100: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        //cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //    &numActiveBlock, transposeTiled<double>, numthread, lc.shmemsize);
      }
    }
    break;

    case TiledCopy:
    {
      if (sizeofType == 4) {
        /*
        DPCT1007:101: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
        //                                              transposeTiledCopy<float>,
        //                                              numthread, lc.shmemsize);
      } else {
        /*
        DPCT1007:102: Migration of this CUDA API is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        //cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //    &numActiveBlock, transposeTiledCopy<double>, numthread,
        //    lc.shmemsize);
      }
    }
    break;
  }

  return numActiveBlock;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//
// Sets up kernel launch configuration
//
// Returns the number of active blocks per SM that can be achieved on the Packed kernel
// NOTE: Returns 0 when kernel execution is not possible
//
// Sets:
// lc.numthread
// lc.numblock
// lc.shmemsize
// lc.numRegStorage  (for Packed method)
//
int librettKernelLaunchConfiguration(const int sizeofType, const TensorSplit &ts,
                                  const int deviceID,
                                  const dpct::device_info &prop,
                                  LaunchConfig &lc) {

  // Return value of numActiveBlock
  int numActiveBlockReturn = -1;

  switch(ts.method) {
    case Trivial:
    {
      // These values don't matter
      lc.numthread[2] = 1;
      lc.numthread[1] = 1;
      lc.numthread[0] = 1;
      lc.numblock[2] = 1;
      lc.numblock[1] = 1;
      lc.numblock[0] = 1;
      lc.numblock[0] = 1;
      lc.numblock[0] = 1;
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;

    case Packed:
    {
      // Amount of shared memory required
      lc.shmemsize = ts.shmemAlloc(sizeofType); //ts.volMmk*sizeofType;

      // Check that we're not using too much shared memory per block
      /*
      DPCT1019:103: local_mem_size in SYCL is not a complete equivalent of
      sharedMemPerBlock in CUDA. You may need to adjust the code.
      */
      if (lc.shmemsize > prop.get_local_mem_size()) {
        // printf("lc.shmemsize %d prop.sharedMemPerBlock %d\n", lc.shmemsize, prop.sharedMemPerBlock);
        return 0;
      }

      // Min and max number of threads we can use
      int warpSize = prop.get_max_sub_group_size();
      int minNumthread = ((ts.volMmk - 1) / (warpSize * MAX_REG_STORAGE) + 1) * warpSize;
      int maxNumthread = ((ts.volMmk - 1) / (warpSize) + 1) * warpSize;
      if (minNumthread > prop.get_max_work_group_size()) return 0;
      maxNumthread = std::min(prop.get_max_work_group_size(), maxNumthread);
      // printf("minNumthread %d maxNumthread %d\n", minNumthread, maxNumthread);

      // Min and max number of register storage we can use
      int minNumRegStorage = (ts.volMmk - 1)/maxNumthread + 1;
      int maxNumRegStorage = (ts.volMmk - 1)/minNumthread + 1;
      // printf("minNumRegStorage %d maxNumRegStorage %d\n", minNumRegStorage, maxNumRegStorage);

      int bestVal = 0;
      int bestNumRegStorage = 0;
      int bestNumActiveBlock = 0;

      lc.numthread[1] = 1;
      lc.numthread[0] = 1;
      lc.numblock[2] = std::max(1, ts.volMbar);
      lc.numblock[2] = std::min<unsigned int>(prop.get_max_compute_units() * 18,
                                              lc.numblock[2]);
      lc.numblock[1] = 1;
      lc.numblock[0] = 1;

      for (lc.numRegStorage=minNumRegStorage;lc.numRegStorage <= maxNumRegStorage;lc.numRegStorage++) {
        lc.numthread[2] = ((ts.volMmk - 1) / (prop.get_max_sub_group_size() *
                                              lc.numRegStorage) +
                           1) *
                          prop.get_max_sub_group_size();

        int numActiveBlock = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
        // int val = numActiveBlock*lc.numthread.x;
        int val = ts.volMmkUsed()*numActiveBlock;
        if (val > bestVal) {
          bestVal = val;
          bestNumRegStorage = lc.numRegStorage;
          bestNumActiveBlock = numActiveBlock;
        }
      }

      if (bestNumRegStorage == 0) return 0;

      lc.numRegStorage = bestNumRegStorage;
      lc.numthread[2] = ((ts.volMmk - 1) / (prop.get_max_sub_group_size() *
                                            lc.numRegStorage) +
                         1) *
                        prop.get_max_sub_group_size();
      numActiveBlockReturn = bestNumActiveBlock;
    }
    break;

    case PackedSplit:
    {
      // Amount of shared memory required
      lc.shmemsize = ts.shmemAlloc(sizeofType);

      // Check that we're not using too much shared memory per block
      /*
      DPCT1019:104: local_mem_size in SYCL is not a complete equivalent of
      sharedMemPerBlock in CUDA. You may need to adjust the code.
      */
      if (lc.shmemsize > prop.get_local_mem_size()) {
        // printf("lc.shmemsize %d prop.sharedMemPerBlock %d\n", lc.shmemsize, prop.sharedMemPerBlock);
        return 0;
      }

      int volMmkWithSplit = (ts.splitDim/ts.numSplit + ((ts.splitDim % ts.numSplit) > 0))*ts.volMmkUnsplit;

      // Min and max number of threads we can use
      int warpSize = prop.get_max_sub_group_size();
      int minNumthread = ((volMmkWithSplit - 1) / (warpSize * MAX_REG_STORAGE) + 1) * warpSize;
      int maxNumthread = ((volMmkWithSplit - 1) / (warpSize) + 1) * warpSize;
      if (minNumthread > prop.get_max_work_group_size()) return 0;
      maxNumthread = std::min(prop.get_max_work_group_size(), maxNumthread);
      // printf("minNumthread %d maxNumthread %d\n", minNumthread, maxNumthread);

      // Min and max number of register storage we can use
      int minNumRegStorage = (volMmkWithSplit - 1)/maxNumthread + 1;
      int maxNumRegStorage = (volMmkWithSplit - 1)/minNumthread + 1;
      // printf("minNumRegStorage %d maxNumRegStorage %d\n", minNumRegStorage, maxNumRegStorage);

      int bestVal = 0;
      int bestNumRegStorage = 0;
      int bestNumActiveBlock = 0;

      lc.numthread[1] = 1;
      lc.numthread[0] = 1;
      lc.numblock[2] = ts.numSplit;
      lc.numblock[1] =
          std::max<unsigned int>(1, std::min<unsigned int>((prop.get_max_compute_units() * 18) /
                                            lc.numblock[2],
                                        ts.volMbar));
      lc.numblock[0] = 1;

      for (lc.numRegStorage=minNumRegStorage;lc.numRegStorage <= maxNumRegStorage;lc.numRegStorage++) {
        lc.numthread[2] =
            ((volMmkWithSplit - 1) /
                 (prop.get_max_sub_group_size() * lc.numRegStorage) +
             1) *
            prop.get_max_sub_group_size();

        int numActiveBlock = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
        // int val = numActiveBlock*lc.numthread.x*lc.numRegStorage;
        int val = ts.volMmkUsed()*numActiveBlock;
        if (val > bestVal) {
          bestVal = val;
          bestNumRegStorage = lc.numRegStorage;
          bestNumActiveBlock = numActiveBlock;
        }
      }

      if (bestNumRegStorage == 0) return 0;

      lc.numRegStorage = bestNumRegStorage;
      lc.numthread[2] =
          ((volMmkWithSplit - 1) /
               (prop.get_max_sub_group_size() * lc.numRegStorage) +
           1) *
          prop.get_max_sub_group_size();
      numActiveBlockReturn = bestNumActiveBlock;
    }
    break;

    case Tiled:
    {
      lc.numthread[2] = TILEDIM;
      lc.numthread[1] = TILEROWS;
      lc.numthread[0] = 1;
      lc.numblock[2] =
          ((ts.volMm - 1) / TILEDIM + 1) * ((ts.volMk - 1) / TILEDIM + 1);
      lc.numblock[1] = 1;
      lc.numblock[0] =
          std::max<unsigned int>(1, std::min<unsigned int>((prop.get_max_compute_units() * 8) /
                                            (lc.numblock[2] * lc.numblock[1]),
                                        ts.volMbar));
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;

    case TiledCopy:
    {
      lc.numthread[2] = TILEDIM;
      lc.numthread[1] = TILEROWS;
      lc.numthread[0] = 1;
      lc.numblock[2] =
          ((ts.volMm - 1) / TILEDIM + 1) * ((ts.volMkBar - 1) / TILEDIM + 1);
      lc.numblock[1] = 1;
      lc.numblock[0] = ts.volMbar;
      lc.numblock[0] = std::min<unsigned int>((prop.get_max_compute_units() * 8) /
                               (lc.numblock[2] * lc.numblock[1]),
                           lc.numblock[0]);
      lc.numblock[0] = std::max<unsigned int>(1, lc.numblock[0]);
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;
  }

  /*
  DPCT1022:105: There is no exact match between the maxGridSize and the
  max_nd_range size. Verify the correctness of the code.
  */
  if (lc.numblock[2] > prop.get_max_nd_range_size()[0] ||
      /*
      DPCT1022:106: There is no exact match between the maxGridSize and the
      max_nd_range size. Verify the correctness of the code.
      */
      lc.numblock[1] > prop.get_max_nd_range_size()[1] ||
      /*
      DPCT1022:107: There is no exact match between the maxGridSize and the
      max_nd_range size. Verify the correctness of the code.
      */
      lc.numblock[0] > prop.get_max_nd_range_size()[2]) return 0;

  // Return the number of active blocks with these settings
  if (numActiveBlockReturn == -1) {
    // Not set, get it
    numActiveBlockReturn = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
  }
  return numActiveBlockReturn;
}

bool librettKernel(librettPlan_t &plan, void *dataIn, void *dataOut) try {

  LaunchConfig& lc = plan.launchConfig;
  TensorSplit& ts = plan.tensorSplit;

  switch(ts.method) {
    case Trivial:
    {
      gpuCheck(plan.stream->memcpy(dataOut, dataIn,
                                   ts.volMmk * ts.volMbar * plan.sizeofType));
    }
    break;

    case Packed:
    {
      switch(lc.numRegStorage) {
#define CALL0(TYPE, NREG)                                                      \
  {plan.stream->submit([&](sycl::handler &cgh) {                               \
    localAcc<uint8_t, 1> local_acc_ct1(sycl::range<1>(lc.shmemsize), cgh);\
                                                                               \
    auto ts_volMmk_ct0 = ts.volMmk;                                            \
    auto ts_volMbar_ct1 = ts.volMbar;                                          \
    auto ts_sizeMmk_ct2 = ts.sizeMmk;                                          \
    auto ts_sizeMbar_ct3 = ts.sizeMbar;                                        \
    auto plan_Mmk_ct4 = plan.Mmk;                                              \
    auto plan_Mbar_ct5 = plan.Mbar;                                            \
    auto plan_Msh_ct6 = plan.Msh;                                              \
    auto dataIn_ct7 = (TYPE *)dataIn;                                          \
    auto dataOut_ct8 = (TYPE *)dataOut;                                        \
                                                                               \
    cgh.parallel_for(                                                          \
        sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),           \
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {    \
          transposePacked<TYPE, NREG>(                                         \
              ts_volMmk_ct0, ts_volMbar_ct1, ts_sizeMmk_ct2, ts_sizeMbar_ct3,  \
              plan_Mmk_ct4, plan_Mbar_ct5, plan_Msh_ct6, dataIn_ct7,           \
              dataOut_ct8, item, local_acc_ct1.get_pointer());        \
        });                                                                    \
    }); plan.stream->wait(); \
  }
#define CALL(ICASE) case ICASE: if (plan.sizeofType == 4) CALL0(float,  ICASE); if (plan.sizeofType == 8) CALL0(double, ICASE); break
#include "calls.h"
        default:
        printf("librettKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case PackedSplit:
    {
      switch(lc.numRegStorage) {
#define CALL0(TYPE, NREG)                                                      \
  plan.stream->submit([&](sycl::handler &cgh) {                                \
    localAcc<uint8_t, 1> local_acc(sycl::range<1>(lc.shmemsize), cgh);     \
                                                                               \
    auto ts_splitDim_ct0 = ts.splitDim;                                        \
    auto ts_volMmkUnsplit_ct1 = ts.volMmkUnsplit;                              \
    auto ts_volMbar_ct2 = ts.volMbar;                                          \
    auto ts_sizeMmk_ct3 = ts.sizeMmk;                                          \
    auto ts_sizeMbar_ct4 = ts.sizeMbar;                                        \
    auto plan_cuDimMm_ct5 = plan.cuDimMm;                                      \
    auto plan_cuDimMk_ct6 = plan.cuDimMk;                                      \
    auto plan_Mmk_ct7 = plan.Mmk;                                              \
    auto plan_Mbar_ct8 = plan.Mbar;                                            \
    auto plan_Msh_ct9 = plan.Msh;                                              \
    auto dataIn_ct10 = (TYPE *)dataIn;                                         \
    auto dataOut_ct11 = (TYPE *)dataOut;                                       \
                                                                               \
    cgh.parallel_for(                                                          \
        sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),           \
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {    \
          transposePackedSplit<TYPE, NREG>(                                    \
              ts_splitDim_ct0, ts_volMmkUnsplit_ct1, ts_volMbar_ct2,           \
              ts_sizeMmk_ct3, ts_sizeMbar_ct4, plan_cuDimMm_ct5,               \
              plan_cuDimMk_ct6, plan_Mmk_ct7, plan_Mbar_ct8, plan_Msh_ct9,     \
              dataIn_ct10, dataOut_ct11, item,                             \
              local_acc.get_pointer());                               \
        });                                                                    \
  }); plan.stream->wait();
#define CALL(ICASE) case ICASE: if (plan.sizeofType == 4) CALL0(float,  ICASE); if (plan.sizeofType == 8) CALL0(double, ICASE); break
#include "calls.h"
#include <cmath>

#include <algorithm>

        default:
        printf("librettKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case Tiled:
    {
#define CALL(TYPE)                                                             \
  plan.stream->submit([&](sycl::handler &cgh) {                                \
    sycl::range<2> shTile_range_ct1(32 /*TILEDIM*/, 33 /*TILEDIM+1*/);         \
                                                                               \
    localAcc<TYPE, 2> shTile_acc(shTile_range, cgh);                   \
                                                                               \
    auto ts_volMm_TILEDIM_ct0 = ((ts.volMm - 1) / TILEDIM + 1);                \
    auto ts_volMbar = ts.volMbar;                                          \
    auto ts_sizeMbar_ct2 = ts.sizeMbar;                                        \
    auto plan_tiledVol_ct3 = plan.tiledVol;                                    \
    auto plan_cuDimMk_ct4 = plan.cuDimMk;                                      \
    auto plan_cuDimMm_ct5 = plan.cuDimMm;                                      \
    auto plan_Mbar_ct6 = plan.Mbar;                                            \
    auto dataIn_ct7 = (TYPE *)dataIn;                                          \
    auto dataOut_ct8 = (TYPE *)dataOut;                                        \
                                                                               \
    cgh.parallel_for(                                                          \
        sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),           \
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {    \
          transposeTiled<TYPE>(                                                \
              ts_volMm_TILEDIM_ct0, ts_volMbar, ts_sizeMbar_ct2,           \
              plan_tiledVol_ct3, plan_cuDimMk_ct4, plan_cuDimMm_ct5,           \
              plan_Mbar_ct6, dataIn_ct7, dataOut_ct8, item,                \
              shTile_acc));
        });                                                                    \
  }); plan.stream->wait();
      if (plan.sizeofType == 4) CALL(float) if (plan.sizeofType == 8) CALL(double)
#undef CALL
    }
    break;

    case TiledCopy:
    {
/*
DPCT1049:112: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
#define CALL(TYPE)                                                             \
  plan.stream->submit([&](sycl::handler &cgh) {                                \
    auto ts_volMm_TILEDIM_ct0 = ((ts.volMm - 1) / TILEDIM + 1);                \
    auto ts_volMbar_ct1 = ts.volMbar;                                          \
    auto ts_sizeMbar_ct2 = ts.sizeMbar;                                        \
    auto plan_cuDimMk_ct3 = plan.cuDimMk;                                      \
    auto plan_cuDimMm_ct4 = plan.cuDimMm;                                      \
    auto plan_tiledVol_ct5 = plan.tiledVol;                                    \
    auto plan_Mbar_ct6 = plan.Mbar;                                            \
    auto dataIn_ct7 = (TYPE *)dataIn;                                          \
    auto dataOut_ct8 = (TYPE *)dataOut;                                        \
                                                                               \
    cgh.parallel_for(                                                          \
        sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),           \
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {    \
          transposeTiledCopy<TYPE>(                                            \
              ts_volMm_TILEDIM_ct0, ts_volMbar_ct1, ts_sizeMbar_ct2,           \
              plan_cuDimMk_ct3, plan_cuDimMm_ct4, plan_tiledVol_ct5,           \
              plan_Mbar_ct6, dataIn_ct7, dataOut_ct8, item);               \
        });                                                                    \
  }); plan.stream->wait();
      if (plan.sizeofType == 4) CALL(float) if (plan.sizeofType == 8) CALL(double)
#undef CALL
    }
    break;

  }

  plan.stream->throw_asynchronous();
  return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
