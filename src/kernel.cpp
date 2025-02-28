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

#include "GpuUtils.h"
#include "LRUCache.h"
#include "kernel.h"
#include <iostream>
#include "unistd.h"

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
__global__ void transposeTiled(const int numMm, const int volMbar, const int sizeMbar,
  const int2_t tiledVol, const int cuDimMk, const int cuDimMm,
  const TensorConvInOut* RESTRICT glMbar, const T* RESTRICT dataIn, T* RESTRICT dataOut
#if LIBRETT_USES_SYCL
  , sycl::nd_item<3>& item
#endif
  )
{
  // Shared memory
#if LIBRETT_USES_SYCL
  sycl::group wrk_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();
  #ifdef LIBRETT_SUBGROUP_SIZE64
  using tile_t = T[TILEDIM][TILEDIM];
  #else
  using tile_t = T[TILEDIM][TILEDIM+1];
  #endif
  tile_t& shTile = *sycl::ext::oneapi::group_local_memory_for_overwrite<tile_t>(wrk_grp);
  const int warpSize = sg.get_local_range().get(0);
#elif LIBRETT_USES_HIP
  __shared__ T shTile[TILEDIM][TILEDIM];
#elif LIBRETT_USES_CUDA
  __shared__ T shTile[TILEDIM][TILEDIM+1];
#endif

  const int warpLane = threadIdx_x & (warpSize - 1);

  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  const int bx = (blockIdx_x % numMm)*TILEDIM;
  const int by = (blockIdx_x / numMm)*TILEDIM;

  const int xin = bx + threadIdx_x;
  const int yin = by + threadIdx_y;

  const int xout = bx + threadIdx_y;
  const int yout = by + threadIdx_x;

#if LIBRETT_USES_SYCL
  const unsigned long long int maskIny = ballot(sg, (yin + warpLane < tiledVol.y())).s0() * (xin < tiledVol.x());
  const unsigned long long int maskOutx = ballot(sg, (xout + warpLane < tiledVol.x())).s0() * (yout < tiledVol.y());
  const unsigned long long int one = 1;
#elif LIBRETT_USES_HIP
  // AMD change
  const unsigned long long int maskIny = __ballot((yin + warpLane < tiledVol.y))*(xin < tiledVol.x);
  const unsigned long long int maskOutx = __ballot((xout + warpLane < tiledVol.x))*(yout < tiledVol.y);
  const unsigned long long int one = 1;
#elif LIBRETT_USES_CUDA
  const unsigned int maskIny = __ballot_sync(0xffffffff,(yin + warpLane < tiledVol.y))*(xin < tiledVol.x);
  const unsigned int maskOutx = __ballot_sync(0xffffffff,(xout + warpLane < tiledVol.x))*(yout < tiledVol.y);
  const unsigned int one = 1;
#endif

  const int posMinorIn = xin + yin*cuDimMk;
  const int posMinorOut = yout + xout*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int posMbar=blockIdx_z; posMbar < volMbar; posMbar += gridDim_z)
  {
    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#if LIBRETT_USES_SYCL
    posMajorIn  = sycl::reduce_over_group(sg, posMajorIn,  sycl::plus<int>());
    posMajorOut = sycl::reduce_over_group(sg, posMajorOut, sycl::plus<int>());
#else // FOR CUDA, HIP only
    #pragma unroll
    for (int i=warpSize/2; i >= 1; i/=2) {  // AMD change
        posMajorIn += gpu_shfl_xor(posMajorIn,i);
        posMajorOut += gpu_shfl_xor(posMajorOut,i);
    }
#endif // SYCL

    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

    // Read from global memory
    #if LIBRETT_USES_SYCL
    sycl::group_barrier( wrk_grp );
    #else
    syncthreads();
    #endif

    // Read data into shared memory tile
#pragma unroll
    for (int j=0; j < TILEDIM; j += TILEROWS) {
      // int pos = posIn + j*cuDimMk;
      // if (xin < readVol.x && yin + j < readVol.y) {
      if ((maskIny & (one << j)) != 0) {   // AMD change
        shTile[threadIdx_y + j][threadIdx_x] = dataIn[posIn];
      }
      posIn += posInAdd;
    }

    // Write to global memory
    #if LIBRETT_USES_SYCL
    sycl::group_barrier( wrk_grp );
    #else
    syncthreads();
    #endif

#pragma unroll
    for (int j=0; j < TILEDIM; j += TILEROWS) {
      // int pos = posOut + j*cuDimMm;
      // if (xout + j < readVol.x && yout < readVol.y) {
      if ((maskOutx & (one << j)) != 0 ) {   // AMD change
        dataOut[posOut] = shTile[threadIdx_x][threadIdx_y + j];
      }
      posOut += posOutAdd;
    }

  }

}

//
// Packed transpose. Thread block loads plan.volMmk number of elements
//
template <typename T, int numRegStorage>
__global__ void transposePacked(
  const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* RESTRICT gl_Mmk,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const TensorConv* RESTRICT gl_Msh,
  const T* RESTRICT dataIn, T* RESTRICT dataOut
  #if LIBRETT_USES_SYCL
  , sycl::nd_item<3> item, uint8_t *dpct_local
  #endif
  )
{
  // Shared memory. volMmk elements
#if LIBRETT_USES_SYCL
  auto shBuffer_char = (char *)dpct_local;
  sycl::group wrk_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();
  const int warpSize = sg.get_local_range().get(0);
#elif LIBRETT_USES_HIP
  HIP_DYNAMIC_SHARED( char, shBuffer_char)
#elif LIBRETT_USES_CUDA
  extern __shared__ char shBuffer_char[];
#endif
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = threadIdx_x & (warpSize - 1);

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
  for (int j=0; j < numRegStorage; j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
    posSh[j] = 0;
  }
  for (int i=0; i < sizeMmk; i++) {
#pragma unroll
    for (int j=0; j < numRegStorage; j++) {
      int posMmk = threadIdx_x + j*blockDim_x;

      posMmkIn[j]  += ((posMmk / gpu_shuffle(Mmk.c_in,i))  % gpu_shuffle(Mmk.d_in,i))  * gpu_shuffle(Mmk.ct_in,i);
      posMmkOut[j] += ((posMmk / gpu_shuffle(Mmk.c_out,i)) % gpu_shuffle(Mmk.d_out,i)) * gpu_shuffle(Mmk.ct_out,i);
      posSh[j]     += ((posMmk / gpu_shuffle(Msh.c,i))     % gpu_shuffle(Msh.d,i))     * gpu_shuffle(Msh.ct,i);
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

  for (int posMbar=blockIdx_x; posMbar < volMbar; posMbar += gridDim_x)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
    int posMbarIn  = ((posMbar/Mbar.c_in)  % Mbar.d_in) *Mbar.ct_in;
#if LIBRETT_USES_SYCL
    posMbarOut = sycl::reduce_over_group(sg, posMbarOut, sycl::plus<int>());
    posMbarIn  = sycl::reduce_over_group(sg, posMbarIn,  sycl::plus<int>());
#else // for CUDA, HIP only
    #pragma unroll
    for (int i=warpSize/2; i >= 1; i/=2) {   // AMD change
      posMbarOut += gpu_shfl_xor(posMbarOut,i);
      posMbarIn  += gpu_shfl_xor(posMbarIn,i);
    }
#endif

    #if LIBRETT_USES_SYCL
    sycl::group_barrier( wrk_grp );
    #else
    syncthreads();
    #endif

    // Read from global memory
#pragma unroll
    for (int j=0; j < numRegStorage; j++) {
      int posMmk = threadIdx_x + j*blockDim_x;
      int posIn = posMbarIn + posMmkIn[j];
      if (posMmk < volMmk) shBuffer[posMmk] = dataIn[posIn];
    }

    #if LIBRETT_USES_SYCL
    sycl::group_barrier( wrk_grp );
    #else
    syncthreads();
    #endif

    // Write to global memory
#pragma unroll
    for (int j=0; j < numRegStorage; j++) {
      int posMmk = threadIdx_x + j*blockDim_x;
      int posOut = posMbarOut + posMmkOut[j];
      if (posMmk < volMmk) dataOut[posOut] = shBuffer[posSh[j]];
    }

  }

}

//
// Packed method with a split rank
//
// dim nthread(((volMmkWithSplit - 1)/(gpuWarpSize*lc.numRegStorage) + 1)*gpuWarpSize, 1, 1)
// dim nblock(ts.numSplit, min(256, max(1, ts.volMbar)), 1)
//
template <typename T, int numRegStorage>
__global__ void transposePackedSplit(
  const int splitDim, const int volMmkUnsplit, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const int cMmSplit, const int cMkSplit,
  const TensorConvInOut* RESTRICT glMmk,
  const TensorConvInOut* RESTRICT glMbar,
  const TensorConv* RESTRICT glMsh,
  const T* RESTRICT dataIn, T* RESTRICT dataOut
  #if LIBRETT_USES_SYCL
  , sycl::nd_item<3>& item, uint8_t *dpct_local
  #endif
  )
{
  // Shared memory. max(volSplit)*volMmkUnsplit T elements
#if LIBRETT_USES_SYCL
  auto shBuffer_char = (char *)dpct_local;
  sycl::group wrk_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();
  const int warpSize = sg.get_local_range().get(0);
#elif LIBRETT_USES_HIP
  HIP_DYNAMIC_SHARED( char, shBuffer_char)
#elif LIBRETT_USES_CUDA
  extern __shared__ char shBuffer_char[];
#endif
  T* shBuffer = (T *)shBuffer_char;

  const int warpLane = threadIdx_x & (warpSize - 1);

  // const int plusone = (blockIdx.x < (splitDim % gridDim.x));
  const int p0 = blockIdx_x*splitDim/gridDim_x;
  const int volSplit = (blockIdx_x + 1)*splitDim/gridDim_x - p0;
  const int plusone = volSplit - splitDim/gridDim_x;

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
  for (int j=0; j < numRegStorage; j++) {
    posMmkIn[j]  = posMmkIn0;
    posMmkOut[j] = posMmkOut0;
    posSh[j] = 0;
  }
  for (int i=0; i < sizeMmk; i++) {
#pragma unroll
    for (int j=0; j < numRegStorage; j++) {
      int t = threadIdx_x + j*blockDim_x;

      posMmkIn[j]  += ((t/gpu_shuffle(Mmk.c_in,i))  % gpu_shuffle(Mmk.d_in,i))  * gpu_shuffle(Mmk.ct_in,i);
      posMmkOut[j] += ((t/gpu_shuffle(Mmk.c_out,i)) % gpu_shuffle(Mmk.d_out,i)) * gpu_shuffle(Mmk.ct_out,i);
      posSh[j]     += ((t/gpu_shuffle(Msh.c,i))     % gpu_shuffle(Msh.d,i))     * gpu_shuffle(Msh.ct,i);
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

  const int posMbar0 = blockIdx_y*volMbar/gridDim_y;
  const int posMbar1 = (blockIdx_y + 1)*volMbar/gridDim_y;
  for (int posMbar=posMbar0; posMbar < posMbar1; posMbar++)
  // for (int posMbar=blockIdx.y;posMbar < volMbar;posMbar+=gridDim.y)
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#if LIBRETT_USES_SYCL
    posMbarOut = sycl::reduce_over_group(sg, posMbarOut, sycl::plus<int>());
    posMbarIn  = sycl::reduce_over_group(sg, posMbarIn,  sycl::plus<int>());
#else // HIP, CUDA only
    #pragma unroll
    for (int i=warpSize/2; i >= 1; i/=2) {   // AMD change
      posMbarOut += gpu_shfl_xor(posMbarOut,i);
      posMbarIn += gpu_shfl_xor(posMbarIn,i);
    }
#endif

    // Read from global memory
    #if LIBRETT_USES_SYCL
    sycl::group_barrier( wrk_grp );
    #else
    syncthreads();
    #endif

#pragma unroll
    for (int j=0; j < numRegStorage; j++) {
      int posMmk = threadIdx_x + j*blockDim_x;
      int posIn = posMbarIn + posMmkIn[j];
      if (posMmk < volMmkSplit) shBuffer[posMmk] = dataIn[posIn];
    }

    // Write to global memory
    #if LIBRETT_USES_SYCL
    sycl::group_barrier( wrk_grp );
    #else
    syncthreads();
    #endif

#pragma unroll
    for (int j=0; j < numRegStorage; j++) {
      int posMmk = threadIdx_x + j*blockDim_x;
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
__global__ void transposeTiledCopy(
  const int numMm, const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const int2_t tiledVol,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const T* RESTRICT dataIn, T* RESTRICT dataOut
  #if LIBRETT_USES_SYCL
  , sycl::nd_item<3>& item
  #endif
  )
{
#if LIBRETT_USES_SYCL
  sycl::sub_group sg = item.get_sub_group();
  const int warpSize = sg.get_local_range().get(0);
#endif
  const int warpLane = threadIdx_x & (warpSize - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  const int bx = (blockIdx_x % numMm)*TILEDIM;
  const int by = (blockIdx_x / numMm)*TILEDIM;

  const int x = bx + threadIdx_x;
  const int y = by + threadIdx_y;

#if LIBRETT_USES_SYCL
  const unsigned int mask = ballot(sg, (y + warpLane < tiledVol.y())).s0() * (x < tiledVol.x());
  const unsigned int one = 1;
#elif LIBRETT_USES_HIP // AMD change
  const unsigned long long int mask = __ballot((y + warpLane < tiledVol.y))*(x < tiledVol.x);
  const unsigned long long int one = 1;
#elif LIBRETT_USES_CUDA
  const unsigned int mask = __ballot_sync(0xffffffff,(y + warpLane < tiledVol.y))*(x < tiledVol.x);
  const unsigned int one = 1;
#endif

  const int posMinorIn = x + y*cuDimMk;
  const int posMinorOut = x + y*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  for (int posMbar=blockIdx_z; posMbar < volMbar; posMbar += gridDim_z)
  {

    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#if LIBRETT_USES_SYCL
    posMajorIn  = sycl::reduce_over_group(sg, posMajorIn,  sycl::plus<int>());
    posMajorOut = sycl::reduce_over_group(sg, posMajorOut, sycl::plus<int>());
#else // for CUDA, HIP only
    #pragma unroll
    for (int i=warpSize/2; i >= 1; i/=2) {   // AMD change
      posMajorIn += gpu_shfl_xor(posMajorIn,i);
      posMajorOut += gpu_shfl_xor(posMajorOut,i);
    }
#endif // SYCL
    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

    // Variables where values are stored
    T val[TILEDIM/TILEROWS];

    // Read global memory
#pragma unroll
    for (int j=0; j < TILEDIM; j += TILEROWS) {
      // if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
      if ((mask & (one << j)) != 0) {   // AMD change
        val[j/TILEROWS] = dataIn[posIn];
      }
      posIn += posInAdd;
    }

    // Write global memory
#pragma unroll
    for (int j=0; j < TILEDIM; j += TILEROWS) {
      // if ((x < tiledVol.x) && (y + j < tiledVol.y)) {
      if ((mask & (one << j)) != 0) {   // AMD change
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
  const int2_t tiledVol,
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
void librettKernelSetSharedMemConfig() {
#if LIBRETT_USES_CUDA
  #define CALL(NREG) cudaCheck(cudaFuncSetSharedMemConfig(transposePacked<float, NREG>, cudaSharedMemBankSizeFourByte ))
  #include "calls.h"
  #undef CALL

  #define CALL(NREG) cudaCheck(cudaFuncSetSharedMemConfig(transposePacked<double, NREG>, cudaSharedMemBankSizeEightByte ))
  #include "calls.h"
  #undef CALL

  #define CALL(NREG) cudaCheck(cudaFuncSetSharedMemConfig(transposePackedSplit<float, NREG>, cudaSharedMemBankSizeFourByte ))
  #include "calls.h"
  #undef CALL

  #define CALL(NREG) cudaCheck(cudaFuncSetSharedMemConfig(transposePackedSplit<double, NREG>, cudaSharedMemBankSizeEightByte ))
  #include "calls.h"
  #undef CALL

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiled<float>, cudaSharedMemBankSizeFourByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledCopy<float>, cudaSharedMemBankSizeFourByte));

  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiled<double>, cudaSharedMemBankSizeEightByte));
  cudaCheck(cudaFuncSetSharedMemConfig(transposeTiledCopy<double>, cudaSharedMemBankSizeEightByte));

#endif // LIBRETT_USES_CUDA
}

// Caches for PackedSplit kernels. One cache for all devices
// NOTE: Not thread safe
const int CACHE_SIZE = 100000;
#if LIBRETT_USES_HIP
  const int MAX_NUMWARP = (1024/64);  // AMD change
#elif LIBRETT_USES_SYCL
  #if LIBRETT_SUBGROUP_SIZE16
  const int MAX_NUMWARP = (1024/16);
  #elif LIBRETT_SUBGROUP_SIZE32
  const int MAX_NUMWARP = (1024/32);
  #elif LIBRETT_SUBGROUP_SIZE64
  const int MAX_NUMWARP = (1024/64);
  #else
  const int MAX_NUMWARP = (1024/32);
  #endif
#elif LIBRETT_USES_CUDA
  const int MAX_NUMWARP = (1024/32);
#endif
const int MAX_NUMTYPE = 2;
static int numDevices = -1;
LRUCache<unsigned long long int, int> nabCache(CACHE_SIZE, -1);

//
// Returns the maximum number of active blocks per SM
//
int getNumActiveBlock(const int method, const int sizeofType, const LaunchConfig &lc,
  const int deviceID, const gpuDeviceProp_t &prop)
{
  //int numActiveBlock = 1;
  int numActiveBlock;
  int numthread = lc.numthread_x * lc.numthread_y * lc.numthread_z;
  switch(method) {
    case Trivial:
    {
      // This value does not matter, but should be > 0
      numActiveBlock = 1;
    }
    break;

    case Packed:
    {
    #ifndef LIBRETT_USES_SYCL
      #define CALL0(TYPE, NREG) \
        generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, \
          transposePacked<TYPE, NREG>, numthread, lc.shmemsize))
      switch(lc.numRegStorage) {
        #define CALL(ICASE) case ICASE: if (sizeofType == 4) CALL0(float,  ICASE); \
	                                if (sizeofType == 8) CALL0(double, ICASE); \
                                        if (sizeofType == 16) CALL0(librett_complex, ICASE); break;
        #include "calls.h"
      }
      #undef CALL
      #undef CALL0
    #endif // CUDA or HIP
    }
    break;

    case PackedSplit:
    {
      // Allocate cache structure if needed
      if (numDevices == -1) {
        #if LIBRETT_USES_SYCL
          Librett::syclGetDeviceCount(&numDevices);
        #elif LIBRETT_USES_HIP
          hipCheck(hipGetDeviceCount(&numDevices));
        #elif LIBRETT_USES_CUDA
          cudaCheck(cudaGetDeviceCount(&numDevices));
        #endif
      }
      // Build unique key for cache
      int key_warp = (numthread/gpuWarpSize - 1);
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
        #ifndef LIBRETT_USES_SYCL
          #define CALL0(TYPE, NREG) \
            generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock, \
              transposePackedSplit<TYPE, NREG>, numthread, lc.shmemsize))
          switch(lc.numRegStorage) {
            #define CALL(ICASE) case ICASE: if (sizeofType == 4) CALL0(float,  ICASE); \
		                            if (sizeofType == 8) CALL0(double, ICASE); \
                                            if (sizeofType == 16) CALL0(librett_complex,ICASE); break;
            #include "calls.h"
          }
          #undef CALL
          #undef CALL0
        #endif // CUDA or HIP
        nabCache.set(key, numActiveBlock);
      }
    }
    break;

    case Tiled:
    {
    #ifndef LIBRETT_USES_SYCL
      if (sizeofType == 4) {
        generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiled<float>, numthread, lc.shmemsize));
      } else if (sizeofType == 8) {
        generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiled<double>, numthread, lc.shmemsize));
      }
      else if (sizeofType == 16) {
        generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiled<librett_complex>, numthread, lc.shmemsize));
      }
    #endif // CUDA or HIP
    }
    break;

    case TiledCopy:
    {
    #ifndef LIBRETT_USES_SYCL
      if (sizeofType == 4) {
        generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledCopy<float>, numthread, lc.shmemsize));
      } else if (sizeofType == 8) {
        generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledCopy<double>, numthread, lc.shmemsize));
      } else if (sizeofType == 16) {
        generalCheck(gpuOccupancyMaxActiveBlocksPerMultiprocessor(&numActiveBlock,
          transposeTiledCopy<librett_complex>, numthread, lc.shmemsize));
      }
    #endif // CUDA or HIP
    }
    break;
  }

  return numActiveBlock;
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
  const int deviceID, const gpuDeviceProp_t &prop, LaunchConfig &lc) {

  // Return value of numActiveBlock
  int numActiveBlockReturn = -1;

  switch(ts.method) {
    case Trivial:
    {
      // These values don't matter
      lc.numthread_x = 1;
      lc.numthread_y = 1;
      lc.numthread_z = 1;
      lc.numblock_x = 1;
      lc.numblock_y = 1;
      lc.numblock_z = 1;
      lc.numblock_z = 1;
      lc.numblock_z = 1;
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;

    case Packed:
    {
      // Amount of shared memory required
      lc.shmemsize = ts.shmemAlloc(sizeofType); //ts.volMmk*sizeofType;

      // Check that we're not using too much shared memory per block
      /* DPCT1019:103: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. */
      if (lc.shmemsize > gpuSharedMemPerBlock) {
        // printf("lc.shmemsize %d prop.sharedMemPerBlock %d\n", lc.shmemsize, prop.sharedMemPerBlock);
        return 0;
      }

      // Min and max number of threads we can use
      int minNumthread = ((ts.volMmk - 1)/(gpuWarpSize*MAX_REG_STORAGE) + 1)*gpuWarpSize;
      int maxNumthread = ((ts.volMmk - 1)/(gpuWarpSize) + 1)*gpuWarpSize;
      if (minNumthread > gpuMaxThreadsPerBlock) return 0;
      maxNumthread = min(gpuMaxThreadsPerBlock, maxNumthread);
      // printf("minNumthread %d maxNumthread %d\n", minNumthread, maxNumthread);

      // Min and max number of register storage we can use
      int minNumRegStorage = (ts.volMmk - 1)/maxNumthread + 1;
      int maxNumRegStorage = (ts.volMmk - 1)/minNumthread + 1;
      // printf("minNumRegStorage %d maxNumRegStorage %d\n", minNumRegStorage, maxNumRegStorage);

      int bestVal = 0;
      int bestNumRegStorage = 0;
      int bestNumActiveBlock = 0;

      lc.numthread_y = 1;
      lc.numthread_z = 1;
      lc.numblock_x = max(1, ts.volMbar);
      lc.numblock_x = std::min<unsigned int>(gpuMultiProcessorCount * 18, lc.numblock_x);
      lc.numblock_y = 1;
      lc.numblock_z = 1;

      for (lc.numRegStorage=minNumRegStorage; lc.numRegStorage <= maxNumRegStorage; lc.numRegStorage++) {
        lc.numthread_x = ((ts.volMmk - 1) / (gpuWarpSize * lc.numRegStorage) + 1) * gpuWarpSize;

        int numActiveBlock = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
        // int val = numActiveBlock*lc.numthread.x;
        int val = ts.volMmkUsed()*numActiveBlock;
        if (val > bestVal) {
          bestVal = val;
          bestNumRegStorage = lc.numRegStorage;
          bestNumActiveBlock = numActiveBlock;
        }
      }

      //if (bestNumRegStorage == 0) return 0; // cuTT
      if (bestNumRegStorage < 9) return 0;    // avoid small workgroup size; suggested by Xinmin Tian, Intel

      lc.numRegStorage = bestNumRegStorage;
      lc.numthread_x = ((ts.volMmk - 1)/(gpuWarpSize * lc.numRegStorage) + 1) * gpuWarpSize;
      numActiveBlockReturn = bestNumActiveBlock;
    }
    break;

    case PackedSplit:
    {
      // Amount of shared memory required
      lc.shmemsize = ts.shmemAlloc(sizeofType);

      // Check that we're not using too much shared memory per block
      /* DPCT1019:104: local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. */
      if (lc.shmemsize > gpuSharedMemPerBlock) {
        // printf("lc.shmemsize %d prop.sharedMemPerBlock %d\n", lc.shmemsize, prop.sharedMemPerBlock);
        return 0;
      }

      int volMmkWithSplit = (ts.splitDim/ts.numSplit + ((ts.splitDim % ts.numSplit) > 0))*ts.volMmkUnsplit;

      // Min and max number of threads we can use
      int minNumthread = ((volMmkWithSplit - 1)/(gpuWarpSize*MAX_REG_STORAGE) + 1)*gpuWarpSize;
      int maxNumthread = ((volMmkWithSplit - 1)/(gpuWarpSize) + 1)*gpuWarpSize;
      if (minNumthread > gpuMaxThreadsPerBlock) return 0;
      maxNumthread = min(gpuMaxThreadsPerBlock, maxNumthread);
      // printf("minNumthread %d maxNumthread %d\n", minNumthread, maxNumthread);

      // Min and max number of register storage we can use
      int minNumRegStorage = (volMmkWithSplit - 1)/maxNumthread + 1;
      int maxNumRegStorage = (volMmkWithSplit - 1)/minNumthread + 1;
      // printf("minNumRegStorage %d maxNumRegStorage %d\n", minNumRegStorage, maxNumRegStorage);

      int bestVal = 0;
      int bestNumRegStorage = 0;
      int bestNumActiveBlock = 0;

      lc.numthread_y = 1;
      lc.numthread_z = 1;
      lc.numblock_x = ts.numSplit;
      lc.numblock_y = std::max<unsigned int>(1, std::min<unsigned int>((gpuMultiProcessorCount*18)/lc.numblock_x,
			                                                ts.volMbar));
      lc.numblock_z = 1;

      for (lc.numRegStorage=minNumRegStorage; lc.numRegStorage <= maxNumRegStorage; lc.numRegStorage++) {
        lc.numthread_x = ((volMmkWithSplit - 1)/(gpuWarpSize*lc.numRegStorage) + 1)*gpuWarpSize;

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
      lc.numthread_x = ((volMmkWithSplit - 1)/(gpuWarpSize*lc.numRegStorage) + 1)*gpuWarpSize;
      numActiveBlockReturn = bestNumActiveBlock;
    }
    break;

    case Tiled:
    {
      lc.numthread_x = TILEDIM;
      lc.numthread_y = TILEROWS;
      lc.numthread_z = 1;
      lc.numblock_x = ((ts.volMm - 1)/TILEDIM + 1)*((ts.volMk - 1)/TILEDIM + 1);
      lc.numblock_y = 1;
      lc.numblock_z = std::max<unsigned int>(1, std::min<unsigned int>((gpuMultiProcessorCount*8) /
			                    (lc.numblock_x*lc.numblock_y), ts.volMbar));
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;

    case TiledCopy:
    {
      lc.numthread_x = TILEDIM;
      lc.numthread_y = TILEROWS;
      lc.numthread_z = 1;
      lc.numblock_x = ((ts.volMm - 1)/TILEDIM + 1)*((ts.volMkBar - 1)/TILEDIM + 1);
      lc.numblock_y = 1;
      lc.numblock_z = ts.volMbar;
      lc.numblock_z = min((gpuMultiProcessorCount*8)/(lc.numblock_x*lc.numblock_y), lc.numblock_z);
      lc.numblock_z = std::max<unsigned int>(1, lc.numblock_z);
      lc.shmemsize = 0;
      lc.numRegStorage = 0;
    }
    break;
  }

  // /* DPCT1022:105: There is no exact match between the maxGridSize and the max_nd_range size. */
  // if (lc.numblock_x > gpuMaxGridSize[0] ||
  //     lc.numblock_y > gpuMaxGridSize[1] ||
  //     lc.numblock_z > gpuMaxGridSize[2]) return 0;

  // Return the number of active blocks with these settings
  if (numActiveBlockReturn == -1) {
    // Not set, get it
    numActiveBlockReturn = getNumActiveBlock(ts.method, sizeofType, lc, deviceID, prop);
  }
  return numActiveBlockReturn;
}

bool librettKernel(librettPlan_t &plan, void *dataIn, void *dataOut)
{
  LaunchConfig& lc = plan.launchConfig;
  TensorSplit& ts = plan.tensorSplit;

  switch(ts.method) {
    case Trivial:
    {
#if LIBRETT_USES_SYCL
      plan.stream->memcpy(dataOut, dataIn, ts.volMmk * ts.volMbar * plan.sizeofType);
#elif LIBRETT_USES_HIP
      hipCheck(hipMemcpyAsync(dataOut, dataIn, ts.volMmk*ts.volMbar*plan.sizeofType,
        hipMemcpyDefault, plan.stream));
#elif LIBRETT_USES_CUDA
      cudaCheck(cudaMemcpyAsync(dataOut, dataIn, ts.volMmk*ts.volMbar*plan.sizeofType,
        cudaMemcpyDefault, plan.stream));
#endif
    }
    break;

    case Packed:
    {
      switch(lc.numRegStorage) {
        #if LIBRETT_USES_SYCL
        #define CALL0(TYPE, NREG)                                       \
        {auto event = plan.stream->submit([&](sycl::handler &cgh) {     \
          sycl::local_accessor<uint8_t, 1>                              \
            dpct_local_acc_ct1(sycl::range<1>(lc.shmemsize), cgh);      \
                                                                        \
          auto ts_volMmk_ct0 = ts.volMmk;                               \
          auto ts_volMbar_ct1 = ts.volMbar;                             \
          auto ts_sizeMmk_ct2 = ts.sizeMmk;                             \
          auto ts_sizeMbar_ct3 = ts.sizeMbar;                           \
          auto plan_Mmk_ct4 = plan.Mmk;                                 \
          auto plan_Mbar_ct5 = plan.Mbar;                               \
          auto plan_Msh_ct6 = plan.Msh;                                 \
          auto dataIn_ct7 = (TYPE *)dataIn;                             \
          auto dataOut_ct8 = (TYPE *)dataOut;                           \
                                                                        \
          cgh.parallel_for(                                             \
            sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread), \
            [=](sycl::nd_item<3> item) { \
              transposePacked<TYPE, NREG>(                              \
                ts_volMmk_ct0, ts_volMbar_ct1, ts_sizeMmk_ct2, ts_sizeMbar_ct3, \
                plan_Mmk_ct4, plan_Mbar_ct5, plan_Msh_ct6, dataIn_ct7,  \
                dataOut_ct8, item, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());   \
            });                                                         \
        });                                                             \
          event.wait();                                                 \
        }
        #else // CUDA or HIP
          #define CALL0(TYPE, NREG)                                                                \
          transposePacked<TYPE, NREG> <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
              (ts.volMmk, ts.volMbar, ts.sizeMmk, ts.sizeMbar,                                     \
              plan.Mmk, plan.Mbar, plan.Msh, (TYPE *)dataIn, (TYPE *)dataOut)
        #endif // SYCL

        #define CALL(ICASE) case ICASE: if (plan.sizeofType == 4) CALL0(float,  ICASE); \
	                                if (plan.sizeofType == 8) CALL0(double, ICASE); \
                                        if (plan.sizeofType == 16) CALL0(librett_complex,ICASE); break;
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
        #if LIBRETT_USES_SYCL
          #define CALL0(TYPE, NREG)                                                 \
          plan.stream->submit([&](sycl::handler &cgh) {                             \
            sycl::local_accessor<uint8_t, 1>                                        \
                dpct_local_acc_ct1(sycl::range<1>(lc.shmemsize), cgh);              \
                                                                                    \
            auto ts_splitDim_ct0 = ts.splitDim;                                     \
            auto ts_volMmkUnsplit_ct1 = ts.volMmkUnsplit;                           \
            auto ts_volMbar_ct2 = ts.volMbar;                                       \
            auto ts_sizeMmk_ct3 = ts.sizeMmk;                                       \
            auto ts_sizeMbar_ct4 = ts.sizeMbar;                                     \
            auto plan_cuDimMm_ct5 = plan.cuDimMm;                                   \
            auto plan_cuDimMk_ct6 = plan.cuDimMk;                                   \
            auto plan_Mmk_ct7 = plan.Mmk;                                           \
            auto plan_Mbar_ct8 = plan.Mbar;                                         \
            auto plan_Msh_ct9 = plan.Msh;                                           \
            auto dataIn_ct10 = (TYPE *)dataIn;                                      \
            auto dataOut_ct11 = (TYPE *)dataOut;                                    \
                                                                                    \
            cgh.parallel_for(                                                       \
                sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),        \
                [=](sycl::nd_item<3> item) { \
                  transposePackedSplit<TYPE, NREG>(                                 \
                      ts_splitDim_ct0, ts_volMmkUnsplit_ct1, ts_volMbar_ct2,        \
                      ts_sizeMmk_ct3, ts_sizeMbar_ct4, plan_cuDimMm_ct5,            \
                      plan_cuDimMk_ct6, plan_Mmk_ct7, plan_Mbar_ct8, plan_Msh_ct9,  \
                      dataIn_ct10, dataOut_ct11, item,                          \
                      dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());                            \
                });                                                                 \
          }); plan.stream->wait();
        #else // CUDA or HIP
          #define CALL0(TYPE, NREG)                                                                     \
          transposePackedSplit<TYPE, NREG> <<< lc.numblock, lc.numthread, lc.shmemsize, plan.stream >>> \
              (ts.splitDim, ts.volMmkUnsplit, ts. volMbar, ts.sizeMmk, ts.sizeMbar,                     \
              plan.cuDimMm, plan.cuDimMk, plan.Mmk, plan.Mbar, plan.Msh, (TYPE *)dataIn, (TYPE *)dataOut)
        #endif
        #define CALL(ICASE) case ICASE: if (plan.sizeofType == 4) CALL0(float,  ICASE); \
	                                if (plan.sizeofType == 8) CALL0(double, ICASE); \
                                        if (plan.sizeofType == 16) CALL0(librett_complex, ICASE); break;
        #include "calls.h"
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
      #if LIBRETT_USES_SYCL
        #define CALL(TYPE)                                                        \
        plan.stream->submit([&](sycl::handler &cgh) {                             \
                                                                                  \
          auto ts_volMm_TILEDIM_ct0 = ((ts.volMm - 1) / TILEDIM + 1);             \
          auto ts_volMbar_ct1 = ts.volMbar;                                       \
          auto ts_sizeMbar_ct2 = ts.sizeMbar;                                     \
          auto plan_tiledVol_ct3 = plan.tiledVol;                                 \
          auto plan_cuDimMk_ct4 = plan.cuDimMk;                                   \
          auto plan_cuDimMm_ct5 = plan.cuDimMm;                                   \
          auto plan_Mbar_ct6 = plan.Mbar;                                         \
          auto dataIn_ct7 = (TYPE *)dataIn;                                       \
          auto dataOut_ct8 = (TYPE *)dataOut;                                     \
                                                                                  \
          cgh.parallel_for(                                                       \
              sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),        \
              [=](sycl::nd_item<3> item) { \
                transposeTiled<TYPE>(                                             \
                    ts_volMm_TILEDIM_ct0, ts_volMbar_ct1, ts_sizeMbar_ct2,        \
                    plan_tiledVol_ct3, plan_cuDimMk_ct4, plan_cuDimMm_ct5, \
                    plan_Mbar_ct6, dataIn_ct7, dataOut_ct8, item);      \
              });                                                       \
        }); plan.stream->wait();
      #else // CUDA or HIP
        #define CALL(TYPE)                                                                                     \
        transposeTiled<TYPE> <<< lc.numblock, lc.numthread, 0, plan.stream >>>                                 \
            (((ts.volMm - 1)/TILEDIM + 1), ts.volMbar, ts.sizeMbar, plan.tiledVol, plan.cuDimMk, plan.cuDimMm, \
            plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)
      #endif
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
      if (plan.sizeofType == 16) CALL(librett_complex);
      #undef CALL
    }
    break;

    case TiledCopy:
    {
      #if LIBRETT_USES_SYCL
        #define CALL(TYPE)                                                           \
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
              [=](sycl::nd_item<3> item) {    \
                transposeTiledCopy<TYPE>(                                            \
                    ts_volMm_TILEDIM_ct0, ts_volMbar_ct1, ts_sizeMbar_ct2,           \
                    plan_cuDimMk_ct3, plan_cuDimMm_ct4, plan_tiledVol_ct5,           \
                    plan_Mbar_ct6, dataIn_ct7, dataOut_ct8, item);               \
              });                                                                    \
        }); plan.stream->wait();
      #else // CUDA or HIP
        #define CALL(TYPE)                                                                                     \
        transposeTiledCopy<TYPE> <<< lc.numblock, lc.numthread, 0, plan.stream >>>                             \
            (((ts.volMm - 1)/TILEDIM + 1), ts.volMbar, ts.sizeMbar, plan.cuDimMk, plan.cuDimMm, plan.tiledVol, \
            plan.Mbar, (TYPE *)dataIn, (TYPE *)dataOut)
      #endif
      if (plan.sizeofType == 4) CALL(float);
      if (plan.sizeofType == 8) CALL(double);
      if (plan.sizeofType == 16) CALL(librett_complex);
      #undef CALL
    }
    break;

  }

#if LIBRETT_USES_CUDA
  cudaCheck(cudaGetLastError());
#elif LIBRETT_USES_HIP
  hipCheck(hipGetLastError());
#endif
  return true;
}
