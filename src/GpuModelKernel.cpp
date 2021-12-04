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
#include "dpct/dpct.hpp"
#include "Utils.h"
#include "Mem.h"
#include "GpuModelKernel.h"

#define RESTRICT //__restrict__

// suppress Clang warning about it being unable to unroll a loop
#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wpass-failed"
#endif

#define __sycl_inline__ __inline__ __attribute__((always_inline))

//
// Global memory access statistics
//
struct MemStat {
  int gld_tran;
  int gst_tran;
  int gld_req;
  int gst_req;
  int cl_full_l2;
  int cl_part_l2;
  int cl_full_l1;
  int cl_part_l1;
  // int l1_tran;
  __sycl_inline__ void clear() {
    gld_tran = 0;
    gst_tran = 0;
    gld_req = 0;
    gst_req = 0;
    cl_full_l2 = 0;
    cl_part_l2 = 0;
    cl_full_l1 = 0;
    cl_part_l1 = 0;
    // l1_tran = 0;
  }
};

//
// Returns scalar tensor position. Each lane has the same p
// NOTE: c and d on inactive warps must be 1 !!
//
__sycl_inline__ int tensorPos(const int p, const int rank, const int c,
                              const int d, const int ct,
                              sycl::nd_item<3>& item) {
  sycl::sub_group sg = item.get_sub_group();
  const int numLane = sg.get_local_range().get(0);
  int r = ((p / c) % d) * ct;
#pragma unroll
  for (int i=numLane/2;i >= 1;i/=2) {
    r += sg.shuffle_xor(r, i);
  }
  return r;

}

//
// Counts number of global memory transactions for a warp that accesses
// memory at pos using warp lanes 0, ..., n - 1
//
__sycl_inline__ int countGlTransactions(const int pos, const int n,
                                        const int accWidth, const int warpLane,
                                        sycl::nd_item<3>& item) {
  sycl::sub_group sg = item.get_sub_group();

  int seg0 = pos/accWidth;
  int srcLane = (warpLane == 0 || warpLane >= n) ? (warpLane) : (warpLane - 1);
  int seg1 = sg.shuffle(seg0, srcLane);
  int count = sycl::popcount(ballot(sg, seg0 != seg1)[0]) + 1;
  count = (n == 0) ? 0 : count;
  return count;
}

//
// Counts number of global memory transactions for a warp that accesses
// memory at pos using warp lanes 0, ..., n - 1
//
__sycl_inline__ int countGlTransactions(const int *segbuf, const int n,
                                        sycl::nd_item<3>& item) {
  int count = 0;
  for (int i = item.get_local_id(2); i < n;
       i += item.get_local_range().get(2)) {
    int seg      = segbuf[i];
    int seg_prev = (i - 1 >= 0) ? segbuf[i - 1] : -1;
    count += (seg != seg_prev);
  }
  return count;
}

//
// Counts number of full and partial cache lines for a warp that accesses per warp
// memory at pos using warp lanes 0, ..., n - 1
//
__sycl_inline__ void countCacheLines(const int pos, const int n,
                                     const int cacheWidth, const int warpLane,
                                     int &cl_full, int &cl_part,
                                     sycl::nd_item<3>& item) {

  sycl::sub_group sg = item.get_sub_group();
  size_t warpSize = sg.get_local_range().get(0);

  int seg = pos/cacheWidth;
  // Lane is at the beginning of a full cache line, if seg0 matches seg0 cacheWidth - 1 away
  int readLane = warpLane + (cacheWidth - 1);
  int val = (seg == sg.shuffle(seg, readLane));
  val = (readLane < n) ? val : 0;
  cl_full += val;

  unsigned int valbit = (((val << cacheWidth) - 1)*val) << warpLane;
  // Perform warpSize-way bitwise or
#pragma unroll
  for (int i = warpSize / 2; i >= 1; i /= 2) {
    valbit |= sg.shuffle_xor(valbit, i);
  }
  // Now: lanes with valbit set are part of a full cache line,
  //      lanes with valbit unset are part of a partial cache line
  int full = (valbit >> warpLane) & 1;

  seg = (warpLane < n) ? seg : -1;
  int segP1 = sg.shuffle_down(seg, 1);
  segP1 = (warpLane + 1 < warpSize) ? segP1 : -1;
  int val2 = ((!full) && seg != segP1);
  cl_part += val2;
}

//
// Counts number of full and partial cache lines for a warp that accesses
// memory at cachelines segbuf[0] ... segbuf[n - 1]
//
__sycl_inline__ void countCacheLines(int *segbuf, const int n,
                                     const int cacheWidth, int &cl_full,
                                     int &cl_part, sycl::nd_item<3>& item) {

  sycl::group work_grp = item.get_group();
  size_t threadIdx_x = item.get_local_id(2);
  size_t blockDim_x = item.get_local_range(2);

  const int topbit = (1 << 31);
  const int lowbits = ~(1 << 31);

  for (int i = threadIdx_x; i < n; i += blockDim_x) {
    // seg[i] is at the beginning of a full cache line, if seg[i] matches seg[i + cacheWidth - 1]
    int i1 = i + (cacheWidth - 1);
    int val = 0;
    if (i1 < n) val = ((segbuf[i] & lowbits) == (segbuf[i1] & lowbits));
    cl_full += val;
    // Mark full cache lines with top bit set to 1
    if (val) {
      for (int j=0;j < cacheWidth;j++) {
        if (i + j < n) segbuf[i + j] |= topbit;
      }
    }
  }
  sycl::group_barrier(work_grp);

  for (int i = threadIdx_x; i < n; i += blockDim_x) {
    int seg = segbuf[i];
    int segP1 = (i + 1 < n) ? segbuf[i + 1] : -1;
    int part = ((seg & topbit) == 0);
    int val2 = (part && seg != segP1);
    cl_part += val2;
  }

  // Clear top bits
  sycl::group_barrier(work_grp);
  for (int i = threadIdx_x; i < n; i += blockDim_x) {
    segbuf[i] &= lowbits;
  }

}

//
// Runs countGlTransactions and countCacheLines counters for testing
// Unused values in posData[] are marked with "-1"
//
void runCountersKernel(const int* posData, const int numPosData,
  const int accWidth, const int cacheWidth, int* tranData, int* cl_fullData, int* cl_partData,
  sycl::nd_item<1>& item) {

  sycl::sub_group sg = item.get_sub_group();

  const int warpSize = sg.get_local_range().get(0);
  const int warpLane = item.get_local_id(0) & (warpSize - 1);

  for (int i = item.get_global_id(0); i < numPosData; i += item.get_global_range(0)) {
    int pos = posData[i];
    int flag = (pos == -1);
    int ffsval = __builtin_ffs(ballot(sg, flag)[0]) - 1;
    int n = (sycl::ONEAPI::any_of(sg, flag)) ? ffsval : warpSize;
    int tran = countGlTransactions(pos, n, accWidth, warpLane, item);
    int cl_full = 0;
    int cl_part = 0;
    countCacheLines(pos, n, cacheWidth, warpLane, cl_full, cl_part, item);
#pragma unroll
    for (int k = warpSize / 2; k >= 1; k /= 2) {
      cl_full += sg.shuffle_xor(cl_full, k);
      cl_part += sg.shuffle_xor(cl_part, k);
    }
    // avoid multiple threads writing to the same address space
    if(sg.get_local_id()[0] == 0) {
      int j = i / warpSize;
      tranData[j] = tran;
      cl_fullData[j] = cl_full;
      cl_partData[j] = cl_part;
    }
  }

}

//
// Reduce memStat within warp and write result to global memory
// NOTE: Not super-efficient since every warp does atomicAdd().
//
__sycl_inline__ void writeMemStat(const int warpLane, MemStat memStat,
                                  MemStat *RESTRICT glMemStat,
                                  sycl::nd_item<3>& item) {
  sycl::sub_group sg = item.get_sub_group();

  for (int i=16;i >= 1;i/=2) {
    // memStat.gld_tran += __shfl_xor_sync(0xffffffff,memStat.gld_tran,i);
    // memStat.gst_tran += __shfl_xor_sync(0xffffffff,memStat.gst_tran,i);
    // memStat.gld_req  += __shfl_xor_sync(0xffffffff,memStat.gld_req,i);
    // memStat.gst_req  += __shfl_xor_sync(0xffffffff,memStat.gst_req,i);
    memStat.cl_full_l2 += sg.shuffle_xor(memStat.cl_full_l2, i);
    memStat.cl_part_l2 += sg.shuffle_xor(memStat.cl_part_l2, i);
    memStat.cl_full_l1 += sg.shuffle_xor(memStat.cl_full_l1, i);
    memStat.cl_part_l1 += sg.shuffle_xor(memStat.cl_part_l1, i);
    // memStat.l1_tran     += __shfl_xor_sync(0xffffffff,memStat.l1_tran,i);
  }
  if (warpLane == 0) {
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->gld_tran)))
        .fetch_add(memStat.gld_tran);
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->gst_tran)))
        .fetch_add(memStat.gst_tran);
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->gld_req)))
        .fetch_add(memStat.gld_req);
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->gst_req)))
        .fetch_add(memStat.gst_req);
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->cl_full_l2)))
        .fetch_add(memStat.cl_full_l2);
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->cl_part_l2)))
        .fetch_add(memStat.cl_part_l2);
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->cl_full_l1)))
        .fetch_add(memStat.cl_full_l1);
    sycl::atomic<int>(sycl::global_ptr<int>(&(glMemStat->cl_part_l1)))
        .fetch_add(memStat.cl_part_l1);
    // atomicAdd(&(glMemStat->l1_tran), memStat.l1_tran);
  }
}

//
// Transpose when Mm and Mk don't overlap and contain only single rank
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMk-1)/TILEDIM+1), 1, plan.volMbar);
//
void
countTiled(
  const int numMm, const int volMbar, const int sizeMbar,
  const sycl::int2 tiledVol, const int cuDimMk, const int cuDimMm,
  const TensorConvInOut* RESTRICT glMbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat, sycl::nd_item<3>& item) {

  sycl::group work_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();
  size_t warpSize = sg.get_local_range().get(0);
  size_t threadIdx_x = item.get_local_id(2);
  size_t threadIdx_y = item.get_local_id(1);

  const int warpLane = threadIdx_x & (warpSize - 1);
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

  const int xin = bx + threadIdx_x;
  const int yin = by + threadIdx_y;

  const int xout = bx + threadIdx_y;
  const int yout = by + threadIdx_x;

  const unsigned int maskIny =
      ballot(sg, (yin + warpLane < tiledVol.y()))[0] * (xin < tiledVol.x());
  const unsigned int maskOutx =
      ballot(sg, (xout + warpLane < tiledVol.x()))[0] * (yout < tiledVol.y());

  const int posMinorIn = xin + yin*cuDimMk;
  const int posMinorOut = yout + xout*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  MemStat memStat;
  memStat.clear();

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

    // Read data into shared memory tile
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int n = sycl::popcount(ballot(sg, maskIny & (1 << j))[0]);
      memStat.gld_tran +=
          countGlTransactions(posIn, n, accWidth, warpLane, item);
      memStat.gld_req += sycl::ONEAPI::any_of(work_grp, n > 0);
      posIn += posInAdd;
    }

#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int n = sycl::popcount(ballot(sg, maskOutx & (1 << j))[0]);
      memStat.gst_tran +=
          countGlTransactions(posOut, n, accWidth, warpLane, item);
      memStat.gst_req += sycl::ONEAPI::any_of(work_grp, n > 0);
      countCacheLines(posOut, n, cacheWidth, warpLane, memStat.cl_full_l2,
                      memStat.cl_part_l2, item);
      posOut += posOutAdd;
    }

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item);
}

//
// Packed transpose. Thread block loads plan.volMmk number of elements
//
template <int numRegStorage>
void

countPacked(
  const int volMmk, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const TensorConvInOut* RESTRICT gl_Mmk,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat, sycl::nd_item<3>& item, uint8_t *dpct_local) {

  sycl::group work_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();

  auto shSegOut = (int *)dpct_local;

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

  // Pre-compute tensor positions in Mmk
  // 3*numRegStorage registers
  int posMmkIn[numRegStorage];
  int posMmkOut[numRegStorage];
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j] = 0;
    posMmkOut[j] = 0;
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

  MemStat memStat;
  memStat.clear();

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

    // Read from global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item.get_local_id(2) + j * item.get_local_range().get(2);
      int posIn = posMbarIn + posMmkIn[j];
      int n = sycl::popcount(ballot(sg, posMmk < volMmk)[0]);
      memStat.gld_tran +=
          countGlTransactions(posIn, n, accWidth, warpLane, item);
      memStat.gld_req += sycl::ONEAPI::any_of(work_grp, n > 0);
    }

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item.get_local_id(2) + j * item.get_local_range().get(2);
      int posOut = posMbarOut + posMmkOut[j];
      int n = sycl::popcount(ballot(sg, posMmk < volMmk)[0]);
      memStat.gst_tran +=
          countGlTransactions(posOut, n, accWidth, warpLane, item);
      memStat.gst_req += sycl::ONEAPI::any_of(work_grp, n > 0);
      if (posMmk < volMmk) shSegOut[posMmk] = posOut/cacheWidth;
    }

    sycl::group_barrier(work_grp);
    countCacheLines(shSegOut, volMmk, cacheWidth, memStat.cl_full_l2, memStat.cl_part_l2, item);
    // Go from L2 segments to L1 segments
    sycl::group_barrier(work_grp);
    const int L2toL1 = accWidth/cacheWidth;
    for (int i = item.get_local_id(2); i < volMmk;
         i += item.get_local_range().get(2)) {
      shSegOut[i] /= L2toL1;
    }
    sycl::group_barrier(work_grp);
    countCacheLines(shSegOut, volMmk, accWidth, memStat.cl_full_l1,
                    memStat.cl_part_l1, item);

    // __syncthreads();
    // memStat.l1_tran += countGlTransactions(shSegOut, volMmk);

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item);
}

//
// Packed method with a split rank
//
// dim nthread(((volMmkWithSplit - 1)/(prop.warpSize*lc.numRegStorage) + 1)*prop.warpSize, 1, 1)
// dim nblock(ts.numSplit, min(256, max(1, ts.volMbar)), 1)
//
template <int numRegStorage>
void
countPackedSplit(
  const int splitDim, const int volMmkUnsplit, const int volMbar,
  const int sizeMmk, const int sizeMbar,
  const int cMmSplit, const int cMkSplit,
  const TensorConvInOut* RESTRICT glMmk,
  const TensorConvInOut* RESTRICT glMbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat, sycl::nd_item<1>& item, uint8_t *dpct_local) {

  sycl::group work_grp = item.get_group();
  sycl::sub_group sg = item.get_sub_group();
  size_t threadIdx_x = item.get_local_id(2);

  auto shSegOut = (int *)dpct_local;

  const int warpLane = threadIdx_x &
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
#pragma unroll
  for (int j=0;j < numRegStorage;j++) {
    posMmkIn[j]  = posMmkIn0;
    posMmkOut[j] = posMmkOut0;
  }
  for (int i=0;i < sizeMmk;i++) {
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int t = threadIdx_x + j * item.get_local_range().get(2);
      posMmkIn[j] += ((t / sg.shuffle(Mmk.c_in, i)) %
                      sg.shuffle(Mmk.d_in, i)) *
                     sg.shuffle(Mmk.ct_in, i);
      posMmkOut[j] += ((t / sg.shuffle(Mmk.c_out, i)) %
                       sg.shuffle(Mmk.d_out, i)) *
                      sg.shuffle(Mmk.ct_out, i);
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

  MemStat memStat;
  memStat.clear();

  for (int posMbar = item.get_group(1); posMbar < volMbar;
       posMbar += item.get_group_range(1))
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
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          threadIdx_x + j * item.get_local_range().get(2);
      int posIn = posMbarIn + posMmkIn[j];
      int n = sycl::popcount(ballot(sg, posMmk < volMmkSplit)[0]);
      memStat.gld_tran +=
          countGlTransactions(posIn, n, accWidth, warpLane, item);
      memStat.gld_req += sycl::ONEAPI::any_of(work_grp, n > 0);
    }

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          threadIdx_x + j * item.get_local_range().get(2);
      int posOut = posMbarOut + posMmkOut[j];
      int n = sycl::popcount(ballot(sg, posMmk < volMmkSplit)[0]);
      memStat.gst_tran +=
          countGlTransactions(posOut, n, accWidth, warpLane, item);
      memStat.gst_req += sycl::ONEAPI::any_of(work_grp, n > 0);
      if (posMmk < volMmkSplit) shSegOut[posMmk] = posOut / cacheWidth;
      // countCacheLines(posOut, n, cacheWidth, warpLane, memStat.cl_full, memStat.cl_part);
    }

    sycl::group_barrier(work_grp);
    countCacheLines(shSegOut, volMmkSplit, cacheWidth, memStat.cl_full_l2,
                    memStat.cl_part_l2, item);
    // Go from L2 segments to L1 segments
    sycl::group_barrier(work_grp);
    const int L2toL1 = accWidth/cacheWidth;
    for (int i = threadIdx_x; i < volMmkSplit;
         i += item.get_local_range().get(2)) {
      shSegOut[i] /= L2toL1;
    }
    sycl::group_barrier(work_grp);
    countCacheLines(shSegOut, volMmkSplit, accWidth, memStat.cl_full_l1,
                    memStat.cl_part_l1, item);

    // sycl::group_barrier(work_grp);
    // memStat.l1_tran += countGlTransactions(shSegOut, volMmkSplit);

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item);
}

//
// Transpose when the lead dimension is the same, e.g. (1, 2, 3) -> (1, 3, 2)
//
//  dim3 numthread(TILEDIM, TILEROWS, 1);
//  dim3 numblock( ((plan.volMm-1)/TILEDIM+1)*((plan.volMkBar-1)/TILEDIM+1), 1, plan.volMbar);
//
void

countTiledCopy(
  const int numMm, const int volMbar, const int sizeMbar,
  const int cuDimMk, const int cuDimMm,
  const sycl::int2 tiledVol,
  const TensorConvInOut* RESTRICT gl_Mbar,
  const int accWidth, const int cacheWidth,
  MemStat* RESTRICT glMemStat, sycl::nd_item<3>& item) {

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

  MemStat memStat;
  memStat.clear();

  for (int posMbar = item.get_group(0); posMbar < volMbar;
       posMbar += item.get_group_range(0))
  {

    // Read global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in,
                           item);
      pos0 += x + y*cuDimMk;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos  = pos0  + j*cuDimMk;
	int n = sycl::popcount(
            ballot(sg, (x < tiledVol.x()) && (y + j < tiledVol.y()))[0]);
        memStat.gld_tran +=
            countGlTransactions(pos, n, accWidth, warpLane, item);
        memStat.gld_req += sycl::ONEAPI::any_of(item.get_group(), n > 0);
      }
    }

    // Write global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out,
                           Mbar.ct_out, item);
      pos0 += x + y*cuDimMm;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
	int n = sycl::popcount(
            ballot(sg, (x < tiledVol.x()) && (y + j < tiledVol.y()))[0]);
        memStat.gst_tran +=
            countGlTransactions(pos, n, accWidth, warpLane, item);
        memStat.gst_req += sycl::ONEAPI::any_of(item.get_group(), n > 0);
        countCacheLines(pos, n, cacheWidth, warpLane, memStat.cl_full_l2,
                        memStat.cl_part_l2, item);
      }
    }

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item);
}

//######################################################################################
//######################################################################################
//######################################################################################

void runCounters(const int warpSize, const int *hostPosData,
                 const int numPosData, const int accWidth, const int cacheWidth,
                 int *host_tran, int *host_cl_full, int *host_cl_part) try {

  const int numWarp = numPosData/warpSize;

  int* devPosData;
  allocate_device<int>(&devPosData, numPosData);
  sycl::queue q = librett::get_default_queue();
  copy_HtoD<int>(hostPosData, devPosData, numPosData, &q);
  q.wait();

  int* dev_tran;
  int* dev_cl_full;
  int* dev_cl_part;
  allocate_device<int>(&dev_tran, numWarp);
  allocate_device<int>(&dev_cl_full, numWarp);
  allocate_device<int>(&dev_cl_part, numWarp);

  //int nthread = 512;
  int nthread = 64;
  int nblock = (numPosData - 1)/nthread + 1;

  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nblock * nthread),
				   sycl::range<1>(nthread)),
		 [=](sycl::nd_item<1>& item) [[intel::reqd_sub_group_size(32)]] {
		   runCountersKernel(devPosData, numPosData, accWidth,
				     cacheWidth, dev_tran, dev_cl_full,
				     dev_cl_part, item);
		 });
  q.wait();

  copy_DtoH<int>(dev_tran,    host_tran,    numWarp, &q);
  copy_DtoH<int>(dev_cl_full, host_cl_full, numWarp, &q);
  copy_DtoH<int>(dev_cl_part, host_cl_part, numWarp, &q);
  /*
  DPCT1003:54: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //gpuCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
  q.wait();

  deallocate_device<int>(&dev_tran);
  deallocate_device<int>(&dev_cl_full);
  deallocate_device<int>(&dev_cl_part);

  deallocate_device<int>(&devPosData);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool librettGpuModelKernel(librettPlan_t &plan, const int accWidth,
                        const int cacheWidth, int &gld_tran, int &gst_tran,
                        int &gld_req, int &gst_req, int &cl_full_l2,
                        int &cl_part_l2, int &cl_full_l1, int &cl_part_l1) try {

  LaunchConfig& lc = plan.launchConfig;
  TensorSplit& ts = plan.tensorSplit;

  MemStat* devMemStat;
  allocate_device<MemStat>(&devMemStat, 1);
  set_device_array<MemStat>(devMemStat, 0, 1, plan.stream);

  switch(ts.method) {
    case Trivial:
    {
      return false;
    }

    case Packed:
    {
      switch(lc.numRegStorage) {
#define CALL0(NREG)                                                            \
  plan.stream->submit([&](sycl::handler &cgh) {                                \
    localAcc<uint8_t, 1> local_acc(sycl::range<1>(ts.volMmk * sizeof(int)), cgh);      \
                                                                               \
    auto ts_volMmk_ct0 = ts.volMmk;                                            \
    auto ts_volMbar = ts.volMbar;                                          \
    auto ts_sizeMmk_ct2 = ts.sizeMmk;                                          \
    auto ts_sizeMbar_ct3 = ts.sizeMbar;                                        \
    auto plan_Mmk_ct4 = plan.Mmk;                                              \
    auto plan_Mbar_ct5 = plan.Mbar;                                            \
    auto accWidth_ct6 = accWidth;                                              \
    auto cacheWidth_ct7 = cacheWidth;                                          \
    auto devMemStat_ct8 = devMemStat;                                          \
                                                                               \
    cgh.parallel_for(                                                          \
        sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),           \
        [=](sycl::nd_item<3> item) {                                       \
          countPacked<NREG>(ts_volMmk_ct0, ts_volMbar, ts_sizeMmk_ct2,     \
                            ts_sizeMbar_ct3, plan_Mmk_ct4, plan_Mbar_ct5,      \
                            accWidth_ct6, cacheWidth_ct7, devMemStat_ct8,      \
                            item, local_acc.get_pointer());       \
        });                                                                    \
  });
#define CALL(ICASE) case ICASE: CALL0(ICASE); break
#include "calls.h"
        default:
        printf("librettGpuModelKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case PackedSplit:
    {

      // Calculate max. volume of split Mmk
      const int volSplit = (ts.splitDim/ts.numSplit) + ((ts.splitDim % ts.numSplit) != 0);
      const int volMmkSplit = volSplit*ts.volMmkUnsplit;

      switch(lc.numRegStorage) {
#define CALL0(NREG)                                                            \
  plan.stream->submit([&](sycl::handler &cgh) {                                \
    localAcc<uint8_t, 1> local_acc(sycl::range<1>(volMmkSplit * sizeof(int)), cgh);    \
                                                                               \
    auto ts_splitDim_ct0 = ts.splitDim;                                        \
    auto ts_volMmkUnsplit = ts.volMmkUnsplit;                              \
    auto ts_volMbar_ct2 = ts.volMbar;                                          \
    auto ts_sizeMmk_ct3 = ts.sizeMmk;                                          \
    auto ts_sizeMbar_ct4 = ts.sizeMbar;                                        \
    auto plan_cuDimMm_ct5 = plan.cuDimMm;                                      \
    auto plan_cuDimMk_ct6 = plan.cuDimMk;                                      \
    auto plan_Mmk_ct7 = plan.Mmk;                                              \
    auto plan_Mbar_ct8 = plan.Mbar;                                            \
    auto accWidth_ct9 = accWidth;                                              \
    auto cacheWidth0 = cacheWidth;                                         \
    auto devMemStat1 = devMemStat;                                         \
                                                                               \
    q.parallel_for(							\
        sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),           \
        [=](sycl::nd_item<3> item) {                                       \
          countPackedSplit<NREG>(                                              \
              ts_splitDim_ct0, ts_volMmkUnsplit, ts_volMbar_ct2,           \
              ts_sizeMmk_ct3, ts_sizeMbar_ct4, plan_cuDimMm_ct5,               \
              plan_cuDimMk_ct6, plan_Mmk_ct7, plan_Mbar_ct8, accWidth_ct9,     \
              cacheWidth0, devMemStat1, item,                      \
              local_acc.get_pointer());                               \
        });                                                                    \
#define CALL(ICASE) case ICASE: CALL0(ICASE); break
#include "calls.h"
        default:
        printf("librettGpuModelKernel no template implemented for numRegStorage %d\n", lc.numRegStorage);
        return false;
#undef CALL
#undef CALL0
      }

    }
    break;

    case Tiled:
    {
    plan.stream->submit([&](sycl::handler &cgh) {
      auto ts_volMm_TILEDIM_ct0 = ((ts.volMm - 1) / TILEDIM + 1);

      auto tiledVol = plan.tiledVol;
      auto cuDimMk  = plan.cuDimMk;
      auto cuDimMm  = plan.cuDimMm;
      auto Mbar     = plan.Mbar;

      cgh.parallel_for(
          sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),
          [=](sycl::nd_item<3>& item) {
            countTiled(ts_volMm_TILEDIM_ct0, ts.volMbar, ts.sizeMbar,
                       tiledVol, cuDimMk, cuDimMm, Mbar,
                       accWidth, cacheWidth, devMemStat, item);
          });
    });
    }
    break;

    case TiledCopy:
    {
    plan.stream->submit([&](sycl::handler &cgh) {
      auto ts_volMm_TILEDIM_ct0 = ((ts.volMm - 1) / TILEDIM + 1);

      auto tiledVol = plan.tiledVol;
      auto cuDimMk  = plan.cuDimMk;
      auto cuDimMm  = plan.cuDimMm;
      auto Mbar     = plan.Mbar;

      cgh.parallel_for(
          sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),
          [=](sycl::nd_item<3> item) {
            countTiledCopy(ts_volMm_TILEDIM_ct0, ts.volMbar, ts.sizeMbar,
                           cuDimMk, cuDimMm, tiledVol, Mbar,
                           accWidth, cacheWidth, devMemStat, item);
          });
    });
    }
    break;

  }

  /*
  DPCT1010:59: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  gpuCheck(0);

  MemStat hostMemStat;
  copy_DtoH<MemStat>(devMemStat, &hostMemStat, 1, plan.stream);
  /*
  DPCT1003:60: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  gpuCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
  deallocate_device<MemStat>(&devMemStat);

  gld_tran   = hostMemStat.gld_tran;
  gst_tran   = hostMemStat.gst_tran;
  gld_req    = hostMemStat.gld_req;
  gst_req    = hostMemStat.gst_req;
  cl_full_l2 = hostMemStat.cl_full_l2;
  cl_part_l2 = hostMemStat.cl_part_l2;
  cl_full_l1 = hostMemStat.cl_full_l1;
  cl_part_l1 = hostMemStat.cl_part_l1;
  // l1_tran    = hostMemStat.l1_tran;

  return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
