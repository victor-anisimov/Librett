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
#include "GpuMem.h"
#include "GpuModelKernel.h"

#define RESTRICT //__restrict__

// suppress Clang warning about it being unable to unroll a loop
#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wpass-failed"
#endif

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
  __dpct_inline__ void clear() {
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
__dpct_inline__ int tensorPos(const int p, const int rank, const int c,
                              const int d, const int ct,
                              sycl::nd_item<3> item_ct1) {

  const int numLane = item_ct1.get_sub_group().get_local_range().get(0);
  int r = ((p / c) % d) * ct;
#pragma unroll
  for (int i=numLane/2;i >= 1;i/=2) {
    /*
    DPCT1023:0: The DPC++ sub-group does not support mask options for
    shuffle_xor.
    */
    r += item_ct1.get_sub_group().shuffle_xor(r, i);
  }
  return r;

}

//
// Counts number of global memory transactions for a warp that accesses
// memory at pos using warp lanes 0, ..., n - 1
//
__dpct_inline__ int countGlTransactions(const int pos, const int n,
                                        const int accWidth, const int warpLane,
                                        sycl::nd_item<3> item_ct1) {
  int seg0 = pos/accWidth;
  int srcLane = (warpLane == 0 || warpLane >= n) ? (warpLane) : (warpLane - 1);
  int seg1 = item_ct1.get_sub_group().shuffle(seg0, srcLane);
  /*
  DPCT1023:1: The DPC++ sub-group does not support mask options for shuffle.
  */
  int count = sycl::popcount(ballot(item_ct1.get_sub_group(), seg0 != seg1)[0]) + 1;
  count = (n == 0) ? 0 : count;
  return count;
}

//
// Counts number of global memory transactions for a warp that accesses
// memory at pos using warp lanes 0, ..., n - 1
//
__dpct_inline__ int countGlTransactions(const int *segbuf, const int n,
                                        sycl::nd_item<3> item_ct1) {
  int count = 0;
  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range().get(2)) {
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
__dpct_inline__ void countCacheLines(const int pos, const int n,
                                     const int cacheWidth, const int warpLane,
                                     int &cl_full, int &cl_part,
                                     sycl::nd_item<3> item_ct1) {

  int seg = pos/cacheWidth;
  // Lane is at the beginning of a full cache line, if seg0 matches seg0 cacheWidth - 1 away
  int readLane = warpLane + (cacheWidth - 1);
  /*
  DPCT1023:3: The DPC++ sub-group does not support mask options for shuffle.
  */
  int val = (seg == item_ct1.get_sub_group().shuffle(seg, readLane));
  val = (readLane < n) ? val : 0;
  cl_full += val;

  unsigned int valbit = (((val << cacheWidth) - 1)*val) << warpLane;
  // Perform warpSize-way bitwise or
#pragma unroll
  for (int i = item_ct1.get_sub_group().get_local_range().get(0) / 2; i >= 1;
       i /= 2) {
    /*
    DPCT1023:4: The DPC++ sub-group does not support mask options for
    shuffle_xor.
    */
    valbit |= item_ct1.get_sub_group().shuffle_xor(valbit, i);
  }
  // Now: lanes with valbit set are part of a full cache line,
  //      lanes with valbit unset are part of a partial cache line
  int full = (valbit >> warpLane) & 1;

  seg = (warpLane < n) ? seg : -1;
  /*
  DPCT1023:5: The DPC++ sub-group does not support mask options for
  shuffle_down.
  */
  int segP1 = item_ct1.get_sub_group().shuffle_down(seg, 1);
  segP1 = (warpLane + 1 < item_ct1.get_sub_group().get_local_range().get(0))
              ? segP1
              : -1;
  int val2 = ((!full) && seg != segP1);
  cl_part += val2;
}

//
// Counts number of full and partial cache lines for a warp that accesses
// memory at cachelines segbuf[0] ... segbuf[n - 1]
//
__dpct_inline__ void countCacheLines(int *segbuf, const int n,
                                     const int cacheWidth, int &cl_full,
                                     int &cl_part, sycl::nd_item<3> item_ct1) {

  const int topbit = (1 << 31);
  const int lowbits = ~(1 << 31);

  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range().get(2)) {
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
  /*
  DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  item_ct1.barrier();

  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range().get(2)) {
    int seg = segbuf[i];
    int segP1 = (i + 1 < n) ? segbuf[i + 1] : -1;
    int part = ((seg & topbit) == 0);
    int val2 = (part && seg != segP1);
    cl_part += val2;
  }

  // Clear top bits
  /*
  DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance, if there is no access to global memory.
  */
  item_ct1.barrier();
  for (int i = item_ct1.get_local_id(2); i < n;
       i += item_ct1.get_local_range().get(2)) {
    segbuf[i] &= lowbits;
  }

}

//
// Runs countGlTransactions and countCacheLines counters for testing
// Unused values in posData[] are marked with "-1"
//
void runCountersKernel(const int* posData, const int numPosData,
  const int accWidth, const int cacheWidth, int* tranData, int* cl_fullData, int* cl_partData,
  sycl::nd_item<3> item_ct1) {

  const int warpSize = item_ct1.get_sub_group().get_local_range().get(0);
  const int warpLane = item_ct1.get_local_id(2) & (warpSize - 1);

  for (int i = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
       i < numPosData;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    int pos = posData[i];
    int flag = (pos == -1);
    int ffsval = __builtin_ffs(ballot(item_ct1.get_sub_group(), flag)[0]) - 1;
    /*
    DPCT1023:10: The DPC++ sub-group does not support mask options for
    sycl::ext::oneapi::any_of.
    */
    int n = (sycl::ext::oneapi::any_of(item_ct1.get_sub_group(), flag)) ? ffsval : warpSize;
    int tran = countGlTransactions(pos, n, accWidth, warpLane, item_ct1);
    int cl_full = 0;
    int cl_part = 0;
    countCacheLines(pos, n, cacheWidth, warpLane, cl_full, cl_part, item_ct1);
#pragma unroll
    for (int k = warpSize / 2; k >= 1;
         k /= 2) {
      /*
      DPCT1023:11: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      cl_full += item_ct1.get_sub_group().shuffle_xor(cl_full, k);
      /*
      DPCT1023:12: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      cl_part += item_ct1.get_sub_group().shuffle_xor(cl_part, k);
    }
    // avoid multiple threads writing to the same address space
    if(item_ct1.get_sub_group().get_local_id()[0] == 0) {
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
__dpct_inline__ void writeMemStat(const int warpLane, MemStat memStat,
                                  MemStat *RESTRICT glMemStat,
                                  sycl::nd_item<3> item_ct1) {
  for (int i=16;i >= 1;i/=2) {
    // memStat.gld_tran += __shfl_xor_sync(0xffffffff,memStat.gld_tran,i);
    // memStat.gst_tran += __shfl_xor_sync(0xffffffff,memStat.gst_tran,i);
    // memStat.gld_req  += __shfl_xor_sync(0xffffffff,memStat.gld_req,i);
    // memStat.gst_req  += __shfl_xor_sync(0xffffffff,memStat.gst_req,i);
    /*
    DPCT1023:13: The DPC++ sub-group does not support mask options for
    shuffle_xor.
    */
    memStat.cl_full_l2 +=
        item_ct1.get_sub_group().shuffle_xor(memStat.cl_full_l2, i);
    /*
    DPCT1023:14: The DPC++ sub-group does not support mask options for
    shuffle_xor.
    */
    memStat.cl_part_l2 +=
        item_ct1.get_sub_group().shuffle_xor(memStat.cl_part_l2, i);
    /*
    DPCT1023:15: The DPC++ sub-group does not support mask options for
    shuffle_xor.
    */
    memStat.cl_full_l1 +=
        item_ct1.get_sub_group().shuffle_xor(memStat.cl_full_l1, i);
    /*
    DPCT1023:16: The DPC++ sub-group does not support mask options for
    shuffle_xor.
    */
    memStat.cl_part_l1 +=
        item_ct1.get_sub_group().shuffle_xor(memStat.cl_part_l1, i);
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
  MemStat* RESTRICT glMemStat, sycl::nd_item<3> item_ct1) {

  const int warpLane = item_ct1.get_local_id(2) &
                       (item_ct1.get_sub_group().get_local_range().get(0) - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = glMbar[warpLane];
  }

  const int bx = (item_ct1.get_group(2) % numMm) * TILEDIM;
  const int by = (item_ct1.get_group(2) / numMm) * TILEDIM;

  const int xin = bx + item_ct1.get_local_id(2);
  const int yin = by + item_ct1.get_local_id(1);

  const int xout = bx + item_ct1.get_local_id(1);
  const int yout = by + item_ct1.get_local_id(2);

  const unsigned int maskIny = 
      ballot(item_ct1.get_sub_group(), (yin + warpLane < tiledVol.y()))[0] * (xin < tiledVol.x());
  const unsigned int maskOutx =
      ballot(item_ct1.get_sub_group(), (xout + warpLane < tiledVol.x()))[0] * (yout < tiledVol.y());

  const int posMinorIn = xin + yin*cuDimMk;
  const int posMinorOut = yout + xout*cuDimMm;
  const int posInAdd = TILEROWS*cuDimMk;
  const int posOutAdd = TILEROWS*cuDimMm;

  MemStat memStat;
  memStat.clear();

  for (int posMbar = item_ct1.get_group(0); posMbar < volMbar;
       posMbar += item_ct1.get_group_range(0))
  {

    // Compute global memory positions
    int posMajorIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
    int posMajorOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      /*
      DPCT1023:19: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      posMajorIn += item_ct1.get_sub_group().shuffle_xor(posMajorIn, i);
      /*
      DPCT1023:20: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      posMajorOut += item_ct1.get_sub_group().shuffle_xor(posMajorOut, i);
    }
    int posIn = posMajorIn + posMinorIn;
    int posOut = posMajorOut + posMinorOut;

    // Read data into shared memory tile
#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int n = sycl::popcount(ballot(item_ct1.get_sub_group(), maskIny & (1 << j))[0]);
      memStat.gld_tran +=
          countGlTransactions(posIn, n, accWidth, warpLane, item_ct1);
      /*
      DPCT1023:22: The DPC++ sub-group does not support mask options for
      sycl::ext::oneapi::any_of.
      */
      memStat.gld_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
      posIn += posInAdd;
    }

#pragma unroll
    for (int j=0;j < TILEDIM;j += TILEROWS) {
      int n = sycl::popcount(ballot(item_ct1.get_sub_group(), maskOutx & (1 << j))[0]);
      memStat.gst_tran +=
          countGlTransactions(posOut, n, accWidth, warpLane, item_ct1);
      /*
      DPCT1023:24: The DPC++ sub-group does not support mask options for
      sycl::ext::oneapi::any_of.
      */
      memStat.gst_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
      countCacheLines(posOut, n, cacheWidth, warpLane, memStat.cl_full_l2,
                      memStat.cl_part_l2, item_ct1);
      posOut += posOutAdd;
    }

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item_ct1);
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
  MemStat* RESTRICT glMemStat, sycl::nd_item<3> item_ct1, uint8_t *dpct_local) {

  auto shSegOut = (int *)dpct_local;

  const int warpLane = item_ct1.get_local_id(2) &
                       (item_ct1.get_sub_group().get_local_range().get(0) - 1);

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
          item_ct1.get_local_id(2) + j * item_ct1.get_local_range().get(2);
      /*
      DPCT1023:25: The DPC++ sub-group does not support mask options for
      shuffle.
      */
      posMmkIn[j] += ((posMmk / item_ct1.get_sub_group().shuffle(Mmk.c_in, i)) %
                      item_ct1.get_sub_group().shuffle(Mmk.d_in, i)) *
                     item_ct1.get_sub_group().shuffle(Mmk.ct_in, i);
      /*
      DPCT1023:26: The DPC++ sub-group does not support mask options for
      shuffle.
      */
      posMmkOut[j] +=
          ((posMmk / item_ct1.get_sub_group().shuffle(Mmk.c_out, i)) %
           item_ct1.get_sub_group().shuffle(Mmk.d_out, i)) *
          item_ct1.get_sub_group().shuffle(Mmk.ct_out, i);
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

  for (int posMbar = item_ct1.get_group(2); posMbar < volMbar;
       posMbar += item_ct1.get_group_range(2))
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      /*
      DPCT1023:30: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      posMbarOut += item_ct1.get_sub_group().shuffle_xor(posMbarOut, i);
    }

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      /*
      DPCT1023:31: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      posMbarIn += item_ct1.get_sub_group().shuffle_xor(posMbarIn, i);
    }

    // Read from global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item_ct1.get_local_id(2) + j * item_ct1.get_local_range().get(2);
      int posIn = posMbarIn + posMmkIn[j];
      int n = sycl::popcount(ballot(item_ct1.get_sub_group(), posMmk < volMmk)[0]);
      memStat.gld_tran +=
          countGlTransactions(posIn, n, accWidth, warpLane, item_ct1);
      /*
      DPCT1023:33: The DPC++ sub-group does not support mask options for
      sycl::ext::oneapi::any_of.
      */
      memStat.gld_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
    }

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item_ct1.get_local_id(2) + j * item_ct1.get_local_range().get(2);
      int posOut = posMbarOut + posMmkOut[j];
      int n = sycl::popcount(ballot(item_ct1.get_sub_group(), posMmk < volMmk)[0]);
      memStat.gst_tran +=
          countGlTransactions(posOut, n, accWidth, warpLane, item_ct1);
      /*
      DPCT1023:35: The DPC++ sub-group does not support mask options for
      sycl::ext::oneapi::any_of.
      */
      memStat.gst_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
      if (posMmk < volMmk) shSegOut[posMmk] = posOut/cacheWidth;
    }

    /*
    DPCT1065:27: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    countCacheLines(shSegOut, volMmk, cacheWidth, memStat.cl_full_l2,
                    memStat.cl_part_l2, item_ct1);
    // Go from L2 segments to L1 segments
    /*
    DPCT1065:28: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    const int L2toL1 = accWidth/cacheWidth;
    for (int i = item_ct1.get_local_id(2); i < volMmk;
         i += item_ct1.get_local_range().get(2)) {
      shSegOut[i] /= L2toL1;
    }
    /*
    DPCT1065:29: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    countCacheLines(shSegOut, volMmk, accWidth, memStat.cl_full_l1,
                    memStat.cl_part_l1, item_ct1);

    // __syncthreads();
    // memStat.l1_tran += countGlTransactions(shSegOut, volMmk);

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item_ct1);
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
  MemStat* RESTRICT glMemStat, sycl::nd_item<3> item_ct1, uint8_t *dpct_local) {

  auto shSegOut = (int *)dpct_local;

  const int warpLane = item_ct1.get_local_id(2) &
                       (item_ct1.get_sub_group().get_local_range().get(0) - 1);

  // const int plusone = (blockIdx.x < (splitDim % gridDim.x));
  const int p0 = item_ct1.get_group(2) * splitDim / item_ct1.get_group_range(2);
  const int volSplit =
      (item_ct1.get_group(2) + 1) * splitDim / item_ct1.get_group_range(2) - p0;
  const int plusone = volSplit - splitDim / item_ct1.get_group_range(2);

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
      int t = item_ct1.get_local_id(2) + j * item_ct1.get_local_range().get(2);
      /*
      DPCT1023:36: The DPC++ sub-group does not support mask options for
      shuffle.
      */
      posMmkIn[j] += ((t / item_ct1.get_sub_group().shuffle(Mmk.c_in, i)) %
                      item_ct1.get_sub_group().shuffle(Mmk.d_in, i)) *
                     item_ct1.get_sub_group().shuffle(Mmk.ct_in, i);
      /*
      DPCT1023:37: The DPC++ sub-group does not support mask options for
      shuffle.
      */
      posMmkOut[j] += ((t / item_ct1.get_sub_group().shuffle(Mmk.c_out, i)) %
                       item_ct1.get_sub_group().shuffle(Mmk.d_out, i)) *
                      item_ct1.get_sub_group().shuffle(Mmk.ct_out, i);
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

  for (int posMbar = item_ct1.get_group(1); posMbar < volMbar;
       posMbar += item_ct1.get_group_range(1))
  {

    int posMbarOut = ((posMbar/Mbar.c_out) % Mbar.d_out)*Mbar.ct_out;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      /*
      DPCT1023:41: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      posMbarOut += item_ct1.get_sub_group().shuffle_xor(posMbarOut, i);
    }

    int posMbarIn = ((posMbar/Mbar.c_in) % Mbar.d_in)*Mbar.ct_in;
#pragma unroll
    for (int i=16;i >= 1;i/=2) {
      /*
      DPCT1023:42: The DPC++ sub-group does not support mask options for
      shuffle_xor.
      */
      posMbarIn += item_ct1.get_sub_group().shuffle_xor(posMbarIn, i);
    }

    // Read from global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item_ct1.get_local_id(2) + j * item_ct1.get_local_range().get(2);
      int posIn = posMbarIn + posMmkIn[j];
      int n = sycl::popcount(ballot(item_ct1.get_sub_group(), posMmk < volMmkSplit)[0]);
      memStat.gld_tran +=
          countGlTransactions(posIn, n, accWidth, warpLane, item_ct1);
      /*
      DPCT1023:44: The DPC++ sub-group does not support mask options for
      sycl::ext::oneapi::any_of.
      */
      memStat.gld_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
    }

    // Write to global memory
#pragma unroll
    for (int j=0;j < numRegStorage;j++) {
      int posMmk =
          item_ct1.get_local_id(2) + j * item_ct1.get_local_range().get(2);
      int posOut = posMbarOut + posMmkOut[j];
      int n = sycl::popcount(ballot(item_ct1.get_sub_group(), posMmk < volMmkSplit)[0]);
      memStat.gst_tran +=
          countGlTransactions(posOut, n, accWidth, warpLane, item_ct1);
      /*
      DPCT1023:46: The DPC++ sub-group does not support mask options for
      sycl::ext::oneapi::any_of.
      */
      memStat.gst_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
      if (posMmk < volMmkSplit) shSegOut[posMmk] = posOut / cacheWidth;
      // countCacheLines(posOut, n, cacheWidth, warpLane, memStat.cl_full, memStat.cl_part);
    }

    /*
    DPCT1065:38: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    countCacheLines(shSegOut, volMmkSplit, cacheWidth, memStat.cl_full_l2,
                    memStat.cl_part_l2, item_ct1);
    // Go from L2 segments to L1 segments
    /*
    DPCT1065:39: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    const int L2toL1 = accWidth/cacheWidth;
    for (int i = item_ct1.get_local_id(2); i < volMmkSplit;
         i += item_ct1.get_local_range().get(2)) {
      shSegOut[i] /= L2toL1;
    }
    /*
    DPCT1065:40: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    countCacheLines(shSegOut, volMmkSplit, accWidth, memStat.cl_full_l1,
                    memStat.cl_part_l1, item_ct1);

    // __syncthreads();
    // memStat.l1_tran += countGlTransactions(shSegOut, volMmkSplit);

  }

  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item_ct1);
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
  MemStat* RESTRICT glMemStat, sycl::nd_item<3> item_ct1) {

  const int warpLane = item_ct1.get_local_id(2) &
                       (item_ct1.get_sub_group().get_local_range().get(0) - 1);
  TensorConvInOut Mbar;
  Mbar.c_in = 1;
  Mbar.d_in = 1;
  Mbar.c_out = 1;
  Mbar.d_out = 1;
  if (warpLane < sizeMbar) {
    Mbar = gl_Mbar[warpLane];
  }

  const int bx = (item_ct1.get_group(2) % numMm) * TILEDIM;
  const int by = (item_ct1.get_group(2) / numMm) * TILEDIM;

  const int x = bx + item_ct1.get_local_id(2);
  const int y = by + item_ct1.get_local_id(1);

  MemStat memStat;
  memStat.clear();

  for (int posMbar = item_ct1.get_group(0); posMbar < volMbar;
       posMbar += item_ct1.get_group_range(0))
  {

    // Read global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_in, Mbar.d_in, Mbar.ct_in,
                           item_ct1);
      pos0 += x + y*cuDimMk;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos  = pos0  + j*cuDimMk;
	int n = sycl::popcount(
            ballot(item_ct1.get_sub_group(), (x < tiledVol.x()) && (y + j < tiledVol.y()))[0]);
        memStat.gld_tran +=
            countGlTransactions(pos, n, accWidth, warpLane, item_ct1);
        /*
        DPCT1023:48: The DPC++ sub-group does not support mask options for
        sycl::ext::oneapi::any_of.
        */
        memStat.gld_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
      }
    }

    // Write global memory
    {
      int pos0 = tensorPos(posMbar, sizeMbar, Mbar.c_out, Mbar.d_out,
                           Mbar.ct_out, item_ct1);
      pos0 += x + y*cuDimMm;

#pragma unroll
      for (int j=0;j < TILEDIM;j += TILEROWS) {
        int pos = pos0 + j*cuDimMm;
	int n = sycl::popcount(
            ballot(item_ct1.get_sub_group(), (x < tiledVol.x()) && (y + j < tiledVol.y()))[0]);
        memStat.gst_tran +=
            countGlTransactions(pos, n, accWidth, warpLane, item_ct1);
        /*
        DPCT1023:50: The DPC++ sub-group does not support mask options for
        sycl::ext::oneapi::any_of.
        */
        memStat.gst_req += sycl::ext::oneapi::any_of(item_ct1.get_group(), n > 0);
        countCacheLines(pos, n, cacheWidth, warpLane, memStat.cl_full_l2,
                        memStat.cl_part_l2, item_ct1);
      }
    }

  }
  
  // Reduce memStat within thread block and write result to global memory
  writeMemStat(warpLane, memStat, glMemStat, item_ct1);
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
  sycl::queue q = dpct::get_default_queue();
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
  /*
  DPCT1049:51: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  //dpct::get_default_queue().submit([&](sycl::handler &cgh) {
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblock) *
                                           sycl::range<3>(1, 1, nthread),
                                       sycl::range<3>(1, 1, nthread)),
                     [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                       runCountersKernel(devPosData, numPosData, accWidth,
                                         cacheWidth, dev_tran, dev_cl_full,
                                         dev_cl_part, item_ct1);
                     });
  });
  q.wait();
  /*
  DPCT1010:52: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);

  copy_DtoH<int>(dev_tran,    host_tran,    numWarp, &q);
  copy_DtoH<int>(dev_cl_full, host_cl_full, numWarp, &q);
  copy_DtoH<int>(dev_cl_part, host_cl_part, numWarp, &q);
  /*
  DPCT1003:54: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  //cudaCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
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
/*
DPCT1049:55: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
#define CALL0(NREG)                                                            \
  plan.stream->submit([&](sycl::handler &cgh) {                                \
    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,                 \
                   sycl::access::target::local>                                \
        dpct_local_acc_ct1(sycl::range<1>(ts.volMmk * sizeof(int)), cgh);      \
                                                                               \
    auto ts_volMmk_ct0 = ts.volMmk;                                            \
    auto ts_volMbar_ct1 = ts.volMbar;                                          \
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
        [=](sycl::nd_item<3> item_ct1) {                                       \
          countPacked<NREG>(ts_volMmk_ct0, ts_volMbar_ct1, ts_sizeMmk_ct2,     \
                            ts_sizeMbar_ct3, plan_Mmk_ct4, plan_Mbar_ct5,      \
                            accWidth_ct6, cacheWidth_ct7, devMemStat_ct8,      \
                            item_ct1, dpct_local_acc_ct1.get_pointer());       \
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
/*
DPCT1049:56: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
#define CALL0(NREG)                                                            \
  plan.stream->submit([&](sycl::handler &cgh) {                                \
    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,                 \
                   sycl::access::target::local>                                \
        dpct_local_acc_ct1(sycl::range<1>(volMmkSplit * sizeof(int)), cgh);    \
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
    auto accWidth_ct9 = accWidth;                                              \
    auto cacheWidth_ct10 = cacheWidth;                                         \
    auto devMemStat_ct11 = devMemStat;                                         \
                                                                               \
    cgh.parallel_for(                                                          \
        sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),           \
        [=](sycl::nd_item<3> item_ct1) {                                       \
          countPackedSplit<NREG>(                                              \
              ts_splitDim_ct0, ts_volMmkUnsplit_ct1, ts_volMbar_ct2,           \
              ts_sizeMmk_ct3, ts_sizeMbar_ct4, plan_cuDimMm_ct5,               \
              plan_cuDimMk_ct6, plan_Mmk_ct7, plan_Mbar_ct8, accWidth_ct9,     \
              cacheWidth_ct10, devMemStat_ct11, item_ct1,                      \
              dpct_local_acc_ct1.get_pointer());                               \
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

    case Tiled:
    {
      /*
      DPCT1049:57: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
    plan.stream->submit([&](sycl::handler &cgh) {
      auto ts_volMm_TILEDIM_ct0 = ((ts.volMm - 1) / TILEDIM + 1);

      auto tiledVol = plan.tiledVol;
      auto cuDimMk  = plan.cuDimMk;
      auto cuDimMm  = plan.cuDimMm;
      auto Mbar     = plan.Mbar;

      cgh.parallel_for(
          sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),
          [=](sycl::nd_item<3> item_ct1) {
            countTiled(ts_volMm_TILEDIM_ct0, ts.volMbar, ts.sizeMbar,
                       tiledVol, cuDimMk, cuDimMm, Mbar,
                       accWidth, cacheWidth, devMemStat, item_ct1);
          });
    });
    }
    break;

    case TiledCopy:
    {
      /*
      DPCT1049:58: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
    plan.stream->submit([&](sycl::handler &cgh) {
      auto ts_volMm_TILEDIM_ct0 = ((ts.volMm - 1) / TILEDIM + 1);

      auto tiledVol = plan.tiledVol;
      auto cuDimMk  = plan.cuDimMk;
      auto cuDimMm  = plan.cuDimMm;
      auto Mbar     = plan.Mbar;

      cgh.parallel_for(
          sycl::nd_range<3>(lc.numblock * lc.numthread, lc.numthread),
          [=](sycl::nd_item<3> item_ct1) {
            countTiledCopy(ts_volMm_TILEDIM_ct0, ts.volMbar, ts.sizeMbar,
                           cuDimMk, cuDimMm, tiledVol, Mbar,
                           accWidth, cacheWidth, devMemStat, item_ct1);
          });
    });
    }
    break;

  }

  /*
  DPCT1010:59: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);

  MemStat hostMemStat;
  copy_DtoH<MemStat>(devMemStat, &hostMemStat, 1, plan.stream);
  /*
  DPCT1003:60: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((dpct::get_current_device().queues_wait_and_throw(), 0));
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
