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
#include "Memcpy.h"

// suppress Clang warning about it being unable to unroll a loop
#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wpass-failed"
#endif

const int numthread = 64;

// -----------------------------------------------------------------------------------
//
// Copy using scalar loads and stores
//
template <typename T>
void scalarCopyKernel(const int n, const T* data_in, T* data_out,
                      sycl::nd_item<3> item_ct1) {

  for (int i = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
       i < n;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    data_out[i] = data_in[i];
  }

}
template <typename T>
void scalarCopy(const int n, const T *data_in, T *data_out,
                sycl::queue *stream) {

  int numblock = (n - 1)/numthread + 1;
  // numblock = min(65535, numblock);
  // numblock = min(256, numblock);

  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numblock) *
                                           sycl::range<3>(1, 1, numthread),
                                       sycl::range<3>(1, 1, numthread)),
                     [=](sycl::nd_item<3> item_ct1) {
                       scalarCopyKernel<T>(n, data_in, data_out, item_ct1);
                     });
  });

  /*
  DPCT1010:123: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);
}
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
//
// Copy using vectorized loads and stores
//
template <typename T>
void vectorCopyKernel(const int n, T* data_in, T* data_out,
                      sycl::nd_item<3> item_ct1) {

  // Maximum vector load is 128 bits = 16 bytes
  const int vectorLength = 16/sizeof(T);

  int idx = item_ct1.get_local_id(2) +
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2);

  // Vector elements
  for (int i = idx; i < n / vectorLength;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    reinterpret_cast<sycl::int4 *>(data_out)[i] =
        reinterpret_cast<sycl::int4 *>(data_in)[i];
  }

  // Remaining elements
  for (int i = idx + (n / vectorLength) * vectorLength; i < n;
       i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2) +
            item_ct1.get_local_id(2)) {
    data_out[i] = data_in[i];
  }

}

template <typename T>
void vectorCopy(const int n, T *data_in, T *data_out, sycl::queue *stream) {

  const int vectorLength = 16/sizeof(T);

  int numblock = (n/vectorLength - 1)/numthread + 1;
  // numblock = min(65535, numblock);
  int shmemsize = 0;

  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numblock) *
                                           sycl::range<3>(1, 1, numthread),
                                       sycl::range<3>(1, 1, numthread)),
                     [=](sycl::nd_item<3> item_ct1) {
                       vectorCopyKernel<T>(n, data_in, data_out, item_ct1);
                     });
  });

  /*
  DPCT1010:124: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);
}
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
//
// Copy using vectorized loads and stores
//
template <int numElem>
void memcpyFloatKernel(const int n, sycl::float4 *data_in,
                       sycl::float4 *data_out, sycl::nd_item<3> item_ct1) {
  int index = item_ct1.get_local_id(2) + numElem * item_ct1.get_group(2) *
                                             item_ct1.get_local_range().get(2);
  sycl::float4 a[numElem];
#pragma unroll
  for (int i=0;i < numElem;i++) {
    if (index + i * item_ct1.get_local_range().get(2) < n) a[i] =
        data_in[index + i * item_ct1.get_local_range().get(2)];
  }
#pragma unroll
  for (int i=0;i < numElem;i++) {
    if (index + i * item_ct1.get_local_range().get(2) < n)
        data_out[index + i * item_ct1.get_local_range().get(2)] = a[i];
  }
}

template <int numElem>
void memcpyFloatLoopKernel(const int n, sycl::float4 *data_in,
                           sycl::float4 *data_out, sycl::nd_item<3> item_ct1) {
  for (int index =
           item_ct1.get_local_id(2) +
           item_ct1.get_group(2) * numElem * item_ct1.get_local_range().get(2);
       index < n; index += numElem * item_ct1.get_group_range(2) *
                           item_ct1.get_local_range().get(2))
  {
    sycl::float4 a[numElem];
#pragma unroll
    for (int i=0;i < numElem;i++) {
      if (index + i * item_ct1.get_local_range().get(2) < n) a[i] =
          data_in[index + i * item_ct1.get_local_range().get(2)];
    }
#pragma unroll
    for (int i=0;i < numElem;i++) {
      if (index + i * item_ct1.get_local_range().get(2) < n)
          data_out[index + i * item_ct1.get_local_range().get(2)] = a[i];
    }
  }
}

#define NUM_ELEM 2
void memcpyFloat(const int n, float *data_in, float *data_out,
                 sycl::queue *stream) {

  int numblock = (n/(4*NUM_ELEM) - 1)/numthread + 1;
  int shmemsize = 0;
  stream->submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, numblock) *
                                           sycl::range<3>(1, 1, numthread),
                                       sycl::range<3>(1, 1, numthread)),
                     [=](sycl::nd_item<3> item_ct1) {
                       memcpyFloatKernel<NUM_ELEM>(
                           n / 4, (sycl::float4 *)data_in,
                           (sycl::float4 *)data_out, item_ct1);
                     });
  });

  // int numblock = 64;
  // int shmemsize = 0;
  // memcpyFloatLoopKernel<NUM_ELEM> <<< numblock, numthread, shmemsize, stream >>>
  // (n/4, (float4 *)data_in, (float4 *)data_out);

  /*
  DPCT1010:125: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);
}
// -----------------------------------------------------------------------------------

// Explicit instances
template void scalarCopy<int>(const int n, const int *data_in, int *data_out,
                              sycl::queue *stream);
template void scalarCopy<long long int>(const int n,
                                        const long long int *data_in,
                                        long long int *data_out,
                                        sycl::queue *stream);
template void vectorCopy<int>(const int n, int *data_in, int *data_out,
                              sycl::queue *stream);
template void vectorCopy<long long int>(const int n, long long int *data_in,
                                        long long int *data_out,
                                        sycl::queue *stream);
void memcpyFloat(const int n, float *data_in, float *data_out,
                 sycl::queue *stream);
