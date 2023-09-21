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

#include <stdio.h>

#ifdef ENABLE_NVTOOLS
  #include <nvToolsExtCuda.h>
#endif

#include "GpuUtils.h"

//----------------------------------------------------------------------------------------

void set_device_array_async_T(void *data, int value, const size_t ndata,
                              gpuStream_t& stream, const size_t sizeofT)
{
  #if LIBRETT_USES_SYCL
    stream->memset(data, value, sizeofT * ndata);
  #elif LIBRETT_USES_HIP
    hipCheck(hipMemsetAsync(data, value, sizeofT*ndata, stream));
  #elif LIBRETT_USES_CUDA
    cudaCheck(cudaMemsetAsync(data, value, sizeofT*ndata, stream));
  #endif
}

void set_device_array_sync_T(void *data, int value, const size_t ndata,
                             gpuStream_t& stream, const size_t sizeofT)
{
  #if LIBRETT_USES_SYCL
    stream->memset(data, value, sizeofT * ndata).wait();
  #elif LIBRETT_USES_HIP
    hipCheck(hipMemset(data, value, sizeofT*ndata));
  #elif LIBRETT_USES_CUDA
    cudaCheck(cudaMemset(data, value, sizeofT*ndata));
  #endif
}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
void copy_HtoD_async_T(const void *h_array, void *d_array, size_t array_len,
                       gpuStream_t& stream, const size_t sizeofT)
{
  #if LIBRETT_USES_SYCL
    stream->memcpy(d_array, h_array, sizeofT * array_len);
  #elif LIBRETT_USES_HIP
    hipCheck(hipMemcpyAsync(d_array, h_array, sizeofT*array_len, hipMemcpyDefault, stream));
  #elif LIBRETT_USES_CUDA
    cudaCheck(cudaMemcpyAsync(d_array, h_array, sizeofT*array_len, cudaMemcpyDefault, stream));
  #endif
}

void copy_HtoD_sync_T(const void *h_array, void *d_array, size_t array_len,
                       gpuStream_t& stream, const size_t sizeofT)
{
  #if LIBRETT_USES_SYCL
    stream->memcpy(d_array, h_array, sizeofT * array_len).wait();
  #elif LIBRETT_USES_HIP
    hipCheck(hipMemcpy(d_array, h_array, sizeofT*array_len, hipMemcpyDefault));
  #elif LIBRETT_USES_CUDA
    cudaCheck(cudaMemcpy(d_array, h_array, sizeofT*array_len, cudaMemcpyDefault));
  #endif
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
void copy_DtoH_async_T(const void *d_array, void *h_array, const size_t array_len,
		       gpuStream_t& stream, const size_t sizeofT)
{
  #if LIBRETT_USES_SYCL
    stream->memcpy(h_array, d_array, sizeofT * array_len);
  #elif LIBRETT_USES_HIP
    hipCheck(hipMemcpyAsync(h_array, d_array, sizeofT*array_len, hipMemcpyDefault, stream));
  #elif LIBRETT_USES_CUDA
    cudaCheck(cudaMemcpyAsync(h_array, d_array, sizeofT*array_len, cudaMemcpyDefault, stream));
  #endif
}

void copy_DtoH_sync_T(const void *d_array, void *h_array, const size_t array_len,
                      gpuStream_t& stream, const size_t sizeofT)
{
  #if LIBRETT_USES_SYCL
    stream->memcpy(h_array, d_array, sizeofT * array_len).wait();
  #elif LIBRETT_USES_HIP
    hipCheck(hipMemcpy(h_array, d_array, sizeofT*array_len, hipMemcpyDefault));
  #else
    cudaCheck(cudaMemcpy(h_array, d_array, sizeofT*array_len, cudaMemcpyDefault));
  #endif
}

//----------------------------------------------------------------------------------------
#ifdef ENABLE_NVTOOLS
void gpuRangeStart(const char *range_name) {
  static int color_id=0;
  nvtxEventAttributes_t att;
  att.version = NVTX_VERSION;
  att.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  att.colorType = NVTX_COLOR_ARGB;
  if (color_id == 0) {
    att.color = 0xFFFF0000;
  } else if (color_id == 1) {
    att.color = 0xFF00FF00;
  } else if (color_id == 2) {
    att.color = 0xFF0000FF;
  } else if (color_id == 3) {
    att.color = 0xFFFF00FF;
  }
  color_id++;
  if (color_id > 3) color_id = 0;
  att.messageType = NVTX_MESSAGE_TYPE_ASCII;
  att.message.ascii = range_name;
  nvtxRangePushEx(&att);
}

void gpuRangeStop() {
  nvtxRangePop();
}
#endif

void DeviceReset() {
  #if LIBRETT_USES_SYCL
  // does nothing
  #elif LIBRETT_USES_HIP
    hipCheck(hipSetDevice(0));
    hipCheck(hipDeviceReset());
  #elif LIBRETT_USES_CUDA
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaDeviceReset());
  #endif
}
