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
#include <stdio.h>
#ifdef ENABLE_NVTOOLS
#include <nvToolsExtCuda.h>
#endif
#include "CudaUtils.h"

//----------------------------------------------------------------------------------------

void set_device_array_async_T(void *data, int value, const size_t ndata,
                              sycl::queue *stream, const size_t sizeofT) try {
  /*
  DPCT1003:126: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((stream->memset(data, value, sizeofT * ndata), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void set_device_array_T(void *data, int value, const size_t ndata,
                        const size_t sizeofT) try {
  /*
  DPCT1003:127: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck(
      (dpct::get_default_queue().memset(data, value, sizeofT * ndata).wait(),
       0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//----------------------------------------------------------------------------------------
//
// Copies memory Host -> Device
//
void copy_HtoD_async_T(const void *h_array, void *d_array, size_t array_len,
                       sycl::queue *stream, const size_t sizeofT) try {
  /*
  DPCT1003:128: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(d_array, h_array, sizeofT * array_len), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void copy_HtoD_T(const void *h_array, void *d_array, size_t array_len,
                 const size_t sizeofT) try {
  /*
  DPCT1003:129: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue()
                 .memcpy(d_array, h_array, sizeofT * array_len)
                 .wait(),
             0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

//----------------------------------------------------------------------------------------
//
// Copies memory Device -> Host
//
void copy_DtoH_async_T(const void *d_array, void *h_array,
                       const size_t array_len, sycl::queue *stream,
                       const size_t sizeofT) try {
  /*
  DPCT1003:130: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(h_array, d_array, sizeofT * array_len), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void copy_DtoH_T(const void *d_array, void *h_array, const size_t array_len,
                 const size_t sizeofT) try {
  /*
  DPCT1003:131: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue()
                 .memcpy(h_array, d_array, sizeofT * array_len)
                 .wait(),
             0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
