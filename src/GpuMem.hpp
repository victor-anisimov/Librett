/******************************************************************************
MIT License

Copyright (c) 2016 Chong Peng Virginia Tech

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

#ifndef LIBRETTMEM_HPP
#define LIBRETTMEM_HPP

#include "GpuUtils.h"

#ifdef LIBRETT_HAS_UMPIRE
#include <umpire/Umpire.hpp>

// defined in librett.cpp
extern umpire::Allocator librett_umpire_allocator;
#endif

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
template <class T>
void allocate_device(T **pp, const size_t len, gpuStream_t gpuStream) {

#ifdef LIBRETT_HAS_UMPIRE
  *pp = librett_umpire_allocator.allocate(sizeof(T)*len);
#else  // LIBRETT_HAS_UMPIRE
  #if SYCL
  *((void **)pp) = (void *)sycl::malloc_device( sizeof(T)*len, *gpuStream);
  #elif HIP
  hipCheck(hipMalloc((void **)pp, sizeof(T)*len));
  #else // CUDA
  cudaCheck(cudaMalloc((void **)pp, sizeof(T)*len));
  #endif
#endif // LIBRETT_HAS_UMPIRE

}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
template <class T>
void deallocate_device(T **pp, gpuStream_t gpuStream) {

#ifdef LIBRETT_HAS_UMPIRE
  librett_umpire_allocator.deallocate((void *) (*pp) );
#else
  if (*pp != NULL) {
    #if SYCL
      sycl::free( (void *)(*pp), *gpuStream );
    #elif HIP
      hipCheck(hipFree((void *)(*pp)));
    #else // CUDA
      cudaCheck(cudaFree((void *)(*pp)));
    #endif
    *pp = NULL;
  }
#endif // LIBRETT_HAS_UMPIRE

}


#endif //LIBRETTMEM_HPP
