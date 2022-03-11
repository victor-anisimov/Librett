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
#ifndef LIBRETT_H
#define LIBRETT_H

#ifdef SYCL
  #include <CL/sycl.hpp>
#elif HIP
  #include <hip/hip_runtime.h>
#else // CUDA
  #include <cuda_runtime.h>
#endif
#include "uniapi.h"

// Handle type that is used to store and access librett plans
typedef unsigned int librettHandle;

// Return value
typedef enum librettResult_t {
  LIBRETT_SUCCESS,            // Success
  LIBRETT_INVALID_PLAN,       // Invalid plan handle
  LIBRETT_INVALID_PARAMETER,  // Invalid input parameter
  LIBRETT_INVALID_DEVICE,     // Execution tried on device different than where plan was created
  LIBRETT_INTERNAL_ERROR,     // Internal error
  LIBRETT_UNDEFINED_ERROR,    // Undefined error
} librettResult;

// Initializes LIBRETT
//
// This is only needed for the Umpire allocator's lifetime management:
// - if LIBRETT_HAS_UMPIRE is defined, will grab Umpire's allocator;
// - otherwise this is a no-op
void librettInitialize();

// Finalizes LIBRETT
//
// This is currently a no-op
void librettFinalize();

//
// Create plan
//
// Parameters
// handle            = Returned handle to LIBRETT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
// stream            = CUDA stream (0 if no stream is used)
//
// Returns
// Success/unsuccess code
//
librettResult librettPlan(librettHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  gpuStream_t stream);

//
// Create plan and choose implementation by measuring performance
//
// Parameters
// handle            = Returned handle to LIBRETT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// sizeofType        = Size of the elements of the tensor in bytes (=4 or 8)
// stream            = CUDA stream (0 if no stream is used)
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
//
// Returns
// Success/unsuccess code
//
librettResult librettPlanMeasure(librettHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  gpuStream_t stream, void* idata, void* odata);

//
// Destroy plan
//
// Parameters
// handle            = Handle to the LIBRETT plan
// 
// Returns
// Success/unsuccess code
//
librettResult librettDestroy(librettHandle handle);

//
// Execute plan out-of-place
//
// Parameters
// handle            = Returned handle to LIBRETT plan
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
// 
// Returns
// Success/unsuccess code
//
librettResult librettExecute(librettHandle handle, void* idata, void* odata);

#endif // LIBRETT_H
