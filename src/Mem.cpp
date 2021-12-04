
#ifdef SYCL
#include <CL/sycl.hpp>
#include "dpct/dpct.hpp"
#endif
#include "Mem.h"

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
#ifdef SYCL
void allocate_device_T(void **pp, const size_t len, const size_t sizeofT) try {
#ifdef LIBRETT_HAS_UMPIRE
  *pp = librett_umpire_allocator.allocate(sizeofT*len);
#else
  /*
  DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  gpuCheck((*pp = (void *)sycl::malloc_device(sizeofT * len, dpct::get_default_queue()), 0));
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#else // CUDA
void allocate_device_T(void **pp, const size_t len, const size_t sizeofT) {
#ifdef LIBRETT_HAS_UMPIRE
  *pp = librett_umpire_allocator.allocate(sizeofT*len);
#else
  gpuCheck(cudaMalloc(pp, sizeofT*len));
#endif
}
#endif

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
#ifdef SYCL
void deallocate_device_T(void **pp) try {
#ifdef LIBRETT_HAS_UMPIRE
  librett_umpire_allocator.deallocate((void *) (*pp) );
#else
  if (*pp != NULL) {
    /*
    DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    gpuCheck((sycl::free((void *)(*pp), dpct::get_default_queue()), 0));
    *pp = NULL;
  }
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#else // CUDA
void deallocate_device_T(void **pp) {
#ifdef LIBRETT_HAS_UMPIRE
  librett_umpire_allocator.deallocate((void *) (*pp) );
#else
  if (*pp != NULL) {
    gpuCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }
#endif
}
#endif
