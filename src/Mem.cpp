
#include "Mem.h"

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_device_T(void **pp, const size_t len, const size_t sizeofT) {
#ifdef LIBRETT_HAS_UMPIRE
  *pp = librett_umpire_allocator.allocate(sizeofT*len);
#elif defined(SYCL)
  *pp = (void *)sycl::malloc_device(sizeofT * len, librett::get_default_queue());
#else
  gpuCheck(cudaMalloc(pp, sizeofT*len));
#endif
}

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_device_T(void **pp) {
#ifdef LIBRETT_HAS_UMPIRE
  librett_umpire_allocator.deallocate((void *) (*pp) );
#elif defined(SYCL)
  if (*pp != NULL) {
    sycl::free((void *)(*pp), librett::get_default_queue());
    *pp = NULL;
  }
#else
  if (*pp != NULL) {
    gpuCheck(cudaFree((void *)(*pp)));
    *pp = NULL;
  }
#endif
}

