
#ifdef SYCL
#include <sycl/sycl.hpp>
#endif

#include "GpuMem.h"

//----------------------------------------------------------------------------------------
//
// Allocate gpu memory
// pp = memory pointer
// len = length of the array
//
void allocate_device_T(void **pp, const size_t len, const size_t sizeofT)
#ifdef SYCL
try
#endif
{
#ifdef LIBRETT_HAS_UMPIRE
    *pp = librett_umpire_allocator.allocate(sizeofT*len);
#else  // LIBRETT_HAS_UMPIRE
  #if SYCL
    *pp = (void *)sycl::malloc_device(sizeofT * len, dpct::get_default_queue());
  #elif HIP
    hipCheck(hipMalloc(pp, sizeofT*len));
  #else // CUDA
    cudaCheck(cudaMalloc(pp, sizeofT*len));
  #endif
#endif // LIBRETT_HAS_UMPIRE
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

//----------------------------------------------------------------------------------------
//
// Deallocate gpu memory
// pp = memory pointer
//
void deallocate_device_T(void **pp)
#ifdef SYCL
try 
#endif
{
#ifdef LIBRETT_HAS_UMPIRE
  librett_umpire_allocator.deallocate((void *) (*pp) );
#else
  if (*pp != NULL) {
    #if SYCL
      sycl::free((void *)(*pp), dpct::get_default_queue());
    #elif HIP
      hipCheck(hipFree((void *)(*pp)));
    #else // CUDA
      cudaCheck(cudaFree((void *)(*pp)));
    #endif
    *pp = NULL;
  }
#endif // LIBRETT_HAS_UMPIRE
}
#ifdef SYCL
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif
