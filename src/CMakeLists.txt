
set(LIBRETT_SOURCE_FILES
  calls.h
  GpuMem.cpp
  GpuMem.h
  GpuMemcpy.cpp
  GpuMemcpy.h
  GpuUtils.cpp
  GpuUtils.h
  librett.cpp
  librett.h
  GpuModel.cpp
  GpuModel.h
  GpuModelKernel.cpp
  GpuModelKernel.h
  kernel.cpp
  kernel.h
  plan.cpp
  plan.h
  Timer.cpp
  Timer.h
  Types.h
  int_vector.h
  TensorTester.cpp
  TensorTester.h
  LRUCache.h)

set(DEVICE_SOURCE_FILES
  GpuModelKernel.cpp
  TensorTester.cpp
  GpuMemcpy.cpp
  GpuUtils.cpp
  kernel.cpp)

add_library(librett ${LIBRETT_SOURCE_FILES})

if(ENABLE_CUDA)
  set_source_files_properties(${DEVICE_SOURCE_FILES} PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "-x cu")
endif(ENABLE_CUDA)

if(ENABLE_HIP)
  set_source_files_properties(${DEVICE_SOURCE_FILES} PROPERTIES LANGUAGE HIP)
  # set_property(TARGET librett PROPERTY HIP_ARCHITECTURES gfx906 gfx908)
endif(ENABLE_HIP)

if(ENABLE_UMPIRE)
  target_link_libraries(librett umpire)
  target_compile_definitions(librett PUBLIC LIBRETT_HAS_UMPIRE -DLIBRETT_USES_THIS_UMPIRE_ALLOCATOR=${LIBRETT_USES_THIS_UMPIRE_ALLOCATOR})
endif(ENABLE_UMPIRE)



