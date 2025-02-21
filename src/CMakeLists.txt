
set(LIBRETT_SOURCE_FILES
  calls.h
  GpuMem.hpp
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

set(LIBRETT_HEADERS librett.h)

add_library(librett ${LIBRETT_SOURCE_FILES})

if (BUILD_SHARED_LIBS)
  set_target_properties(librett PROPERTIES LIBRARY_OUTPUT_NAME rett)
else()
  set_target_properties(librett PROPERTIES ARCHIVE_OUTPUT_NAME librett)
endif()

if(ENABLE_SYCL)
  if(NOT DEFINED ENABLE_SYCL_HIP OR NOT DEFINED ENABLE_SYCL_CUDA)
    target_compile_options(librett PRIVATE -fsycl -fsycl-targets=intel_gpu_pvc)
    target_link_options(librett PRIVATE -fsycl -fsycl-targets=intel_gpu_pvc)
  endif()
endif(ENABLE_SYCL)

if(ENABLE_CUDA)
  set_source_files_properties(${DEVICE_SOURCE_FILES} PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "-x cu")
endif(ENABLE_CUDA)

if(ENABLE_HIP)
  set_source_files_properties(${DEVICE_SOURCE_FILES} PROPERTIES LANGUAGE HIP)
endif(ENABLE_HIP)

if(ENABLE_UMPIRE)
  target_link_libraries(librett umpire)
  target_compile_definitions(librett PUBLIC LIBRETT_HAS_UMPIRE -DLIBRETT_USES_THIS_UMPIRE_ALLOCATOR=${LIBRETT_USES_THIS_UMPIRE_ALLOCATOR})
endif(ENABLE_UMPIRE)


add_library(librett::librett ALIAS librett)
target_compile_definitions(librett PUBLIC ${LIBRETT_COMPILE_DEFS})
if(ENABLE_CUDA)
  if(LIBRETT_LINK_STATIC_GPU_LIBS)
    target_link_libraries(librett PUBLIC CUDA::cudart_static)
  else()
    target_link_libraries(librett PUBLIC CUDA::cudart)
  endif()
endif()
if(ENABLE_HIP)
  target_link_libraries(librett PUBLIC hip::host)
endif()

# Install

install(TARGETS librett
        EXPORT librett-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

set_target_properties(librett PROPERTIES EXPORT_NAME librett)

# Export include directories. Fixes compatibility with FetchContent.
target_include_directories(librett PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<INSTALL_INTERFACE:include>)

#Install headers
install(FILES ${LIBRETT_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/librett)


