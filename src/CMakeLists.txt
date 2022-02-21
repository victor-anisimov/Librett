
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

if(ENABLE_CUDA)
    set_source_files_properties(TensorTester.cpp   PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "-x cu")
    set_source_files_properties(GpuModelKernel.cpp PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "-x cu")
    set_source_files_properties(GpuMemcpy.cpp      PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "-x cu")
    set_source_files_properties(GpuUtils.cpp       PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "-x cu")
    set_source_files_properties(kernel.cpp         PROPERTIES LANGUAGE CUDA COMPILE_FLAGS "-x cu")
endif(ENABLE_CUDA)

add_library(librett ${LIBRETT_SOURCE_FILES})
if(ENABLE_UMPIRE)
    target_link_libraries(librett umpire)
    target_compile_definitions(librett PUBLIC LIBRETT_HAS_UMPIRE -DLIBRETT_USES_THIS_UMPIRE_ALLOCATOR=${LIBRETT_USES_THIS_UMPIRE_ALLOCATOR})
endif()

# install library and include
install(TARGETS librett
        ARCHIVE
            DESTINATION lib
        LIBRARY
            DESTINATION lib
        )
install(FILES librett.h
        DESTINATION include
        )

add_executable(librett_bench EXCLUDE_FROM_ALL librett_bench.cpp)
target_link_libraries(librett_bench PUBLIC librett)

add_executable(librett_test EXCLUDE_FROM_ALL librett_test.cpp)
target_link_libraries(librett_test PUBLIC librett)

add_executable(test EXCLUDE_FROM_ALL test.cpp)
target_link_libraries(test PUBLIC librett)

