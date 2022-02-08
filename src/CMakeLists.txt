
if(ENABLE_CUDA)
  set(LIBRETT_SOURCE_FILES
    calls.h
    GpuMem.cpp
    GpuMem.h
    GpuMemcpy.cu
    GpuMemcpy.h
    Utils.cu
    Utils.h
    librett.cpp
    librett.h
    GpuModel.cpp
    GpuModel.h
    GpuModelKernel.cu
    GpuModelKernel.h
    kernel.cu
    kernel.h
    plan.h
    plan.cpp
    Timer.cpp
    Timer.h
    Types.h
    int_vector.h
    TensorTester.cu
    TensorTester.h
    LRUCache.h)
endif(ENABLE_CUDA)

if(ENABLE_SYCL)
  set(LIBRETT_SOURCE_FILES
    calls.h
    GpuMem.cpp
    GpuMem.h
    GpuMemcpy.cpp
    GpuMemcpy.h
    Utils.cpp
    Utils.h
    librett.cpp
    librett.h
    GpuModel.cpp
    GpuModel.h
    GpuModelKernel.cpp
    GpuModelKernel.h
    kernel.cpp
    kernel.h
    plan.h
    plan.cpp
    Timer.cpp
    Timer.h
    Types.h
    int_vector.h
    TensorTester.cpp
    TensorTester.h
    LRUCache.h)
endif(ENABLE_SYCL)

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


