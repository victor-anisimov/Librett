
set(LIBRETT_SOURCE_FILES
    calls.h
    CudaMem.cpp
    CudaMem.h
    CudaMemcpy.cu
    CudaMemcpy.h
    CudaUtils.cu
    CudaUtils.h
    cutt.cpp
    cutt.h
    cuttGpuModel.cpp
    cuttGpuModel.h
    cuttGpuModelKernel.cu
    cuttGpuModelKernel.h
    cuttkernel.cu
    cuttkernel.h
    cuttplan.h
    cuttplan.cpp
    cuttTimer.cpp
    cuttTimer.h
    cuttTypes.h
    int_vector.h
    TensorTester.cu
    TensorTester.h
    LRUCache.h)


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
install(FILES cutt.h
        DESTINATION include
        )

add_executable(cutt_bench EXCLUDE_FROM_ALL cutt_bench.cpp)
target_link_libraries(cutt_bench PUBLIC librett)

add_executable(cutt_test EXCLUDE_FROM_ALL cutt_test.cpp)
target_link_libraries(cutt_test PUBLIC librett)


