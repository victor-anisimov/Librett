
set(LIBRETT_TESTS librett_test librett_bench example)

# build the following executables
foreach(_exec ${LIBRETT_TESTS})
  add_executable(${_exec} "${_exec}.cpp")
  target_link_libraries(${_exec} PUBLIC librett)
  target_include_directories(${_exec} PUBLIC ${PROJECT_SOURCE_DIR}/src)
endforeach()

# add AoT compilation for Intel GPU
if(ENABLE_SYCL)
  if(ENABLE_SYCL_HIP)
    foreach(_exec ${LIBRETT_TESTS})
      target_compile_options(${_exec} PRIVATE -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${CMAKE_HIP_ARCHITECTURES} -DPERFTEST)
      target_link_options(${_exec} PRIVATE -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${CMAKE_HIP_ARCHITECTURES})
    endforeach()
  elseif(ENABLE_SYCL_CUDA)
    foreach(_exec ${LIBRETT_TESTS})
      target_compile_options(${_exec} PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend '--cuda-gpu-arch=sm_${CMAKE_CUDA_ARCHITECTURES}' -DPERFTEST)
      target_link_options(${_exec} PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend '--cuda-gpu-arch=sm_${CMAKE_CUDA_ARCHITECTURES}')
    endforeach()
  else()
    foreach(_exec ${LIBRETT_TESTS})
      target_compile_options(${_exec} PRIVATE -fsycl -fsycl-targets=intel_gpu_pvc -DPERFTEST)
      target_link_options(${_exec} PRIVATE -fsycl -fsycl-targets=intel_gpu_pvc)
    endforeach()
  endif(ENABLE_SYCL_HIP)
endif(ENABLE_SYCL)

foreach(_exec ${LIBRETT_TESTS})
  add_test(NAME ${_exec} COMMAND ${_exec})
endforeach()
