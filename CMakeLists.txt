cmake_minimum_required(VERSION 3.10)
project(librett)
set(CMAKE_STATIC_LIBRARY_PREFIX "")

# enable CXX
enable_language(CXX)
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif (NOT CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif (NOT CMAKE_CXX_EXTENSIONS)

# select platform
if(ENABLE_CUDA AND ENABLE_SYCL)
  set(ENABLE_SYCL OFF)
endif(ENABLE_CUDA AND ENABLE_SYCL)

if(NOT ENABLE_CUDA AND NOT ENABLE_SYCL)
  set(ENABLE_CUDA ON)
  set(ENABLE_SYCL OFF)
endif(NOT ENABLE_CUDA AND NOT ENABLE_SYCL)

# enable CUDA
if(ENABLE_CUDA)
  message(STATUS "Compiling for CUDA platform")
  enable_language(CUDA)

  if (NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
  endif (NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if (NOT CMAKE_CUDA_EXTENSIONS)
    set(CMAKE_CUDA_EXTENSIONS OFF)
  endif (NOT CMAKE_CUDA_EXTENSIONS)

  message(STATUS "CUDA Include Dirs: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CUDA Host Compiler: " ${CMAKE_CUDA_HOST_COMPILER})

  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)
  option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in libreTT" OFF)
  option(ENABLE_UMPIRE "Enable umpire for memory management" OFF)
  include(CheckFunctionExists)

  # ENABLE_NVTOOLS
  if(ENABLE_NVTOOLS)
    add_definitions(-DENABLE_NVTOOLS)
    list(APPEND CUDA_NVCC_FLAGS -lnvToolsExt)
    link_libraries(-lnvToolsExt)
  endif()

  message(STATUS "Current CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
endif(ENABLE_CUDA)

#enable SYCL
if(ENABLE_SYCL)
  message(STATUS "Compiling for SYCL platform")

  option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in libreTT" OFF)
  option(ENABLE_UMPIRE "Enable umpire for memory management" OFF)
  include(CheckFunctionExists)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-unnamed-lambda -O3 -DSYCL")
endif(ENABLE_SYCL)

# ENABLE_NO_ALIGNED_ALLOC
if(ENABLE_NO_ALIGNED_ALLOC)
    add_definitions(-DNO_ALIGNED_ALLOC)
else()
    # this is available in C11 N.B. std::aligned_alloc is available in C++17
    CHECK_FUNCTION_EXISTS(aligned_alloc HAVE_ALIGNED_ALLOC)
    if(NOT HAVE_ALIGNED_ALLOC)
        add_definitions(-DNO_ALIGNED_ALLOC)
    endif()
endif()

# ENABLE_UMPIRE
if (ENABLE_UMPIRE)
    find_package(umpire REQUIRED)
    if (NOT LIBRETT_USES_THIS_UMPIRE_ALLOCATOR)
	    set(LIBRE_USES_THIS_UMPIRE_ALLOCATOR "UM")
    endif(NOT LIBRETT_USES_THIS_UMPIRE_ALLOCATOR)
    message(STATUS "Will use Umpire allocator named \"${LIBRETT_USES_THIS_UMPIRE_ALLOCATOR}\"")
endif ()

add_subdirectory(src)

