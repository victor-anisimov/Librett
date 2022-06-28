cmake_minimum_required(VERSION 3.22)
project(librett VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_STATIC_LIBRARY_PREFIX "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)

option(ENABLE_CUDA OFF)
option(ENABLE_HIP  OFF)
option(ENABLE_SYCL OFF)

if (NOT (ENABLE_CUDA OR ENABLE_HIP OR ENABLE_SYCL))
  message(FATAL_ERROR "Need to set one of ENABLE_CUDA/ENABLE_HIP/ENABLE_SYCL")
endif()

include(CheckFunctionExists)
option(ENABLE_NO_ALIGNED_ALLOC "Enable aligned_alloc() function implemented in libreTT" OFF)
option(ENABLE_UMPIRE "Enable umpire for memory management" OFF)

# select platform
if(ENABLE_CUDA)
  set(ENABLE_HIP OFF)
  set(ENABLE_SYCL OFF)
  option(CMAKE_CUDA_ARCHITECTURES 70)
  set(LIBRETT_COMPILE_DEFS CUDA)
endif()

if(ENABLE_HIP)
  set(ENABLE_CUDA OFF)
  set(ENABLE_SYCL OFF)
  set(LIBRETT_COMPILE_DEFS HIP)
endif()

if(ENABLE_SYCL)
  set(ENABLE_CUDA OFF)
  set(ENABLE_HIP OFF)
  set(LIBRETT_COMPILE_DEFS SYCL)
endif()

# enable CUDA
if(ENABLE_CUDA)
  message(STATUS "Compiling for CUDA platform")
  enable_language(CUDA)

  set(_CUDA_MIN "10.1")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${_CUDA_MIN})
    message(FATAL_ERROR "CUDA version provided \
    (${CMAKE_CUDA_COMPILER_VERSION}) \
    is insufficient. Need CUDA >= ${_CUDA_MIN})")
  endif()

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

  message(STATUS "CUDA Include Dirs: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  message(STATUS "CUDA Host Compiler: " ${CMAKE_CUDA_HOST_COMPILER})

  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  option(ENABLE_NVTOOLS "Enable nvvp profiling of CPU code" OFF)

  # ENABLE_NVTOOLS
  if(ENABLE_NVTOOLS)
    add_definitions(-DENABLE_NVTOOLS)
    list(APPEND CUDA_NVCC_FLAGS -lnvToolsExt)
    link_libraries(-lnvToolsExt)
    message(STATUS "Current CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
  endif()

endif(ENABLE_CUDA)

#enable HIP
if(ENABLE_HIP)
  message(STATUS "Compiling for HIP platform")
  enable_language(HIP)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D_FORCE_INLINES")
endif(ENABLE_HIP)

#enable SYCL
if(ENABLE_SYCL)
  message(STATUS "Compiling for SYCL platform")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-unnamed-lambda -fsycl-default-sub-group-size=32")
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

add_library(librett::librett ALIAS librett)
target_compile_definitions(librett PUBLIC ${LIBRETT_COMPILE_DEFS})

set(LIBRETT_HEADERS
	src/librett.h
)


# Install

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/librett)

install(TARGETS librett
    EXPORT librett-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties(librett PROPERTIES EXPORT_NAME librett)

#Install headers
install(FILES ${LIBRETT_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/librett)

#Export the targets to a script
install(EXPORT librett-targets
    FILE
        librett-targets.cmake
    NAMESPACE
        librett::
    DESTINATION
        ${INSTALL_CONFIGDIR}
)

#Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/librett-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Setup librett-config.cmake
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/librett-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/librett-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR
)

#Install the config and configversion files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/librett-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/librett-config-version.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

#Export build tree
export(EXPORT librett-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/librett-targets.cmake
    NAMESPACE librett::)
