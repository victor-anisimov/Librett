set( CMAKE_C_COMPILER   clang   )
set( CMAKE_CXX_COMPILER clang++ )

set( CMAKE_CXX_FLAGS "-v -std=c++17 -sycl-std=2020 -fsycl -fsycl-default-sub-group-size 64 -fno-sycl-id-queries-fit-in-int -Wsycl-strict -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a" )

set( CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Default build type: RelWithDebInfo" FORCE )
set( ENABLE_SYCL ON CACHE BOOL "" )
set( ENABLE_TESTS OFF CACHE BOOL "" )
