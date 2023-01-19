# LibreTT - Tensor Transpose Library

LibreTT is a free Tensor Transpose Library for nVidia, AMD, and Intel GPU accelerators.

## About

LibreTT is a portable Tensor Transpose library that incorporates and superseds the original 
cuTT library, https://github.com/ap-hynninen/cutt, which is no longer supported due to tragic 
death of its author, Antti-Pekka Hynninen in a car accident. Enhancements include adding thread 
safety by Valeyev Lab at Virginia Tech, https://github.com/ValeevGroup/cutt, incorporating 
AMD GPU support by Dmitry Lyakh, Oak Ridge National Laboratory, https://github.com/DmitryLyakh/hipTT 
followed by implementation of warpsize 64 by Alessandro Fanfarillo, Advanced Micro Devices, Inc., 
and adding SYCL support by Argonne National Laboratory.

## Credits
```
cuTT     algorithms: Dmitry Lyakh (ORNL), Antti-Pekka Hynninen (ORNL)
cuTT     original implementation: Antti-Pekka Hynninen (ORNL)
cuTT     thread safety: Ed Valeev (VT), Dmitry Lyakh (ORNL)
hipTT    port of cuTT: Luke Roskop (HPE), Dmitry Lyakh (ORNL)
hipTT    working version: Alessandro Fanfarillo (AMD)
LibreTT  SYCL port and CUDA,AMD,INTEL platform integration: Victor Anisimov (ANL)
LibreTT  CMake enhancements and complex-double datatype: Ajay Panyala (PNNL)
LibreTT  SYCL portability and remove dependency on DPCT: Abhishek Bagusetty (ANL)
```

## Directory Structure
```
CMakeLists.txt      CMake script
LICENSE.md          License to use this software
Makefile            Top-level Makefile
Makefile.cuda       Makefile for CUDA platform
Makefile.hip        Makefile for HIP  platform
Makefile.sycl       Makefile for SYCL platform
README.md           This file
src/                Source code directory
```

## Manual Compilation

`make platform [option=-DPERFTEST]` where platform = cuda, hip, or sycl. The option=-DPERFTEST compiles the light version of librett_test suitable for performance analysis. EXAMPLE: `make hip option=-DPERFTEST`

## The outcome of manual build

* `bin/`     librett_test, librett_bench, example
* `build/`   temporary placeholder for object files
* `include/` librett.h
* `lib/`     librett.a

## CMake compilation:

Example of CUDA compilation: `cmake -H. -Bbuild -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`

Example of HIP compilation: `cmake -H. -Bbuild -DENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_HIP_ARCHITECTURES=gfx908`

Example of SYCL compilation: `cmake -H. -Bbuild -DENABLE_SYCL=ON -DCMAKE_CXX_COMPILER=icpx`

Testing options: `-DENABLE_TESTS=ON` (default)

## Testing

`Manual build`: Execute `bin/librett_test` without arguments.  
`CMake build`: Execute `ctest`

## Description

To be added.
