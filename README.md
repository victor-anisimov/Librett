# LibreTT - Tensor Transpose Library

LibreTT is a free Tensor Transpose Library for nVidia and Intel GPU accelerators.

## About

LibreTT incorporates and enhances the original cuTT library, https://github.com/ap-hynninen/cutt, which is no longer supported due to tragic death of its author, Antti-Pekka Hynninen in a car accident. Enhancements include adding thread safety implemented by Valeyev Lab at Virginia Tech, https://github.com/ValeevGroup/cutt and incorporating SYCL support by Argonne National Laboratory.

## Directory Structure
```
CMakeLists.txt      CMake script
LICENSE.md          License to use this software
Makefile            Top-level Makefile
Makefile.cuda       Makefile for CUDA platform
Makefile.sycl       Makefile for SYCL platform
README.md           This file
src/                Source code directory
```

## Compilation

Manual compilation of a stand-alone library:
make cuda    or    make sycl

CMake compilation:
Add -DCMAKE_CUDA_ARCHITECTURES=70 for nVidia V100.
To do: Add -DENABLE_CUDA and -DENABLE_SYCL for nVidia and Intel GPU platforms, respectively.
```
cmake -H. -Bbuild
cd build; make; make librett_test; make librett_bench
```

## The outcome of manual build

* `bin/`     librett_test, librett_bench
* `build/`   temporary placeholder for object files
* `include/` librett.h
* `lib/`     librett.a

## Description

To be added.
