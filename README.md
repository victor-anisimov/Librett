# Librett - Tensor Transpose Library

Librett is a free Tensor Transpose Library for nVidia and Intel GPU accelerators.

## About

Librett incorporates and enhances the original cuTT library.

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
Add -DENABLE_CUDA and -DENABLE_SYCL for nVidia and Intel GPU platforms, respectively.
Add -DCMAKE_CUDA_ARCHITECTURES=70 for nVidia V100
```
cmake -H. -Bbuild
cd build; make; make librett_test; make librett_bench
```

## The outcome of manual build, make -f Makefile.[platform], where [platform] is cuda or sycl

* `bin/`     librett_test, librett_bench
* `build/`   temporary placeholder for object files
* `include/` librett.h
* `lib/`     librett.a

## Description

To be added.
