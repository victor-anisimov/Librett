# Librett - Tensor Transpose Library

Librett is a free Tensor Transpose Library for nVidia and Intel GPU accelerators.

## About

Librett incorporates and enhances the original cuTT library.

## Directory Structure
```
CMakeLists.txt      CMake script
LICENSE.md          License to use this software
Makefile.cuda       Makefile for CUDA platform
Makefile.sycl       Makefile for SYCL platform
README.md           This file
src/                Source code directory
```

## Compilation
Add -DENABLE_CUDA and -DENABLE_SYCL for nVidia and Intel GPU platforms, respectively.
```
cmake -Bbuild ./CMakeLists.txt
cd build; make
```

## The outcome of manual build, make -f Makefile.[platform], where [platform] is cuda or sycl

* `bin/`     cutt_test, cutt_bench
* `build/`   temporary placeholder for object files
* `include/` cutt.h
* `lib/`     libcutt.a

## Description

To be added.
