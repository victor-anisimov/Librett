# LibreTT - Tensor Transpose Library

LibreTT is a free Tensor Transpose Library for nVidia and Intel GPU accelerators.

## About

LibreTT is a portable Tensor Transpose library that incorporates and superseds the original 
cuTT library, https://github.com/ap-hynninen/cutt, which is no longer supported due to tragic 
death of its author, Antti-Pekka Hynninen in a car accident. Enhancements include adding thread 
safety by Valeyev Lab at Virginia Tech, https://github.com/ValeevGroup/cutt and incorporating 
SYCL support by Argonne National Laboratory.

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

## Manual Compilation

`make cuda`    or    `make sycl`

## The outcome of manual build

* `bin/`     librett_test, librett_bench
* `build/`   temporary placeholder for object files
* `include/` librett.h
* `lib/`     librett.a

## CMake compilation:

CUDA platform is set by default. 

Example of CUDA compilation: `cmake -H. -Bbuild -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`

Example of SYCL compilation: `cmake -H. -Bbuild -DENABLE_SYCL=ON -DCMAKE_CXX_COMPILER=icpx`

After that: `cd build; make; make librett_test; make librett_bench; make test`

## Description

To be added.
