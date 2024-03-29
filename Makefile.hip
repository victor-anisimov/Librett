#******************************************************************************
#MIT License
#
#Copyright (c) 2016 Antti-Pekka Hynninen
#Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#*******************************************************************************

#################### User Settings ####################

# C++ compiler
HOST_CC = hipcc
GPU_CC = hipcc

# CUDA compiler
CUDAC = hipcc

# Enable nvvp profiling of CPU code by using "make ENABLE_NVTOOLS=1"
# If aligned_alloc() is not available, use "make NO_ALIGNED_ALLOC=1"

# SM versions for which code is generated must be sm_30 and above
GENCODE_SM35  := -gencode arch=compute_35,code=sm_35
GENCODE_SM50  := -gencode arch=compute_50,code=sm_50
GENCODE_SM52  := -gencode arch=compute_52,code=sm_52
GENCODE_SM61  := -gencode arch=compute_61,code=sm_61
GENCODE_SM70  := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS := $(GENCODE_SM50) $(GENCODE_SM52) $(GENCODE_SM61) $(GENCODE_SM70)

#######################################################

# Detect OS
ifeq ($(shell uname -a|grep Linux|wc -l|tr -d ' '), 1)
OS = linux
endif

ifeq ($(shell uname -a|grep titan|wc -l|tr -d ' '), 1)
OS = linux
endif

ifeq ($(shell uname -a|grep Darwin|wc -l|tr -d ' '), 1)
OS = osx
endif

# Detect x86_64 vs. Power
CPU = unknown

ifeq ($(shell uname -a|grep x86_64|wc -l|tr -d ' '), 1)
CPU = x86_64
endif

ifeq ($(shell uname -a|grep ppc64|wc -l|tr -d ' '), 1)
CPU = ppc64
endif

# Set optimization level
OPTLEV = -O3 -g

# Defines
DEFS = -DLIBRETT_USES_HIP

ifdef ENABLE_NVTOOLS
DEFS += -DENABLE_NVTOOLS
endif

ifdef NO_ALIGNED_ALLOC
DEFS += -DNO_ALIGNED_ALLOC
endif

OBJSLIB = build/librett.o build/plan.o build/kernel.o build/GpuModel.o build/GpuUtils.o build/Timer.o build/GpuModelKernel.o
OBJSTEST1 = build/example.o build/TensorTester.o build/GpuUtils.o build/Timer.o
OBJSTESTX = build/librett_test.o build/TensorTester.o build/GpuUtils.o build/Timer.o
OBJSBENCH = build/librett_bench.o build/TensorTester.o build/GpuUtils.o build/Timer.o build/GpuMemcpy.o
OBJS = $(OBJSLIB) $(OBJSTEST1) $(OBJSTESTX) $(OBJSBENCH)

CUDAROOT = $(subst /bin/,,$(dir $(shell which $(CUDAC))))

CFLAGS = -I${CUDAROOT}/include -Isrc -std=c++17 $(DEFS) $(OPTLEV) -fPIC -D__HIP_PLATFORM_HCC__ -D__HIP_ROCclr__
ifeq ($(CPU),x86_64)
CFLAGS += -march=native
endif

# make hip tests=all
# compile all tests allowed by the hardware platform (sycl, hip, cuda)
ifneq ($(tests),all)
  # compile portable tests only; default
CFLAGS += -DPERFTEST
endif

CUDA_CFLAGS = --amdgpu-target=gfx906,gfx908,gfx90a -std=c++17 $(DEFS) $(OPTLEV) -D_FORCE_INLINES

ifeq ($(OS),osx)
CUDA_LFLAGS = -L$(CUDAROOT)/lib
else
#CUDA_LFLAGS = -L$(CUDAROOT)/lib64
endif

CUDA_LFLAGS += -fPIC -Llib -lrett

ifdef ENABLE_NVTOOLS
CUDA_LFLAGS += -lnvToolsExt
endif

all: create_build lib/librett.a bin/example bin/librett_test bin/librett_bench

create_build:
	mkdir -p build

lib/librett.a: $(OBJSLIB)
	mkdir -p lib
	rm -f lib/librett.a
	ar -cvq lib/librett.a $(OBJSLIB)
	mkdir -p include
	cp -f src/librett.h include/librett.h

bin/example : lib/librett.a $(OBJSTEST1)
	mkdir -p bin
	$(HOST_CC) -o bin/example -L/opt/rocm-3.9.0/lib/ -lamdhip64 $(OBJSTEST1) $(CUDA_LFLAGS)

bin/librett_test : lib/librett.a $(OBJSTESTX)
	mkdir -p bin
	$(HOST_CC) -o bin/librett_test -L/opt/rocm-3.9.0/lib/ -lamdhip64 $(OBJSTESTX) $(CUDA_LFLAGS)

bin/librett_bench : lib/librett.a $(OBJSBENCH)
	mkdir -p bin
	$(HOST_CC) -o bin/librett_bench -L/opt/rocm-3.9.0/lib/ -lamdhip64 $(OBJSBENCH) $(CUDA_LFLAGS)

clean:
	rm -f $(OBJS)
	rm -f build/*.d
	rm -f *~
	rm -f lib/librett.a
	rm -f bin/example
	rm -f bin/librett_test
	rm -f bin/librett_bench

# Pull in dependencies that already exist
-include $(OBJS:.o=.d)

# build/%.o : src/%.cu
# 	$(CUDAC) -c $(CUDA_CFLAGS) -o build/$*.o $<
# 	echo -e 'build/\c' > build/$*.d
# 	$(CUDAC) -M $(CUDA_CFLAGS) $< >> build/$*.d


build/GpuMemcpy.o : src/GpuMemcpy.cpp
	$(CUDAC) -c $(CUDA_CFLAGS) -o $*.o $<
	echo -e 'build/\c' > $*.d
	$(CUDAC) -M $(CUDA_CFLAGS) $< >> $*.d

build/GpuUtils.o : src/GpuUtils.cpp
	$(CUDAC) -c $(CUDA_CFLAGS) -o $*.o $<
	echo -e 'build/\c' > $*.d
	$(CUDAC) -M $(CUDA_CFLAGS) $< >> $*.d

build/TensorTester.o : src/TensorTester.cpp
	$(CUDAC) -c $(CUDA_CFLAGS) -o $*.o $<
	echo -e 'build/\c' > $*.d
	$(CUDAC) -M $(CUDA_CFLAGS) $< >> $*.d

build/GpuModelKernel.o : src/GpuModelKernel.cpp
	$(CUDAC) -c $(CUDA_CFLAGS) -o $*.o $<
	echo -e 'build/\c' > $*.d
	$(CUDAC) -M $(CUDA_CFLAGS) $< >> $*.d

build/kernel.o : src/kernel.cpp
	$(CUDAC) -c $(CUDA_CFLAGS) -o $*.o $<
	echo -e 'build/\c' > $*.d
	$(CUDAC) -M $(CUDA_CFLAGS) $< >> $*.d

build/%.o : src/%.cpp
	$(HOST_CC) -c $(CFLAGS) -o build/$*.o $<
	echo -e 'build/\c' > build/$*.d
	$(HOST_CC) -M $(CFLAGS) $< >> build/$*.d

build/%.o : tests/%.cpp
	$(HOST_CC) -c $(CFLAGS) -o build/$*.o $<
	echo -e 'build/\c' > build/$*.d
	$(HOST_CC) -M $(CFLAGS) $< >> build/$*.d
