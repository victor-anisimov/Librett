/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/
#ifndef LIBRETTGPUMODELKERNEL_H
#define LIBRETTGPUMODELKERNEL_H
#include "plan.h"

void runCounters(const int warpSize, const int* hostPosData, const int numPosData,
  const int accWidth, const int cacheWidth, int* host_tran, int* host_cl_full, int* host_cl_part, gpuStream_t gpustream);

bool librettGpuModelKernel(librettPlan_t& plan, const int accWidth, const int cacheWidth,
  int& gld_tran, int& gst_tran, int& gld_req, int& gst_req,
  int& cl_full_l2, int& cl_part_l2, int& cl_full_l1, int& cl_part_l1);

#endif // LIBRETTGPUMODELKERNEL_H
