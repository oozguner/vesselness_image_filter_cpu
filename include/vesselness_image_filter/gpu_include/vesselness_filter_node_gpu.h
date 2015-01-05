/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014 Case Western Reserve University
 *    Russell C Jackson <rcj33@case.edu>
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Case Western Reserve Univeristy, nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef IMAGESEGMENTGPUH
#define IMAGESEGMENTGPUH

#include <vesselness_image_filter_base.h>
#include <vesselness_image_filter_kernels.h>



//Converts a single image into a displayable RGB format.
void convertSegmentImageGPU(const Mat&,Mat&);

//This class extends the basic VesselnessNode based on using a GPU to complete the actual processing.
class VesselnessNodeGPU: public VesselnessNodeBase {

private:
    /* private semi-static class members */

    //Input and output information
    gpu::GpuMat inputG;

    gpu::GpuMat outputG;



    //Intermediates:
    gpu::GpuMat cXX;
    gpu::GpuMat cXY;
    gpu::GpuMat cYY;

    gpu::GpuMat inputGreyG;
    gpu::GpuMat inputFloat255G;
    gpu::GpuMat ones;
    gpu::GpuMat inputFloat1G;

    gpu::GpuMat preOutput;

    gpu::GpuMat scaled;
    gpu::GpuMat scaledU8;
    gpu::GpuMat dispOut;


    //Gauss kernels
    gpu::GpuMat tempGPU_XX;
    gpu::GpuMat tempGPU_XY;
    gpu::GpuMat tempGPU_YY;
    gpu::GpuMat gaussG;


    //Mat topKernel;
    Mat tempCPU_XX;
    Mat tempCPU_XY;
    Mat tempCPU_YY;


    Mat srcMats;
    Mat dstMats;

    //status booleans
    bool kernelReady;
    bool allocatedPageLock;
    bool allocatedGPUMem;
    bool allocatedKernels;


    void  setKernels();
    void  updateKernels();


    void allocatePageLock(int,int);
    void deallocatePageLock();

    void allocateGPUMem();
    void deallocateGPUMem();


    gpu::CudaMem srcMatMem;
    gpu::CudaMem dstMatMem;
    //gpu::CudaMem dispMatMem;

    cv::gpu::Stream streamInfo;
    cudaStream_t cudaStream;


    //segmenting thread.
    void segmentingThread();

    //Update object parameters.
    //
    void updateKernels(const segmentThinParam &);


    //this is a paged locked output data.
    Mat pageLockedOutput;
    //this is a paged locked output data.
    Mat pageLockedInput;


public:


    //This function needs to operate at peak speed:
    VesselnessNodeGPU(segmentThinParam); //constructor
    VesselnessNodeGPU();    //default constructor
    ~VesselnessNodeGPU();   //deconstructor

    //inherited required functions:
    void segmentImage(const Mat &, Mat &);
    void allocateMem(int,int);

    void deallocateMem();
    void  initKernels();


};





#endif
