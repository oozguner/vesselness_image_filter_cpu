/* 
* thinSegmentationKernels.cuh:
* 
* Created By Russell Jackson
*  2014/7/24
*/


/*This file relies on the following external libraries:
OpenCV 
Eigen  (?????)
*/

//This file defines functions for segmenting the suture thread and the needle from the
//camera images.
#ifndef GPUKERNELS_H
#define GPUKERNELS_H

#include "Third_Party/LibraryIncludesCUDA.local.h"
#include "Third_Party/opencv.h"
#include "../cameraModel.h"
#include "../imageProcCommon.h"
#include "cuda_runtime.h"
#include "opencv2/gpu/gpu.hpp"
#include <opencv2/gpu/stream_accessor.hpp> 

/*GPU kernel defs:*/

using namespace cv::gpu;


__global__ void genGaussHessKernel_XX(PtrStepSzf output,float var,int offset);

__global__ void genGaussHessKernel_XY(PtrStepSzf output,float var,int offset);

__global__ void genGaussHessKernel_YY(PtrStepSzf output,float var,int offset);

__global__ void generateEigenValues(const PtrStepSzf XX,const PtrStepSzf XY,const PtrStepSzf YY,PtrStepSz<float3> output,float betaParam,float cParam);

__global__ void gaussAngBlur(const PtrStepSz<float3> srcMat,PtrStepSz<float3> dstMat,PtrStepSzf gMat,int gaussOff);


#endif