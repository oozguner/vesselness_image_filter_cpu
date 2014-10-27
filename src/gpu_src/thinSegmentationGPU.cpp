/* 
* imageSegmentationGPU.cpp:
* 
* Created By Russell Jackson
*  07/24/2014
*/

#include "thinSegmentationGPU.h"
#include "thinSegmentationKernels.cuh"
#include <opencv2/gpu/stream_accessor.hpp> 
#include "thinSegmentationGPU.h"
#include "../boost.h"

/*This file relies on the following external libraries:
OpenCV
Eigen
cuda
*/

//This file defines functions for segmenting the suture thread and the needle from the
//camera images using an object with GPU support.
//This contains the C++ compilable functions.

ThinSegmentationGPU::ThinSegmentationGPU(segmentThinParam inputParams){

    //preProcess Params
    hessParam.variance = inputParams.preProcess.variance;
    hessParam.side = inputParams.preProcess.side;

    //Process Params
    betaParam = 2*inputParams.betaParam*inputParams.betaParam;
    cParam = 2*inputParams.cParam*inputParams.cParam;

    //postProcess Params
    postProcess.variance = inputParams.postProcess.variance;
    postProcess.side = inputParams.postProcess.side;


    this->segStatus = -1;
    this->allocatedGPUMem = false;
    initKernels();

}







