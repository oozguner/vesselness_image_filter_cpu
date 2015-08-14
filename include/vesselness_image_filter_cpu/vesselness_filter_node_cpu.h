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
 *   * Neither the name of Case Western Reserve University, nor the names of its
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

#ifndef IMAGESEGMENTCPUH
#define IMAGESEGMENTCPUH


#include <vesselness_image_filter_common/vesselness_image_filter_common.h>

//Converts a single image into a displayable RGB format.
void convertSegmentImageCPU(const Mat&,Mat&);

//This class extends the basic VesselnessNode based on using a CPU to complete the actual processing.
class VesselnessNodeCPU: public VesselnessNodeBase {

private:

    //Input and output information
    Mat input;
    Mat output;

    //Intermediates:
    Mat cXX;
    Mat cXY;
    Mat cYY;

    Mat greyImage_xx;
    Mat greyImage_xy;
    Mat greyImage_yy;

    Mat inputGreyG;
    Mat inputFloat255G;
    Mat ones;
    Mat inputFloat1G;

    Mat preOutput;

    Mat scaled;
    Mat scaledU8;
    Mat dispOut;

    //Gauss kernels
    Mat gaussKernel_XX;
    Mat gaussKernel_XY;
    Mat gaussKernel_YY;
    Mat imageMask;

    Mat greyFloat;
    Mat greyImage;


    Mat srcMats;
    Mat dstMats;


    //status booleans
    bool kernelReady;
    bool allocatedKernels;


    void  setKernels();
    void  initKernels();
    void  updateKernels();



    //declare the memory management functions
    //void allocateMem(Size); (declared in the abstract base class)
    void allocateMem(int,int);
    void deallocateMem();

    /*TODO void VesselnessNodeGPU::findOutputCutoffs(float*,int = 10); */

    //blocking image segmentation
    void segmentImage(const Mat &, Mat &);

    
    //Update object parameters.
    void updateKernels(const segmentThinParam &);
	

public:
   
    //This function needs to operate at peak speed:
    VesselnessNodeCPU(const char*,const char*); //constructor
    VesselnessNodeCPU();    //default constructor
    ~VesselnessNodeCPU();   //deconstructor

};





#endif
