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


#include <vesselness_filter_node_gpu.h>


/*This file relies on the following external libraries:
OpenCV
Eigen
cuda
*/

//This file defines functions for segmenting the suture thread and the needle from the
//camera images using an object with GPU support.


VesselnessNodeGpu::VesselnessNodeGpu(const char* subscriptionChar):VesselnessNodeBase(subscriptionChar)
{

    //predetermined init values. (sorta random)
	hessParam.variance = 1.5;
	hessParam.side = 5;
	betaParam = 0.1;    //  betaParamIn;
	cParam    = 0.005;     //  cParamIn;

	postProcess.variance = 2.0;
	postProcess.side = 7;

	//initialize the kernels.
	imgAllocSize = Size(-1,-1);
    	initKernels();



}


//destructorfunction
VesselnessNodeGpu::~VesselnessNodeGpu(){
    


}

void VesselnessNodeGpu::updateKernels(const segmentThinParam &inputParams){

    //preProcess Params
    hessParam.variance = inputParams.preProcess.variance;
    hessParam.side = inputParams.preProcess.side;

    //Process Params
    betaParam = 2*inputParams.betaParam*inputParams.betaParam;
    cParam = 2*inputParams.cParam*inputParams.cParam;

    //postProcess Params
    postProcess.variance = inputParams.postProcess.variance;
    postProcess.side = inputParams.postProcess.side;


    initKernels();

}



void VesselnessNodeGpu::initKernels(){


	//reallocate the GpuMats
	tempGPU_XX.create(hessParam.side,hessParam.side,CV_32FC1);
	tempGPU_XY.create(hessParam.side,hessParam.side,CV_32FC1);
	tempGPU_YY.create(hessParam.side,hessParam.side,CV_32FC1);

	//initialize the hessian kernels variables:
	int offset =  (int) floor((float)hessParam.side/2);

	dim3 kBlock(1,1,1);
	dim3 kThread(hessParam.side,hessParam.side,1);
	genGaussHessKernel_XX<<<kBlock,kThread>>>(tempGPU_XX,hessParam.variance,offset);
	genGaussHessKernel_XY<<<kBlock,kThread>>>(tempGPU_XY,hessParam.variance,offset);
	genGaussHessKernel_YY<<<kBlock,kThread>>>(tempGPU_YY,hessParam.variance,offset);


	tempGPU_XX.download(tempCPU_XX);
	tempGPU_XY.download(tempCPU_XY);
	tempGPU_YY.download(tempCPU_YY);

	//initialize the postProcess Kernel:
	Mat gaussKernel = getGaussianKernel(postProcess.side,postProcess.variance,CV_32FC1);
	Mat gaussOuter  = gaussKernel*gaussKernel.t();

	//topKernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));

	gaussG.upload(gaussOuter);


	//Finished...
	this->kernelReady = true;
}


//This function allocates the GPU mem to save time
void VesselnessNodeGpu::allocateMem(int rows,int cols){


    //allocate the convolved hessian matrices.
    cXX.create(rows,cols,CV_32FC1);
    cXY.create(rows,cols,CV_32FC1);
    cYY.create(rows,cols,CV_32FC1);


    //allocate the other matrices.
    preOutput.create(rows,cols,CV_32FC3);
    outputG.create(rows,cols,CV_32FC3);
    inputG.create(rows,cols,CV_8UC3);
    inputGreyG.create(rows,cols,CV_8UC1);
    inputFloat255G.create(rows,cols,CV_32FC1);
    inputFloat1G.create(rows,cols,CV_32FC1);
    scaled.create(rows,cols,CV_32FC3);
    scaledU8.create(rows,cols,CV_8UC3);
    dispOut.create(rows,cols,CV_8UC3);

    ones.create(rows,cols,CV_32FC1);
    ones.setTo(Scalar(255.0));


    //allocate the page lock memory
    srcMatMem.create(rows, cols, CV_8UC3, HostMem::ALLOC_PAGE_LOCKED);
    dstMatMem.create(rows, cols, CV_32FC2, HostMem::ALLOC_PAGE_LOCKED);

}

//This function allocates the GPU mem to save time
void VesselnessNodeGpu::deallocateMem(){

   //input data
   inputG.release();
   inputGreyG.release();
   inputFloat255G.release();
   inputFloat1G.release();

   //intermediaries.
   cXX.release();
   cXY.release();
   cYY.release();

    //output data
    preOutput.release();
    outputG.release();

    ones.release();

    srcMatMem.release();
    dstMatMem.release();

}



void VesselnessNodeGpu::segmentImage(const Mat &srcMat,Mat &dstMat)
{
    //compute the size of the image
    int iX,iY;

    iX = srcMat.cols;
    iY = srcMat.rows;

    cv::cuda::Stream streamInfo;
    cudaStream_t cudaStream;

    //upload &  convert image to gray scale with a max of 1.0;
    inputG.upload(srcMat,streamInfo);


    cuda::cvtColor(inputG,inputGreyG,CV_BGR2GRAY,0,streamInfo);

    //perform a top hat operation.
    //cuda::morphologyEx(inputGreyG[lr],inputGreyG2[lr],MORPH_BLACKHAT,topKernel,inputBuff1[lr],inputBuff2[lr],Point(-1,-1),1,streamInfo);

    //cuda::cvtColor(inputG[lr],inputGreyG[lr],CV_BGR2GRAY,0,streamInfo[lr]);
    //streamInfo.enqueueConvert(inputGreyG[lr], inputFloat255G[lr], CV_32FC1,1.0,0.0);
    inputGreyG.convertTo(inputFloat255G, CV_32FC1,1.0,0.0,streamInfo);

    //inputGreyG[lr].convertTo(inputFloat255G[lr],CV_32FC1,1.0,0.0);
    //cuda::divide(1/255,inputFloat255G[lr],inputFloat1G[lr],CV_32F,streamInfo);

    cuda::divide(inputFloat255G,ones,inputFloat1G,1.0,CV_32F,streamInfo);


    //cuda::divide(inputFloat255G[lr],Scalar(255.0,255.0,255.0),inputFloat1G[lr]);

    cuda::filter2D(inputFloat1G,cXX,-1,tempCPU_XX,Point(-1,-1),BORDER_DEFAULT,streamInfo);
    cuda::filter2D(inputFloat1G,cYY,-1,tempCPU_YY,Point(-1,-1),BORDER_DEFAULT,streamInfo);
    cuda::filter2D(inputFloat1G,cXY,-1,tempCPU_XY,Point(-1,-1),BORDER_DEFAULT,streamInfo);


    //cuda::filter2D(inputFloat1G[lr],cXX[lr],-1,tempCPU_XX);
    //cuda::filter2D(inputFloat1G[lr],cYY[lr],-1,tempCPU_YY);
    //cuda::filter2D(inputFloat1G[lr],cXY[lr],-1,tempCPU_XY);


    int blockX = (int) ceil((double) iX /(16.0f));
    int blockY = (int) ceil((double) iY /(16.0f));


    dim3 eigBlock(blockX,blockY,1);
    dim3 eigThread(16,16,1); 

    //What about here?
    //get the stream access first
    cudaStream = StreamAccessor::getStream(streamInfo);

    generateEigenValues<<<eigBlock,eigThread,0,cudaStream>>>(cXX,cXY,cYY,preOutput,betaParam,cParam);
        //preOutput[lr].create(iY,iX,CV_32FC3);
        //generateEigenValues<<<eigBlock,eigThread>>>(cXX[lr],cXY[lr],cYY[lr],preOutput[lr],betaParam,cParam);

        //Blur the result:
        int gaussOff = (int) floor(((float) postProcess.side)/2.0f);

        //outputG[lr] = preOutput[lr].clone();
        //streamInfo.enqueueCopy(preOutput[lr],outputG[lr]);
        gaussAngBlur<<<eigBlock,eigThread,0,cudaStream>>>(preOutput,outputG,gaussG,gaussOff);

        //compute the display output.
    /*  multiply(outputG[lr], Scalar(1/3.141,1.0,1.0),scaled[lr],255.0,-1,streamInfo);
        streamInfo.enqueueConvert(scaled[lr],scaledU8[lr],CV_8UC3,1.0,0.0);
        cuda::cvtColor(scaledU8[lr],dispOut[lr],CV_HSV2BGR,0,streamInfo);
        streamInfo.enqueueDownload(outputG[lr],dstMatMem[lr]);

        streamInfo.enqueueDownload(dispOut[lr],dispMatMem[lr]); */


	outputG.download(dstMatMem,streamInfo);
        //streamInfo.enqueueDownload(outputG,dstMatMem);

        streamInfo.waitForCompletion();

        Mat tempDst;
        tempDst = dstMatMem;
        dstMat = tempDst.clone(); 
    }

/*
void VesselnessNodeGpu::::findOutputCutoffs(float* guess,int iters)
{

    if(this->segStatus > 1)
    {

        for(int lr = 0; lr < 2; lr++)
        {
            float tVal = guess[lr];
            for(int ind = 0; ind < iters; ind++)
            {
                Scalar rangeL(0,tVal,0);
                Scalar rangeU(10.0,1.0,0);
                cuda::GpuMat sideL;
                cuda::GpuMat sideU;
                cuda::GpuMat inRange3,outRange;
                cuda::GpuMat inRange[3];
                cuda::GpuMat buffer;

                buffer.create(outputG[lr].size(),outputG[lr].type());

                cuda::compare(outputG[lr],rangeL, inRange3, CMP_GT);
                //cuda::compare(outputG[lr],rangeU, sideU, CMP_LT);
                cuda::split(inRange3, inRange);

                //cuda::bitwise_and(sideL,sideU, inRange);

                cuda::bitwise_not(inRange[1],outRange);

                Scalar highSum = cuda::sum(outputG[lr], inRange[1],buffer);
                Scalar lowSum = cuda::sum(outputG[lr], outRange,buffer);

                //get the norms:
                Scalar inSum = cuda::sum(inRange[1],buffer);
                Scalar outSum = cuda::sum(outRange,buffer);

                float hValf;
                if(inSum[0] == 0.0)
                    hValf = 0.0;
                else
                    hValf = highSum[1]/(inSum[0]/(255));

                float lValf;
                if(lowSum[0] == 0.0)
                    lValf = 0.0;
                else
                    lValf = lowSum[1]/(outSum[0]/255);

                float newT = (lValf + hValf)/2;


                if(fabs(newT-tVal) < 0.0001) break;

                tVal = newT;
            }
            guess[lr] = tVal;
        }
    }




void VesselnessNodeGpu::::getSegmentImagePair(Mat &stDst){
		for(int lr = 0; lr <2; lr++)
		{
			stDst[lr] = dstMats[lr].clone();
		}
}


void VesselnessNodeGpu::::getSegmentDisplayPair(Mat &stDisp){

        for(int lr = 0; lr <2; lr++)
        {
            stDisp[lr] = dispMats[lr].clone();
        }
}
*/
