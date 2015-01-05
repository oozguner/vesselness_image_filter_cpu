/*
* vesselness_image_filter_gpu.cpp
*
* Created By Russell Jackson
*  07/24/2014
*/


#include "thinSegmentationGPU.h"
#include "thinSegmentationKernels.cuh"
#include <opencv2/gpu/stream_accessor.hpp> 
#include "thinSegmentationGPU.h"


/*This file relies on the following external libraries:
OpenCV
Eigen
cuda
*/

//This file defines functions for segmenting the suture thread and the needle from the
//camera images using an object with GPU support.



//destructorfunction
VesselnessNodeGPU::~VesselnessNodeGPU(){

    //release ALL GPU Mats
    tempGPU_XX.release();
    tempGPU_XY.release();
    tempGPU_YY.release();

    this->deallocateGPUMem();

    //release the page lock
    if(this->allocatedPageLock)
    {
        for(int lr = 0; lr < 2; lr++)
        {
            Mat temp = srcMatMem[lr];
            temp.release();
            temp = dstMatMem[lr];
            temp.release();
            this->allocatedPageLock = false;
        }
    }


    tempCPU_XX.release();
    tempCPU_XY.release();
    tempCPU_YY.release();


    this->memStatus = -1;

}

void VesselnessNodeGPU::updateKernels(const segmentThinParam &inputParams){

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



void VesselnessNodeGPU::ProcessImage(const Mat & src,Mat & dst)
{









}




void VesselnessNodeGPU::initKernels(){


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

	if(this->kernelReady && this->allocatedGPUMem) this->segStatus = 0;

}


//This function allocates the GPU mem to save time
void VesselnessNodeGPU::allocateGPUMem(int rows,int cols){

	for(int lr = 0; lr < 2; lr++)
	{
		cXX[lr].create(rows,cols,CV_32FC1);
		cXY[lr].create(rows,cols,CV_32FC1);
		cYY[lr].create(rows,cols,CV_32FC1);
		preOutput[lr].create(rows,cols,CV_32FC3);
		outputG[lr].create(rows,cols,CV_32FC3);
		inputG[lr].create(rows,cols,CV_8UC3);
		inputGreyG[lr].create(rows,cols,CV_8UC1);
		inputFloat255G[lr].create(rows,cols,CV_32FC1);
		inputFloat1G[lr].create(rows,cols,CV_32FC1);
		scaled[lr].create(rows,cols,CV_32FC3);
		scaledU8[lr].create(rows,cols,CV_8UC3);
		dispOut[lr].create(rows,cols,CV_8UC3);
	}
	ones.create(rows,cols,CV_32FC1);
	ones.setTo(Scalar(255.0));


    //allocate the page lock memory
	srcMatMem.create(rows, cols, CV_8UC3, CudaMem::ALLOC_PAGE_LOCKED);
	dstMatMem.create(rows, cols, CV_32FC3, CudaMem::ALLOC_PAGE_LOCKED);

    /*don't bother with display for now*/
    dispMatMem[0].create(rows, cols, CV_8UC3, CudaMem::ALLOC_PAGE_LOCKED);
    dispMatMem[1].create(rows, cols, CV_8UC3, CudaMem::ALLOC_PAGE_LOCKED);





}

//This function allocates the GPU mem to save time
void VesselnessNodeGPU::deallocateGPUMem(){

	for(int lr = 0; lr < 2; lr++)
	{

		//input data
		inputG[2].release();
		inputGreyG[2].release();
		inputFloat255G[2].release();
		inputFloat1G[2].release();

		//intermediaries.
		cXX[2].release();
		cXY[2].release();
		cYY[2].release();

		//output data
		preOutput[2].release();
		outputG[2].release();
	}
	ones.release();
	allocatedGPUMem = false;
	this->segStatus = -1;

}

void VesselnessNodeGPU::allocatePageLock(int rows,int cols)
{


	srcMatMem.create(rows, cols, CV_8UC3, CudaMem::ALLOC_PAGE_LOCKED);
	dstMatMem.create(rows, cols, CV_32FC3, CudaMem::ALLOC_PAGE_LOCKED);

    /*don't bother with display for now*/
    dispMatMem[0].create(rows, cols, CV_8UC3, CudaMem::ALLOC_PAGE_LOCKED);
    dispMatMem[1].create(rows, cols, CV_8UC3, CudaMem::ALLOC_PAGE_LOCKED);

	this->allocatedPageLock = true;

}


void VesselnessNodeGPU::deallocatePageLock()
{


    srcMatMem.release();
    dstMatMem.release();

    this->allocatedPageLock = false;

}


void VesselnessNodeGPU::segmentImagePairBlocking(const Mat &stSrc, Mat &stDst){

    //compute the size of the image
    int iX,iY;

    iX = stSrc.cols;
    iY = stSrc.rows;

    if(allocatedPageLock == false)
    {
        this->allocatePageLock(iY,iX);
    }


    if(allocatedGPUMem == false)
    {
        this->allocateGPUMem(iY,iX);
    }

    cv::gpu::Stream streamInfo;
    cudaStream_t cudaStream;
    for(int lr = 0; lr < 2; lr++)
    {
        Mat srcMat = srcMatMem[lr];
        stSrc[lr].copyTo(srcMat);
        //inputG[lr].upload(stSrc[lr]);

        //convert image to gray scale witha max of 1.0;
        streamInfo.enqueueUpload(srcMat, inputG[lr]);

        gpu::cvtColor(inputG[lr],inputGreyG[lr],CV_BGR2GRAY,0,streamInfo);

        //perform a top hat operation.
        //gpu::morphologyEx(inputGreyG[lr],inputGreyG2[lr],MORPH_BLACKHAT,topKernel,inputBuff1[lr],inputBuff2[lr],Point(-1,-1),1,streamInfo);

        //gpu::cvtColor(inputG[lr],inputGreyG[lr],CV_BGR2GRAY,0,streamInfo[lr]);
        //streamInfo.enqueueConvert(inputGreyG[lr], inputFloat255G[lr], CV_32FC1,1.0,0.0);
        streamInfo.enqueueConvert(inputGreyG[lr], inputFloat255G[lr], CV_32FC1,1.0,0.0);

        //inputGreyG[lr].convertTo(inputFloat255G[lr],CV_32FC1,1.0,0.0);
        //gpu::divide(1/255,inputFloat255G[lr],inputFloat1G[lr],CV_32F,streamInfo);

        gpu::divide(inputFloat255G[lr],ones,inputFloat1G[lr],1.0,CV_32F,streamInfo);


        //gpu::divide(inputFloat255G[lr],Scalar(255.0,255.0,255.0),inputFloat1G[lr]);

        gpu::filter2D(inputFloat1G[lr],cXX[lr],-1,tempCPU_XX,Point(-1,-1),BORDER_DEFAULT,streamInfo);
        gpu::filter2D(inputFloat1G[lr],cYY[lr],-1,tempCPU_YY,Point(-1,-1),BORDER_DEFAULT,streamInfo);
        gpu::filter2D(inputFloat1G[lr],cXY[lr],-1,tempCPU_XY,Point(-1,-1),BORDER_DEFAULT,streamInfo);
        //^^^this takes 0.7 seconds
        //convolve the filters together to take less time?

        //gpu::filter2D(inputFloat1G[lr],cXX[lr],-1,tempCPU_XX);
        //gpu::filter2D(inputFloat1G[lr],cYY[lr],-1,tempCPU_YY);
        //gpu::filter2D(inputFloat1G[lr],cXY[lr],-1,tempCPU_XY);


        int blockX = (int) ceil((double) iX /(16.0f));
        int blockY = (int) ceil((double) iY /(16.0f));


        dim3 eigBlock(blockX,blockY,1);
        dim3 eigThread(16,16,1); 

        //What about here?
        //get the stream access first
        cudaStream = StreamAccessor::getStream(streamInfo);

        generateEigenValues<<<eigBlock,eigThread,0,cudaStream>>>(cXX[lr],cXY[lr],cYY[lr],preOutput[lr],betaParam,cParam);
        //preOutput[lr].create(iY,iX,CV_32FC3);
        //generateEigenValues<<<eigBlock,eigThread>>>(cXX[lr],cXY[lr],cYY[lr],preOutput[lr],betaParam,cParam);

        //Blur the result:
        int gaussOff = (int) floor(((float) postProcess.side)/2.0f);

        //outputG[lr] = preOutput[lr].clone();
        //streamInfo.enqueueCopy(preOutput[lr],outputG[lr]);
        gaussAngBlur<<<eigBlock,eigThread,0,cudaStream>>>(preOutput[lr],outputG[lr],gaussG,gaussOff);

        //compute the display output.
    /*  multiply(outputG[lr], Scalar(1/3.141,1.0,1.0),scaled[lr],255.0,-1,streamInfo);
        streamInfo.enqueueConvert(scaled[lr],scaledU8[lr],CV_8UC3,1.0,0.0);
        gpu::cvtColor(scaledU8[lr],dispOut[lr],CV_HSV2BGR,0,streamInfo);
        streamInfo.enqueueDownload(outputG[lr],dstMatMem[lr]);

        streamInfo.enqueueDownload(dispOut[lr],dispMatMem[lr]); */



        streamInfo.enqueueDownload(outputG[lr],dstMatMem[lr]);


        streamInfo.waitForCompletion();


        Mat dstMat = dstMatMem[lr];
        stDst[lr] = dstMat.clone(); 

        /*Mat dispMat = dispMatMem[lr];
        dispMats[lr] = dispMat.clone(); */

    }
    segStatus = 3;
}


void VesselnessNodeGPU::findOutputCutoffs(float* guess,int iters)
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
                gpu::GpuMat sideL;
                gpu::GpuMat sideU;
                gpu::GpuMat inRange3,outRange;
                gpu::GpuMat inRange[3];
                gpu::GpuMat buffer;

                buffer.create(outputG[lr].size(),outputG[lr].type());

                gpu::compare(outputG[lr],rangeL, inRange3, CMP_GT);
                //gpu::compare(outputG[lr],rangeU, sideU, CMP_LT);
                gpu::split(inRange3, inRange);

                //gpu::bitwise_and(sideL,sideU, inRange);

                gpu::bitwise_not(inRange[1],outRange);

                Scalar highSum = gpu::sum(outputG[lr], inRange[1],buffer);
                Scalar lowSum = gpu::sum(outputG[lr], outRange,buffer);

                //get the norms:
                Scalar inSum = gpu::sum(inRange[1],buffer);
                Scalar outSum = gpu::sum(outRange,buffer);

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




void VesselnessNodeGPU::getSegmentImagePair(Mat &stDst){
		for(int lr = 0; lr <2; lr++)
		{
			stDst[lr] = dstMats[lr].clone();
		}
}


void VesselnessNodeGPU::getSegmentDisplayPair(Mat &stDisp){

		for(int lr = 0; lr <2; lr++)
		{
			stDisp[lr] = dispMats[lr].clone();
		}
}
