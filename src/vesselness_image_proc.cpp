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


#include "imageSegmentation.h"


//Converts a simgle image into a displayable RGB format.
void convertSegmentImage(const Mat&src,Mat&dst){
	
	Mat tempDisplay1,tempDisplay2;
	
	tempDisplay1 = src.mul(Scalar(1/3.14159,1.0,1.0));
	convertScaleAbs(tempDisplay1,tempDisplay2,255.0);
	cvtColor(tempDisplay2,dst,CV_HSV2BGR);
}

//This function takes the angle and intensity and creates an average angle.
 double meanAngle(const Mat & inputDir,const Mat & inputMag)
{
	

	Point2f finalDir(0,0);

	//float magSum = 0.0;

	int pixelCount1 = inputDir.cols*inputDir.rows;

	int pixelCount2 = inputMag.cols*inputMag.rows;

	if(pixelCount1 != pixelCount2) return -1;

	
	char* dirImagePtr = (char*) inputDir.data;

	int   dirImageStep0 =  inputDir.step[0];

	int   dirImageStep1 =  inputDir.step[1];



	char* magImagePtr = (char*) inputMag.data;

	int   magImageStep0 =  inputMag.step[0];

	int   magImageStep1 =  inputMag.step[1];



	for(int i = 0; i < pixelCount1; i++)

	{

		int xPos =  i%inputDir.cols;

		int yPos =  (int) floor(((float) i)/((float) inputDir.cols));

		//int xVect = xPos-center.x;
		//int yVect = yPos-center.y;

		//Point2f newPtP(xVect,yVect);


		float* dirPtr =  (float*) (dirImagePtr+ dirImageStep0*yPos + dirImageStep1*xPos);

		float* magPtr =  (float*) (magImagePtr+ magImageStep0*yPos + magImageStep1*xPos);


		//magSum += magPtr[0];

		Point2f newPt(magPtr[0]*cos(dirPtr[0]),magPtr[0]*sin(dirPtr[0]));

		//float dProd2 = newPtP.dot(newPt);

		float dProd = finalDir.dot(newPt)/(norm(finalDir)*norm(newPt));

		if(dProd < 0.00) finalDir -= newPt;
		else finalDir += newPt;

	}


	//Final angle calculation;


	float angle = atan2(finalDir.y,finalDir.x);

	if(angle < 0) angle += 3.1415;

	return angle;

}


 double angleVar(const Mat & inputDir,const Mat & inputMag)
 {
	 
	 double meanAngleE = meanAngle(inputDir,inputMag);

	 double meanVar = 0.0;
	 double sumMag  = 0.0;

	 Point2f meanPt(cos(meanAngleE),sin(meanAngleE));


	 int pixelCount1 = inputDir.cols*inputDir.rows;

	int pixelCount2 = inputMag.cols*inputMag.rows;

	if(pixelCount1 != pixelCount2) return -1;

	
	char* dirImagePtr = (char*) inputDir.data;

	int   dirImageStep0 =  inputDir.step[0];

	int   dirImageStep1 =  inputDir.step[1];



	char* magImagePtr = (char*) inputMag.data;

	int   magImageStep0 =  inputMag.step[0];

	int   magImageStep1 =  inputMag.step[1];



	for(int i = 0; i < pixelCount1; i++)

	{

		int xPos =  i%inputDir.cols;

		int yPos =  (int) floor(((float) i)/((float) inputDir.cols));

		
		float* dirPtr =  (float*) (dirImagePtr+ dirImageStep0*yPos + dirImageStep1*xPos);

		float* magPtr =  (float*) (magImagePtr+ magImageStep0*yPos + magImageStep1*xPos);


		//magSum += magPtr[0];

		Point2f newPt(cos(dirPtr[0]),sin(dirPtr[0]));

		float dProd2 = abs(newPt.dot(meanPt));

		float angleErr =  acos(dProd2);
		
		meanVar += angleErr*angleErr*magPtr[0];
		sumMag  += magPtr[0];
	}

	//average out:
	double meanVarF = meanVar/sumMag;

	return meanVarF; //returns the angle variance:

 }

void matAngleAlignment(const Mat & src,Mat & dst,Point2d baseDir)
{
	
	dst.create(src.size(),CV_32FC1);

	unsigned char * srcPtr=  (unsigned char*) src.data;
	int  srcStep0 = src.step[0];
	int  srcStep1 = src.step[1];

	unsigned char * dstPtr=  (unsigned char*) dst.data;
	int  dstStep0 = dst.step[0];
	int  dstStep1 = dst.step[1];
	

	

	int pixCount = src.rows*src.cols;

	for(int i = 0; i < pixCount; i++)
	{

		int XPos =  i%src.cols;
		int YPos =  (int) floor(((double) i)/((double) src.cols));

		float* dstPointer =  (float*) (dstPtr+ dstStep0*XPos + dstStep1*YPos); 
		float* srcPointer =  (float*) (srcPtr+ srcStep0*XPos + srcStep1*YPos); 

		//compute inner product...
		Point2f srcDir(cos(srcPointer[0]),sin(srcPointer[0]));

		float ip = srcDir.dot(baseDir);

		dstPointer[0] = abs(ip)*srcPointer[1];
	}

}


//gaussian blurring function for angles


void angleMagBlur(const Mat &src,Mat &dst, const gaussParam inputParam)

{

	

	//reallocate the dst matrix

	dst.create(src.size(),src.type());

	

	//define a gaussian kernel

	Mat gaussKernelA = getGaussianKernel(inputParam.side,inputParam.variance,CV_32F);

	Mat gaussKernel = gaussKernelA*gaussKernelA.t();

	int gaussOffset = floor((float) inputParam.side/2);


	int imagePixCount = src.rows*src.cols;

	int gaussPixCount = gaussKernel.rows*gaussKernel.cols;


	char * gPtr =  (char*) gaussKernel.data;
	int  gStep0 = gaussKernel.step[0];
	int  gStep1 = gaussKernel.step[1];

    char * srcPtr=  (char*) src.data;
	int  srcStep0 = src.step[0];
	int  srcStep1 = src.step[1];

	char * dstPtr =  (char*) dst.data;
	int  dstStep0 = dst.step[0];
	int  dstStep1 = dst.step[1];


	//This is a convolution...of sorts..

	//This will be painfully slow until the GPU computation is sorted out.

	for(int i = 0; i < imagePixCount; i++)
	{

		int dstXPos =  i%src.cols;
		int dstYPos =  (int) floor(((double) i)/((double) src.cols));

		float* dstPointer =  (float*) (dstPtr+ dstStep0*dstYPos + dstStep1*dstXPos); 


		float val = 0.0;

		Point2f dirPt(0,0);


		for(int j = 0; j < gaussPixCount; j++)
		{

			int gXPos = j%gaussKernel.cols;

			int gYPos = (int) floor(((double) j)/((double) gaussKernel.cols));

			float* gPointer =  (float*) (gPtr+ gStep0*gYPos + gStep1*gXPos); 

			int srcXPos =dstXPos-gaussOffset+gXPos;
			int srcYPos =dstYPos-gaussOffset+gYPos;

			//constant corner assumption:
			if(srcXPos < 0) srcXPos = 0;
			if(srcYPos < 0) srcYPos = 0;
			
			if(srcXPos >= src.cols) srcXPos = src.cols-1;
			if(srcYPos >= src.rows) srcYPos = src.rows-1;

			float* srcPointer =  (float*) (srcPtr+ srcStep0*srcYPos + srcStep1*srcXPos); 

			val +=srcPointer[1]*gPointer[0];
			
			Point2f newDir(srcPointer[1]*gPointer[0]*cos(srcPointer[0]),srcPointer[1]*gPointer[0]*sin(srcPointer[0]));

			//find the cos between the two vectors;

			float dotResult = newDir.dot(dirPt)/(norm(newDir)*norm(dirPt));

			if(dotResult < -0.707) dirPt-=newDir;
			else dirPt+=newDir;
		}
		dstPointer[2] = 0.5;
		dstPointer[1]  = val;
		float newAngle = atan2(dirPt.y,dirPt.x);
		if(newAngle < 0.0) dstPointer[0] = (newAngle+3.1415);
		else dstPointer[0] = (newAngle);
	}
	return;
}
	
//constructor fnc

ThinSegmentation::ThinSegmentation(float betaParamIn,float cParamIn,int kSizeHessIn,float varHessIn,int kSizePostIn,float varPostIn){


	

	

	hessParam.variance = varHessIn;

	hessParam.side = kSizeHessIn;

	

	betaParam = 2*betaParamIn*betaParamIn;

	cParam    = 2*cParamIn*betaParamIn;


	postProcess.variance = varPostIn;

	postProcess.side = kSizePostIn;


	initKernels();


	//Init the status parameters:

	sharedStatus = 1; //Ready to go.

	localStatus[0] = 0;

	localStatus[1] = 1;


	//Now start the two threads

	//thrd0 = new

	boost::thread threadOut0(boost::bind(&ThinSegmentation::segmentingLoop, this, 0)); //left side loop thread;

}



ThinSegmentation::ThinSegmentation(segmentThinParam inputParams){

	
	hessParam.variance = inputParams.preProcess.variance;

	hessParam.side = inputParams.preProcess.side;

	

	betaParam = 2*inputParams.betaParam*inputParams.betaParam; //  betaParamIn;

	cParam    = 2*inputParams.cParam*inputParams.cParam;    //  cParamIn;


	postProcess.variance = inputParams.postProcess.variance;

	postProcess.side = inputParams.postProcess.side;


	initKernels();


	//Init the status parameters:

	sharedStatus = 1; //Ready to go.

	localStatus[0] = 0;

	localStatus[1] = 0;


	//Now start the two threads

	boost::thread thrd0(boost::bind(&ThinSegmentation::segmentingLoop, this, 0)); //left side loop thread;

	boost::thread thrd1(boost::bind(&ThinSegmentation::segmentingLoop, this, 1)); //right side loop thread;

}



inline float gaussFnc(float var,float x,float y){

	

	return 1/(3.1415*2*var)*exp(-x*x/(2*var)-y*y/(2*var));



}


//adds the image mask 

void ThinSegmentation::setImageMask(const stereoImage &inputMasks_){


	boost::mutex::scoped_lock updateLock(segmentingMutex);

	for(int lr = 0; lr <2; lr++)

	{

		inputMasks[lr] = inputMasks_[lr].clone();


	}


	initKernels();

	sharedStatus = 5;  //update image Mask;

	updateLock.unlock();

	while(true){

		boost::this_thread::sleep(boost::posix_time::milliseconds(10));

		if(localStatus[0] ==5 && localStatus[1] ==5){ //completed loading the parameters in

			updateLock.lock();

			sharedStatus = 1;

			std::cout << "Updated mask matrix for Both threads \n";

			updateLock.unlock();

			break;

		}

	}




}


void ThinSegmentation::initKernels(){

	

	float var = hessParam.variance;

	int kSize =  hessParam.side;

	int kSizeEnd = (int) floor((float) kSize/2);


	//Allocate the matrices

	gaussKernel_XX =Mat(kSize,kSize,CV_32F);

	gaussKernel_XY =Mat(kSize,kSize,CV_32F);

	gaussKernel_YY =Mat(kSize,kSize,CV_32F);


	for(int ix = -kSizeEnd; ix < kSizeEnd+1; ix++){

		for(int iy = -kSizeEnd; iy < kSizeEnd+1; iy++){


			float ixD = (float) ix;

			float iyD = (float) iy;


			gaussKernel_XX.at<float>(iy+kSizeEnd,ix+kSizeEnd) = (ixD*ixD)/(var*var)*gaussFnc(var,ixD,iyD)

															-1/(var)*gaussFnc(var,ixD,iyD);


			gaussKernel_YY.at<float>(iy+kSizeEnd,ix+kSizeEnd) = (iyD*iyD)/(var*var)*gaussFnc(var,ixD,iyD)

															-1/(var)*gaussFnc(var,ixD,iyD);


			gaussKernel_XY.at<float>(iy+kSizeEnd,ix+kSizeEnd) = (iyD*ixD)/(var*var)*gaussFnc(var,ixD,iyD);

		}

	}
	std::cout << "Updated Kernel matrix for CPU \n";
	std::cout << gaussKernel_XX << std::endl<< std::endl;
	std::cout << gaussKernel_XY << std::endl<< std::endl;
	std::cout << gaussKernel_YY << std::endl<< std::endl;
}


void  ThinSegmentation::segmentingLoop(int lr){


	//thread globals: 

	Mat inputImage;

	Mat inputMask = Mat();

	Mat greyImage;

	Mat greyFloat;

	Mat greyImage_xx;

	Mat greyImage_xy;

	Mat greyImage_yy;

	Mat det;

	Mat preOutput;

	Mat outputImage;

	boost::mutex::scoped_lock loopLock(segmentingMutex);

	//boost::mutex::scoped_lock localLock(segmentingMutex);

	//clone out the kernels

	Mat localKernel_XX = gaussKernel_XX.clone(); 

	Mat localKernel_XY = gaussKernel_XY.clone();

	Mat localKernel_YY = gaussKernel_YY.clone();

	localStatus[lr] = 1;  //ready to run

	loopLock.unlock();


	int sharedStatusIn;

	int localLocalStatus = 1;

	//free running mode

	while(true){

		

		//Use a switch statment?

		loopLock.lock();

		sharedStatusIn = sharedStatus;

		localStatus[lr] = localLocalStatus;

		loopLock.unlock();


		//update the kernel:

		if(sharedStatusIn == 4 && localStatus[lr] != 4){

			

			loopLock.lock();


			localKernel_XX = gaussKernel_XX.clone(); 

			localKernel_XY = gaussKernel_XY.clone();

			localKernel_YY = gaussKernel_YY.clone();


			std::cout << "Updated Kernel matrix for thread # : " << lr << std::endl;

			loopLock.unlock();

			//std::cout << localKernel_XX << std::endl;

			//std::cout << localKernel_XY << std::endl;

			//std::cout << localKernel_YY << std::endl;


			localLocalStatus = 4;


			

			

		}

		//update the Mask

		if(sharedStatusIn == 5 && localStatus[lr] != 5){

			

			loopLock.lock();


			inputMask = inputMasks[lr].clone();


			std::cout << "Updated mask matrix for thread # : " << lr << std::endl;

			loopLock.unlock();

			//std::cout << localKernel_XX << std::endl;

			//std::cout << localKernel_XY << std::endl;

			//std::cout << localKernel_YY << std::endl;


			localLocalStatus = 5;


		}


		//update the kernel:

		if(sharedStatusIn == 1 && localLocalStatus != 1){

			localLocalStatus = 1;

		}


		//central thread has the item

		if(sharedStatusIn == 3){

			localLocalStatus = 1;

		}


		//Quitting time

		if(sharedStatusIn < 0){

			break;	

		}


		//Actual process segmentation code:

		if(sharedStatusIn == 2 && localLocalStatus == 1){

			//lock mutex and load in the image.

			loopLock.lock();

			inputImage = inputPair[lr].clone(); 

			//std::cout << "Beginning Processing with thread # : " << lr << std::endl;

			localLocalStatus = 2;

			localStatus[lr] = localLocalStatus;

			loopLock.unlock();

			
			//for now only use one image:

			//float imageGain = (2*3.14*postProcess.variance);
			
				

			//Use an ROI for speed?

			//Not as of right now.

			cvtColor(inputImage,greyImage,CV_BGR2GRAY);

			greyImage.convertTo(greyFloat,CV_32FC1,1.0,0.0);

			float *greyFloatPtr = (float*) greyFloat.data;
			
			
			greyFloat /= 255.0;
			
			//Ideally this is now Faster

			filter2D(greyFloat,greyImage_xx,-1,localKernel_XX);

			filter2D(greyFloat,greyImage_xy,-1,localKernel_XY);

			filter2D(greyFloat,greyImage_yy,-1,localKernel_YY);

		

			//det = greyImage_xx.mul(greyImage_yy)-greyImage_xy.mul(greyImage_xy);

			//Now that the Hessian components have been calculated, move through

			//The image points and compute the eigen values.

		

			//Compute the number of total pixels

			int pixCount = greyImage_xx.rows*greyImage_xx.cols;

		

			float *gradPtr_xx = (float*)  greyImage_xx.data;

			float *gradPtr_yx = (float*)  greyImage_xy.data;

			float *gradPtr_xy = (float*)  greyImage_xy.data;

			float *gradPtr_yy = (float*)  greyImage_yy.data;

			//float *detPtr     = (float*)  det.data;


			preOutput.create(greyImage_xx.rows,greyImage_xx.cols,CV_32FC3);


			char* preOutputImagePtr = (char*) preOutput.data;

			int preOutputImageStep0 =  preOutput.step[0];

			int preOutputImageStep1 =  preOutput.step[1];

		


			char* inputMaskPtr = (char*) inputMask.data;

			int inputMaskStep0 =  inputMask.step[0];

			int inputMaskStep1 =  inputMask.step[1];



			//for each image, evaluate its eigen vectors, then look at the cost

			//function from Frangi et al.	

			for(int i =0 ; i < pixCount; i=i++){

				int xPos =  i%greyImage_xx.cols;

				int yPos =  (int) floor(((float) i)/((float) greyImage.cols));

				float* finalPointer =  (float*) (preOutputImagePtr+ preOutputImageStep0*yPos + preOutputImageStep1*xPos); 

	 	

				if(inputMask.rows == preOutput.rows && inputMask.cols == preOutput.cols){

					char* maskVal = (inputMaskPtr+ inputMaskStep0*yPos + inputMaskStep1*xPos); 

					if(maskVal[0] == 0)
					{
						finalPointer[0] = 0.0;
						finalPointer[1] = 0.0;
						continue;
					}
				} //if(inputMask.rows == preOutput.rows && inputMask.cols == preOutput.cols)
				

				float vMag;
				float v_y;
				float v_x;
				float a2;

				vMag = 0.0;
				v_y = 0.0;
				v_x = 1.0;
				a2 = 0;

				
				float det = gradPtr_xx[i]*gradPtr_yy[i]-gradPtr_yx[i]*gradPtr_yx[i];
				float b = -gradPtr_xx[i]-gradPtr_yy[i];
				float c =  det;
				float descriminant = sqrt(b*b-4*c);

				float eig0;
				float eig1;
				float r_Beta;

					
				//adding safety for small values of the descriminant.
				if(descriminant > 0.000000001) 
				{

					eig0 = (-b+descriminant)/(2);

					eig1 = (-b-descriminant)/(2);

					r_Beta = eig0/eig1;

					//find the dominant eigenvector:
					if(abs(r_Beta) > 1.0){  //indicates that eig0 is larger.

						r_Beta = 1/r_Beta;

						
						v_y = (eig0-gradPtr_xx[i])*v_x/(gradPtr_xy[i]);
					}
					else //indicates that eig1 is larger.
					{
						
						v_y = (eig1-gradPtr_xx[i])*v_x/(gradPtr_xy[i]);

					}
				} //if(descriminant > 0.000000001) 
				else
				{
						
					eig0 = eig1 = -b/2;
					r_Beta = 1.0;
					v_y = 0.00;
					v_x = 1.0;

				}

				//In this formulation, the image peak is 1.0;	
				vMag = exp(-r_Beta*r_Beta/(betaParam))*(1-exp(-(eig0*eig0+eig1*eig1)/(cParam)));

					
				//include the eigenVector:
				float a = atan2(v_y,v_x);


				if(a > 0.00)
				{
					a2 = (a); ///3.1415;
				}
				else
				{
					a2 = (a+3.1415); ///3.1415;
				}



				if(!(vMag <= 1) || !(vMag >= 0))
				{
					float test = 1;
					std::cout << "Bad number here\n";
				}

				//HSV space
				finalPointer[0] = a2;
				finalPointer[1] = vMag;
				finalPointer[2] = 0.5;
				//finalPointer[0] = eig0;
				//finalPointer[1] = eig1;
				//finalPointer[2] = a2/(3.1415);
			}

		//Once all is said and done, blur the final image using a gaussian.

		//Need to reincorporate this aspect:
		angleMagBlur(preOutput,outputImage,this->postProcess);
			
		//outputImage = preOutput.clone();

		
#ifdef DEBUGSUTURESEGMENT

	Mat outputImageDisp,preOutputImageDisp;

	convertScaleAbs(outputImage,outputImageDisp,(255.0/imageGain));

	convertScaleAbs(preOutputImage,preOutputImageDisp,(255.0/imageGain));

	//cout << segmentMean << endl;

	//cout << segmentStdDev << endl;

	//imshow((i? "left original":"right original"),inputImage[i]);

	//imshow((i? "left blur":"right blur"),imageBlur[i]);

	//imshow("lab",lab[i]);

	//cv::setMouseCallback("lab",uicallback::displayPixel,&lab);

	imshow("OutputResults",outputImageDisp);

	imshow("preOutputResults",preOutputImageDisp);

	//	cv::setMouseCallback("segment",uicallback::displayPixel,&segment);

	//	imshow((i? "left Mask":"right mask"),segmentMask);

			

	/*imshow((i? "left original":"right original"),inputImage[i]);

	imshow((i? "left blur":"right blur"),imageBlur[i]);

	imshow((i? "left hsv":"right hsv"),hsv[i]);

	imshow((i? "left lab":"right lab"),lab[i]);

	imshow((i? "left hsv hue":"right hsv hue"),hueHSV[i]);

			

	imshow((i? "left lab light":"right lab light"),lightLAB[i]);

	imshow((i? "left lab b":"right lab b"),bLAB[i]);

	imshow((i? "left lab hue":"right lab hue"),hueLAB[i]);

	//imshow((i? "left maska":"right maska"),maska[i]);

	//imshow((i? "left maskb":"right maskb"),maskb[i]);*/

	//	imshow((i? "left mask":"right mask"),mask[i]); 

	waitKey(00);

	//	cv::destroyAllWindows();

	destroyWindow("OutputResults");

	destroyWindow("preOutputResults");


#endif



			//lock mutex and load in the image.

			loopLock.lock();

			outputPair[lr] = outputImage.clone(); 

			localLocalStatus = 3;

			localStatus[lr] = localLocalStatus;

			//std::cout << "finished Processing with thread # : " << lr << std::endl;

			loopLock.unlock();

		} //	if(sharedStatus == 2){

   //At the end of the if statment

		


		boost::this_thread::sleep(boost::posix_time::milliseconds(10));

		//boost::this_thread::sleep_for(sleepTime);  //sleep for 30 ms.

		

		loopLock.lock();

		if(localLocalStatus == 4  && sharedStatus == 2) 

		{


			localLocalStatus = 1;

			localStatus[lr] = localLocalStatus;

		}

		loopLock.unlock();



	}//while(true)


	loopLock.lock();

	std::cout << "Quitting segmentation thread # : " << lr << std::endl;

	localStatus[lr] = -1;

	loopLock.unlock();


	return;

}



ThinSegmentation::~ThinSegmentation(){

	//The destructor should deallocate the memory as needed.
	
}


void ThinSegmentation::updateKernels(const segmentThinParam &inputParams){



	//boost::chrono::milliseconds threadSleepTime(30);

	boost::mutex::scoped_lock updateLock(segmentingMutex);


	hessParam.variance = inputParams.preProcess.variance;

	hessParam.side = inputParams.preProcess.side;

	

	betaParam = 2*inputParams.betaParam*inputParams.betaParam; //  betaParamIn;

	cParam    = 2*inputParams.cParam*inputParams.cParam;    //  cParamIn;


	postProcess.variance = inputParams.postProcess.variance;

	postProcess.side = inputParams.postProcess.side;


	

	initKernels();




int reloadMatchingParameters(const char * filename,stereoMatchingParam &loadedParameters){



	FILE* pFile = fopen (filename,"r");

	if (pFile!=NULL)

	{

		//The file is valid read in the numbers.

		float stereoVariance,expGain;

		int gaussSide;

		int results = fscanf(pFile, "%f, %f",&stereoVariance,&expGain);

		fclose(pFile);

		//verify that all parameters are loaded.

		if(results == 2)

		{

			//Ensure that the parameters work

	

			loadedParameters.stereoVariance = stereoVariance;

			loadedParameters.expGain = expGain;

			//loadedParameters.endCostGain  = matchGain;

			

			printf("The parameters were loaded successfully.\n");

			printf("Parameter summary:\n");

			printf("stereoVariance parameter: %f \n",stereoVariance);

			printf("exp Gain  parameter: %f \n",expGain);

			//printf("end cost gain parameter: %f \n",matchGain);

			printf("\n");


		

			return 1;

		}

		else return 0;

	}


	else return -1; 

}


int reloadSegmentParameters(const char * filename,segmentThinParam & loadedParameters){



	FILE* pFile = fopen (filename,"r");

	if (pFile!=NULL)

	{

		//The file is valid read in the numbers.

		float betaP,cP,preGaussSigma,postGaussSigma;
		int sobelParam,preGaussSide,postGaussSide;

		

		int results = fscanf (pFile, "%f, %f, %f, %d, %d, %f, %d", &betaP,&cP,&preGaussSigma,&preGaussSide,&sobelParam,&postGaussSigma,&postGaussSide);

		fclose(pFile);

		//verify that all 7 parameters are loaded.

		if(results == 7)

		{

			//Ensure that the parameters work

			if(preGaussSide%2 == 0)

			{

				preGaussSide++;

			}

			if(postGaussSide%2 == 0)

			{

				postGaussSide++;

			}

			//int sobelList[] = {1,3,5,7};

			if(sobelParam > 7)

			{

				sobelParam = 7;

			}

			if(sobelParam%2 ==0)

			{

				sobelParam++;

			}


			loadedParameters.betaParam = betaP;

			loadedParameters.cParam = cP;

			loadedParameters.sobelParam = sobelParam;

			loadedParameters.preProcess.side = preGaussSide;

			loadedParameters.preProcess.variance = preGaussSigma;

			loadedParameters.postProcess.side = postGaussSide;

			loadedParameters.postProcess.variance = postGaussSigma;


			printf("The parameters were loaded successfully.\n");

			printf("Parameter summary:\n");

			printf("beta parameter: %f \n",betaP);

			printf("c parameter: %f \n",cP);

			printf("sobel parameter: %d \n",sobelParam);

			printf("Pre gaussian variance: %f \n",preGaussSigma);

			printf("Pre gaussian kernal size: %d \n",preGaussSide);

			printf("Post gaussian variance: %f \n",postGaussSigma);

			printf("Post gaussian kernal size: %d \n",postGaussSide);

			printf("\n");


		

			return 1;

		}

		else return 0;

	}


	else return -1; 

}



void reshapeROIRect(const Mat& imageBase, Rect &rectInput,int buffer){


	//For safety

	//

	//offset x lower bound.

	if((rectInput.x-buffer) < 0)

	{

		rectInput.x = 0;

	}

	else rectInput.x -=buffer;



	//offset y lower bound

	if((rectInput.y-buffer) < 0)

	{

		rectInput.y = 0;

	}

	else rectInput.y -=buffer;

	



	//Offset x width

	if(rectInput.width > (imageBase.cols-rectInput.x-2*buffer) || rectInput.width < 0) {

		

		rectInput.width = imageBase.cols-rectInput.x;

		if(rectInput.width < 0) rectInput.width = imageBase.cols;

	}

	else rectInput.width +=2*buffer;



	//Offset y height

	if(rectInput.height > (imageBase.rows-rectInput.y-2*buffer) || rectInput.height < 0) {

		

		rectInput.height = imageBase.rows-rectInput.y;

		if(rectInput.height < 0) rectInput.height = imageBase.rows;

	}

	else rectInput.height +=2*buffer;

}



/*void computeDirectionKernel(Mat &kernel,int x, int y){


	//Use the int x and int y to define the kernel shape

	//

	kernel = Mat(5,5,CV_32F);


	double* kernelP = (double*) kernel.data;

	

	if( x > 0 && y > 0){


		for(int i = 0; i <= 7; i++)

		{


			for(int j = 0; j <= 7; j++)

			{

			


		



			}



		}


} */




void generateThreadMask(const stereoImage &inputImages,stereoImage &maskOutput,const Scalar &lowerRange,const Scalar &upperRange, stereoROI inputROI)

{

	

	Mat labSpace,labSpaceFiltered;

	

	Mat inputInterest,outputInterest;

	


	for(int i =0; i < 2; i++)

	{

		reshapeROIRect(inputImages[i],inputROI[i],25);

		inputInterest = inputImages[i](inputROI[i]);

		maskOutput[i]= Mat::zeros(inputImages[i].rows,inputImages[i].cols,CV_8U);

		outputInterest = maskOutput[i](inputROI[i]);

		cvtColor(inputInterest,labSpace,CV_BGR2Lab);

		inRange(labSpace,lowerRange,upperRange,outputInterest);


		

#ifdef MASKDEBUG

		imshow("labImage",labSpace);

		imshow("labImageFilt",maskOutput[i]);

		waitKey(0);

		destroyWindow("labImage");

		destroyWindow("labImageFilt");

#endif




	}





}




void segmentThin(const Mat &inputImage,Mat &outputImage,const segmentThinParam &inputParams,Rect regionROI){

	//This function segments the image based on Hessians. 

	//The Hessians' eigenvalues are used to infer whether an object is shaped like a needle or thread.


	reshapeROIRect(inputImage,regionROI,25);


	if((outputImage.cols != inputImage.cols) ||  (outputImage.rows != inputImage.rows))

	{

		outputImage = Mat::zeros(inputImage.rows,inputImage.cols,CV_32F);

	}

	else outputImage.setTo(0.0);


	Mat preOutputImage = outputImage.clone();



	//for now only use one image:

	Mat inputROI =  inputImage(regionROI);

	Mat preOutputROI = preOutputImage(regionROI);


	double imageGain = (2*3.14*inputParams.postProcess.variance);


	//Will static cause problems?

    static	Mat greyImage,greyBlur;

	static  Mat greyFloat(regionROI.height,regionROI.width,CV_32FC1);


	static  Mat grad_x,grad_y,grad_xx,grad_yx,grad_yy;

	 


	//Use an ROI for speed?



	cvtColor(inputROI,greyImage,CV_BGR2GRAY);

	greyImage.convertTo(greyFloat,CV_32FC1);

	double *greyFloatPtr = (double*) greyFloat.data;


	greyFloat /= 255.0;

	

#ifdef DEBUGSUTURESEGMENT

	imshow("greyImage",greyImage);

	imshow("greyfloat",greyFloat);

	waitKey(00);

	destroyWindow("greyImage");

	destroyWindow("greyfloat");

#endif


	//enter a for loop for the kernal sizing;

	Size gaussSize = Size(inputParams.preProcess.side,inputParams.preProcess.side);

	GaussianBlur(greyFloat,greyBlur,gaussSize,inputParams.preProcess.variance,inputParams.preProcess.variance);

	//Compute the hessian

	Sobel(greyBlur,grad_xx,-1,2,0,inputParams.sobelParam);

	Sobel(greyBlur,grad_yx,-1,1,1,inputParams.sobelParam);

	//grad_xy = grad_yx.clone();

	Sobel(greyBlur,grad_yy,-1,0,2,inputParams.sobelParam);

		

	//Now that the Hessian components have been calculated, move through

	//The image points and compute the eigen values.

		

	//Compute the number of total pixels

	int pixCount = greyBlur.rows*greyBlur.cols;

		

	double *gradPtr_xx = (double*)  grad_xx.data;

	double *gradPtr_yx = (double*)  grad_yx.data;

	double *gradPtr_xy = (double*)  grad_yx.data;

	double *gradPtr_yy = (double*)  grad_yy.data;



	char* preOutputImagePtr = (char*) preOutputROI.data;

	int preOutputImageStep0 =  preOutputROI.step[0];

	int preOutputImageStep1 =  preOutputROI.step[1];

		




	//for each image, evaluate its eigen vectors, then look at the cost

	//function from Frangi et al.	

	for(int i =0 ; i < pixCount; i=i++){

	

		double eigLow;

		double eigHigh;

			//cout << hessian << endl;

			

		if(sqrt( gradPtr_xx[i]*gradPtr_xx[i]+gradPtr_yx[i]*gradPtr_yx[i]+gradPtr_yy[i]*gradPtr_yy[i]) < 0.0001) continue;


			

		double a = 1.0f;

		double b = -gradPtr_xx[i]-gradPtr_yy[i];

		double c = gradPtr_xx[i]*gradPtr_yy[i]-gradPtr_yx[i]*gradPtr_yx[i];


		double eig0 = (-b+sqrt(b*b-4*a*c))/(2*a);

		double eig1 = (-b-sqrt(b*b-4*a*c))/(2*a);


		//eigen(hessian,eigenVects,eigenVals);


				

		//cout << eigenVects << endl;

		//cout << eigTest0 << " " << eigTest1 << endl;

		//cout << eigenVals << endl;

		double fNorm = sqrt(eig0*eig0+eig1*eig1);  //norm(eigenVals);


		//Now find the low and the high eigenvalues;

			

		//double eig0 = eigenVals.at<double>(0);

		//double eig1 = eigenVals.at<double>(1);


		if(abs(eig0) > abs(eig1))

		{

			eigLow = eig1;

			eigHigh = eig0;

		}	

		else

		{

			eigLow = eig0;

			eigHigh = eig1;

		}

		double vMag;

		if(eigHigh == 0.00) vMag = 0.0;


		else{

				

			double r_Beta = eigLow/eigHigh;

			

			vMag = imageGain*exp(-r_Beta*r_Beta/(2*inputParams.betaParam*inputParams.betaParam))*(1-exp(-fNorm*fNorm/(2*inputParams.cParam*inputParams.cParam)));

			

		}

		



		int xPos =  i%greyBlur.cols;

		int yPos =  (int) floor(((double) i)/((double) greyBlur.cols));

		

		double* finalPointer =  (double*) (preOutputImagePtr+ preOutputImageStep0*yPos + preOutputImageStep1*xPos); 

	 	

		finalPointer[0] = vMag;

		



	}

	//Once all is said and done, blur the final image using a gaussian.


	

	GaussianBlur(preOutputImage,outputImage,Size(inputParams.postProcess.side,inputParams.postProcess.side), inputParams.postProcess.variance,inputParams.postProcess.variance);


	//convertScaleAbs(outputImage,outputImageDisp);


			

#ifdef DEBUGSUTURESEGMENT

	Mat outputImageDisp,preOutputImageDisp;

	convertScaleAbs(outputImage,outputImageDisp,(255.0/imageGain));

	convertScaleAbs(preOutputImage,preOutputImageDisp,(255.0/imageGain));

	//cout << segmentMean << endl;

	//cout << segmentStdDev << endl;

	//imshow((i? "left original":"right original"),inputImage[i]);

	//imshow((i? "left blur":"right blur"),imageBlur[i]);

	//imshow("lab",lab[i]);

	//cv::setMouseCallback("lab",uicallback::displayPixel,&lab);

	imshow("OutputResults",outputImageDisp);

	imshow("preOutputResults",preOutputImageDisp);

	//	cv::setMouseCallback("segment",uicallback::displayPixel,&segment);

	//	imshow((i? "left Mask":"right mask"),segmentMask);

			

	/*imshow((i? "left original":"right original"),inputImage[i]);

	imshow((i? "left blur":"right blur"),imageBlur[i]);

	imshow((i? "left hsv":"right hsv"),hsv[i]);

	imshow((i? "left lab":"right lab"),lab[i]);

	imshow((i? "left hsv hue":"right hsv hue"),hueHSV[i]);

			

	imshow((i? "left lab light":"right lab light"),lightLAB[i]);

	imshow((i? "left lab b":"right lab b"),bLAB[i]);

	imshow((i? "left lab hue":"right lab hue"),hueLAB[i]);

	//imshow((i? "left maska":"right maska"),maska[i]);

	//imshow((i? "left maskb":"right maskb"),maskb[i]);*/

	//	imshow((i? "left mask":"right mask"),mask[i]); 

	waitKey(00);

	//	cv::destroyAllWindows();

	destroyWindow("OutputResults");

	destroyWindow("preOutputResults");


#endif



	//return outputImage;

}


/*void segmentThinVec(const Mat &inputImage,Mat &outputImage,const segmentThinParam &inputParams){

	//This function segments the image based on Hessians. 

	//The Hessians' eigenvalues are used to infer whether an object is shaped like a needle or thread.

	if((outputImage.cols != inputImage.cols) ||  (outputImage.rows != inputImage.rows))
	{
		outputImage = Mat::zeros(inputImage.rows,inputImage.cols,CV_32F);
	}

	else outputImage.setTo(0.0);


	Mat preOutputImage = outputImage.clone();



	//for now only use one image:

	Mat inputROI =  inputImage(regionROI);

	Mat preOutputROI = preOutputImage(regionROI);


	double imageGain = (2*3.14*inputParams.postProcess.variance);


	//Will static cause problems?

    static	Mat greyImage,greyBlur;

	static  Mat greyFloat(regionROI.height,regionROI.width,CV_32FC1);


	static  Mat grad_x,grad_y,grad_xx,grad_yx,grad_yy;

	 


	//Use an ROI for speed?



	cvtColor(inputROI,greyImage,CV_BGR2GRAY);

	greyImage.convertTo(greyFloat,CV_32FC1);

	double *greyFloatPtr = (double*) greyFloat.data;


	greyFloat /= 255.0;

	

#ifdef DEBUGSUTURESEGMENT

	imshow("greyImage",greyImage);

	imshow("greyfloat",greyFloat);

	waitKey(00);

	destroyWindow("greyImage");

	destroyWindow("greyfloat");

#endif


	//enter a for loop for the kernal sizing;

	Size gaussSize = Size(inputParams.preProcess.side,inputParams.preProcess.side);

	GaussianBlur(greyFloat,greyBlur,gaussSize,inputParams.preProcess.variance,inputParams.preProcess.variance);

	//Compute the hessian

	Sobel(greyBlur,grad_xx,-1,2,0,inputParams.sobelParam);

	Sobel(greyBlur,grad_yx,-1,1,1,inputParams.sobelParam);

	//grad_xy = grad_yx.clone();

	Sobel(greyBlur,grad_yy,-1,0,2,inputParams.sobelParam);

		

	//Now that the Hessian components have been calculated, move through

	//The image points and compute the eigen values.

		

	//Compute the number of total pixels

	int pixCount = greyBlur.rows*greyBlur.cols;

		

	double *gradPtr_xx = (double*)  grad_xx.data;

	double *gradPtr_yx = (double*)  grad_yx.data;

	double *gradPtr_xy = (double*)  grad_yx.data;

	double *gradPtr_yy = (double*)  grad_yy.data;



	char* preOutputImagePtr = (char*) preOutputROI.data;

	int preOutputImageStep0 =  preOutputROI.step[0];

	int preOutputImageStep1 =  preOutputROI.step[1];

		




	//for each image, evaluate its eigen vectors, then look at the cost

	//function from Frangi et al.	

	for(int i =0 ; i < pixCount; i=i++){

	

		double eigLow;

		double eigHigh;

			//cout << hessian << endl;

			

		if(sqrt( gradPtr_xx[i]*gradPtr_xx[i]+gradPtr_yx[i]*gradPtr_yx[i]+gradPtr_yy[i]*gradPtr_yy[i]) < 0.0001) continue;


			

		double a = 1.0f;

		double b = -gradPtr_xx[i]-gradPtr_yy[i];

		double c = gradPtr_xx[i]*gradPtr_yy[i]-gradPtr_yx[i]*gradPtr_yx[i];


		double eig0 = (-b+sqrt(b*b-4*a*c))/(2*a);

		double eig1 = (-b-sqrt(b*b-4*a*c))/(2*a);


		//eigen(hessian,eigenVects,eigenVals);


				

		//cout << eigenVects << endl;

		//cout << eigTest0 << " " << eigTest1 << endl;

		//cout << eigenVals << endl;

		double fNorm = sqrt(eig0*eig0+eig1*eig1);  //norm(eigenVals);


		//Now find the low and the high eigenvalues;
		//double eig0 = eigenVals.at<double>(0);
		//double eig1 = eigenVals.at<double>(1);
		if(abs(eig0) > abs(eig1))

		{

			eigLow = eig1;

			eigHigh = eig0;

		}	

		else

		{

			eigLow = eig0;

			eigHigh = eig1;

		}

		double vMag;

		if(eigHigh == 0.00) vMag = 0.0;


		else{

				

			double r_Beta = eigLow/eigHigh;

			

			vMag = imageGain*exp(-r_Beta*r_Beta/(2*inputParams.betaParam*inputParams.betaParam))*(1-exp(-fNorm*fNorm/(2*inputParams.cParam*inputParams.cParam)));

			

		}

		



		int xPos =  i%greyBlur.cols;

		int yPos =  (int) floor(((double) i)/((double) greyBlur.cols));

		

		double* finalPointer =  (double*) (preOutputImagePtr+ preOutputImageStep0*yPos + preOutputImageStep1*xPos); 

	 	

		finalPointer[0] = vMag;

		



	}

	//Once all is said and done, blur the final image using a gaussian.


	

	GaussianBlur(preOutputImage,outputImage,Size(inputParams.postProcess.side,inputParams.postProcess.side), inputParams.postProcess.variance,inputParams.postProcess.variance);


	//convertScaleAbs(outputImage,outputImageDisp);


			

#ifdef DEBUGSUTURESEGMENT

	Mat outputImageDisp,preOutputImageDisp;

	convertScaleAbs(outputImage,outputImageDisp,(255.0/imageGain));

	convertScaleAbs(preOutputImage,preOutputImageDisp,(255.0/imageGain));

	//cout << segmentMean << endl;

	//cout << segmentStdDev << endl;

	//imshow((i? "left original":"right original"),inputImage[i]);

	//imshow((i? "left blur":"right blur"),imageBlur[i]);

	//imshow("lab",lab[i]);

	//cv::setMouseCallback("lab",uicallback::displayPixel,&lab);

	imshow("OutputResults",outputImageDisp);

	imshow("preOutputResults",preOutputImageDisp);

	//	cv::setMouseCallback("segment",uicallback::displayPixel,&segment);

	//	imshow((i? "left Mask":"right mask"),segmentMask);

			

	/*imshow((i? "left original":"right original"),inputImage[i]);

	imshow((i? "left blur":"right blur"),imageBlur[i]);

	imshow((i? "left hsv":"right hsv"),hsv[i]);

	imshow((i? "left lab":"right lab"),lab[i]);

	imshow((i? "left hsv hue":"right hsv hue"),hueHSV[i]);

			

	imshow((i? "left lab light":"right lab light"),lightLAB[i]);

	imshow((i? "left lab b":"right lab b"),bLAB[i]);

	imshow((i? "left lab hue":"right lab hue"),hueLAB[i]);

	//imshow((i? "left maska":"right maska"),maska[i]);

	//imshow((i? "left maskb":"right maskb"),maskb[i]);* /

	//	imshow((i? "left mask":"right mask"),mask[i]); 

	waitKey(00);

	//	cv::destroyAllWindows();

	destroyWindow("OutputResults");

	destroyWindow("preOutputResults");


#endif



	//return outputImage;

} */



double endPointCosts(const Mat &segmentedImage,const std::vector<cv::Point> &interestPoints,double offsetLength){


	int n = (int) interestPoints.size();

	double totalSum = 0.0;


	for(int i = 0; i < 2; i++){

		Point2d initialVect;

		Point2d initialPoint;


		switch(i)

		{

		case 0: 

			initialVect  = interestPoints[0]-interestPoints[1];

			initialPoint = interestPoints[0];

			break;


		case 1:

			initialVect  = interestPoints[n-1]-interestPoints[n-2];

			initialPoint = interestPoints[n-1];

			break;

		}

		double initialInvNorm =1/norm(initialVect);


		if(initialInvNorm > 1000000) initialInvNorm = 10000;


		Point2d initialOffset = initialPoint+initialVect*initialInvNorm*offsetLength;

		

		

		LineIterator it(segmentedImage, initialOffset, initialPoint, 8);


		for(int j = 0; j < it.count; j++, ++it)

	

		{


			//double val0 = *(const double*)*it;

			

		

			//double val1 = segmentedImage.at<double>(it.pos());

		

			totalSum+= *(const double*)*it;


		}

	

	}

	return totalSum;

}



//This function score an image as it is matched against a model of points.

//The point model is a vector.

//This function, draws the point sequence and then blurs it to help make the

//matching more like climbing a hill

double scoreImageMatch(const Mat &segmentedImage,const std::vector<cv::Point> &interestPoints,Rect rectROI,bool displayPause){



	//static containers to save allocation deallocation time. 

	reshapeROIRect(segmentedImage,rectROI,0);


	static Mat generatedImage;

	static Mat generatedImageGauss;

	static Mat outputImage(segmentedImage.rows,segmentedImage.cols,CV_32FC1);

	

	double totalLength=0.0;

	


	//The image is assumed to be a single channel unsigned charactor matrix.

	std::vector<std::vector<cv::Point>> ptList;

	ptList.clear();

	

	//ROI

	static Mat imageROI,generatedROI, outputROI;


	generatedImage = Mat::zeros(segmentedImage.rows,segmentedImage.cols,CV_32FC1);


	ptList.resize(1,interestPoints);	


	int n = (int) interestPoints.size();


	int rows = segmentedImage.rows;

	int cols = segmentedImage.cols;


	//Instead of poly lines, use a line iterator.


	double totalSum = 0.0;

	int ptCounter = 0;


	for(int i = 1; i < n; i++)

	{

		LineIterator it(segmentedImage, interestPoints[i], interestPoints[i-1], 8);

		ptCounter += it.count;

		for(int j = 0; j < it.count; j++, ++it)

		{

			//double val0 = *(const double*)*it;

			

			//double val1 = segmentedImage.at<double>(it.pos());

			totalSum+= *(const double*)*it;

			

		}


	}




	



	/*polylines(generatedImage, ptList,false,Scalar(1.0));

	

	

	//Done generating the image, 1. Use and ROI?

	//generate ROI:

	generatedROI = Mat(generatedImage, rectROI);

	imageROI     = Mat(segmentedImage, rectROI);

	outputROI    = Mat(outputImage,rectROI);


	//At this point, Blur the image based on the gaussian infromation. 


	cv::multiply(generatedROI,imageROI,outputROI);

	

	//GaussianBlur(generatedImage,generatedImageGauss, gaussSize, blurMatchInfo.variance,blurMatchInfo.variance);

	//cv::multiply(generatedImageGauss,segmentedImage,outputImage);


	if(displayPause)

	{

		Mat imageROIChar,generatedROIChar,generatedImageGaussChar,outputROIChar;

		convertScaleAbs(imageROI,imageROIChar,255.0);

		convertScaleAbs(generatedROI,generatedROIChar,255.0);

		convertScaleAbs(outputROI,outputROIChar,255.0);


		/*convertScaleAbs(segmentedImage,imageROIChar,255.0);

		convertScaleAbs(generatedImage,generatedROIChar,255.0);

		convertScaleAbs(generatedImageGauss,generatedImageGaussChar,255.0);

		convertScaleAbs(outputImage,outputROIChar,255.0);* /


		imshow("image ROI",imageROIChar);

		imshow("generated ROI",generatedROIChar);

		imshow("Results ROI",outputROIChar);


		waitKey(0);


		destroyWindow("image ROI");

		destroyWindow("generated ROI");

		destroyWindow("Results ROI");

	}



	//Compute the sum? 

	

	Scalar imageSum = sum(outputROI);

	//Scalar imageSum = sum(outputImage);


	//double factor =  1/((double) (rows*cols)) ;

	//double factor =  1/(totalLength) ;



	double matching= imageSum[0]; //*factor; */


	double meanMatch = totalSum/((double) ptCounter);


	return meanMatch;

}



/*double scoreImageMatch(const Mat &segmentedImage,const std::vector<stereoCorrespondence> &interestPoints,int index,gaussParam blurMatchInfo,bool displayPause){



	//static containers to save allocation deallocation time. 

	static Mat generatedImage;

	static Mat generatedImageGauss;

	static Mat outputImage;

	

	double totalLength=0.0;

	


	//The image is assumed to be a single channel unsigned charactor matrix.

	std::vector<std::vector<cv::Point>> ptList;

	ptList.clear();

	ptList.resize(1);


	//ROI

	Mat imageROI,generatedROI, outputROI;


	generatedImage = Mat::zeros(segmentedImage.rows,segmentedImage.cols,CV_32FC1);


	


	int n = (int) interestPoints.size();


	int rows = segmentedImage.rows;

	int cols = segmentedImage.cols;



	double scalarMag = (2*3.14*blurMatchInfo.variance);


	int maxRow = 0;

	int minRow = rows;

	int maxCol = 0;

	int minCol = cols;


	for(int i = 0; i < n; i++)

	{

		/*if(i > 0)

		{

			double dist = norm(interestPoints[i]-interestPoints[i-1]);

			totalLength+=dist;



		}* /

		int j,k;

		


		if(index == 0){

			j = (int) interestPoints[i].left.x;

			k = (int) interestPoints[i].left.y;

		}

		else{	

			j = (int) interestPoints[i].right.x;

			k = (int) interestPoints[i].right.y;

		}


		if( j < cols && k < rows){

			ptList[0].push_back(Point(j,k));

			//double* matAddress = (double*) (generatedImage.data+generatedImage.step[0]*k+generatedImage.step[1]*j); 

			// imageValue = segmentedImage.data(k,j);


			//matAddress[0] = 1.0;

			

			if(j >=  maxCol) maxCol = j+1;

			if(j < minCol) minCol = j;

			if(k >=  maxRow) maxRow = k+1;

			if(k < minRow) minRow = k;

		}

	}

	if(ptList[0].size() > 0)

	{

		polylines(generatedImage, ptList,false,Scalar(scalarMag));

	}

	int yROI = minRow-blurMatchInfo.side*2;


	if((minRow-blurMatchInfo.side) <0) yROI = 0;


	int heightROI = maxRow+blurMatchInfo.side*2-minRow;

	if((heightROI+yROI) > (rows)) heightROI =  (rows-yROI);


	int xROI = minCol-blurMatchInfo.side;


	if((minCol-blurMatchInfo.side) <0) xROI = 0;


	int widthROI = maxCol+blurMatchInfo.side*2-minCol;

	if((widthROI+xROI)  > (cols)) widthROI =  (cols-xROI);



	//Generated the Rect for the ROI:

	

	Rect rectROI;


	rectROI.x = xROI;

	rectROI.y = yROI;

	rectROI.width = widthROI;

	rectROI.height = heightROI;

	


	//Done generating the image, 1. Use and ROI?

	//generate ROI:

	generatedROI = Mat(generatedImage, rectROI);

	imageROI     = Mat(segmentedImage, rectROI);


	//At this point, Blur the image based on the gaussian infromation. 


	Size gaussSize(blurMatchInfo.side,blurMatchInfo.side);


	GaussianBlur(generatedROI,generatedImageGauss, gaussSize, blurMatchInfo.variance

,blurMatchInfo.variance);



	cv::multiply(generatedImageGauss,imageROI,outputROI);


	if(displayPause)

	{

		Mat imageROIChar,generatedROIChar,generatedImageGaussChar,outputROIChar;

		convertScaleAbs(imageROI,imageROIChar,255.0);

		convertScaleAbs(generatedROI,generatedROIChar,255.0);

		convertScaleAbs(generatedImageGauss,generatedImageGaussChar,255.0);

		convertScaleAbs(outputROI,outputROIChar,255.0);

		imshow("image ROI",imageROIChar);

		imshow("generated ROI",generatedROIChar);

		imshow("blured ROI",generatedImageGaussChar);

		imshow("Results ROI",outputROIChar);


		waitKey(0);


		destroyWindow("image ROI");

		destroyWindow("generated ROI");

		destroyWindow("blured ROI");

		destroyWindow("Results ROI");

	}



	//Compute the sum? 

	

	Scalar imageSum = sum(outputROI);


	//double factor =  1/((double) (rows*cols)) ;

	//double factor =  1/(totalLength) ;



	double matching= imageSum[0]; //*factor;



	return matching;

} */





