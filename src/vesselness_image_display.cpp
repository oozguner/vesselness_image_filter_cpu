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
 *   * Neither the name of Case Western Reserve Univeristy, Inc. nor the names of its
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


// Converts a simgle image into a displayable RGB format.
void convertSegmentImage(const cv::Mat&src,cv::Mat&dst){
	
	Mat tempDisplay1,tempDisplay2;
	
	tempDisplay1 = src.mul(Scalar(1/3.14159,1.0,1.0));
	convertScaleAbs(tempDisplay1,tempDisplay2,255.0);
	cvtColor(tempDisplay2,dst,CV_HSV2BGR);
}


void findOutputCutoff(const cv::Mat&src, double *cuttOff, int iters)
{
    // this refines the cuttoff mean of the image.
	Scalar meanOut;

    if (cuttoff[0] <= 0)
    {
	    double mean = mean(src)[1];
	    cuttoff[0] = meanOut[lr]/2;
    } 

	for (int i(0); i < 10; i++)
	{
		inRange(thresh32f,Scalar(-7,cuttoffMean[lr],0),Scalar(7,1,1),threshMask);

		double mean0 = mean(segmentedIn[lr],threshMask < 100)[1];
		double mean1 = mean(segmentedIn[lr],threshMask > 100)[1];

		double newCuttoffMean = mean0/2+mean1/2;

		if(abs(newCuttoffMean-cuttoffMean[lr]) < 0.005){
			lowMean[lr] = mean0;
			highMean[lr] = mean1;
			cuttoffMean[lr] = newCuttoffMean;
			break;
		}
		else cuttoffMean[lr] = newCuttoffMean;
	}
}
