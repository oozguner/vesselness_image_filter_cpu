/*  Copyright (c) 2014 Case Western Reserve University
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


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vesselness_image_filter_common/vesselness_image_filter_common.h>
#include <vesselness_image_filter_cpu/vesselness_filter_node_cpu.h>
 #include <vesselness_image_filter_cpu/vesselness_lib.h>


// Converts a simgle image into a displayable RGB format.
void convertSegmentImageCPU(const cv::Mat&src,cv::Mat&dst)
{
	cv::Mat temp1 = src.mul(cv::Scalar(1/3.14159,1.0));
	cv::Mat temp2,temp3;
    cv::convertScaleAbs(temp1,temp2,255.0);
   
    
    temp3.create(src.rows,src.cols,CV_8UC3);

	cv::Mat tempHalf=cv::Mat::ones(src.rows,src.cols,CV_8UC1)*127;
	
	
	cv::Mat in[] = {temp2,tempHalf};

    // forming an array of matrices is a quite efficient operation,
	  // because the matrix data is not copied, only the headers
	  // rgba[0] -> bgr[2], rgba[1] -> bgr[1],
	  // rgba[2] -> bgr[0], rgba[3] -> alpha[0]
   	int from_to[] = {0,0, 1,1, 2,2};
	


    cv::mixChannels(in, 2, &temp3, 1, from_to, 3 );
	cv::cvtColor(temp3,dst,CV_HSV2BGR);

}

//
void convertSegmentImageCPUBW(const cv::Mat&src,cv::Mat&dst)
{
	double maxVal;
	cv::minMaxLoc(src,NULL,&maxVal, NULL, NULL);
    
    cv::convertScaleAbs(src,dst,(255.0/maxVal));
}

void findOutputCutoff(const cv::Mat&src, double *cuttOff, int iters)
{
    // this refines the cuttoff mean of the image.
	cv::Scalar meanOut;

    if (cuttOff[0] <= 0)
    {
	    double mean = cv::mean(src)[1];
	    cuttOff[0] = mean/2;
    } 

    cv::Mat threshMask;

	for (int i(0); i < 10; i++)
	{
		cv::inRange(src,cv::Scalar(-7,cuttOff[0],0),cv::Scalar(7,10,1),threshMask);

		double mean0 = cv::mean(src,threshMask < 100)[1];
		double mean1 = cv::mean(src,threshMask > 100)[1];

		double newCuttoffMean = mean0/2+mean1/2;

		if(abs(newCuttoffMean-cuttOff[0]) < 0.005){
			cuttOff[0] = newCuttoffMean;
			break;
		}
		else cuttOff[0] = newCuttoffMean;
	}
}
