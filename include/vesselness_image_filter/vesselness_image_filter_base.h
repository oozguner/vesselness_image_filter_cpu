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


#ifndef VESSELNESSNODEH
#define VESSELNESSNODEH

#include <vector>
#include <stdio.h>
#include <iostream>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

/*TODO-introduce the file and its function*/

struct gaussParam{
	
	float variance;
	int side;

};

struct segmentThinParam{

	gaussParam preProcess;
	gaussParam postProcess;

	float betaParam;
	float cParam;
	int sobelParam;
};


double meanAngle(const Mat & ,const Mat &);
double angleVar(const Mat & ,const Mat &);
void matAngleAlignment(const Mat &,Mat &,Point2d);
void convertSegmentImage(const Mat&,Mat&);

/*TODO-discuss the base class*/
class VesselnessNodeBase{

private:
    //ROS variables
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Subscriber settings_sub_;
    image_transport::Publisher image_pub_;


protected:

    
    //
    Mat outputImage;
    Size imgAllocSize;

	
    //Hessian Kernel Parameters:
    gaussParam hessParam;
	
    //Eigen weight params:
    float betaParam;
    float cParam;
    //Output post processing Parameters:
    gaussParam postProcess;
	
    
	


public:
	
    //main and only constructor
    VesselnessNodeBase(const char*);


    //default destructor: none
    ~VesselnessNodeBase(); 


   //image masking function... in order to improve efficiency and speed.
   //virtual void setImageMask(const Mat &) = 0;

	//callback hook
    void  imgTopicCallback(const sensor_msgs::ImageConstPtr&);
    //void  updateParameters(const segmentThinParam&);

    
//image processing functions
    virtual void segmentImage(const Mat&, Mat &)=0;

    //memory allocation function.
    virtual void allocateMem(int,int) = 0;
    virtual void initKernels()= 0;

};





#endif
