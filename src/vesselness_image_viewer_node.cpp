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

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
//Converts a simgle image into a displayable RGB format.
void convertSegmentImage(const Mat&src,Mat&dst){
	
		
    std::cout << "Converting the image" << std::endl;
	Mat temp1 = src.mul(Scalar(1/3.14159,1.0));
	Mat temp2,temp3;

    std::cout << "Scaled the image" << std::endl;
    temp3.create(src.rows,src.cols,CV_8UC3);

	Mat tempHalf=Mat::ones(src.rows,src.cols,CV_8UC1)*127;
	
    convertScaleAbs(temp1,temp2,255.0);
	
	Mat in[] = {temp2,tempHalf};


	// forming an array of matrices is a quite efficient operation,
	// because the matrix data is not copied, only the headers
	// rgba[0] -> bgr[2], rgba[1] -> bgr[1],
	// rgba[2] -> bgr[0], rgba[3] -> alpha[0]
   	int from_to[] = {0,0, 1,1, 2,2};
	
     std::cout << "mix the image" << std::endl;

    mixChannels(in, 2, &temp3, 1, from_to, 3 );
	cvtColor(temp3,dst,CV_HSV2BGR);

}



static const std::string OPENCV_WINDOW = "Image window";

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/vesselness/output", 1, 
      &ImageConverter::imageCb, this);
  
    cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    Mat outputImage;
    convertSegmentImage(cv_ptr->image,outputImage);

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, outputImage);
    cv::waitKey(3);

    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_viewer");
  ImageConverter ic;
  ros::spin();
  return 0;
}
