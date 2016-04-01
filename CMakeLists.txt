cmake_minimum_required(VERSION 2.8.3)
project(vesselness_image_filter_cpu)
find_package(OpenCV REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  vesselness_image_filter_common
)


SET(LIB_TYPE STATIC) 

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES vesselness_lib
  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs vesselness_image_filter_common
)

add_library(vesselness_image_filter_cpu_lib
   src/vesselness_image_filter_cpu.cpp
 )

add_library(vesselness_lib
   src/vesselness_lib.cpp
 )

add_library(vesselness_image_filter_bw_cpu_lib
   src/vesselness_image_filter_cpu_bw.cpp
 )


ADD_EXECUTABLE(vesselness_image_filter_cpu_node src/vesselness_image_filter_cpu_node.cpp)

ADD_EXECUTABLE(vesselness_image_filter_cpu_bw_node src/vesselness_image_filter_cpu_bw_node.cpp)

ADD_EXECUTABLE(vesselness_image_filter_cpu_viewer_node src/vesselness_image_viewer_node.cpp)


target_link_libraries(vesselness_image_filter_cpu_lib
  vesselness_image_filter_common_lib
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
 )

target_link_libraries(vesselness_image_filter_cpu_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_cpu_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )


target_link_libraries(vesselness_image_filter_bw_cpu_lib
    vesselness_image_filter_common_lib
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )

target_link_libraries(vesselness_image_filter_cpu_bw_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_bw_cpu_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )

target_link_libraries(vesselness_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )

target_link_libraries(vesselness_image_filter_cpu_viewer_node
   vesselness_lib
   ${OpenCV_LIBRARIES}
   ${catkin_LIBRARIES}
 )
