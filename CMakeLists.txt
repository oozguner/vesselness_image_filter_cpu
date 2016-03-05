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


include_directories(
  include
  include/vesselness_image_filter_cpu
  ${catkin_INCLUDE_DIRS}
)

catkin_package(
)

add_library(vesselness_image_filter_cpu_lib
   src/vesselness_image_filter_cpu.cpp
 )

add_library(vesselness_image_filter_bw_cpu_lib
   src/vesselness_image_filter_cpu_bw.cpp
 )



ADD_EXECUTABLE(vesselness_image_filter_cpu_node src/vesselness_image_filter_cpu_node.cpp)

ADD_EXECUTABLE(vesselness_image_filter_cpu_bw_node src/vesselness_image_filter_cpu_bw_node.cpp)


target_link_libraries(vesselness_image_filter_cpu_lib
  vesselness_image_filter_common_lib
   ${catkin_LIBRARIES}
 )

target_link_libraries(vesselness_image_filter_cpu_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_cpu_lib
   ${catkin_LIBRARIES}
 )


target_link_libraries(vesselness_image_filter_bw_cpu_lib
    vesselness_image_filter_common_lib
    ${catkin_LIBRARIES}
    )

target_link_libraries(vesselness_image_filter_cpu_bw_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_bw_cpu_lib
   ${catkin_LIBRARIES}
 )

