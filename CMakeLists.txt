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


catkin_package(
    #   INCLUDE_DIRS include
    # LIBRARIES vesselness_image_filter
    # CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
    #   DEPENDS system_lib
)


#SET(CUDA_NVCC_FLAGS "-arch=sm_13" CACHE STRING "nvcc flags" FORCE)
#SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(LIB_TYPE STATIC) 


include_directories(
  include
  include/vesselness_image_filter_cpu
  ${catkin_INCLUDE_DIRS}
)


#CUDA_ADD_LIBRARY(vesselness_image_filter_gpu_kernels src/gpu_src/vesselness_image_filter_kernels.cu)
#CUDA_ADD_LIBRARY(vesselness_image_gpu src/gpu_src/vesselness_image_filter_gpu.cu)



add_library(vesselness_image_filter_cpu_lib
   src/vesselness_image_filter_cpu.cpp
 )

add_library(vesselness_image_filter_bw_cpu_lib
   src/vesselness_image_filter_cpu_bw.cpp
 )

#cuda_add_library( 

#add_library(vesselness_image_filter
#   src/${PROJECT_NAME}/src/vesselness_image_filter.cpp
# )



ADD_EXECUTABLE(vesselness_image_filter_cpu_node src/vesselness_image_filter_cpu_node.cpp)

ADD_EXECUTABLE(vesselness_image_filter_cpu_bw_node src/vesselness_image_filter_cpu_bw_node.cpp)

#ADD_EXECUTABLE(vesselness_image_filter_cpu_node_bw src/cpu_src/vesselness_image_filter_cpu_node_bw.cpp)


#ADD_EXECUTABLE(vesselness_image_filter_viewer_node src/vesselness_image_viewer_node.cpp)

#ADD_EXECUTABLE(vesselness_image_filter_gpu_node src/gpu_src/vesselness_image_filter_gpu_node.cpp)

#ADD_EXECUTABLE(vesselness_image_filter_cpu src/cpu_src/vesselness_image_filter_gpu.cpp)
#ADD_DEPENDENCIES(beginner_tutorials_node T)

#TARGET_LINK_LIBRARIES(
#   #vesselness_image_filter
#   ${catkin_LIBRARIES}
#  #${PCL_LIBRARIES}
#)


#target_link_libraries(vesselness_image_filter_cpu_node_bw
#   vesselness_image_base
#   vesselness_image_cpu
#   ${catkin_LIBRARIES}
# )

target_link_libraries(vesselness_image_filter_cpu_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_cpu_lib
   ${catkin_LIBRARIES}
 )


target_link_libraries(vesselness_image_filter_cpu_bw_node
   vesselness_image_filter_common_lib
   vesselness_image_filter_bw_cpu_lib
   ${catkin_LIBRARIES}
 )
#target_link_libraries(vesselness_image_filter_gpu_node
#   vesselness_image_base
#  vesselness_image_gpu
#  vesselness_image_filter_gpu_kernels
#  ${CUDA_LIBRARIES}
#  ${OpenCV_LIBRARIES}
#  ${catkin_LIBRARIES}
# )

target_link_libraries(vesselness_image_filter_cpu_lib
  vesselness_image_filter_common_lib
   ${catkin_LIBRARIES}
 )



target_link_libraries(vesselness_image_filter_bw_cpu_lib
    ${catkin_LIBRARIES}
    )


#target_link_libraries(vesselness_image_gpu
# vesselness_image_gpu
# vesselness_image_filter_gpu_kernels
#  vesselness_image_base
#   ${CUDA_LIBRARIES}
#   ${catkin_LIBRARIES}
# )



#target_link_libraries(vesselness_image_filter_viewer_node
#   ${catkin_LIBRARIES}
# )
