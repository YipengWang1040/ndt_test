cmake_minimum_required(VERSION 3.0.2)
project(ndt_matching)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
	include
	${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_executable(ndt_matching src/main.cpp src/debug.cpp src/voxel_grid_covariance_rgb.cpp)
target_link_libraries(ndt_matching ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES}  )
