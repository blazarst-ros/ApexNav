cmake_minimum_required(VERSION 2.8.3)
project(exploration_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  pcl_conversions
  plan_env
  path_searching
  vis_utils
  cv_bridge
  lkh_mtsp_solver
  trajectory_manager
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES exploration_manager
  CATKIN_DEPENDS plan_env path_searching vis_utils lkh_mtsp_solver sensor_msgs pcl_conversions
  #  DEPENDS system_lib
)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(exploration_node
  src/exploration_node.cpp
  src/dynamic_voronoi_allocator.cpp
  src/exploration_fsm.cpp
  src/exploration_fsm_traj.cpp
  src/exploration_manager.cpp
)
target_link_libraries(exploration_node
  ${catkin_LIBRARIES}
)

if(CATKIN_ENABLE_TESTING)
  add_executable(dynamic_voronoi_allocator_test
    ${PROJECT_SOURCE_DIR}/../../../tests/dynamic_voronoi_allocator_test.cpp
    src/dynamic_voronoi_allocator.cpp
  )
  add_test(NAME exploration_manager_dynamic_voronoi_allocator
    COMMAND dynamic_voronoi_allocator_test)
endif()
