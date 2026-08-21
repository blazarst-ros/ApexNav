cmake_minimum_required(VERSION 3.0.2)
project(apexnav_gazebo)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge diagnostic_msgs gazebo_msgs geometry_msgs mavros_msgs message_filters nav_msgs
  plan_env rospy sensor_msgs std_msgs std_srvs visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/apexnav_state_logger.py
  scripts/gazebo_sensor_adapter.py
  scripts/gazebo_vision_height_bridge.py
  scripts/px4_mission_supervisor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS scripts/lite_perception_launcher.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_hm3dv2_camera.py)
  catkin_add_nosetests(test/test_gazebo_height_filter_launch.py)
  catkin_add_nosetests(test/test_apexnav_state_logger.py)
  catkin_add_nosetests(test/test_px4_mission_supervisor.py)
  catkin_add_nosetests(test/test_gazebo_sensor_adapter.py)
  catkin_add_nosetests(test/test_gazebo_vision_height_bridge.py)
endif()
