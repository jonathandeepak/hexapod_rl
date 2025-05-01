cmake_minimum_required(VERSION 3.0.2)
project(hexapod_rl)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  gazebo_msgs

)

# enable setuptools-based installation of your Python modules
catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/hexapod_rl_training.py
  scripts/rl_inference.py
  scripts/hexapod_rl_training_headless.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(
  DIRECTORY hexapod_rl/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/hexapod_rl
  FILES_MATCHING PATTERN "*.py"
)
