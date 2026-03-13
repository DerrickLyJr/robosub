cmake_minimum_required(VERSION 3.12.2)
project(bluerov2_gz)

find_package(ament_cmake_auto REQUIRED)

# Update GZ_SIM_RESOURCE_PATH
# Reference: https://colcon.readthedocs.io/en/released/developer/environment.html#dsv-files
ament_environment_hooks(hooks/setup.dsv)

ament_auto_find_build_dependencies()

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(teleop_node scripts/teleop_node.cpp)
ament_target_dependencies(teleop_node rclcpp std_msgs)

find_package(sensor_msgs REQUIRED)

add_executable(imu_listener scripts/imu_listener.cpp)
ament_target_dependencies(imu_listener rclcpp sensor_msgs)

# Install launch files from the launch directory to the share directory
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY scripts/computer_vision
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/depth_calculator.py
  scripts/stereo_vision.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  teleop_node
  imu_listener
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE models worlds)