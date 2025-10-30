#!/usr/bin/env bash
set -e


sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt-get update
sudo apt-get upgrade -y



# Install ROS 2 from debs + build tools so colcon exists later
sudo apt-get install -y ros-jazzy-ros-base python3-colcon-common-extensions

# Persist system-level sourcing (do this once)
grep -q 'source /opt/ros/jazzy/setup.bash' ~/.bashrc || \
  echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
# Source for current shell
source /opt/ros/jazzy/setup.bash

# Make (blank) workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Only build if packages exist; otherwise skip
if [ "$(ls -A src)" ]; then
  colcon build --symlink-install
  grep -q 'source ~/ros2_ws/install/setup.bash' ~/.bashrc || \
    echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
else
  echo "No packages in src — skipping colcon build (this is OK)."
fi

# Persist ROS env vars
grep -q 'export ROS_DOMAIN_ID=' ~/.bashrc || echo 'export ROS_DOMAIN_ID=1' >> ~/.bashrc
grep -q 'export ROS_NAMESPACE=' ~/.bashrc || echo 'export ROS_NAMESPACE=hornet_robotics_robosub' >> ~/.bashrc

source ~/.bashrc"
