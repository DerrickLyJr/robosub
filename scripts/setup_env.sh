#!/bin/bash

# 1. Create Workspace if it doesn't exist
echo "📂 Setting up ROS 2 workspace..."
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 2. Clone the Repositories
echo "📥 Cloning BlueROV2 Gazebo and required tools..."
# Use -p to ignore error if directory already exists
if [ ! -d "bluerov2_gz" ]; then
    git clone https://github.com/clydemcqueen/bluerov2_gz.git
else
    echo "✅ bluerov2_gz already exists, skipping clone."
fi

# 3. Install System Dependencies
echo "📦 Installing ROS 2 Jazzy & Gazebo Harmonic dependencies..."
sudo apt update
sudo apt install -y ros-jazzy-ros-gz ros-jazzy-gz-sim-vendor python3-ruby

# 4. Build the Workspace
echo "🛠️ Building the workspace..."
cd ~/ros2_ws
colcon build --packages-select bluerov2_gz --symlink-install
source install/setup.bash

# 5. Generate the Model
echo "🤖 Generating the SDF model..."
cd ~/ros2_ws/src/bluerov2_gz/scripts
python3 generate_model.py ../models/bluerov2/model.sdf.in ../models/bluerov2/model.sdf ../models/bluerov2/configs.yaml

# 6. Set Environment Variables (Permanent)
if ! grep -q "GZ_SIM_RESOURCE_PATH" ~/.bashrc; then
    echo 'export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:~/ros2_ws/src/bluerov2_gz/models' >> ~/.bashrc
    echo "🌐 Added GZ_SIM_RESOURCE_PATH to .bashrc"
fi

echo "✅ Setup Complete! Please run 'source ~/.bashrc' or restart your terminal."