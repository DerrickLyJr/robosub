import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('bluerov2_gz')
    
    # 1. The Gazebo Bridge (Make sure your bridge_config.yaml is updated)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        # This assumes your YAML is in a 'config' folder
        parameters=[{'config_file': os.path.join(pkg_path, 'config', 'bridge_config.yaml')}],
        output='screen'
    )

    # 2. Your Python Depth Calculator
    depth_calc = Node(
        package='bluerov2_gz',
        executable='depth_calculator.py',
        output='screen'
    )

    # 3. Your C++ Teleop Node
    teleop = Node(
        package='bluerov2_gz',
        executable='teleop_node',
        output='screen',
        prefix="xterm -e", # Opens teleop in a separate window to capture keys
        emulate_tty=True
    )

    camera = Node(
    package='bluerov2_gz',
    executable='stereo_vision.py',
    name='stereo_vision',
    output='screen'
    )   

    return LaunchDescription([
        bridge,
        depth_calc,
        teleop,
        camera
    ])