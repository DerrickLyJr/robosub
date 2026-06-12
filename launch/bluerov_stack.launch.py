import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('bluerov2_gz')
    sdf_file = os.path.join(pkg_path, 'models', 'bluerov2', 'model.sdf')
    config_file = os.path.join(pkg_path, 'config', 'subParams.yaml')

    with open(sdf_file, 'r') as infp:
        robot_description_content = infp.read()
        robot_description_content = robot_description_content.replace(
            'model://bluerov2', 
            'package://bluerov2_gz/models/bluerov2'
        )
    print(f"DEBUG: Read SDF file, length: {len(robot_description_content)} characters")
    
    # 1. The Gazebo Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/model/bluerov2/pressure@sensor_msgs/msg/FluidPressure[gz.msgs.FluidPressure',
            
            '/world/bluerov2_underwater/dynamic_pose/info@geometry_msgs/msg/PoseArray[gz.msgs.Pose_V',

            # Simple path—no messy inline remapping strings
            '/world/bluerov2_underwater/model/bluerov2/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
            
            # Camera Feed Bridges
            '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/left_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/right_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            
            # Thruster Command Targets
            '/model/bluerov2/joint/thruster1_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/bluerov2/joint/thruster2_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/bluerov2/joint/thruster3_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/bluerov2/joint/thruster4_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/bluerov2/joint/thruster5_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/bluerov2/joint/thruster6_joint/cmd_thrust@std_msgs/msg/Float64]gz.msgs.Double',
        ],
        output='screen'
    )

    # 2. Robot State Publisher (Interprets the long track directly)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True,
            'ignore_timestamp': False
        }],
        # REMAPPING: Intercepts its internal lookups and points it to the raw bridge data
        remappings=[
            ('/joint_states', '/world/bluerov2_underwater/model/bluerov2/joint_state')
        ]
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'frequency': 30.0,
            'sensor_timeout': 0.1,
            'two_d_mode': False,
            'publish_tf': True,
            
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_link_frame': 'base_link_estimated',
            'world_frame': 'odom',
            
            # 1. POINT TO THE CONDITIONED IMU STREAM
            'imu0': '/model/bluerov2/conditioned_imu',
            'imu0_config': [False, False, False, 
                            False, False, False, 
                            False, False, False, 
                            True,  True,  True,   # Track angular velocities (Gyros)
                            False, False, False],  # Track linear accelerations (Accelerometers)
            'imu0_differential': False,
            'imu0_relative': True,
            'imu0_remove_gravitational_acceleration': True,
            'gravitational_acceleration': 9.8011, # FIXED: Matches your exact log baseline perfectly
            
            # 2. DEDICATED DEPTH DATA TRACK
            'pose0': '/model/bluerov2/pose_depth', 
            'pose0_config': [False, False, True,  
                             False, False, False, 
                             False, False, False, 
                             False, False, False, 
                             False, False, False],
            'pose0_differential': False,
            
            
        }]
    )

    ground_truth_node = Node(
        package='bluerov2_gz',
        executable='ground_truth_bridge.py',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 2. Processing Pipeline Nodes (Syncing their time parameters to the simulation)
    depth_calc = Node(
        package='bluerov2_gz',
        executable='depth_calculator.py',
        output='screen',
        parameters=[{'use_sim_time': True}] # Added
    )

    camera = Node(
        package='bluerov2_gz',
        executable='stereo_vision.py',
        name='stereo_vision',
        output='screen',
        parameters=[{'use_sim_time': True}] # Added
    )

    pressure_Bridge = Node(
        package='bluerov2_gz',
        executable='pressure_bridge.py',
        name='pressure_bridge',
        parameters=[{'use_sim_time': True}] # Added
    )

    My_localization_Master = Node(
        package='bluerov2_gz',
        executable='ekf_9dof_localizer.py',
        name='ekf_localization_node',
        output='screen',
        parameters=[{'use_sim_time': True,
                     'depth_topic': '/world/bluerov2_underwater/dynamic_pose/info'},
                     config_file] # Added
    )  

    localization_validator_node = Node(
        package='bluerov2_gz',
        executable='localization_validator.py',
        name='localization_validator_node',
        output='screen',
        parameters=[config_file]
    )

    localization_Master = Node(
        package='bluerov2_gz',
        executable='localization_master.py',
        name='localization_master',
        output='screen',
        parameters=[{'use_sim_time': True}] # Added
    )  

    # 3. Interactive Teleop Node
    teleop = Node(
        package='bluerov2_gz',
        executable='teleop_node',
        output='screen',
        prefix="xterm -e",
        emulate_tty=True
        # Left without use_sim_time so your keyboard hardware stream remains independent
    )

    world_to_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # Append the explicit ROS argument parameter to the end of the arguments list
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map', '--ros-args', '-p', 'use_sim_time:=true']
    )

    map_to_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # Append the explicit ROS argument parameter to the end of the arguments list
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom', '--ros-args', '-p', 'use_sim_time:=true']
    )

    controller_node = Node(
        package='bluerov2_gz',
        executable='flight_controller.py',
        name='bluerov_flight_controller',
        output='screen',
        parameters=[config_file] # Added
    )
    
    

    return LaunchDescription([
        controller_node,
        #ekf_node,
        bridge,
        robot_state_publisher,
        world_to_map,
        #map_to_odom,
        depth_calc,
        teleop,
        #camera,
        #pressure_Bridge,
        My_localization_Master,
        localization_validator_node,
        ground_truth_node
    ])