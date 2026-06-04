#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, TransformStamped, PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
import numpy as np


class GroundTruthBridge(Node):
    def __init__(self):
        super().__init__('ground_truth_bridge', automatically_declare_parameters_from_overrides=True)
        self.br = TransformBroadcaster(self)
        
        self.subscription = self.create_subscription(
            PoseArray, 
            '/world/bluerov2_underwater/dynamic_pose/info', 
            self.listener_callback, 
            10)

        self.pub_virtual_depth = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/model/bluerov2/virtual_depth', 
            10)
        
        self.get_logger().info('Ground Truth Bridge Listening to Dynamic Pose Array Track...')

    def listener_callback(self, msg):
        if len(msg.poses) > 0:
            chassis_pose = msg.poses[0]
            
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = 'odom'
            t.child_frame_id = 'base_link'
            
            t.transform.translation.x = chassis_pose.position.x
            t.transform.translation.y = chassis_pose.position.y
            t.transform.translation.z = chassis_pose.position.z
            t.transform.rotation = chassis_pose.orientation
            
            self.br.sendTransform(t)

            #----------------------------------------------------------
            # SIMULATED PHYSICAL DEPTH SENSOR NOISE ENGINE
            # -------------------------------------------------------------
            # Parameters representing a standard commercial subsea pressure transducer
            sensor_accuracy_std_dev = 0.04  # +/- 4 centimeters of high-frequency white noise jitter
            sensor_constant_bias = -0.02    # -2 centimeters of structural sensor calibration bias
            
            # Generate White Gaussian Jitter
            white_noise = np.random.normal(0.0, sensor_accuracy_std_dev)
            
            # Blend pristine data with physical environmental degradations
            noisy_z = chassis_pose.position.z + white_noise + sensor_constant_bias

            # 2. Package the ground truth coordinate as a virtual depth sensor packet
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.header.frame_id = 'odom'
            
            # Feed the noisy Gazebo Z position straight into the message
            pose_msg.pose.pose.position.z = noisy_z
            
            # Fill the 36-element covariance array (Index 14 is Z-variance)
            full_covariance = [0.0] * 36
            full_covariance[14] = 0.05  
            pose_msg.pose.covariance = full_covariance
            
            self.pub_virtual_depth.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()