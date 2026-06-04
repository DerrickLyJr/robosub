#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

class PressureToDepthNode(Node):
    def __init__(self):
        super().__init__('pressure_to_depth_bridge')
        self.get_logger().info("Pressure Bridge Node has started and is listening...")
        
        # Subscribe to Gazebo's bridged pressure topic
        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        depth=10
)

        self.sub = self.create_subscription(
        FluidPressure,
        '/model/bluerov2/pressure',
        self.callback,
        qos_profile=qos_profile)
            
        # Publish to the localization input topic
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/model/bluerov2/depth_pose', 10)

    def callback(self, msg):
        depth = (msg.fluid_pressure - 101325.0) / (1025.0 * 9.81)
       # self.get_logger().info(f"Calculated Depth: {depth:.2f}m", throttle_duration_sec=1.0)
        
        # Gazebo depth is often negative (z-down), 
        # check your coordinate frame and flip sign if needed.
        depth_msg = PoseWithCovarianceStamped()
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.header.frame_id = 'world'
        
        # We only fill the Z component
        depth_msg.pose.pose.position.z = -depth 
        
        # Covariance: How much we trust this sensor (Standard Deviation squared)
        # Pressure sensors are usually very stable, so we use a small value.
        depth_msg.pose.covariance[14] = 0.001 # index 14 is the Z-Z variance
        
        self.pub.publish(depth_msg)

def main():
    rclpy.init()
    rclpy.spin(PressureToDepthNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()