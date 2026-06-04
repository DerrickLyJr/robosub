#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import FluidPressure, Imu
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class DepthCalculator(Node):
    def __init__(self):
        super().__init__('sensor_conditioner')
        
        # --- BEST-EFFORT SENSOR QOS PROFILE ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # --- DEPTH PIPELINE ---
        self.sub_pressure = self.create_subscription(
            FluidPressure, 
            '/model/bluerov2/pressure', 
            self.pressure_callback, 
            sensor_qos)
        
        self.pub_depth = self.create_publisher(Float64, '/model/bluerov2/calculated_depth', 10)
        self.pub_ekf_pose = self.create_publisher(PoseWithCovarianceStamped, '/model/bluerov2/pose_depth', sensor_qos)
        
        # --- IMU PIPELINE ---
        self.sub_raw_imu = self.create_subscription(
            Imu,
            '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/imu_sensor/imu',
            self.imu_callback,
            sensor_qos)
        
        self.pub_conditioned_imu = self.create_publisher(Imu, '/model/bluerov2/conditioned_imu', 10)
        
        self.rho = 997.0    
        self.g = 9.80
        self.p_atm = None

        self.br = TransformBroadcaster(self)
        self.p_atm = 101325.0  # Standard atmospheric pressure constant baseline (Pa)
        self.initial_tare_complete = False

        self.get_logger().info('Depth Calculator (Sensor Conditioner) Operational.')

    def pressure_callback(self, msg):
        # Calculate depth relative to standard atmospheric baseline
        measured_depth = (msg.fluid_pressure - self.p_atm) / (self.rho * self.g)
        
        # Capture the very first reading to establish our spatial tree tare
        if not self.initial_tare_complete:
            self.starting_immersion_z = measured_depth # e.g., 0.25m
            self.get_logger().info(f'Spatial Tare Locked! Sub submerged by: {self.starting_immersion_z:.3f}m')
            self.initial_tare_complete = True
            
        # Broadcast the map->odom transform to shift the world coordinate grid cleanly
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        # Invert the starting depth so odom shifts upward to the true surface line
        t.transform.translation.z = -self.starting_immersion_z 
        t.transform.rotation.w = 1.0
        self.br.sendTransform(t)

        # Standard depth tracking outputs continue natively
        relative_depth = measured_depth - self.starting_immersion_z
        
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = msg.header.stamp 
        pose_msg.header.frame_id = 'odom' 
        pose_msg.pose.pose.position.z = relative_depth 
        full_covariance = [0.0] * 36
        # Position Z variance sits at index 14 (Row 2, Column 2 of a 6x6 matrix)
        full_covariance[14] = 0.01 
        # Assign the fully initialized array back to the message field
        pose_msg.pose.covariance = full_covariance
        # Publish the pristine data package
        self.pub_ekf_pose.publish(pose_msg)
        

    def imu_callback(self, msg):
        corrected_msg = Imu()
        corrected_msg.header.stamp = msg.header.stamp
        corrected_msg.header.frame_id = 'base_link_estimated'
        
        corrected_msg.orientation = msg.orientation
        corrected_msg.angular_velocity = msg.angular_velocity
        
        # Invert specific axis profiles if your simulation coordinate framework flips directions
        corrected_msg.linear_acceleration.x = msg.linear_acceleration.x
        corrected_msg.linear_acceleration.y = -msg.linear_acceleration.y 
        corrected_msg.linear_acceleration.z = -msg.linear_acceleration.z

        corrected_msg.orientation_covariance[0] = 0.05
        corrected_msg.orientation_covariance[4] = 0.05
        corrected_msg.orientation_covariance[8] = 0.05
        
        corrected_msg.linear_acceleration_covariance[0] = 0.02
        corrected_msg.linear_acceleration_covariance[4] = 0.02
        corrected_msg.linear_acceleration_covariance[8] = 0.02

        self.pub_conditioned_imu.publish(corrected_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DepthCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()