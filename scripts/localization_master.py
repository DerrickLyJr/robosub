    #!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math
import os

class BlueROVLocalization(Node):
    def __init__(self):
        super().__init__('bluerov_localization_node')

        # State Vector: [x, y, z, vx, vy, vz, roll, pitch, yaw] initialized to zero with a starting depth of -2.0m
        self.state = [0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # State Covariance Matrix P: Initialized with 0.5 variance down the diagonal
        # Meaning: "I have mild uncertainty about my starting states."
        self.P = np.eye(9) * 0.5  
        
        self.Q = np.eye(9) * 0.05
        self.Q[2, 2] = 0.5   # Position Z Process Noise
        self.Q[5, 5] = 0.5   # Velocity Z Process Noise
        
        self.print_counter = 0
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Calibration Baselines
        self.gravity_baseline = None  
        self.calibration_samples = 0
        self.gravity_accumulator = 0.0
        
        # Subscribers (Standard matching queue settings)
        self.imu_sub = self.create_subscription(Imu, '/model/bluerov2/conditioned_imu', self.imu_callback, 10)
        self.depth_sub = self.create_subscription(PoseWithCovarianceStamped, '/model/bluerov2/pose_depth', self.depth_callback, 10)

        # Publisher
        self.odom_pub = self.create_publisher(Odometry, '/odometry/filtered', 10)
        
        self.last_time = self.get_clock().now()
        self.get_logger().info("BlueROV2 Master Localizer: Full 3D Coordinate Transformations Live.")

    def imu_callback(self, msg):
        """ Prediction Step: Full 3D Direction Cosine Matrix (DCM) body-to-world rotation pass """
        if self.gravity_baseline is None:
            self.gravity_accumulator += msg.linear_acceleration.z
            self.calibration_samples += 1
            if self.calibration_samples >= 20:
                self.gravity_baseline = self.gravity_accumulator / 20.0
                self.get_logger().info(f'IMU Calibrated! Base locked at: {self.gravity_baseline:.4f} m/s^2')
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        if dt <= 0.0001:
            return

        # Extract local body accelerations
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z - self.gravity_baseline

        # Parse full 3D Quaternion into Euler angles (Roll, Pitch, Yaw)
        q = msg.orientation
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(sinp) if abs(sinp) <= 1.0 else math.copysign(math.pi/2, sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Save orientation tracking into state vector registers
        self.state[6] = roll
        self.state[7] = pitch
        self.state[8] = yaw

        # 3D Direction Cosine Matrix Transformation Pass
        world_accel_x = ax * (math.cos(yaw) * math.cos(pitch)) + \
                        ay * (math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll)) + \
                        az * (math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll))

        world_accel_y = ax * (math.sin(yaw) * math.cos(pitch)) + \
                        ay * (math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll)) + \
                        az * (math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll))
                        
        # World vertical translation projection mapping
        world_accel_z = ax * (-math.sin(pitch)) + \
                        ay * (math.cos(pitch) * math.sin(roll)) + \
                        az * (math.cos(pitch) * math.cos(roll))

        # Update global Velocity registers safely
        self.state[3] += world_accel_x * dt  # Vel X
        self.state[4] += world_accel_y * dt  # Vel Y
        self.state[5] += world_accel_z * dt  # Vel Z
        
        # Integrate Position registers dynamically
        self.state[0] += self.state[3] * dt  # Pos X
        self.state[1] += self.state[4] * dt  # Pos Y
        self.state[2] += self.state[5] * dt  # Pos Z
        
        self.P += self.Q * dt
        self.last_time = current_time
        
        self.print_counter += 1
        if self.print_counter % 15 == 0:
            self.print_diagnostics(ax, ay, az)

        self.publish_odometry()

    def depth_callback(self, msg):
        """ Correction Step: Direct, inverted mapping matching the negative depth coordinate grid """
        z_measured = -msg.pose.pose.position.z  # Maps positive sensor input to negative coordinate grid
        
        R_depth = 0.05 
        K_z = self.P[2, 2] / (self.P[2, 2] + R_depth)
        
        self.state[2] += K_z * (z_measured - self.state[2])
        self.P[2, 2] *= (1.0 - K_z)

    def print_diagnostics(self, ax, ay, az):
        """ High-visibility terminal matrix monitor """
        os.system('clear' if os.name == 'nt' else 'clear')
        print("="*50)
        print("          FILTER MATRIX DIAGNOSTIC MONITOR         ")
        print("="*50)
        print(f"RAW IMU INPUTS:  Accel X: {ax:7.4f} | Accel Y: {ay:7.4f} | Net Accel Z: {az:7.4f}")
        print("-"*50)
        print(f"CALCULATED VEL:  Vel X:   {self.state[3]:7.4f} | Vel Y:   {self.state[4]:7.4f} | Vel Z:     {self.state[5]:7.4f}")
        print("-"*50)
        print(f"ESTIMATED POSE:  Pos X:   {self.state[0]:7.4f} | Pos Y:   {self.state[1]:7.4f} | Pos Z:     {self.state[2]:7.4f}")
        print("="*50)

    def publish_odometry(self):
        current_time = self.get_clock().now().to_msg()

        msg = Odometry()
        msg.header.stamp = current_time
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link_estimated'
        msg.pose.pose.position.x = self.state[0]
        msg.pose.pose.position.y = self.state[1]
        msg.pose.pose.position.z = self.state[2]
        
        cy = math.cos(self.state[8] * 0.5)
        sy = math.sin(self.state[8] * 0.5)
        msg.pose.pose.orientation.w = cy
        msg.pose.pose.orientation.z = sy
        self.odom_pub.publish(msg)

        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link_estimated' 
        t.transform.translation.x = self.state[0]
        t.transform.translation.y = self.state[1]
        t.transform.translation.z = self.state[2]
        t.transform.rotation.w = cy
        t.transform.rotation.z = sy
        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = BlueROVLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()