#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import numpy as np
import math

class EKFLocalization(Node):
    def __init__(self):
        super().__init__('ekf_localization_node')

        self.declare_parameter('topics.imu', '/model/bluerov2/conditioned_imu')
        self.declare_parameter('topics.depth', '/model/bluerov2/virtual_depth')
        self.declare_parameter('topics.odometry_out', '/odometry/filtered')
        #self.declare_parameter('topics.depth', '/world/bluerov2_underwater/dynamic_pose/info')
        
        self.declare_parameter('process_noise.position_xy', 0.05)
        self.declare_parameter('process_noise.position_z', 0.001)
        self.declare_parameter('process_noise.velocity_xy', 0.15)
        self.declare_parameter('process_noise.velocity_z', 0.01)
        self.declare_parameter('process_noise.orientation_roll_pitch', 0.005)
        self.declare_parameter('process_noise.orientation_yaw', 0.02)
        
        self.declare_parameter('sensor_noise.depth_variance', 0.04)
        self.declare_parameter('environment.gravity', 9.800)
        self.declare_parameter('environment.min_dt', 0.0001)

        # Resolve Topic Bindings
        self.imu_topic = self.get_parameter('topics.imu').value
        self.depth_topic = self.get_parameter('topics.depth').value
        self.odom_out_topic = self.get_parameter('topics.odometry_out').value
        self.gravity = self.get_parameter('environment.gravity').value
        self.min_dt = self.get_parameter('environment.min_dt').value

        self.print_counter = 0
        # 1. Single Source of Truth Vector (9x1 Column Matrix)
        # Tracking: [x, y, z, vx, vy, vz, roll, pitch, yaw]^T
        self.X = np.zeros((9, 1))
         

        # 2. The State Covariance Matrix (P) 9 x 9 
        self.P = np.eye(9) * 0.1 # initial doubt is low

        # 3. The Process Noise Covariance Matrix (Q)
        """ self.Q = np.eye(9) * 0.01
        self.Q[0:3, 0:3] *= 0.1   # Lower uncertainty for position modeling
        self.Q[3:5, 3:6] *= 1.0   # Higher uncertainty for velocity integration
        self.Q[6:9, 6:9] *= 0.5   # Moderate uncertainty for orientation tracking

        # Open the uncertainty floor so the depth sensor has room to push the state vector
        self.Q[2, 2] = 0.1   # Position Z Process Noise Floor
        self.Q[5, 5] = 0.5   # Velocity Z Process Noise Floor
    """
        # 3. The Process Noise Covariance Matrix (Q) for a FLAT sub
        self.Q = np.eye(9) * 0.01

        self.Q[0, 0] = self.get_parameter('process_noise.position_xy').value # X and Y: High uncertainty. The EKF must know it's drifting horizontally.
        self.Q[1, 1] = self.get_parameter('process_noise.position_xy').value
        self.Q[2, 2] = self.get_parameter('process_noise.position_z').value
        
        self.Q[3, 3] = self.get_parameter('process_noise.velocity_xy').value
        self.Q[4, 4] = self.get_parameter('process_noise.velocity_xy').value
        self.Q[5, 5] = self.get_parameter('process_noise.velocity_z').value
        
        self.Q[6, 6] = self.get_parameter('process_noise.orientation_roll_pitch').value
        self.Q[7, 7] = self.get_parameter('process_noise.orientation_roll_pitch').value
        self.Q[8, 8] = self.get_parameter('process_noise.orientation_yaw').value


        R = self.get_parameter('sensor_noise.depth_variance').value
        self.R_depth = np.array([[R]])
        # -----------------------------------------------------------------
        # 2. ROS 2 NETWORKING & COMMUNICATIONS
        # -----------------------------------------------------------------
        # Best-Effort QoS profile to safely handle high-rate simulator feeds
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        #self.imu_sub = self.create_subscription(Imu, '/model/bluerov2/conditioned_imu', self.imu_callback, sensor_qos)
        #self.depth_sub = self.create_subscription(PoseWithCovarianceStamped, '/model/bluerov2/pose_depth', self.depth_callback, sensor_qos)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.imu_callback, sensor_qos)
        self.depth_sub = self.create_subscription(PoseWithCovarianceStamped, self.depth_topic, self.depth_callback, sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, self.odom_out_topic, 10)

        # Data Publications & Hardware Transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        self.z_sensor_reading = 0.0
        self.z_predicted_guess = 0.0
        self.z_final_estimate = 0.0

        self.last_time = self.get_clock().now()
        self.get_logger().info("BlueROV2 Master 9-DoF EKF Engine Active and Online.")

    def imu_callback(self, msg):
        """ High-Rate Prediction Phase triggered by every incoming IMU telemetry data packet """
        current_time = self.get_clock().now()
        
        dt = (current_time - self.last_time).nanoseconds / 1e9
 
        # Hardened safety gate checking for clock micro-stutters
        if dt < self.min_dt:
            return

        # Fire the mathematical prediction engine
        self.predict_step(dt, msg)
    
        self.last_time = current_time
        self.publish_and_monitor()

        self.print_counter += 1
        if self.print_counter % 15 == 0:
            # Extract the clean World Z acceleration from your calculations to monitor gravity removal
            # We re-calculate or pass it directly from predict_step
            # For simplicity, we grab it by calculating it or caching it. 
            # If your predict_step runs right before this, we can pull the active world z acceleration:
            r, p, y = self.X[6, 0], self.X[7, 0], self.X[8, 0]
            az_raw = msg.linear_acceleration.z
            # Quick horizontal tilt projection extraction to see the live acceleration input:
            az_world = (msg.linear_acceleration.x * -math.sin(p)) + \
                    (msg.linear_acceleration.y * math.cos(p) * math.sin(r)) + \
                    (az_raw * math.cos(p) * math.cos(r)) - 9.800
                    
            #self.print_z_diagnostics(az_world)

    def predict_step(self, dt, msg):
        """ Math Engine: Transforms local bodies and increments physical displacement over time """
        # 1. Extract raw orientation quaternion from the IMU sensor msg
        #The imu produces orientation in quaternion form, Below it is the math to convert it to Euler angles (roll, pitch, yaw)
        qw = msg.orientation.w
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z

        # Mathematical Guard: Block NaN or malformed quaternion streams from destabilizing transformations
        if math.isnan(qw) or math.isnan(qx) or math.isnan(qy) or math.isnan(qz):
            self.get_logger().error("NaN values identified in incoming IMU orientations. Skipping step pass.")
            return
        
        # 2. Convert Quaternion to Euler Angles for our State Vector
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.asin(sinp) if abs(sinp) <= 1.0 else math.copysign(math.pi/2, sinp)

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = -math.atan2(siny_cosp, cosy_cosp)

        

        # 3. Construct the 3D Rotation Matrix (Body-to-World)
        # Using standard Roll (phi), Pitch (theta), Yaw (psi) convention
        #An imu tells you where you are accelerarting from relative to the sub body.
        # (im moving forward)
        #This math converts that into world frame acceleration which is what we need to track velocity and position in the world.
        #(im moving forward from West to East)
        R_body_to_world = np.array([
            [math.cos(yaw)*math.cos(pitch), math.cos(yaw)*math.sin(pitch)*math.sin(roll) - math.sin(yaw)*math.cos(roll), math.cos(yaw)*math.sin(pitch)*math.cos(roll) + math.sin(yaw)*math.sin(roll)],
            [math.sin(yaw)*math.cos(pitch), math.sin(yaw)*math.sin(pitch)*math.sin(roll) + math.cos(yaw)*math.cos(roll), math.sin(yaw)*math.sin(pitch)*math.cos(roll) - math.cos(yaw)*math.sin(roll)],
            [-math.sin(pitch),               math.cos(pitch)*math.sin(roll),                                             math.cos(pitch)*math.cos(roll)]
        ])

        # 4. Rotate Raw Body-Frame Accelerations into World Frame
        #How fast the sub's straight-line speed is changing
        ax_raw = msg.linear_acceleration.x
        ay_raw = msg.linear_acceleration.y
        az_raw = msg.linear_acceleration.z

        A_raw_body = np.array([[ax_raw], [ay_raw], [az_raw]])
        A_world = np.dot(R_body_to_world, A_raw_body) #World acceleration = rotation matrix * body acceleration

        # 5. The Gravity Defeat: Subtract constant Earth gravity from World Z
        # So Gravity acceleration wont effect X and Y, only Z when rotating
        A_world[2, 0] -= 9.800

         # Extract our clean, gravity-free world accelerations
        ax_world = A_world[0, 0]
        ay_world = A_world[1, 0]
        az_world = A_world[2, 0]

        # 1. Update Position States using current Velocity and Acceleration (Kinematics)
        self.X[0, 0] = self.X[0, 0] + (self.X[3, 0] * dt) + (0.5 * ax_world * dt**2)  # Position X New = Old + Velocity*dt + 1/2*Acceleration*dt^2
        self.X[1, 0] = self.X[1, 0] + (self.X[4, 0] * dt) + (0.5 * ay_world * dt**2)  # Position Y New = Old + Velocity*dt + 1/2*Acceleration*dt^2
        self.X[2, 0] = self.X[2, 0] + (self.X[5, 0] * dt) + (0.5 * az_world * dt**2)  # Position Z New = Old + Velocity*dt + 1/2*Acceleration*dt^2

        #self.X[0, 0] = 0.0
        #self.X[1, 0] = 0.0
        self.X[3, 0] = self.X[3, 0] + (ax_world * dt)  # Velocity X New = Old + Acceleration*dt
        self.X[4, 0] = self.X[4, 0] + (ay_world * dt)  # Velocity Y 
        self.X[5, 0] = self.X[5, 0] + (az_world * dt)  # Velocity Z

        # Store these directly in our 9x1 column vector source of truth
        self.X[6, 0] = roll 
        self.X[7, 0] = pitch
        self.X[8, 0] = yaw

        # Construct the State Transition Matrix (F)
        F = np.eye(9)
        F[0, 3] = dt  # x = x + vx * dt
        F[1, 4] = dt  # y = y + vy * dt
        F[2, 5] = dt  # z = z + vz * dt

        # 3. Covariance Prediction (Propagating Paranoia)
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q * dt

      

        # 2. State Prediction (Kinematic Pass)
        # For now, we update position from velocity. 
        # (Acceleration inputs will be cleanly injected right after this!)
        #self.X = np.dot(F, self.X) #Real initial State vector is self.state but we need to convert it to a numpy array for matrix operations.
    

        

    def depth_callback(self, msg):
        """ Low-Rate Correction Phase executed when raw sensor readings arrive """
         # 1. Parse the physical measurement into a 1x1 matrix
        # Maps positive sensor depth into our internal negative coordinate frame
        z_measured = msg.pose.pose.position.z #The negative output of depth sensor

        # Run the correction matrix logic
        self.correct_step(z_measured)
        
    def correct_step(self, z_measured):
        """ Math Engine: Computes multidimensional Kalman Gain vectors to overwrite state trackers """
        # 1. Construct the Measurement Matrix (Z)
        # Matrix Dimensions: 1 Row (1 sensor) x 1 Column (1 measurement)   
        Z = np.array([[z_measured]])
        self.z_sensor_reading = z_measured
        self.z_predicted_guess = self.X[2, 0]

        # 2. Define the Observation Matrix (H)
        # Matrix Dimensions: 1 Row (1 sensor) x 9 Columns (9 states)
        H = np.zeros((1, 9)) # Determines what variable can be effected
        H[0, 2] = 1.0  # In this case its only the z position(depth sensor cant read z velociy or orientation, only depth)

        # 3. Define the Sensor Noise Covariance (R)
        # Matrix Dimensions: 1x1 (1 sensor measurement)
        R = np.array([[0.05]])  # Variance of your physical depth sensor(how much noise the sensor has)

        # 4. Calculate the Innovation (y) - The Reality Gap
        # Matrix Dimensions: (1x1) - (1x1) = 1x1
        y = Z - np.dot(H, self.X) #The difference between calculation and sensor reading(how many meter they are off by)

        # 5. Calculate the Innovation Covariance (S)
        # Matrix Dimensions: (1x9) * (9x9) * (9x1) + (1x1) = 1x1
        S = np.dot(H, np.dot(self.P, H.T)) + R 

        # 6. Calculate the Full Kalman Gain Vector (K)
        # Matrix Dimensions: (9x9) * (9x1) * (1x1)^-1 = 9x1 Column Vector
        # Production Gain Inversion Guard: Guard against system singularities or infinite matrices
        try:
            K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        except np.linalg.LinAlgError:
            self.get_logger().error("Matrix Inversion Singularity error detected inside correct_step. Bypassing state filter pass.")
            return
        
        # 7. Apply the Correction Step to our 9x1 State Vector Source of Truth
        self.X = self.X + np.dot(K, y)

        self.z_final_estimate = self.X[2, 0]

        # 8. Collapse the Cloud of Doubt: Update the Covariance Matrix (P)
        # Matrix Dimensions: (9x9) - (9x1) * (1x9) * (9x9) = 9x9
        I = np.eye(9)
        self.P = np.dot((I - np.dot(K, H)), self.P)

    def publish_and_monitor(self):
        """ Extracts system coordinates from master X array to publish native navigation messages """
        current_time = self.get_clock().now().to_msg()

        # 1. Extract your full 3D Estimated Angles
        r = self.X[6, 0]
        p = self.X[7, 0]
        y = self.X[8, 0]

        # 2. Compute full 3D Quaternion Components (Euler to Quat)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        # Create and populate ROS 2 Odometry tracking packet
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link_estimated'
        
        odom.pose.pose.position.x = self.X[0, 0]
        odom.pose.pose.position.y = self.X[1, 0]
        odom.pose.pose.position.z = self.X[2, 0]
        
        # FIXED: Pass the full 3D spatial rotation coordinates down the pipeline
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz

        pose_covariance = [0.0] * 36
        pose_covariance[0]  = float(self.P[0, 0])  # X variance
        pose_covariance[7]  = float(self.P[1, 1])  # Y variance
        pose_covariance[14] = float(self.P[2, 2])  # Z variance (P_zz)
        pose_covariance[21] = float(self.P[6, 6])  # Roll variance
        pose_covariance[28] = float(self.P[7, 7])  # Pitch variance
        pose_covariance[35] = float(self.P[8, 8])  # Yaw variance
        odom.pose.covariance = pose_covariance
        
        self.odom_pub.publish(odom)

        # Broadcast identical coordinates to your TF tree transformation pipeline
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link_estimated'
        t.transform.translation.x = self.X[0, 0]
        t.transform.translation.y = self.X[1, 0]
        t.transform.translation.z = self.X[2, 0]
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        
        self.tf_broadcaster.sendTransform(t)

    def print_z_diagnostics(self, az_world):
        """ High-visibility terminal monitor tracking the lifecycle of a Z-axis update """
        os.system('clear' if os.name == 'nt' else 'clear')
        
        print("="*60)
        print("             Z-CHANNEL KALMAN FILTER TRACE MONITOR          ")
        print("="*60)
        print(f"WORLD ACCEL Z    :  {az_world:7.4f} m/s^2")
        print(f"CALCULATED VZ    :  {self.X[5, 0]:7.4f} m/s")
        print("-"*60)
        print("                 DEPTH LIFE CYCLE BREAKDOWN                 ")
        print("-"*60)
        # Stage 1: Pure physics guessing
        print(f"1. PREDICTED GUESS (Kinematics) :  {self.z_predicted_guess:7.4f} m")
        
        # Stage 2: Raw data input
        print(f"2. SENSOR READING  (Calculated) :  {self.z_sensor_reading:7.4f} m")
        
        # The Gap: The distance between them
        reality_gap = self.z_sensor_reading - self.z_predicted_guess
        print(f"   REALITY GAP     (Innovation) :  {reality_gap:7.4f} m")
        print("-"*60)
        
        # Stage 3: The mathematical compromise
        print(f"3. FINAL ESTIMATE  (Corrected)  :  {self.z_final_estimate:7.4f} m")
        print("-"*60)
        print(f"DEPTH VARIANCE   :  {self.P[2, 2]:7.4f}")
        print("="*60)

def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()