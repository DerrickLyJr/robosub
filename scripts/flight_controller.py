#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import math

class BlueROVFlightController(Node):
    def __init__(self):
        super().__init__('bluerov_flight_controller')

        # Declare parameters with robust default fallbacks if the YAML is missing
        self.declare_parameter('topics.odometry', '/odometry/filtered')
        self.declare_parameter('topics.cmd_vel', '/cmd_vel')

        # --- PID GAIN SEED CONSTANTS ---
        self.declare_parameter('pid.kp_z', 1.2)
        self.declare_parameter('pid.ki_z', 0.005)
        self.declare_parameter('pid.kd_z', 0.4)
        
        self.declare_parameter('yaw_pid.kp_yaw', 1.5)
        self.declare_parameter('yaw_pid.ki_yaw', 0.0)
        self.declare_parameter('yaw_pid.kd_yaw', 0.3)

        self.declare_parameter('limits.max_thruster_effort', 40.0)
        self.declare_parameter('limits.min_dt', 0.0001)
        self.declare_parameter('limits.max_dt', 0.1)

        # Fetch parameter values safely into memory cache
        self.odom_topic = self.get_parameter('topics.odometry').value
        self.cmd_vel_topic = self.get_parameter('topics.cmd_vel').value

        self.kp_z = self.get_parameter('pid.kp_z').value
        self.ki_z = self.get_parameter('pid.ki_z').value
        self.kd_z = self.get_parameter('pid.kd_z').value

        self.kp_yaw = self.get_parameter('yaw_pid.kp_yaw').value
        self.ki_yaw = self.get_parameter('yaw_pid.ki_yaw').value
        self.kd_yaw = self.get_parameter('yaw_pid.kd_yaw').value
        
        self.max_effort = self.get_parameter('limits.max_thruster_effort').value
        self.min_dt = self.get_parameter('limits.min_dt').value
        self.max_dt = self.get_parameter('limits.max_dt').value

        # --- PID TARGET REGISTERS ---
        # Starts targeted exactly at your initialization tare resting waterline
        self.target_depth = -2.0  
        self.target_yaw = 3.14
   

        # --- CONTROLLER MEMORY CACHES ---
        self.integral_z = 0.0
        self.last_error_z = 0.0
        self.integral_yaw = 0.0
        self.last_error_yaw = 0.0

        self.last_time = self.get_clock().now()

        # --- OPEN-LOOP PILOT OVERRIDES ---
        self.manual_forward = 0.0
        self.manual_strafe = 0.0

        # --- HARDWARE MOTOR PUBLISHERS ---
        self.thruster_pubs = []
        for i in range(1, 7):
            topic = f'/model/bluerov2/joint/thruster{i}_joint/cmd_thrust'
            self.thruster_pubs.append(self.create_publisher(Float64, topic, 10))

        # --- INPUT SUBSCRIPTIONS ---
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10)

        self.get_logger().info('BlueROV2 PID Flight Controller Core Engaged. Depth-Hold Active.')

    def cmd_vel_callback(self, msg):
        """ Captures manual translation requests and steps heading angles safely """
        self.manual_forward = msg.linear.x
        self.manual_strafe = msg.linear.y
        
        # Handle manual climb/dive overrides from keyboard inputs
        if abs(msg.linear.z) > 0.01:
            # FIXED: Increments your locked target tracking plane down/up
            self.target_depth += msg.linear.z * 0.1 
            self.get_logger().info(f'New PID Target Depth Set: {self.target_depth:.2f}m')
            
            # CRITICAL: Decay the input command register instantly so it acts as 
            # a single step adjustment rather than an runaway integration loop!
            msg.linear.z = 0.0

    def odom_callback(self, msg):
        """ Ticks the PID tracking matrices at the exact update rate of your localizer master """
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        #Prevent zero-division or micro-stutters causing derivative explosion
        if dt < self.min_dt:
            return

        # Handle simulator pause or extreme lag spikes gracefully 
        # without jumping states or accumulating massive integrals
        if dt > self.max_dt:
            self.get_logger().warn(f'Excessive time delta detected (dt={dt:.4f}s). Skipping loop step for safety.')
            self.last_time = current_time
            return
        
        
        current_depth = msg.pose.pose.position.z 
        
        q = msg.pose.pose.orientation
        if math.isnan(current_depth) or math.isnan(q.w) or math.isnan(q.z):
            self.get_logger().error('NaN values identified in telemetry message frame. Bypassing execution cycle.')
            return
        
        # 1. Extract Yaw from Localizer Array Matrix
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        current_yaw = -math.atan2(siny_cosp, cosy_cosp)

        # Extract Roll using standard aeronautic sequence definitions
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        current_roll = math.atan2(sinr_cosp, cosr_cosp)

        # Extract Pitch from incoming localizer orientation values
        #Pitch in this BlueROV2 configuration can not be controlled via thrusters so i commented it out
        """sinp = 2 * (q.w * q.y - q.z * q.x)
        current_pitch = math.asin(sinp) if abs(sinp) <= 1.0 else math.copysign(math.pi/2, sinp)"""

        # 2. RUN DEPTH-HOLD PID PASS
        error_z = self.target_depth - current_depth
        derivative_z = (error_z - self.last_error_z) / dt

        # 1. Compute the un-clamped Proportional and Derivative baselines
        u_p = self.kp_z * error_z
        u_d = self.kd_z * derivative_z

        # 2. Test calculate what the total command would be using the EXISTING integral cache
        u_unclamped = u_p + (self.ki_z * self.integral_z) + u_d

        # 3. Check for Actuator Saturation
        is_saturated = abs(u_unclamped) >= self.max_effort

        # 4. Check for Sign Alignment (Is the error making the saturation worse?)
        # np.sign returns -1, 0, or 1 based on the polarity of the variable
        same_direction = (math.copysign(1, u_unclamped) == math.copysign(1, error_z))
        
        # 5. CONDITIONAL INTEGRAL UPDATE RULE
        # Only accumulate memory if we are NOT saturated, OR if the error has 
        # flipped and is actively helping to drain the wound-up memory!
        if not (is_saturated and same_direction):
            self.integral_z += error_z * dt

        # 6. Compute final protected control effort
        f_z = u_p + (self.ki_z * self.integral_z) + u_d
        self.last_error_z = error_z

        # 3. RUN HEADING-HOLD PID PASS
        error_yaw = self.target_yaw - current_yaw
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw)) # Radian boundary wrap filter
        
        self.integral_yaw += error_yaw * dt
        derivative_yaw = (error_yaw - self.last_error_yaw) / dt
        
       # 1. Compute un-clamped rotational baselines
        u_p_yaw = self.kp_yaw * error_yaw
        u_d_yaw = self.kd_yaw * derivative_yaw
        u_unclamped_yaw = u_p_yaw + (self.ki_yaw * self.integral_yaw) + u_d_yaw

        # 2. Conditional Clamping Test for Rotation Frame
        is_saturated_yaw = abs(u_unclamped_yaw) >= self.max_effort
        same_direction_yaw = (math.copysign(1, u_unclamped_yaw) == math.copysign(1, error_yaw))

        if not (is_saturated_yaw and same_direction_yaw):
            self.integral_yaw += error_yaw * dt

        # 3. Final protected torque command
        t_z = u_p_yaw + (self.ki_yaw * self.integral_yaw) + u_d_yaw
        self.last_error_yaw = error_yaw
        t_z = -t_z

        target_roll = 0.0 # Force the vehicle to stay flat at all times
        error_roll = target_roll - current_roll
        
        if not hasattr(self, 'last_error_roll'):
            self.last_error_roll = 0.0
            self.integral_roll = 0.0

        self.integral_roll += error_roll * dt
        # Use conservative baseline gain constants to prevent thruster chatter
        kp_roll = 2.0
        kd_roll = 0.5
        
        derivative_roll = (error_roll - self.last_error_roll) / dt
        u_roll = (kp_roll * error_roll) + (kd_roll * derivative_roll)
        self.last_error_roll = error_roll

         #Pitch in this BlueROV2 configuration can not be controlled via thrusters so i commented it out
        """ target_pitch = 0.0 # Maintain a perfectly level horizon
        error_pitch = target_pitch - current_pitch
        
        if not hasattr(self, 'last_error_pitch'):
            self.last_error_pitch = 0.0
            self.integral_pitch = 0.0

        self.integral_pitch += error_pitch * dt
        # Match your Roll gains as a baseline starting point
        kp_pitch = 2.0
        kd_pitch = 0.5
        
        u_pitch = (kp_pitch * error_pitch) + (kd_pitch * ((error_pitch - self.last_error_pitch) / dt))
        self.last_error_pitch = error_pitch
        """
        
        # 4. THRUSTER ALLOCATION MATRIX MIXING
        # Translates global forces smoothly into individual BlueROV2 vector slots
        thruster_commands = [0.0] * 6
        
        # Horizontal Sub-Group Map (Open-Loop Intent + Closed-Loop Turning Alignment Corrections)
        thruster_commands[0] =  self.manual_forward + self.manual_strafe - t_z  # Thruster 1
        thruster_commands[1] =  self.manual_forward - self.manual_strafe + t_z  # Thruster 2
        thruster_commands[2] = -self.manual_forward + self.manual_strafe + t_z  # Thruster 3
        thruster_commands[3] = -self.manual_forward - self.manual_strafe - t_z  # Thruster 4
        
        # Vertical Sub-Group Map (Closed-Loop Elevation Control Output Matrix)
        thruster_commands[4] = -f_z + u_roll  # Thruster 5
        thruster_commands[5] = -f_z - u_roll  # Thruster 6

        # Package individual Float64 payloads and stream down the Gazebo pipe branches
        for i in range(6):
            out_msg = Float64()
            # Scale and clip motor efforts safely to standard envelope limits
            out_msg.data = max(min(thruster_commands[i], self.max_effort), -self.max_effort)
            self.thruster_pubs[i].publish(out_msg)

        self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    node = BlueROVFlightController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()