#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float64
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DepthCalculator(Node):
    def __init__(self):
        super().__init__('depth_calculator')
        # Subscribing to the raw Gazebo data
        self.sub = self.create_subscription(
            FluidPressure, 
            '/model/bluerov2/pressure', 
            self.callback, 
            10)
        
        # Publishing the calculated result for your C++ Teleop
        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10)

        self.pub = self.create_publisher(
        Float64, 
        '/model/bluerov2/calculated_depth',
        10)
        
        # Constants
        self.rho = 997.0    # Fresh water density
        self.g = 9.80665    # Gravity
        self.p_atm = 101325.0 # Sea level pressure

        self.get_logger().info('Depth Calculator Node has started.')

    def callback(self, msg):
        # The core math: h = (P - P_atm) / (rho * g)
        depth = (msg.fluid_pressure - self.p_atm) / (self.rho * self.g)
        
        # Publish to ROS 2 network
        out_msg = Float64()
        out_msg.data = depth
        self.pub.publish(out_msg)
        
        # Log to terminal for debugging
        self.get_logger().info(f'Depth: {depth:.3f} m | Pressure: {msg.fluid_pressure:.1f} Pa')

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