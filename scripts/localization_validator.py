#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np

class LocalizationValidator(Node):
    def __init__(self):
        super().__init__('localization_validator_node')

        # 1. Setup TF Listener infrastructure to track our frames
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. Subscribe to EKF to pull the live P_zz variance array matrix
        self.declare_parameter('topics.estimate', '/odometry/filtered')
        self.estimate_topic = self.get_parameter('topics.estimate').value
        self.est_sub = self.create_subscription(Odometry, self.estimate_topic, self.estimate_callback, 10)

        # State Caches
        self.current_truth_z = 0.0
        self.current_estimate_z = 0.0
        self.p_matrix_z_variance = 0.01

        self.error_history = []
        self.sample_count = 0

        # 2Hz evaluation execution loop
        self.eval_timer = self.create_timer(0.5, self.evaluate_performance)
        self.get_logger().info("TF-Driven Phase 3 Performance Validator Core Online.")

    def estimate_callback(self, msg):
        # Capture the EKF state and variance natively
        self.current_estimate_z = msg.pose.pose.position.z
        self.p_matrix_z_variance = msg.pose.covariance[14]

    def evaluate_performance(self):
        # 1. Lookup the absolute physical frame position from your GroundTruthBridge
        try:
            truth_tf = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            self.current_truth_z = truth_tf.transform.translation.z
        except TransformException as ex:
            self.get_logger().info('Waiting for GroundTruthBridge TF frame path to stabilize...')
            return

        # 2. Compute absolute spatial tracking metrics
        absolute_error = abs(self.current_truth_z - self.current_estimate_z)
        self.error_history.append(absolute_error)
        self.sample_count += 1
        mae = np.mean(self.error_history)

        # 3. Compute Innovation consistency status
        if self.p_matrix_z_variance > 0.0:
            nis_z = (absolute_error ** 2) / self.p_matrix_z_variance
            consistency_status = "OPTIMAL" if nis_z <= 2.0 else "UNSTABLE (Overconfident)"
        else:
            nis_z = 0.0
            consistency_status = "UNKNOWN"

        self.get_logger().info(
            f"\n"
            f"============================================================\n"
            f"             BLUEROV2 AUTONOMY EVALUATION SCORECARD         \n"
            f"============================================================\n"
            f" Samples Processed : {self.sample_count}\n"
            f" Live Depth Error  : {absolute_error * 1000.0:7.2f} mm\n"
            f" Cumulative MAE    : {mae * 1000.0:7.2f} mm\n"
            f"------------------------------------------------------------\n"
            f" Z-Axis Var (P_zz) : {self.p_matrix_z_variance:7.5f}\n"
            f" NIS Metric        : {nis_z:7.3f}\n"
            f" Filter Status     : {consistency_status}\n"
            f"============================================================"
        )

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()