#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from cv_bridge import CvBridge
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class RoboSubStereoTracker(Node):
    def __init__(self):
        super().__init__('robosub_stereo_tracker')
        self.bridge = CvBridge()

        # Orientation state and topic targets
        self.current_sub_orientation = None
        self.TARGET_WORLD_X = 5.0
        self.TARGET_WORLD_Y = 2.0
        self.TARGET_WORLD_Z = -2.0  

        # 1. CAMERA GEOMETRY DEF OVERRIDES
        fx, cx, cy, baseline = 554.25, 320.0, 240.0, 0.12
        self.Q = np.array([
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0,  fx],
            [0.0, 0.0, 1.0 / baseline, 0.0]
        ], dtype=np.float32)
        self.camera_info_received = True 

        # 2. MATCH CORES CONFIGS QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 3. INTER-NODE LINKS
        self.ekf_sub = self.create_subscription(Odometry, '/odometry/filtered', self.ekf_callback, 10)
        self.vision_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/vision/pose_estimate', 10)
        self.target_pub = self.create_publisher(PointStamped, '/vision/target_relative', 10)

        # 4. STREAM SYNC FILTERS
        self.left_sub = message_filters.Subscriber(self, Image, '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/left_camera/image', qos_profile=sensor_qos)
        self.right_sub = message_filters.Subscriber(self, Image, '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/right_camera/image', qos_profile=sensor_qos)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.stereo_callback)
        self.get_logger().info("RoboSub Stereo Tracker Operational. Decoupled publisher active.")

    def ekf_callback(self, msg):
        self.current_sub_orientation = msg.pose.pose.orientation

    def stereo_callback(self, left_msg, right_msg):
        frame_l = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        frame_r = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # HSV Extraction Logic
        hsv = cv2.cvtColor(frame_l, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        moments = cv2.moments(red_mask)
        if moments["m00"] == 0:
            return  

        u = int(moments["m10"] / moments["m00"])
        v = int(moments["m01"] / moments["m00"])

        # Disparity mapping blocks
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=11, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
        disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        X, Y, Z = points_3d[v, u]
        if not np.isfinite(Z) or Z <= 0.1:
            return 

        # Transform coordinate frames: CV Frame to ROS Robot Frame
        robot_x = float(Z)
        robot_y = float(-X)
        robot_z = float(-Y)

        # Broadcast relative point
        point_msg = PointStamped()
        point_msg.header = left_msg.header
        point_msg.header.frame_id = 'base_link'
        point_msg.point.x, point_msg.point.y, point_msg.point.z = robot_x, robot_y, robot_z
        self.target_pub.publish(point_msg)

        # 🟢 CRITICAL FIXED LINE: Now triggering the absolute state recalculation
        self.publish_absolute_sub_pose(robot_x, robot_y, robot_z, left_msg.header)

        # Render Diagnostics
        cv2.circle(frame_l, (u, v), 8, (0, 255, 0), -1)
        cv2.putText(frame_l, f"Target: [{robot_x:.2f}m, {robot_y:.2f}m, {robot_z:.2f}m]", (u + 15, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Active Tracking Monitor", frame_l)
        cv2.waitKey(1)

    def publish_absolute_sub_pose(self, robot_x, robot_y, robot_z, original_header):
        if self.current_sub_orientation is None:
            return 
            
        q = self.current_sub_orientation
        R_body_to_world = np.array([
            [1 - 2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.w*q.z), 2*(q.x*q.z + q.w*q.y)],
            [2*(q.x*q.y + q.w*q.z), 1 - 2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.w*q.x)],
            [2*(q.x*q.z - q.w*q.y), 2*(q.y*q.z + q.w*q.x), 1 - 2*(q.x**2 + q.y**2)]
        ])

        vec_body = np.array([[robot_x], [robot_y], [robot_z]])
        vec_world = np.dot(R_body_to_world, vec_body)

        sub_world_x = self.TARGET_WORLD_X - vec_world[0, 0]
        sub_world_y = self.TARGET_WORLD_Y - vec_world[1, 0]
        sub_world_z = self.TARGET_WORLD_Z - vec_world[2, 0]

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = original_header
        pose_msg.header.frame_id = 'odom'
        pose_msg.pose.pose.position.x = sub_world_x
        pose_msg.pose.pose.position.y = sub_world_y
        pose_msg.pose.pose.position.z = sub_world_z
        pose_msg.pose.pose.orientation = q

        cov = [0.0] * 36
        cov[0], cov[7], cov[14] = 0.005, 0.005, 0.05
        pose_msg.pose.covariance = cov

        self.vision_pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RoboSubStereoTracker()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()