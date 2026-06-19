#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import message_filters

# Imports from your Hornet Robotics folder
from computer_vision import config

class StereoVisionNode(Node):
    def __init__(self):
        super().__init__('stereo_vision_node')
        self.bridge = CvBridge()
        
        # Load your teammate's calibration data
        self.load_calibration()

        # Subscribers for the topics you found in Gazebo
        self.right_sub = message_filters.Subscriber(self, Image, '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/right_camera/image')
        self.left_sub = message_filters.Subscriber(self, Image, '/world/bluerov2_underwater/model/bluerov2/link/base_link/sensor/left_camera/image')
        
        # Syncing the frames (Simulation needs a small 'slop' for timing)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.stereo_callback)

        self.get_logger().info("Jazzy Stereo Vision Node Started (Sim Mode)")

    def load_calibration(self):
        try:
            path = config.CALIBRATION_PATH + (config.CALIBRATION_VAR_FILE_FISHEYE if config.IS_FISHEYE else config.CALIBRATION_VAR_FILE)
            data = np.load(path)
            # Matrices required for the math
            self.K1, self.D1, self.K2, self.D2 = data["K1"], data["D1"], data["K2"], data["D2"]
            self.R1, self.R2, self.P1, self.P2, self.Q = data["R1"], data["R2"], data["P1"], data["P2"], data["Q"]
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration data: {e}")

    def stereo_callback(self, left_msg, right_msg):
        # Convert ROS to OpenCV
        frame_l = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
        frame_r = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
        
        # Grayscale for processing
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        img_size = gray_l.shape[::-1]

        # Rectify images based on the 'robosub' config
        if config.IS_FISHEYE:
            m1x, m1y = cv2.fisheye.initUndistortRectifyMap(self.K1, self.D1, self.R1, self.P1[:, :3], img_size, cv2.CV_32FC1)
            m2x, m2y = cv2.fisheye.initUndistortRectifyMap(self.K2, self.D2, self.R2, self.P2[:, :3], img_size, cv2.CV_32FC1)
        else:
            m1x, m1y = cv2.initUndistortRectifyMap(self.K1, self.D1, self.R1, self.P1, img_size, cv2.CV_32FC1)
            m2x, m2y = cv2.initUndistortRectifyMap(self.K2, self.D2, self.R2, self.P2, img_size, cv2.CV_32FC1)

        rect_l = cv2.remap(gray_l, m1x, m1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(gray_r, m2x, m2y, cv2.INTER_LINEAR)

        # Create Disparity Map (Depth map)
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=10,
            uniquenessRatio=5,
            speckleWindowSize=100,
            speckleRange=2
        )
        
        disparity = stereo.compute(rect_l, rect_r).astype(np.float32) / 16.0
        
        # Reproject to 3D space
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depths = points_3d[..., 2]
        
        # Filter out noise
        depths[~np.isfinite(depths)] = 0
        depths[depths < 0] = 0

        # Output center distance to terminal
        cy, cx = depths.shape[0] // 2, depths.shape[1] // 2
        center_dist = depths[cy, cx]
        #self.get_logger().info(f"Target Distance: {center_dist:.2f}m", throttle_duration_sec=0.5)

        # Visualize results
        disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("ROV Stereo Vision (Disparity)", disparity_vis)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = StereoVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()