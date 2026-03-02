#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image # Import the Image message type
from plantwall_custom_interfaces.msg import CdprPose
import torch

class LeafDetectionNode(Node):
    def __init__(self):
        super().__init__('leaf_detection')
        self.get_logger().info("Leaf Detection Node has been started.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.control_loop_period = 0.1 # 10 Hz
        self.control_timer = self.create_timer(self.control_loop_period, self.leaf_detector)

        # Subscriber for Color Image
        self.realsense_color_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_image_callback, 
            10)

        # Subscriber for Depth Image
        self.realsense_depth_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/depth/image_rect_raw', 
            self.depth_image_callback, 
            10)

    def color_image_callback(self, msg):
        # 'msg' is a sensor_msgs/Image object
        self.get_logger().info(f"Received color image: {msg.width}x{msg.height}",
                               throttle_duration_sec=0.2)
        # To process this with OpenCV, you'll need cv_bridge later!

    def depth_image_callback(self, msg):
        # 'msg' is a sensor_msgs/Image object (often 16-bit integers for depth)
        self.get_logger().info(f"Received depth image: {msg.width}x{msg.height}",
                               throttle_duration_sec=0.2)
        
    def leaf_detector(self):
        print(f"Using {self.device} for leaf detection.")


def main(args=None):
    rclpy.init(args=args)
    node = LeafDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
