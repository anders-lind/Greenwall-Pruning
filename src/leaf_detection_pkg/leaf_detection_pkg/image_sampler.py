#!/usr/bin/env python3

import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from cv_bridge import CvBridge
from datetime import datetime

class ImageSamplerNode(Node):
    def __init__(self):
        super().__init__('image_sampler_node')
        
        # --- CONFIGURATION ---
        # Saves to ~/Thesis/captured_images/YYYYMMDD_HHMMSS/
        home = os.path.expanduser("~")
        session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = os.path.join(home, "Thesis/captured_images", session_id)
        os.makedirs(self.save_path, exist_ok=True)
        
        self.get_logger().info(f"Storage initialized at: {self.save_path}")
        self.get_logger().info("Ready! Press RB (Button 5) to capture an image.")

        # Internal state
        self.cv_bridge = CvBridge()
        self.current_image = None
        self.rb_pressed_last = False # To prevent multiple saves from one click

        # ROS Subscribers
        self.color_sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_image_callback, 
            10)
            
        self.joy_sub = self.create_subscription(
            Joy, 
            '/joy', 
            self.joy_callback, 
            10)

    def color_image_callback(self, msg):
        # Continuous stream update
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def joy_callback(self, msg: Joy):
        # RB is typically index 5 on most Xbox/Logitech controllers
        # We check for a "rising edge" (button was 0, now is 1)
        rb_button_state = msg.buttons[5]

        if rb_button_state == 1 and not self.rb_pressed_last:
            self.capture_image()
            self.rb_pressed_last = True
        elif rb_button_state == 0:
            self.rb_pressed_last = False

    def capture_image(self):
        if self.current_image is None:
            self.get_logger().warn("Capture requested, but no image received yet!")
            return

        timestamp = datetime.now().strftime("%H-%M-%S_%f")
        filename = f"capture_{timestamp}.png"
        full_path = os.path.join(self.save_path, filename)

        try:
            cv2.imwrite(full_path, self.current_image)
            self.get_logger().info(f"Successfully saved: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to write file: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSamplerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()