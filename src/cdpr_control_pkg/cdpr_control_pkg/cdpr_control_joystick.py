#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_control_joystick')
        self.get_logger().info("CDPR Control Joystick Node has been started.")

def main(args=None):
    rclpy.init(args=args)
    node = CDPRControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
