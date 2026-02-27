#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from plantwall_custom_interfaces.msg import CdprPose

class CDPRPublisher(Node):
    def __init__(self):
        super().__init__('cdpr_publisher')
        self.pub = self.create_publisher(CdprPose, '/cdpr/current_pose', 10)
        self.timer = self.create_timer(0.1, self.publish_pose)
        self.t = 0.0
    

    def publish_pose(self):
        msg = CdprPose()

        center_x = 0.454
        center_y = 0.472

        msg.position = [center_x + 0.3 * math.cos(self.t), center_y + 0.3 * math.sin(self.t)]
        msg.orientation = math.cos(self.t * 0.2)*2*3.14
        self.pub.publish(msg)
        self.get_logger().info(f'Published: x={msg.position[0]:.2f}, y={msg.position[1]:.2f}, θ={msg.orientation:.2f}')
        self.t += 0.1



def main(args=None):
    rclpy.init(args=args)
    node = CDPRPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == 'main':
    main()