#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from plantwall_custom_interfaces.msg import CdprPose


class CDPRPathplannerNode(Node):
    def __init__(self):
        super().__init__('cdpr_pathplanner')

        self.current_pose = None

        self.pathplanner_loop_period = 0.02

        # ROS Infrastructure
        self.goto_pose_publisher = self.create_publisher(CdprPose, '/cdpr/goto_pose', 10)
        self.current_pose_subscriber = self.create_subscription(CdprPose, '/cdpr/current_pose', self.current_pose_callback, 10)
        self.pathplanner_timer = self.create_timer(self.pathplanner_loop_period, self.pathplanner)

        self.get_logger().info("CDPR Pathplanner Node has been started")

    def current_pose_callback(self, msg: CdprPose):
        current_pos = msg.position
        current_ori = msg.orientation
        self.current_pose = np.array([current_pos[0], current_pos[1], current_ori])
        
    def pathplanner(self):
        goto_pose_msg = CdprPose()
        goto_pose_msg.position = [0.5, 0.5]
        goto_pose_msg.orientation = -0.1
        self.goto_pose_publisher.publish(goto_pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CDPRPathplannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
