#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from plantwall_custom_interfaces.msg import CdprPose

class CDPRStubNode(Node):
    def __init__(self):
        super().__init__('cdpr_stub')
        self.get_logger().info("CDPR Kinematic Stub Node has been started.")

        self.CDPR_width = 0.908
        self.CDPR_height = 0.944
        
        self.current_pos = np.array([self.CDPR_width/2.0, 0.545-0.03])
        self.current_ori = 0.0  
        
        self.target_pos = self.current_pos.copy()
        self.target_ori = self.current_ori

        # Simulated velocities
        self.linear_speed = 0.05 # [m/s] 
        self.angular_speed = 0.2 # [rad/s]
        
        self.loop_period = 0.02 # 50 Hz
        
        # ROS2 infrastructure
        self.current_pose_publisher = self.create_publisher(CdprPose, '/cdpr/current_pose', 10)
        self.goto_pose_subscriber = self.create_subscription(CdprPose, '/cdpr/goto_pose', self.goto_pose_callback, 10)
        
        # Simulation loop
        self.sim_timer = self.create_timer(self.loop_period, self.simulation_loop)

    def goto_pose_callback(self, msg: CdprPose):
        self.target_pos = np.array([msg.position[0], msg.position[1]])
        self.target_ori = msg.orientation

    def simulation_loop(self):
        # Move linearly with fixed speed to goal pose
        delta_pos = self.target_pos - self.current_pos
        distance = np.linalg.norm(delta_pos)
        
        max_step_dist = self.linear_speed * self.loop_period
        
        if distance > max_step_dist:
            self.current_pos += (delta_pos / distance) * max_step_dist
        else:
            self.current_pos = self.target_pos.copy()

        delta_ori = self.target_ori - self.current_ori
        max_step_ori = self.angular_speed * self.loop_period
        
        if abs(delta_ori) > max_step_ori:
            self.current_ori += np.sign(delta_ori) * max_step_ori
        else:
            self.current_ori = self.target_ori

        # Publish current pose
        current_pose_msg = CdprPose()
        current_pose_msg.position = self.current_pos.tolist()
        current_pose_msg.orientation = float(self.current_ori)
        
        self.current_pose_publisher.publish(current_pose_msg)

        self.get_logger().info(
            f"self.pose: ["
            f"{self.current_pos[0]:.4f}, {self.current_pos[1]:.4f}, "
            f"{self.current_ori:.4f}]",
            throttle_duration_sec=0.2
        )
        self.get_logger().info(
            f"goal_pose: ["
            f"{self.target_pos[0]:.4f}, {self.target_pos[1]:.4f}, {self.target_ori:.4f}]",
            throttle_duration_sec=0.2
        )
        self.get_logger().info(
            f"-----------------------------------------------",
            throttle_duration_sec=0.2
        )

def main(args=None):
    rclpy.init(args=args)
    node = CDPRStubNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()