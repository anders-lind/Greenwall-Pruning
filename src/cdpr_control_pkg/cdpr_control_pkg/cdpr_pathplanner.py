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

        CDPR_width = 0.908
        CDPR_height = 0.944
        self.initial_pose = np.array([CDPR_width/2, 0.52-0.03])
        self.center_pos = np.array([CDPR_width/2, CDPR_height/2])
        self.current_target_idx = 0
        self.smoothing_radius = 0.005 # 0.5 cm

        self.CDPR_height = 0.944
        self.CDPR_width = 0.908

        # Square path around the initial pose
        # self.square_poses = [
        #     self.initial_pose + np.array([0.00, 0.05]), # 5 cm up
        #     self.initial_pose + np.array([0.05, 0.05]), # 5 cm up and 5 cm right
        #     self.initial_pose + np.array([0.05, -0.05]), # 5 cm right and 5 cm down
        #     self.initial_pose + np.array([-0.05, -0.05]), # 5 cm down and 5 cm left
        #     self.initial_pose + np.array([-0.05, 0.05]), # 5 cm left and 5 cm up
        #     self.initial_pose + np.array([0.0, 0.05]), # 5 cm up
        #     self.initial_pose + np.array([0.0, 0.0]) # back to initial pose
        # ]

        # Workspace test poses
        self.pos = [self.center_pos, np.array([self.CDPR_width, self.CDPR_height])] # top right corner
        # self.pos = [self.center_pos, np.array([self.CDPR_width, 0.0])] # bottom right corner
        # self.pos = [self.center_pos, np.array([0.0, 0.0])] # bottom left corner
        # self.pos = [self.center_pos, np.array([0.0, self.CDPR_height])] # top left corner

        # self.pos = [self.center_pos, np.array([self.CDPR_width, self.CDPR_height/2])] # right side
        # self.pos = [self.center_pos, np.array([self.CDPR_width/2, 0.0])] # bottom side
        # self.pos = [self.center_pos, np.array([0.0, self.CDPR_height/2])] # left side
        # self.pos = [self.center_pos, np.array([self.CDPR_width/2, self.CDPR_height])] # top side

        self.pathplanner_loop_period = 0.02

        # ROS Infrastructure
        self.goto_pose_publisher = self.create_publisher(CdprPose, '/cdpr/goto_pose', 10)
        self.current_pose_subscriber = self.create_subscription(CdprPose, '/cdpr/current_pose', self.current_pose_callback, 10)
        self.pathplanner_timer = self.create_timer(self.pathplanner_loop_period, self.pathplanner_from_poselist)

        self.get_logger().info("CDPR Pathplanner Node has been started")

    def current_pose_callback(self, msg: CdprPose):
        current_pos = msg.position
        current_ori = msg.orientation
        self.current_pose = np.array([current_pos[0], current_pos[1], current_ori])
        
    def pathplanner_from_poselist(self):
        goto_pose_msg = CdprPose()
        if self.current_pose is None:
            return

        if self.current_target_idx >= len(self.square_poses):
            return

        # Go to towards next point until within smoothing radius, then switch to next point
        if np.linalg.norm(self.current_pose[0:2] -self.square_poses[self.current_target_idx]) < self.smoothing_radius:
            self.current_target_idx = self.current_target_idx + 1
            if self.current_target_idx >= len(self.square_poses):
                return

        goto_pose_msg.position = self.square_poses[self.current_target_idx].tolist()
        goto_pose_msg.orientation = 0.0

        self.goto_pose_publisher.publish(goto_pose_msg)

    # def pathplanner_online_circle(self):
    #     goto_pose_msg = CdprPose()
    #     if self.current_pose is None:
    #         return

    #     # Circle path around the initial pose
    #     circle_radius = 0.05 # 5 cm
    #     circle_discretization = 20
    #     angle_step = 2 * np.pi / circle_discretization
        
    #     if self.current_target_idx >= circle_discretization:
    #         target_pose = self.initial_pose
    #         return


    #     target_pose = self.initial_pose + np.array([0, circle_radius])
    #     # Compute next target point on the circle when current_pose is within the smoothing radius
    #     if np.linalg.norm(self.current_pose[0:2] - target_pose) < self.smoothing_radius:
    #         self.get_logger().info(f"Reached target point {self.current_target_idx}, moving to next target")
    #         self.current_target_idx += 1
    #         if self.current_target_idx >= circle_discretization:
    #             target_pose = self.initial_pose
    #             return
            
    #     if self.current_target_idx > 0:
    #         current_angle = angle_step * self.current_target_idx
    #         target_pose = self.initial_pose + np.array([np.cos(current_angle),np.sin(current_angle)]) * circle_radius

    #     goto_pose_msg.position = target_pose.tolist()
    #     goto_pose_msg.orientation = 0.0

    #     self.goto_pose_publisher.publish(goto_pose_msg)

    def pathplanner_online_circle(self):
        if self.current_pose is None:
            return

        # Circle parameters
        circle_radius = 0.1 # 5 cm
        circle_discretization = 20
        angle_step = 2 * np.pi / circle_discretization
        
        # 1. Determine the ACTIVE target pose BEFORE checking distance
        # FIX: Changed >= to > so it includes index 20 (which is 360 degrees, closing the loop)
        if self.current_target_idx > circle_discretization:
            # Trajectory complete: return to center
            target_pose = self.initial_pose
        else:
            # Calculate the specific point on the circle for the current index
            # FIX: Added np.pi/2 to start at the TOP of the circle [0, r]
            current_angle = (angle_step * self.current_target_idx) + (np.pi / 2)
            target_pose = self.initial_pose + np.array([np.cos(current_angle), np.sin(current_angle)]) * circle_radius

        # 2. Check distance to the ACTIVE target pose
        if self.current_target_idx <= circle_discretization:
            distance_to_target = np.linalg.norm(self.current_pose[0:2] - target_pose)
            
            if distance_to_target < self.smoothing_radius:
                self.get_logger().info(f"Reached target point {self.current_target_idx}, moving to next target")
                self.current_target_idx += 1
                
                # 3. Update target_pose immediately for the current frame
                if self.current_target_idx > circle_discretization:
                    target_pose = self.initial_pose
                else:
                    new_angle = (angle_step * self.current_target_idx) + (np.pi / 2)
                    target_pose = self.initial_pose + np.array([np.cos(new_angle), np.sin(new_angle)]) * circle_radius

        # 4. Construct message and publish
        goto_pose_msg = CdprPose()
        goto_pose_msg.position = target_pose.tolist()
        goto_pose_msg.orientation = 0.0

        self.goto_pose_publisher.publish(goto_pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CDPRPathplannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
