#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Joy
from plantwall_custom_interfaces.msg import CdprPose
from plantwall_custom_interfaces.srv import CdprPose as CdprPoseSrv
from std_srvs.srv import SetBool
import time
from enum import Enum
import os

class State(Enum):
    IDLE = 0
    SEARCHING = 1
    GOTO = 2

class CDPRPathplannerNode(Node):
    def __init__(self):
        super().__init__('cdpr_pathplanner')
        self.get_logger().info("CDPR Pathplanner Node has been started")

        # System parameters
        # self.end_effector_height = 0.039161
        self.end_effector_top_margin = 0.03
        self.end_effector_bottom_margin = 0.13
        self.end_effector_width = 0.22
        # self.CDPR_height = 0.944
        # self.CDPR_width = 0.908

        self.CDPR_height = 1.0
        self.CDPR_width = 1.0
        
        # Auxiliary poses
        self.initial_pos = np.array([0.52, 0.545])
        self.clear_homing_stick = self.initial_pos + np.array([0.0, 0.05])
        
        # State variables
        self.current_pose = None
        self.current_target_idx = 0
        self.state = State.IDLE  
        self.active_target_pos = self.initial_pos.copy()
        self.active_target_ori = 0.0
        
        # Variable for storing the resume position for when search is interrupted
        self.resume_pos = None
        self.needs_to_resume = False

        # Tuning variables
        self.smoothing_radius = 0.005 # 0.5 cm
        self.pathplanner_loop_period = 0.02

        # Startup sequence variables
        self.startup_poses = [self.clear_homing_stick]
        self.startup_idx = 0
        self.has_exited_homing = False

        path_file = os.path.join(os.path.expanduser("~"), "Thesis", "cdpr_search_path.csv")
        try:
            # Load the CSV into a numpy array of shape (N, 2)
            self.poselist = np.loadtxt(path_file, delimiter=",")
            self.get_logger().info(f"Successfully loaded {len(self.poselist)} offline waypoints.")
        except Exception as e:
            self.get_logger().error(f"Failed to load path file: {e}. Falling back to default center pose.")

        # # Path poselist
        # self.poselist = [
        #     self.initial_pos + np.array([0.00, 0.05]), 
        #     self.initial_pos + np.array([0.05, 0.05]), 
        #     self.initial_pos + np.array([0.05, -0.05]), 
        #     self.initial_pos + np.array([-0.05, -0.05]), 
        #     self.initial_pos + np.array([-0.05, 0.05]), 
        #     self.initial_pos + np.array([0.0, 0.05]), 
        #     self.initial_pos + np.array([0.0, 0.0]) 
        # ]

        # Callback groups
        self.service_cb_group = MutuallyExclusiveCallbackGroup()

        # Service servers
        self.toggle_search_state_srv = self.create_service(
            SetBool, '/cdpr_pathplanner/toggle', self.toggle_search_state_callback) 
            
        self.goto_pose_srv = self.create_service(
            CdprPoseSrv, '/cdpr_pathplanner/goto_pose', self.goto_pose_callback,
            callback_group=self.service_cb_group) 

        # ROS Infrastructure
        self.goto_pose_publisher = self.create_publisher(CdprPose, '/cdpr/goto_pose', 10)
        self.current_pose_subscriber = self.create_subscription(CdprPose, '/cdpr/current_pose', self.current_pose_callback, 10)
        self.pathplanner_timer = self.create_timer(self.pathplanner_loop_period, self.pathplanner_loop)


    def toggle_search_state_callback(self, request, response):
        if request.data:
            self.get_logger().info("Toggling pathplanner search state to ACTIVE.")
            self.state = State.SEARCHING
        elif self.state == State.IDLE:
            self.get_logger().info("Pathplanner is already in IDLE state. No action taken.")
            response.success = True
            return response
        else:
            self.get_logger().info("Toggling pathplanner search state to INACTIVE.")
            self.state = State.IDLE
            # Log the resume position when search is interrupted, so it can returned to when search is toggled back on
            if self.current_pose is not None:
                self.resume_pos = self.current_pose[0:2].copy()
                self.needs_to_resume = True
                
                goto_pose_msg = CdprPose()
                goto_pose_msg.position = self.resume_pos.tolist()
                goto_pose_msg.orientation = self.current_pose[2]
                self.goto_pose_publisher.publish(goto_pose_msg)
                
        response.success = True
        return response
    
    def goto_pose_callback(self, request, response):
        self.state = State.GOTO
        target_pos = np.array([request.position[0], request.position[1]])
        self.active_target_pos = target_pos
        self.active_target_ori = request.orientation
        
        self.get_logger().info(f"Received goto_pose request: position={target_pos}, orientation={self.active_target_ori}")
        
        # Wait inside GOTO state until the robot is within smooting radius of the target
        while True:
            if self.current_pose is not None:
                distance = np.linalg.norm(self.current_pose[0:2] - target_pos)
                if distance < self.smoothing_radius:
                    break
                # If target_pos is outside of CDPR workspace. 
                min_pos = np.array([self.end_effector_width/2.0, self.end_effector_bottom_margin])
                max_pos = np.array([self.CDPR_width - self.end_effector_width / 2.0, self.CDPR_height - self.end_effector_top_margin])
                clipped_target_pos = np.clip(target_pos, min_pos, max_pos)
                clipped_distance = np.linalg.norm(self.current_pose[0:2] - clipped_target_pos)
                if clipped_distance < self.smoothing_radius:
                    break
            
            time.sleep(self.pathplanner_loop_period) 
            
        self.state = State.IDLE
        self.get_logger().info("Arrived at target pose. Robot is now in IDLE state.")
        return response

    def current_pose_callback(self, msg: CdprPose):
        current_pos = msg.position
        current_ori = msg.orientation
        self.current_pose = np.array([current_pos[0], current_pos[1], current_ori])
        
    def pathplanner_loop(self):
        if self.state == State.IDLE:
            return

        if self.current_pose is None:
            return

        # SEARCHING STATE LOGIC
        if self.state == State.SEARCHING:
            
            # Priority 1: Backtrack to resume_pos if interrupted
            if self.needs_to_resume and self.resume_pos is not None:
                target = self.resume_pos
                self.active_target_pos = target
                self.active_target_ori = 0.0
                
                distance = np.linalg.norm(self.current_pose[0:2] - target)
                if distance < self.smoothing_radius:
                    self.get_logger().info("Successfully returned to pre-interruption pose.")
                    self.needs_to_resume = False 
                    
            # Priority 2: Execute startup sequence if we haven't yet
            elif not self.has_exited_homing:
                target = self.startup_poses[self.startup_idx]
                self.active_target_pos = target
                self.active_target_ori = 0.0
                
                distance = np.linalg.norm(self.current_pose[0:2] - target)
                if distance < self.smoothing_radius:
                    self.startup_idx += 1
                    self.get_logger().info(f"Homing exit waypoint {self.startup_idx} reached.")
                    
                    if self.startup_idx >= len(self.startup_poses):
                        self.has_exited_homing = True
                        self.get_logger().info("Successfully exited homing fixture. Starting offline path tracking.")

            # Priority 3: Normal offline poselist tracking
            else:
                target = self.poselist[self.current_target_idx]
                self.active_target_pos = target
                self.active_target_ori = 0.0

                distance = np.linalg.norm(self.current_pose[0:2] - target)
                if distance < self.smoothing_radius:
                    self.current_target_idx += 1
                    self.get_logger().info(f"Waypoint reached. Moving to offline index {self.current_target_idx}")
                    
                    # Loop the offline path back to the beginning!
                    if self.current_target_idx >= len(self.poselist):
                        self.current_target_idx = 0
                        self.get_logger().info("Search path completed. Looping back to start (bypassing homing).")


        # Clip position and orienation to safe limits
        min_pos = np.array([self.end_effector_width/2.0, self.end_effector_bottom_margin])
        max_pos = np.array([self.CDPR_width - self.end_effector_width / 2.0, self.CDPR_height - self.end_effector_top_margin])
        safe_pos = np.clip(self.active_target_pos, min_pos, max_pos)

        abs_max_ori = 0.3
        safe_ori = np.clip(self.active_target_ori, -abs_max_ori, abs_max_ori)
        
        # Publish target
        goto_pose_msg = CdprPose()
        goto_pose_msg.position = safe_pos.tolist()
        goto_pose_msg.orientation = float(safe_ori)
        self.goto_pose_publisher.publish(goto_pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CDPRPathplannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()