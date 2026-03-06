#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from cdpr_control_pkg.DynamixelSyncDummy import DynamixelSyncDummy
from cdpr_control_pkg.cdpr_control_base_class import CDPRBaseControlNode
from plantwall_custom_interfaces.msg import CdprPose
import scipy
import matplotlib.pyplot as plt
import time


class CDPRForceControlNode(CDPRBaseControlNode):
    def __init__(self):
        super().__init__('cdpr_force_control')

        ## Input type
        self.USE_JOY = False
        self.USE_PATHPLANNER = True

        # CONTROLLER GAINS
        self.joystick_sensitivity = np.array([20.0, 20.0, 0.5]) # Joystick Sensitivity (x, y, theta) [N, N, Nm]

        # Trajectory force
        self.translation_force_norm = 20 # N
        self.rotation_force_norm = 0.5 # Nm

        # Tunable parameters
        self.t_min = -3.0

        # Damping Gains (Newtons per m/s)
        self.Kd = np.array([15.0, 15.0, 1.0]) # (Damp_x, Damp_y, Damp_theta)

        # Motor initialization
        # Set motors to force control
        self.motors.disable_torque(motors=[1,2,3,4])
        self.motors.write(motors=[1,2,3,4], values=OPERATING_MODES.CURRENT_CONTROL_MODE, control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])
    
    def __del__(self):
        print("CDPRForceControlNode destructor")
        
    def command_robot(self):
        self.control_loop_counter += 10
        
        if self.USE_JOY and self.USE_PATHPLANNER:
            print("CANNOT USE BOTH JOY AND PATHPLANNER")
            return
        if not self.USE_JOY and not self.USE_PATHPLANNER:
            print("HAS TO USE JOY OR PATHPLANNER")
            return

        # Update Pose
        raw_pose = self.forward_kinematics(self.cable_lengths, self.pose[0:2], self.pose[2])
        # SAFETY CLAMP
        margin = 0.05 # m
        clamped_x = np.clip(raw_pose[0], margin, self.CDPR_width - margin)
        clamped_y = np.clip(raw_pose[1], margin, self.CDPR_height - margin)
        clamped_th = np.clip(raw_pose[2], -0.3, 0.3) 
        self.pose = np.array([clamped_x, clamped_y, clamped_th])

        # Calculate velocity
        velocity = (self.previous_pose - self.pose) / self.control_loop_period
        self.previous_pose = self.pose

        # Calculate Damping Force
        u_damping = -self.Kd * velocity

        # Total Virtual Wrench
        u_total = np.array([0,0,0])
        if (self.USE_JOY):
            u_joystick = self.joystick_sensitivity * self.input
            u_total = u_joystick #+ u_damping
        if (self.USE_PATHPLANNER):
            goal_pose = self.target_pose.copy()
            delta_pose = goal_pose - self.pose
            if (np.linalg.norm(delta_pose[0:2]) < 0.00001):
                delta_pos_unit = np.array([0,0])
            else:
                delta_pos_unit = delta_pose[0:2] / np.linalg.norm(delta_pose[0:2])
            delta_ori_unit = np.sign(delta_pose[2])
            translation_force = delta_pos_unit * self.translation_force_norm
            rotation_force = delta_ori_unit * self.rotation_force_norm

            u_total = np.array([translation_force[0], translation_force[1], rotation_force])

        # Compute structure matrix
        cable_vectors = self.inverse_kinematics(self.pose[0:2], self.pose[2])
        S = self.compute_structure_matrix(cable_vectors, self.pose[2])

        # Null Space Optimization
        T_move = np.linalg.pinv(S) @ u_total
        S_nullspace = scipy.linalg.null_space(S).flatten()
        # Check for Validity
        valid_null_space = False
        T_final = None
        if S_nullspace.size > 0:
            S_nullspace = S_nullspace / np.linalg.norm(S_nullspace) # Normalize nullvector
            if np.sum(S_nullspace) < 0:
                S_nullspace = -S_nullspace # Allign nullvector
            if np.all(S_nullspace > -0.001): # Check for valid null vector
                valid_null_space = True
        
        # Branching Logic
        if valid_null_space:
            ns_safe = np.maximum(S_nullspace, 1e-6)
            lambda_vec = (self.t_min - T_move) / ns_safe
            lambda_optimal = np.max(lambda_vec)
            T_final = T_move + lambda_optimal * S_nullspace
        else:
            T_final = np.maximum(T_move, self.t_min)

        # Tension safety check
        if not self.tension_safety_check():
            return
    
        # Send to Motors        
        desired_motor_units = self.force_to_motor_units(T_final).astype(int).tolist()
        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_CURRENT,
            values=desired_motor_units
        )

        # Publish pose
        current_pose_msg = CdprPose()
        current_pose_msg.position = self.pose[0:2].tolist()
        current_pose_msg.orientation = self.pose[2]
        self.current_pose_publisher.publish(current_pose_msg)


        ## Prints ##
        # self.get_logger().info(f"u_total: {u_total}")
        # self.get_logger().info(f"T_final: {T_final}")
        self.get_logger().info(f"self.pose: {self.pose}")
        # self.get_logger().info(f"goal_pose: {goal_pose}")
        self.get_logger().info(f"delta_pos_unit: {delta_pos_unit}")
        self.get_logger().info(f"u_total: {u_total}")
        current_forces = self.motor_current_units_to_force(self.get_present_current())
        self.get_logger().info(f"current_forces: {current_forces}")




def main(args=None):
    rclpy.init(args=args)
    node = CDPRForceControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()