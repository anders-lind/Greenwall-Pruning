#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
from cdpr_control_pkg.DynamixelSyncDummy import DynamixelSyncDummy
from cdpr_control_pkg.cdpr_control_base_class import CDPRBaseControlNode
import scipy
import matplotlib.pyplot as plt
import time


class CDPRForceControlNode(CDPRBaseControlNode):
    def __init__(self):
        super().__init__('cdpr_force_control')

        # CONTROLLER GAINS
        self.joystick_sensitivity = np.array([40.0, 40.0, 1]) # Joystick Sensitivity (x, y, theta) [N, N, Nm]

        # Tunable parameters
        self.t_min = -3.0

        # Damping Gains (Newtons per m/s)
        self.Kd = np.array([15.0, 15.0, 1.0]) # (Damp_x, Damp_y, Damp_theta)

        # Motor initialization
        # Set motors to force control
        self.motors.write(motors=[1,2,3,4], values=0, control_type=CONTROL_TABLE.OPERATING_MODE)
    
    def __del__(self):
        print("CDPRControlNode destructor")
        
    def command_robot(self):
        # 1. Update Pose via FK
        raw_pose = self.forward_kinematics(self.cable_lengths, self.pose[0:2], self.pose[2])
        
        # SAFETY CLAMP
        margin = 0.05
        clamped_x = np.clip(raw_pose[0], margin, self.CDPR_width - margin)
        clamped_y = np.clip(raw_pose[1], margin, self.CDPR_height - margin)
        clamped_th = np.clip(raw_pose[2], -0.3, 0.3) 
        self.pose = np.array([clamped_x, clamped_y, clamped_th])

        # 2. Calculate velocity
        velocity = (self.previous_pose - self.pose) / self.control_loop_period
        self.previous_pose = self.pose

        # 3. Calculate Damping Force
        u_damping = -self.Kd * velocity

        # 4. Total Virtual Wrench
        u_joystick = self.joystick_sensitivity * self.input
        u_total = u_joystick + u_damping

        # 5. Compute Matrices
        cable_vectors = self.inverse_kinematics(self.pose[0:2], self.pose[2])
        S = self.compute_structure_matrix(cable_vectors, self.pose[2])

        # 6. Null Space Optimization
        T_move = np.linalg.pinv(S) @ u_total
        S_nullspace = scipy.linalg.null_space(S).flatten()

        # 7. Check for Validity
        valid_null_space = False
        T_final = None
        if S_nullspace.size > 0:
            S_nullspace = S_nullspace / np.linalg.norm(S_nullspace) # Normalize nullvector
            if np.sum(S_nullspace) < 0:
                S_nullspace = -S_nullspace # Allign nullvector
            if np.all(S_nullspace > -0.001): # Check for valid null vector
                valid_null_space = True
        
        # 8. Branching Logic
        if valid_null_space:
            ns_safe = np.maximum(S_nullspace, 1e-6)
            lambda_vec = (self.t_min - T_move) / ns_safe
            lambda_optimal = np.max(lambda_vec)
            T_final = T_move + lambda_optimal * S_nullspace
        else:
            T_final = np.maximum(T_move, self.t_min)
    
        # # 9. Send to Motors        
        desired_currents = self.force_to_current(T_final)
        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_CURRENT,
            values=self.current_to_motor_input(desired_currents)
        )

def main(args=None):
    rclpy.init(args=args)
    node = CDPRForceControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()