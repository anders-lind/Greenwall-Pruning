#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
import scipy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
from plantwall_custom_interfaces.msg import CdprPose, MotorCmd
from cdpr_control_pkg.cdpr_control_base_class import CDPRBaseControlNode


class CDPRSpeedControlJoyNode(CDPRBaseControlNode):
    def __init__(self):
        super().__init__('cdpr_speed_control')

    def command_robot(self):
        current_pose_msg = CdprPose()

        # self.pose = self.forward_kinematics(self.cable_lengths, self.pose[0:2], self.pose[2]).copy()

        actual_cable_lengths = self.cable_lengths - self.cable_errors
        self.pose = self.forward_kinematics(actual_cable_lengths, self.pose[0:2], self.pose[2]).copy()

        self.compute_release_points()

        # Get goal pose from controller
        goal_pose = np.array([0.0, 0.0, 0.0])
        goal_pose[0:2] = self.pose[0:2] + self.input[0:2]
        goal_pose[2:3] = self.pose[2:3] + self.input[2]

        # # Clamp pose
        # margin = 0.05 # m
        # self.pose[0] = np.clip(self.pose[0], margin, self.CDPR_width - margin)
        # self.pose[1] = np.clip(self.pose[1], margin, self.CDPR_height - margin)
        # self.pose[2] = np.clip(self.pose[2], -0.3, 0.3) 

        # Goal relative to current pos and ori
        delta_pos = goal_pose[0:2] - self.pose[0:2]
        delta_ori = goal_pose[2] - self.pose[2]

        # Scale delta pos
        delta_pos_norm = np.linalg.norm(delta_pos)
        if delta_pos_norm < self.stopping_radius_pos:
            delta_pos_scaled = np.array([0.0, 0.0])
        # elif delta_pos_norm < self.slowdown_radius_pos:
        #     delta_pos_scaled = delta_pos
        else:
            delta_pos_scaled = delta_pos# / delta_pos_norm
        
        # Scale delta ori
        if abs(delta_ori) < self.stopping_rot:
            delta_ori_scaled = 0.0
        else:
            delta_ori_scaled = np.sign(delta_ori)


        # Pose estimation with integration
        # self.pose[0:2] = self.pose[0:2] + delta_pos_scaled * self.movement_speed * self.control_loop_period
        # self.pose[2] = self.pose[2] + np.sign(delta_ori) * self.rotation_speed * self.control_loop_period

        # 1. Feedforward: Where we are vs. Where we want to be next
        # Calculate ideal cable lengths for CURRENT pose
        current_ideal_vectors = self.inverse_kinematics(self.pose[0:2], self.pose[2])
        current_ideal_lengths = np.array([np.linalg.norm(l) for l in current_ideal_vectors])

        # Calculate ideal cable lengths for NEXT pose
        next_pos = self.pose[0:2] + delta_pos_scaled * self.movement_speed * self.control_loop_period
        next_ori = self.pose[2] + delta_ori_scaled * self.rotation_speed * self.control_loop_period
        
        desired_cable_vectors = self.inverse_kinematics(next_pos, next_ori)
        desired_cable_lengths = np.array([np.linalg.norm(l) for l in desired_cable_vectors])

        # Feedforward Velocity (The speed required just to execute the movement)
        ff_velocities = -(desired_cable_lengths - current_ideal_lengths) / self.control_loop_period
        # ff_velocities = -(desired_cable_lengths - self.cable_lengths) / self.control_loop_period

        # 2. Feedback: Correcting sensor error
        # Compare where the cables SHOULD be right now vs. where the encoders say they ARE
        self.cable_errors = current_ideal_lengths - self.cable_lengths 
        
        # Proportional Gain (Tune this! Start small. 2.0 means it corrects errors over ~0.5 seconds)
        Kp_feedback = 2.0
        fb_velocities = -(self.cable_errors * Kp_feedback)

        # 3. Total Velocity Command
        desired_cable_velocities = ff_velocities + fb_velocities

        # Convert to RPM and Motor Units
        desired_spool_rpm = (desired_cable_velocities / self.effective_circumferences) * 60 
        desired_motor_units = desired_spool_rpm / 0.229 
        velocity_int_list = [int(v) for v in desired_motor_units]

        # # Tension safety check
        current_forces = self.motor_current_units_to_force(self.present_current)
        

        # Tension safety check
        if not self.tension_safety_check():
            return


        # Prints
        # self.get_logger().info(
        #     f"current_forces: ["
        #     f"{current_forces[0]:.2f}, {current_forces[1]:.2f}, {current_forces[2]:.2f}, {current_forces[3]:.2f}]",
        #     # throttle_duration_sec=0.1
        # )

        # self.get_logger().info(
        #     f"cable_errors: ["
        #     f"{self.cable_errors[0]:.4f}, {self.cable_errors[1]:.4f}, {self.cable_errors[2]:.4f}, {self.cable_errors[3]:.4f}]",
        #     throttle_duration_sec=self.log_period
        # )

        R1 = self.B1 - self.C1
        R2 = self.B2 - self.C2
        R3 = self.B3 - self.C3
        R4 = self.B4 - self.C4

        # self.get_logger().info(
        #     f"R vectors: ["
        #     f"{R1[0]:.4f}, {R1[1]:.4f}; "
        #     f"{R2[0]:.4f}, {R2[1]:.4f}; "
        #     f"{R3[0]:.4f}, {R3[1]:.4f}; "
        #     f"{R4[0]:.4f}, {R4[1]:.4f}]",
        #     throttle_duration_sec=self.log_period
        # )

        # self.get_logger().info(
        #     f"self.pose: ["
        #     f"{self.pose[0]:.4f}, {self.pose[1]:.4f}, "
        #     f"{self.pose[2]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"delta_pose: ["
        #     f"{delta_pos[0]:.4f}, {delta_pos[1]:.4f}, {delta_ori:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"goal_pose: ["
        #     f"{goal_pose[0]:.4f}, {goal_pose[1]:.4f}, {goal_pose[2]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"cable_lengths: ["
        #     f"{self.cable_lengths[0]:.4f}, {self.cable_lengths[1]:.4f}, {self.cable_lengths[2]:.4f}, {self.cable_lengths[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"current_ideal_lengths: ["
        #     f"{current_ideal_lengths[0]:.4f}, {current_ideal_lengths[1]:.4f}, {current_ideal_lengths[2]:.4f}, {current_ideal_lengths[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"desired_cable_velocities: ["
        #     f"{desired_cable_velocities[0]:.4f}, {desired_cable_velocities[1]:.4f}, {desired_cable_velocities[2]:.4f}, {desired_cable_velocities[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )


        # self.get_logger().info(
        #     f"-----------------------------------------------",
        #     throttle_duration_sec=self.log_period
        # )
        
        # Write to motors
        self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
        self.motor_write_continous_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=velocity_int_list, control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))
        # self.get_logger().info(f"velocity_int_list: {velocity_int_list}", throttle_duration_sec = 0.2)

        # Publish pose
        current_pose_msg.position = self.pose[0:2].tolist()
        current_pose_msg.orientation = self.pose[2]
        self.current_pose_publisher.publish(current_pose_msg)
        
        # Update old variables
        self.previous_cable_lengths = desired_cable_lengths

def main(args=None):
    rclpy.init(args=args)
    node = CDPRSpeedControlJoyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
