#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
import scipy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
from cdpr_control_pkg.DynamixelSyncDummy import DynamixelSyncDummy
from plantwall_custom_interfaces.msg import CdprPose, MotorCmd
from cdpr_control_pkg.cdpr_control_base_class import CDPRBaseControlNode


class CDPRSpeedControlJoyNode(CDPRBaseControlNode):
    def __init__(self):
        super().__init__('cdpr_speed_control_joy')

        # CONTROLLER GAINS
        # self.joystick_sensitivity = np.array([1, 1, 1]) # Reference point offset (x, y, theta) [m, m, rad]

    def __del__(self):
        print("CDPRControlNode destructor")
        
    def command_robot(self):
        current_pose_msg = CdprPose()

        # Controller mode
        delta_pos_unscaled = self.input[0:2]
        delta_ori = self.input[2]

        delta_pos_norm = np.linalg.norm(delta_pos_unscaled)
        if delta_pos_norm < 1e-6:
            delta_pos = np.array([0.0, 0.0])
        elif delta_pos_norm < 0.01:
            delta_pos = delta_pos_unscaled
        else:
            delta_pos = delta_pos_unscaled/delta_pos_norm

        # Pose estimation with integration
        self.pose[0:2] = self.pose[0:2] + delta_pos * self.movement_speed * self.control_loop_period
        self.pose[2:3] = self.pose[2:3] + delta_ori * self.rotation_speed * self.control_loop_period

        # Clamping pose
        margin = 0.05 # m
        self.pose[0] = np.clip(self.pose[0], margin, self.CDPR_width - margin)
        self.pose[1] = np.clip(self.pose[1], margin, self.CDPR_height - margin)
        self.pose[2] = np.clip(self.pose[2], -0.3, 0.3) 

        # Pose estimation with forward kinematics
        #self.pose = self.forward_kinematics(self.cable_lengths, self.pose[0:2], self.pose[2])

        desired_cable_vectors = self.inverse_kinematics(self.pose[0:2], self.pose[2])
        desired_cable_lengths = [np.linalg.norm(l) for l in desired_cable_vectors]

        # cable_errors = desired_cable_lengths - self.cable_lengths # meters
        cable_errors = np.array(desired_cable_lengths) - self.previous_cable_lengths # meters

        desired_cable_velocities = -(cable_errors / self.control_loop_period)
        desired_spool_rpm = (desired_cable_velocities / self.effective_circumferences) * 60 # convert to RPM

        desired_motor_units = desired_spool_rpm / 0.229 # convert to motor units (1 unit = 0.229 RPM)

        velocity_int_list = [int(v) for v in desired_motor_units]

        # Tension safety check
        if not self.tension_safety_check():
            return

        # Prints
        current_forces = self.motor_current_units_to_force(self.present_current)
        self.get_logger().info(
            f"current_forces: ["
            f"{current_forces[0]:.2f}, {current_forces[1]:.2f},"
            f"{current_forces[2]:.2f}, {current_forces[3]:.2f}],",
            throttle_duration_sec=0.2
        )
        # self.get_logger().info(
        #     f"self.pose: ["
        #     f"{self.pose[0]:.4f}, {self.pose[1]:.4f}, "
        #     f"{self.pose[2]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"self.cable_lengths: ["
        #     f"{self.cable_lengths[0]:.4f}, {self.cable_lengths[1]:.4f}, "
        #     f"{self.cable_lengths[2]:.4f}, {self.cable_lengths[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"desired_cable_lengths: ["
        #     f"{desired_cable_lengths[0]:.4f}, {desired_cable_lengths[1]:.4f}, "
        #     f"{desired_cable_lengths[2]:.4f}, {desired_cable_lengths[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"desired_velocity_linear: ["
        #     f"{desired_cable_velocities[0]:.4f}, {desired_cable_velocities[1]:.4f}, "
        #     f"{desired_cable_velocities[2]:.4f}, {desired_cable_velocities[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"velocity_int_list: ["
        #     f"{velocity_int_list[0]:.4f}, {velocity_int_list[1]:.4f}, "
        #     f"{velocity_int_list[2]:.4f}, {velocity_int_list[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"-----------------------------------------------",
        #     throttle_duration_sec=0.2
        # )
        
        self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
        self.motor_write_continous_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=velocity_int_list, control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

        current_pose_msg.position = self.pose[0:2].tolist()
        current_pose_msg.orientation = self.pose[2]
        self.current_pose_publisher.publish(current_pose_msg)

        # Update old variables
        self.previous_cable_lengths = desired_cable_lengths
        self.control_loop_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = CDPRSpeedControlJoyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
