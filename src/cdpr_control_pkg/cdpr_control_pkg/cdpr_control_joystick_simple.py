#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_control_joystick')

        # CDPR initial state
        self.initial_pos = np.array([0.4, 0.5, 0.0]) # x, y, theta
        self.initial_cable_vectors = self.inverse_kinematics(self.initial_pos[0:2], self.initial_pos[2])
        self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]

        self.input = np.array([0.0, 0.0, 0.0]) # x, y, theta
        self.pos = self.initial_pos

        # CDPR physical parameters
        self.spool_circumference = 0.021*np.pi

        # Control parameters
        self.Kp_motor = 100
        self.cable_tension = 10

        # Motor initialization
        self.motors = DynamixelSync()
        self.zero_offsets = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        self.motors.write(motors=[1,2,3,4], values=[0,0,0,0], control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])
        
        self.joy_subscriber = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        self.control_timer = self.create_timer(
            0.02, # 50 Hz
            self.command_robot
        )

        self.get_logger().info("CDPR Control Joystick Node has been started.")
    
    def joy_callback(self, msg: Joy):
        x = -msg.axes[3]
        y = -msg.axes[4]
        theta = msg.axes[2]-msg.axes[5]
        self.input = [x,y,theta]
        
    def command_robot(self):
        joystick_sensitivity = 0.001
        for i in range(len(self.pos)):
            self.pos[i] += joystick_sensitivity * self.input[i]
        
        cable_vectors = self.inverse_kinematics(self.pos[0:2], self.pos[2])
        desired_cable_lengths = [np.linalg.norm(l) for l in cable_vectors]
        current_cable_lengths = self.get_current_cable_lengths()

        # Position controller
        length_errors = desired_cable_lengths - current_cable_lengths
        control_current = self.Kp_motor * length_errors
        desired_current = self.cable_tension + control_current

        # self.get_logger().info(
        #     f"Pos: [{self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f}], "
        #     f"Lengths: [{desired_cable_lengths[0]:.3f}, {desired_cable_lengths[1]:.3f}, "
        #     f"{desired_cable_lengths[2]:.3f}, {desired_cable_lengths[3]:.3f}]",
        #     throttle_duration_sec=0.1
        # )
        self.get_logger().info(
            f"Length errors: ["
            f"{length_errors[0]:.4f}, {length_errors[1]:.4f}, "
            f"{length_errors[2]:.4f}, {length_errors[3]:.4f}]",
            throttle_duration_sec=0.2
        )
        # self.get_logger().info(
        #     f"Desired currents: ["
        #     f"{desired_current[0]:.4f}, {desired_current[1]:.4f}, "
        #     f"{desired_current[2]:.4f}, {desired_current[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )

        # Command motors
        desired_current_int_list = [int(v) for v in desired_current]
        self.get_logger().info(
            f"Desired currents: ["
            f"{desired_current_int_list[0]}, {desired_current_int_list[1]}, "
            f"{desired_current_int_list[2]}, {desired_current_int_list[3]}]",
            throttle_duration_sec=0.2
        )


        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_CURRENT,
            values=desired_current_int_list
        )

    def get_current_cable_lengths(self):
        positions_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        motor_encoder_positions = np.array(positions_list) - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = self.initial_cable_lengths + motor_encoder_rotations * self.get_spool_circumference()
        return current_cable_lengths

    def get_spool_circumference(self):
        return self.spool_circumference

    def inverse_kinematics(self, position, orientation):
        q1 = np.array([-0.0425, -0.02])
        q2 = np.array([-0.0425, 0.02])
        q3 = np.array([0.0425, 0.02])
        q4 = np.array([0.0425, -0.02])

        b1 = np.array([0, 0])
        b2 = np.array([0, 1.0175])
        b3 = np.array([0.975, 1.0175])
        b4 = np.array([0.975, 0])

        p = np.array(position)
        theta = orientation

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        l1 = p + R @ q1 - b1
        l2 = p + R @ q2 - b2
        l3 = p + R @ q3 - b3
        l4 = p + R @ q4 - b4

        return np.array([l1, l2, l3, l4])
        

def main(args=None):
    rclpy.init(args=args)
    node = CDPRControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
