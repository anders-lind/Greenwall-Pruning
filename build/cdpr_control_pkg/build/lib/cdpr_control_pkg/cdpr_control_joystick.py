#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_ADDRESS


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_control_joystick')

        self.input = np.array([0.0, 0.0, 0.0]) # x, y, theta
        # Eventuel refactor her til noget med initial pos
        self.pos = np.array([0.4, 0.5, 0.0]) # x, y, theta

        # Constants
        self.initial_cable_length = self.inverse_kinematics(self.pos[0:2], self.pos[2])
        self.spool_circumference = 0.021*np.pi

        motors = DynamixelSync()
        # Motor initialization
        self.motors = DynamixelSync()
        self.zero_offsets = motors.read(motors=[1,2,3,4], control_type=CONTROL_ADDRESS.PRESENT_POSITION)

        self.joy_subscriber = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        control_loop_period = 0.02  # 50 Hz
        self.control_timer = self.create_timer(
            control_loop_period, 
            self.command_robot
        )

        self.get_logger().info("CDPR Control Joystick Node has been started.")
    
    def joy_callback(self, msg: Joy):
        x = -msg.axes[3]
        y = -msg.axes[4]
        theta = msg.axes[2]-msg.axes[5]
        self.input = [x,y,theta]
        
    def command_robot(self):
        Kp = 0.01
        for i in range(len(self.pos)):
            self.pos[i] += Kp * self.input[i]
        
        cable_vectors = self.inverse_kinematics(self.pos[0:2], self.pos[2])
        desired_cable_lengths = [np.linalg.norm(l) for l in cable_vectors]

        # self.get_logger().info(
        #     f"Pos: [{self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f}], "
        #     f"Lengths: [{desired_cable_lengths[0]:.3f}, {desired_cable_lengths[1]:.3f}, "
        #     f"{desired_cable_lengths[2]:.3f}, {desired_cable_lengths[3]:.3f}]",
        #     throttle_duration_sec=0.1
        # )
        
        desired_velocity = None
        Kp_motor = 0.0001

        current_cable_lengths = self.get_current_cable_lengths()

        for i in range(4):
            length_errors = desired_cable_lengths[i] - current_cable_lengths[i]
            self.get_logger().info(f"Length error motor {i+1}: {length_errors:.4f}")
            desired_velocity[i] = Kp_motor * length_errors
            self.get_logger().info(f"Desired velocity motor {i+1}: {desired_velocity[i]:.4f}")

        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_ADDRESS.GOAL_VELOCITY,
            values=desired_velocity
        )

    def get_current_cable_lengths(self):
        motor_encoder_positions = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_ADDRESS.PRESENT_POSITION) - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = motor_encoder_rotations * self.get_spool_circumference()
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

        return l1, l2, l3, l4
        

def main(args=None):
    rclpy.init(args=args)
    node = CDPRControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
