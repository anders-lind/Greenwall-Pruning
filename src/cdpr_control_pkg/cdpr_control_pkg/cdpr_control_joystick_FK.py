#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_control_joystick_FK')

        # Node state variables
        self.last_buttons_state = None

        # CDPR initial state
        self.initial_pos = np.array([0.4, 0.5, 0.0]) # x, y, theta
        self.initial_cable_vectors = self.inverse_kinematics(self.initial_pos[0:2], self.initial_pos[2])
        self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]

        # CDPR state variables
        self.input = np.array([0.0, 0.0, 0.0]) # F_x, F_y, tau_z
        self.pos = self.initial_pos

        # CDPR parameters
        self.spool_circumference = 0.021*np.pi
        self.loop_period = 0.02  # 50 Hz
        self.desired_cable_tension = 25 # 2.69 mA

        self.q1 = np.array([-0.0425, -0.02])
        self.q2 = np.array([-0.0425, 0.02])
        self.q3 = np.array([0.0425, 0.02])
        self.q4 = np.array([0.0425, -0.02])

        self.B1 = np.array([0, 0])
        self.B2 = np.array([0, 1.0175])
        self.B3 = np.array([0.975, 1.0175])
        self.B4 = np.array([0.975, 0])

        # Controller parameters
        self.joystick_sensitivity = np.array([1, 1, 0.1]) # Force and torque sensitivity vector

        # Motor initialization
        self.motors = DynamixelSync()
        self.zero_offsets = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        self.motors.write(motors=[1,2,3,4], values=0, control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])
        
        # Subscriber
        self.joy_subscriber = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        # Main control loop timer
        self.control_timer = self.create_timer(
            self.loop_period,
            self.command_robot
        )

        self.get_logger().info("CDPR Control Joystick Node with FK has been started.")
    
    def joy_callback(self, msg: Joy):
        self.handle_button_events(msg.buttons)

        x = -msg.axes[3]
        y = -msg.axes[4]
        theta = msg.axes[2]-msg.axes[5] 
        self.input = np.array([x,y,theta])
        
        
    def command_robot(self):
        # Position controller:
        u = self.joystick_sensitivity * self.input

        pos = self.pos[0:2]
        theta = self.pos[2]
        cable_vectors = self.inverse_kinematics(pos, theta)
        S = self.compute_structure_matrix(cable_vectors, theta)
        T = np.linalg.pinv(S) @ u
    
        desired_currents = self.force_to_current(T)

        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_CURRENT,
            values=desired_currents
        )

    def get_current_cable_lengths(self):
        positions_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        motor_encoder_positions = np.array(positions_list) - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = self.initial_cable_lengths + motor_encoder_rotations * self.get_spool_circumference()
        return current_cable_lengths
    
    def get_current_tension(self):
        currents_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_CURRENT)
        return np.array(currents_list)

    def get_spool_circumference(self):
        return self.spool_circumference

    def inverse_kinematics(self, position, orientation):
        p = np.array(position)
        theta = orientation
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

        l1 = p + R @ self.q1 - self.B1
        l2 = p + R @ self.q2 - self.B2
        l3 = p + R @ self.q3 - self.B3
        l4 = p + R @ self.q4 - self.B4

        return np.array([l1, l2, l3, l4])
    
    def compute_structure_matrix(self, cable_vectors, theta):
        S = np.zeros((3,4))
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        Rq = np.array([R @ self.q1, R @ self.q2, R @ self.q3, R @ self.q4]).T
        
        for i in range(4):
            l = cable_vectors[i]
            print("l: ",l)
            l_norm = np.linalg.norm(l)
            u = - l / l_norm
            S[0,i] = u[0]
            S[1,i] = u[1]
            S[2,i] = Rq[0,i]*u[1] - Rq[1,i]*u[0]

        return S
    
    def force_to_current(self, force_vector):
        return [int(v*2.69) for v in force_vector]  # 2.69 mA per 1N

    def handle_button_events(self, current_buttons):
        # Initialize button state
        if self.last_buttons_state is None:
            self.last_buttons_state = current_buttons
            return

        # Button A: Start timer on rising edge
        if current_buttons[0] == 1 and self.last_buttons_state[0] == 0:
            if self.control_timer.is_canceled():
                self.control_timer.reset()
                self.motors.enable_torque(motors=[1,2,3,4])
                self.get_logger().info('Control loop STARTED.')

        # Button B: Stop timer on rising edge
        if current_buttons[1] == 1 and self.last_buttons_state[1] == 0:
            if not self.control_timer.is_canceled():
                self.control_timer.cancel()
                self.motors.disable_torque(motors=[1,2,3,4])
                self.get_logger().info('Control loop STOPPED.')

        # Button X
        if current_buttons[2] == 1 and self.last_buttons_state[2] == 0:
            pass

        # Button Y
        if current_buttons[3] == 1 and self.last_buttons_state[3] == 0:
            pass

        # Button LB
        if current_buttons[4] == 1 and self.last_buttons_state[4] == 0:
            pass

        # Button RB
        if current_buttons[5] == 1 and self.last_buttons_state[5] == 0:
            pass
            
        # Update button state
        self.last_buttons_state = current_buttons
        

def main(args=None):
    rclpy.init(args=args)
    node = CDPRControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
