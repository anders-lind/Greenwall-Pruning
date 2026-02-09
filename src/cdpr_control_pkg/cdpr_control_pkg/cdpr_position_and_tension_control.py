#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_position_and_tension_control')

        # Node state variables
        self.last_buttons_state = None

        # CDPR initial state
        self.initial_pos = np.array([0.4, 0.5, 0.0]) # x, y, theta
        self.initial_cable_vectors = self.inverse_kinematics(self.initial_pos[0:2], self.initial_pos[2])
        self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]

        # CDPR state variables
        self.input = np.array([0.0, 0.0, 0.0]) # x, y, theta
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

        # Position controller parameters
        self.pos_Kp = 1000
        self.pos_Ki = 1
        self.pos_Kd = 10
        self.pos_integral_clamp = 50.0
        self.pos_previous_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.pos_integral_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.joystick_sensitivity = 0.001

        # Tension controller parameters
        self.tension_Kp = 1.0
        self.tension_Ki = 0.5
        self.tension_Kd = 0.0
        self.tension_integral_clamp = 50.0
        self.tension_previous_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.tension_integral_error = np.array([0.0, 0.0, 0.0, 0.0])

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

        self.get_logger().info("CDPR Control Joystick Node has been started.")
    
    def joy_callback(self, msg: Joy):
        self.handle_button_events(msg.buttons)

        x = -msg.axes[3]
        y = -msg.axes[4]
        theta = msg.axes[2]-msg.axes[5] 
        self.input = np.array([x,y,theta])
        
        
    def command_robot(self):
        # Position controller:
        self.pos += self.joystick_sensitivity * self.input

        cable_vectors = self.inverse_kinematics(self.pos[0:2], self.pos[2])
        desired_cable_lengths = np.array([np.linalg.norm(l) for l in cable_vectors])
        current_cable_lengths = self.get_current_cable_lengths()
        # Propertional 
        pos_error = desired_cable_lengths - current_cable_lengths
        # Integral with antiwindup
        self.pos_integral_error += pos_error * self.loop_period
        self.pos_integral_error = np.clip(
            self.pos_integral_error, -self.pos_integral_clamp, self.pos_integral_clamp
        )
        # Derivative
        pos_derivative_error = (pos_error - self.pos_previous_error) / self.loop_period
        self.pos_previous_error = pos_error
        # Position PID output:
        pos_control_current = self.pos_Kp*pos_error + self.pos_Ki*self.pos_integral_error + self.pos_Kd*pos_derivative_error
 

        # Tension controller 
        current_cable_tensions = self.get_current_tension()
        # Propertional 
        tension_error = self.desired_cable_tension - current_cable_tensions
        # Integral with antiwindup
        self.tension_integral_error += tension_error * self.loop_period
        self.tension_integral_error = np.clip(
            self.tension_integral_error, -self.tension_integral_clamp, self.tension_integral_clamp
        )
        # Derivative
        tension_derivative_error = (tension_error - self.tension_previous_error) / self.loop_period
        self.tension_previous_error = tension_error
        # Tension PID output:
        tension_control_current = self.tension_Kp*tension_error + self.tension_Ki*self.tension_integral_error + self.tension_Kd*tension_derivative_error

        desired_current = tension_control_current + pos_control_current
        desired_current_int_list = [int(v) for v in desired_current]

        # self.get_logger().info(
        #     f"Pos: [{self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f}], "
        #     f"Lengths: [{desired_cable_lengths[0]:.3f}, {desired_cable_lengths[1]:.3f}, "
        #     f"{desired_cable_lengths[2]:.3f}, {desired_cable_lengths[3]:.3f}]",
        #     throttle_duration_sec=0.1
        # )

        # self.get_logger().info(
        #     f"Current cable lengths: ["
        #     f"{current_cable_lengths[0]:.4f}, {current_cable_lengths[1]:.4f}, "
        #     f"{current_cable_lengths[2]:.4f}, {current_cable_lengths[3]:.4f}]",
        #     #throttle_duration_sec=0.2
        # )

        # self.get_logger().info(
        #     f"Length errors: ["
        #     f"{length_errors[0]:.4f}, {length_errors[1]:.4f}, "
        #     f"{length_errors[2]:.4f}, {length_errors[3]:.4f}]",
        #     #throttle_duration_sec=0.2
        # )
        # self.get_logger().info(
        #     f"Desired currents: ["
        #     f"{desired_current[0]:.4f}, {desired_current[1]:.4f}, "
        #     f"{desired_current[2]:.4f}, {desired_current[3]:.4f}]",
        #     throttle_duration_sec=0.2
        # )

        # Command motors
        # self.get_logger().info(
        #     f"Desired currents int list: ["
        #     f"{desired_current_int_list[0]}, {desired_current_int_list[1]}, "
        #     f"{desired_current_int_list[2]}, {desired_current_int_list[3]}]",
        #     #throttle_duration_sec=0.2
        # )

        self.get_logger().info(
            f"Current cable tensions: ["
            f"{current_cable_tensions[0]:.3f}, {current_cable_tensions[1]:.3f}, "
            f"{current_cable_tensions[2]:.3f}, {current_cable_tensions[3]:.3f}]",
            throttle_duration_sec=0.1
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
