#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
from cdpr_control_pkg.cdpr_control_base_class import CDPRBaseControlNode


class CDPRControlNode(CDPRBaseControlNode):
    def __init__(self):
        super().__init__('cdpr_position_and_tension_control')


        self.desired_cable_tension = 25 # 2.69 mA

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

    def command_robot(self):
        # Position controller:
        self.pose += self.joystick_sensitivity * self.input

        cable_vectors = self.inverse_kinematics(self.pose[0:2], self.pose[2])
        desired_cable_lengths = np.array([np.linalg.norm(l) for l in cable_vectors])
        current_cable_lengths = self.get_current_cable_lengths()
        # Propertional 
        pos_error = desired_cable_lengths - current_cable_lengths
        # Integral with antiwindup
        self.pos_integral_error += pos_error * self.control_loop_period
        self.pos_integral_error = np.clip(
            self.pos_integral_error, -self.pos_integral_clamp, self.pos_integral_clamp
        )
        # Derivative
        pos_derivative_error = (pos_error - self.pos_previous_error) / self.control_loop_period
        self.pos_previous_error = pos_error
        # Position PID output:
        pos_control_current = self.pos_Kp*pos_error + self.pos_Ki*self.pos_integral_error + self.pos_Kd*pos_derivative_error
 

        # Tension controller 
        current_cable_tensions = self.get_present_current()
        # Propertional 
        tension_error = self.desired_cable_tension - current_cable_tensions
        # Integral with antiwindup
        self.tension_integral_error += tension_error * self.control_loop_period
        self.tension_integral_error = np.clip(
            self.tension_integral_error, -self.tension_integral_clamp, self.tension_integral_clamp
        )
        # Derivative
        tension_derivative_error = (tension_error - self.tension_previous_error) / self.control_loop_period
        self.tension_previous_error = tension_error
        # Tension PID output:
        tension_control_current = self.tension_Kp*tension_error + self.tension_Ki*self.tension_integral_error + self.tension_Kd*tension_derivative_error

        desired_current = tension_control_current + pos_control_current
        desired_current_int_list = [int(v) for v in desired_current]

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
        

def main(args=None):
    rclpy.init(args=args)
    node = CDPRControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
