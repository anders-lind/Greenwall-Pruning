#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
import scipy


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_control_joystick_FK')

        # Node state variables
        self.last_buttons_state = None

        # CDPR parameters
        self.spool_radius = 0.027
        self.spool_circumference = 2 * self.spool_radius * np.pi
        self.loop_period = 0.02  # 50 Hz
        self.desired_cable_tension = 25 # 2.69 mA
        
        self.CDPR_height = 0.9600
        self.CDPR_width = 0.9325

        self.q1 = np.array([-0.0425, -0.02])
        self.q2 = np.array([-0.0425, 0.02])
        self.q3 = np.array([0.0425, 0.02])
        self.q4 = np.array([0.0425, -0.02])

        self.B1 = np.array([0, 0])
        self.B2 = np.array([0, self.CDPR_height])
        self.B3 = np.array([self.CDPR_width, self.CDPR_height])
        self.B4 = np.array([self.CDPR_width, 0])

        # Controller parameters
        self.joystick_sensitivity = np.array([10, 10, 1]) # Force and torque sensitivity vector
        self.tension_reference = 20/2.69
        self.Kp_tension = 0.2
        self.Ki_tension = 0.5
        self.Kd_tension = 0
        self.tension_integral = np.zeros(4)
        self.tension_previous_error = np.zeros(4)

        # CDPR initial state
        self.initial_pose = np.array([self.CDPR_width/2, self.CDPR_height/2, 0.0]) # x, y, theta
        self.initial_cable_vectors = self.inverse_kinematics(self.initial_pose[0:2], self.initial_pose[2])
        self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]

        # CDPR state variables
        self.input = np.array([0.0, 0.0, 0.0]) # F_x, F_y, tau_z
        self.pose = self.initial_pose

        # Motor initialization
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=[1,2,3,4], directions=[-1,-1,-1,-1])
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
        if abs(msg.axes[6]) > 0:
            x = -msg.axes[6]
        if abs(msg.axes[7]) > 0:
            y = -msg.axes[7]
        theta = msg.axes[2]-msg.axes[5] 
        self.input = np.array([x,y,theta])
        
        
    def command_robot(self):
        # Force distribution algorithm
        u = self.joystick_sensitivity * self.input

        cable_lenghts = self.get_current_cable_lengths()
        self.pose = self.forward_kinematics(cable_lenghts, self.pose[0:2],self.pose[2])

        x = self.pose[0]
        y = self.pose[1]
        theta = self.pose[2]

        cable_vectors = self.inverse_kinematics([x,y], theta)
        S = self.compute_structure_matrix(cable_vectors, theta)

        #T = scipy.optimize.nnls(S, u)[0]
        T = scipy.optimize.lsq_linear(S, u, bounds=(10,100)).x
        print("u: ", u)
        print("T: ", T)

        #T = np.linalg.pinv(S) @ u 

        # Tension controller

        present_currents = np.array(self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_CURRENT))
        tension_error = np.ones(4)*self.tension_reference-present_currents

        self.tension_integral += tension_error * self.loop_period
        max_integral = 100000
        self.tension_integral = np.maximum(np.zeros(4),np.minimum(self.tension_integral,np.ones(4)*max_integral))

        tension_error_derivative = (tension_error-self.tension_previous_error)/self.loop_period
        self.tension_previous_error = tension_error

        T_tension = self.Kp_tension * tension_error + self.Ki_tension * self.tension_integral + self.Kd_tension * tension_error_derivative

        desired_currents = self.force_to_current(T)#+T_tension)
        # print('desired currents ', desired_currents)

        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_CURRENT,
            values=self.current_to_motor_input(desired_currents)
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
    
    def forward_kinematics(self, cable_lenghts, p0, theta0):
        q = [self.q1, self.q2, self.q3, self.q4]
        B = [self.B1, self.B2, self.B3, self.B4]

        def constraint_equations(input):
            x = input[0]    
            y = input[1]
            theta = input[2]
            l = cable_lenghts
            z_rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            q_world = np.zeros((4,2))
            d_sq = np.zeros(4)
            d = np.zeros(4)
            for i in range(4):
                q_world[i] = (z_rot @ q[i] + [x, y])
                d_sq[i] = (B[i][0] - q_world[i][0])**2 + (B[i][1] - q_world[i][1])**2
                d[i] = np.sqrt(d_sq[i])
            MSE = 0
            for i in range(4):
                MSE += (d[i]-l[i])**2
            MSE = MSE/4
            return MSE

        result = scipy.optimize.minimize(constraint_equations, [p0[0], p0[1], theta0])
        return result.x
    
    def compute_structure_matrix(self, cable_vectors, theta):
        S = np.zeros((3,4))
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        Rq = np.array([R @ self.q1, R @ self.q2, R @ self.q3, R @ self.q4]).T
        
        for i in range(4):
            l = cable_vectors[i]
            l_norm = np.linalg.norm(l)
            u = - l / l_norm
            S[0,i] = u[0]
            S[1,i] = u[1]
            S[2,i] = Rq[0,i]*u[1] - Rq[1,i]*u[0]

        return S
    
    def force_to_current(self, force_vector):
        torque_vector = force_vector*self.spool_radius
        current_vector = torque_vector * 2
        return current_vector
        #return [int(v/0.00269) for v in current_vector]  # 2.69 mA per 1N

    def current_to_motor_input(self, current_vector):
        return [int(v/0.00269) for v in current_vector]  # 2.69 mA per motor unit

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
            new_center = np.array([self.CDPR_width/2, self.CDPR_height/2, 0.0])
            self.pose = new_center
            self.initial_cable_vectors = self.inverse_kinematics(new_center[0:2], new_center[2])
            self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]
            self.get_logger().info('New home set')

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
