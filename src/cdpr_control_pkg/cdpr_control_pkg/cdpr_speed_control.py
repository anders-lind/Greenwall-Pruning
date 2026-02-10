#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
import scipy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_speed_control')

        # Node state variables
        self.last_buttons_state = None
        self.homing_active = False
        self.homing_speed = int(30) # Motor units [0.229 RPM]

        # CDPR parameters
        self.spool_radius = 0.0115 - 0.001 # spool outer radius minus cable radius
        self.spool_circumference = 2 * self.spool_radius * np.pi
        
        self.CDPR_height = 0.944
        self.CDPR_width = 0.908
        self.end_effector_height = 0.03916
        self.end_effector_width = 0.09322

        self.B1 = np.array([0, self.CDPR_height])
        self.B2 = np.array([self.CDPR_width, self.CDPR_height])
        self.B3 = np.array([self.CDPR_width, 0])
        self.B4 = np.array([0, 0])

        self.q1 = np.array([-self.end_effector_width/2, self.end_effector_height/2])  
        self.q2 = np.array([self.end_effector_width/2, self.end_effector_height/2])
        self.q3 = np.array([self.end_effector_width/2, -self.end_effector_height/2])
        self.q4 = np.array([-self.end_effector_width/2, -self.end_effector_height/2])

        # CONTROLLER GAINS
        self.joystick_sensitivity = np.array([1, 1, 0]) # Reference point offset (x, y, theta) [m, m, rad]
        self.home_tension = 5 # Newton

        self.control_loop_period = 0.005  # 200 Hz

        # Motor encoder filter variables
        self.motor_feedback_period = 0.005 # 200 Hz
        self.filter_alpha = 0.7 # Smoothing factor for encoders
        
        # State variables
        self.input = np.array([0.0, 0.0, 0.0]) # Joystick input (u)
        self.initial_pose = np.array([self.CDPR_width/2, 0.52-0.03, 0.0]) # (x, y, theta)
        self.pose = self.initial_pose

        # Motor initialization
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=[1,2,3,4], directions=[-1,-1,1,1])
        self.motors.write(motors=[1,2,3,4], values=1, control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])
        
        # Initialize Logic
        self.initialize_state()

        # ROS Infrastructure
        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.control_timer = self.create_timer(self.control_loop_period, self.command_robot)
        self.control_timer.cancel()
        self.background_timer = self.create_timer(self.control_loop_period, self.background_tasks)
        self.motor_feedback_timer = self.create_timer(self.motor_feedback_period, self.motor_feedback)

        self.get_logger().info("CDPR Control Joystick Node has been started.")

    def initialize_state(self):
        # Assume robot starts at center or user manually centered it
        self.pose = self.initial_pose
        
        # Calculate theoretical lengths for the center
        self.initial_cable_vectors = self.inverse_kinematics(self.initial_pose[0:2], self.initial_pose[2])
        self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]
        
        # Initialize filter with theoretical values
        self.cable_lengths = np.array(self.initial_cable_lengths)
        
        # Zero the encoders at this position
        self.zero_offsets = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
    
    def joy_callback(self, msg: Joy):
        self.handle_button_events(msg.buttons)
        # Xbox controller mapping
        x = -msg.axes[3]
        y = msg.axes[4]
        # D-pad overrides
        if abs(msg.axes[6]) > 0: x = -msg.axes[6]
        if abs(msg.axes[7]) > 0: y = msg.axes[7]
        theta = (msg.axes[5] - msg.axes[2]) / 2 
        self.input = np.array([x,y,theta])
        
    def command_robot(self):
        delta_ref_point = self.joystick_sensitivity[0:2] * self.input[0:2]
        # delta_ref_angle = self.joystick_sensitivity[2] * self.input[2]

        direction_norm = np.linalg.norm(delta_ref_point)
        if direction_norm < 1e-6:
            ref_direction = np.array([0.0, 0.0])
        else:
            ref_direction = delta_ref_point/direction_norm
        
        raw_pose = self.forward_kinematics(self.cable_lengths, self.pose[0:2], self.pose[2])

        discretization_size = 0.001 # meters
        movement_speed = 0.01 # m/s
        movement_time = discretization_size / movement_speed # seconds

        desired_pos = raw_pose[0:2] + ref_direction * discretization_size
        desired_cable_vectors = self.inverse_kinematics(desired_pos, 0)
        desired_cable_lengths = [np.linalg.norm(l) for l in desired_cable_vectors]

        cable_errors = desired_cable_lengths - self.cable_lengths # meters

        desired_velocity_linear = (cable_errors / movement_time)
        desired_velocity_rpm = (desired_velocity_linear / self.spool_circumference) * 60 # convert to RPM

        desired_velocity_motor_units = desired_velocity_rpm / 0.229 # convert to motor units (1 unit = 0.229 RPM)

        velocity_int_list = [int(v) for v in desired_velocity_motor_units]

        self.get_logger().info(
            f": raw_pose["
            f"{raw_pose[0]:.4f}, {raw_pose[1]:.4f}, "
            f"{raw_pose[2]:.4f}]",
            throttle_duration_sec=0.1
        )

        self.get_logger().info(
            f"ref_direction: ["
            f"{ref_direction[0]:.4f}, {ref_direction[1]:.4f}]",
            throttle_duration_sec=0.1
        )

        self.get_logger().info(
            f"desired linear: ["
            f"{desired_velocity_linear[0]:.4f}, {desired_velocity_linear[1]:.4f}, "
            f"{desired_velocity_linear[2]:.4f}, {desired_velocity_linear[3]:.4f}]",
            throttle_duration_sec=0.1
        )

        self.motors.enable_torque(motors=[1,2,3,4])
        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_VELOCITY,
            values=velocity_int_list
        )

    def background_tasks(self):
        # Homing procedure
        if self.homing_active:
            present_current_list = self.get_present_current()
            force_list = self.motor_units_to_force(present_current_list)
            tigthen_array = [0,0,0,0]
            not_tightened = False
            for i in range(4):
                print(f"force {i+1} is {force_list[i]:.2f}N, < desired {self.home_tension}N, tightening...")
                if force_list[i] < self.home_tension:
                    tigthen_array[i] = self.homing_speed
                    not_tightened = True
                else:
                    tigthen_array[i] = 0
            self.motors.enable_torque(motors=[1,2,3,4])
            self.motors.write(
                motors=[1,2,3,4],
                control_type=CONTROL_TABLE.GOAL_VELOCITY,
                values=tigthen_array
            )
            # When all cables reach the desired tension, stop tightening and reinitialize state
            if not not_tightened:
                self.homing_active = False
                self.motors.write(
                    motors=[1,2,3,4],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[0,0,0,0]
                )
                # self.motors.disable_torque(motors=[1,2,3,4])
                self.initialize_state()
                self.get_logger().info('Homing complete.')
            return

    def motor_feedback(self):   
        raw_lengths = self.get_current_cable_lengths()
        # Apply Exponential Moving Average (first order low pass)
        self.cable_lengths = (
            self.filter_alpha * raw_lengths + 
            (1.0 - self.filter_alpha) * self.cable_lengths
        )

    def get_current_cable_lengths(self):
        positions_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        motor_encoder_positions = np.array(positions_list) - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = self.initial_cable_lengths + motor_encoder_rotations * self.get_spool_circumference()
        return current_cable_lengths

    def get_present_current(self):
        currents_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_CURRENT)
        return np.array(currents_list)

    def get_spool_circumference(self):
        return self.spool_circumference

    def force_to_current(self, force_vector):
        torque_vector = force_vector * self.spool_radius
        current_vector = torque_vector * 0.625
        return current_vector
    
    def force_to_motor_units(self, force_vector):
        current_vector = self.force_to_current(force_vector)
        motor_units_mA = current_vector * 1000
        motor_units_vector = motor_units_mA / 2.69 # Convert to motor units (1 unit = 2.69 mA)
        return motor_units_vector
    
    def current_to_force(self, current_vector):
        torque_vector = current_vector / 0.625
        force_vector = torque_vector / self.spool_radius
        return force_vector
    
    def motor_units_to_force(self, motor_units_vector):
        current_vector_mA = motor_units_vector * 2.69
        current_vector = current_vector_mA / 1000 
        force_vector = self.current_to_force(current_vector)
        return force_vector
    
    def inverse_kinematics(self, position, orientation):
        p = np.array(position)
        theta = orientation
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
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
                z_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
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
            
            margin = 0.05
            limit_theta = 0.3 # ~17 degrees
            bnds = (
                (margin, self.CDPR_width - margin), 
                (margin, self.CDPR_height - margin), 
                (-limit_theta, limit_theta)
            )
            # Use 'SLSQP' or 'L-BFGS-B' method which supports bounds
            result = scipy.optimize.minimize(
                constraint_equations, 
                [p0[0], p0[1], theta0], 
                method='SLSQP', 
                bounds=bnds, 
                tol=1e-4
            )
            return result.x

    def handle_button_events(self, current_buttons):
        if self.last_buttons_state is None:
            self.last_buttons_state = current_buttons
            return
        
        # Button A (rising edge)
        if current_buttons[0] == 1 and self.last_buttons_state[0] == 0:
            # Activate CDPR
            if self.control_timer.is_canceled():
                self.control_timer.reset()
                self.motors.enable_torque(motors=[1,2,3,4])
                self.get_logger().info('Control loop STARTED.')

        # Button B (rising edge)
        if current_buttons[1] == 1 and self.last_buttons_state[1] == 0:
            # Deactivate CDPR
            if not self.control_timer.is_canceled():
                self.control_timer.cancel()
                self.motors.disable_torque(motors=[1,2,3,4])
                self.get_logger().info('Control loop STOPPED.')

        # Button X (rising edge)
        if current_buttons[2] == 1 and self.last_buttons_state[2] == 0:
            # Release cables for homing (only when cdpr control loop is inactive)
            if self.control_timer.is_canceled():
                release_speed = -self.homing_speed
                self.motors.enable_torque(motors=[1,2,3,4])
                self.motors.write(
                    motors=[1,2,3,4],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[release_speed,release_speed,release_speed,release_speed]
                )

        # Button X (falling edge)
        if current_buttons[2] == 0 and self.last_buttons_state[2] == 1:
            # Stop cable release
            self.motors.write(
                motors=[1,2,3,4],
                control_type=CONTROL_TABLE.GOAL_VELOCITY,
                values=[0,0,0,0]
            )

        # Button Y (rising edge)
        if current_buttons[3] == 1 and self.last_buttons_state[3] == 0:
            # Start homing procedure (only when cdpr control loop is inactive)
            if self.control_timer.is_canceled():
                self.homing_active = True
                self.get_logger().info('Homing started.')

        # Button LB (rising edge)
        if current_buttons[4] == 1 and self.last_buttons_state[4] == 0:
            pass

        # Button RB (rising edge)
        if current_buttons[5] == 1 and self.last_buttons_state[5] == 0:
            # Cancel homing procedure
            if self.homing_active:
                self.homing_active = False
                self.motors.write(
                    motors=[1,2,3,4],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[0,0,0,0]
                )
                self.motors.disable_torque(motors=[1,2,3,4])
                self.get_logger().info('Homing cancelled.')

        self.last_buttons_state = current_buttons

def main(args=None):
    rclpy.init(args=args)
    node = CDPRControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
