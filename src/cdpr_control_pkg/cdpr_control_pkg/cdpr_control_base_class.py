#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
from cdpr_control_pkg.DynamixelSyncDummy import DynamixelSyncDummy
from plantwall_custom_interfaces.msg import CdprPose
import scipy
import matplotlib.pyplot as plt
import time


class CDPRBaseControlNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        print("CDPRBaseControlNode constructor")

        # Node state variables
        self.homing_active = False
        self.homing_loop_counter = 0
        self.control_loop_counter = 0
        self.last_buttons_state = None
        self.control_loop_period = 0.02 # 50 Hz

        # Homing parameters
        self.homing_speed = int(30) # Motor units [0.229 RPM]
        self.home_tension = 5 # Newton
        self.force_tension_time = 0.1 # S
        self.homing_loop_period = self.control_loop_period

        # Tension safety check parameters
        self.tension_threshold = 40.0 # Newton (40)
        self.tension_thresholds = [self.tension_threshold, self.tension_threshold, self.tension_threshold, self.tension_threshold] # Newton

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
        self.movement_speed = 0.01 # m/s
        self.rotation_speed = 0.1 # rad/s
        self.stopping_radius_pos = 1e-3 # m
        self.slowdown_radius_pos = 0.01 # m
        self.stopping_rot = 1e-3 # rad

        # Motor encoder filter variables
        self.filter_alpha = 0.7 # Smoothing factor for encoders
        self.motor_feedback_period = 0.01 # 100 Hz
        
        # State variables
        self.initial_pose = np.array([self.CDPR_width/2, 0.52-0.03, 0.0]) # (x, y, theta)
        self.input = np.array([0.0, 0.0, 0.0]) # Joystick input (u)
        self.pose = self.initial_pose.copy()
        self.target_pose = self.initial_pose.copy() # (x, y, theta)
        self.previous_pose = self.initial_pose.copy() # (x, y, theta)

        # Motor initialization
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=[1,2,3,4], directions=[-1,-1,1,1])
        self.motors.disable_torque(motors=[1,2,3,4])
        # self.motors.write(motors=[1,2,3,4], values=128, control_type=CONTROL_TABLE.VELOCITY_LIMIT)
        self.motors.write(motors=[1,2,3,4], values=1, control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])
        
        # Initialize Logic
        self.initialize_state()

        # ROS Infrastructure
        self.current_pose_publisher = self.create_publisher(CdprPose, '/cdpr/current_pose', 10)

        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.goto_pose_subscriber = self.create_subscription(CdprPose, '/cdpr/goto_pose', self.goto_pose_callback, 10)
        
        self.control_timer = self.create_timer(self.control_loop_period, self.command_robot)
        self.control_timer.cancel()
        self.homing_timer = self.create_timer(self.homing_loop_period, self.homing)
        self.motor_feedback_timer = self.create_timer(self.motor_feedback_period, self.motor_feedback)

        self.get_logger().info(f"{node_name} Node has been started.")
    
    def __del__(self):
        print("CDPRBaseControlNode destructor")

    def initialize_state(self):
        # Assume robot starts at center or user manually centered it
        self.pose = self.initial_pose.copy()
        self.previous_pose = self.initial_pose.copy()
        
        # Calculate theoretical lengths for the center
        self.initial_cable_vectors = self.inverse_kinematics(self.initial_pose[0:2], self.initial_pose[2])
        self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]
        
        # Initialize filter with theoretical values
        self.cable_lengths = np.array(self.initial_cable_lengths)
        self.previous_cable_lengths = np.array(self.initial_cable_lengths)
        
        # Zero the encoders at this position
        self.zero_offsets = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        while np.any(np.array(self.zero_offsets) == None):
            print("Could not read zero offsets")
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

    def goto_pose_callback(self, msg: Joy):
        target_pos = msg.position
        target_ori = msg.orientation
        self.target_pose = np.array([target_pos[0], target_pos[1], target_ori])
        
    def command_robot(self):
        print("Base Command_robot() call")

    def homing(self):
        if self.homing_active:
            self.homing_loop_counter += 1

            # Begin tightening tensions
            present_current_list = self.get_present_current()
            force_list = self.motor_current_units_to_force(present_current_list)
            tigthen_array = [0,0,0,0]
            not_tightened = False
            for i in range(4):
                # Only check tension after 0.1s
                if (self.homing_loop_counter * self.homing_loop_period <= self.force_tension_time):
                    tigthen_array = [self.homing_speed, self.homing_speed, self.homing_speed, self.homing_speed]
                    not_tightened = True
                    print(f"Slow start tensioning: {force_list[0]:.2f}, {force_list[1]:.2f}, {force_list[2]:.2f}, {force_list[3]:.2f}, ")
                    break
                elif (force_list[i] < self.home_tension):
                    tigthen_array[i] = self.homing_speed
                    not_tightened = True
                else:
                    tigthen_array[i] = 0
                print(f"force {i+1} is {force_list[i]:.2f}N, < desired {self.home_tension}N,  {"tightening..." if tigthen_array[i] != 0 else ""}")
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
                self.initialize_state()
                self.homing_loop_counter = 0
                self.get_logger().info('Homing complete.')

            return

    def tension_safety_check(self):
        current_forces = self.motor_current_units_to_force(self.get_present_current())
        if ((np.any(current_forces > self.tension_threshold)) and (self.control_loop_counter > 10)):
            self.motors.disable_torque(motors=[1,2,3,4])
            self.control_timer.cancel()
            self.get_logger().warn(f"Tension threshold ({self.tension_threshold} N) exceeded! Current forces: {current_forces}")
            return False
        
        return True
        
    def tension_safety_check_advanced(self):
        return
        #TODO: Make this
        current_forces = self.motor_current_units_to_force(self.get_present_current())
        delta_thresholds = self.tension_thresholds - current_forces

        tension_reached = False
        for i in range(len(delta_thresholds)):
            if delta_thresholds[i] <= 0:
                tension_reached = True

        if ((np.any(delta_thresholds)) and (self.control_loop_counter > 10)):
            self.motors.disable_torque(motors=[1,2,3,4])
            self.control_timer.cancel()
            self.get_logger().warn(f"Tension threshold ({self.tension_threshold} N) exceeded! Current forces: {current_forces}")
            return

    def motor_feedback(self):   
        raw_lengths = self.get_current_cable_lengths()
        
        if np.any(raw_lengths == None):
            return
        
        # Apply Exponential Moving Average (first order low pass)
        self.cable_lengths = (
            self.filter_alpha * raw_lengths + 
            (1.0 - self.filter_alpha) * self.cable_lengths
        )

    def get_current_cable_lengths(self):
        positions_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        positions_list = np.array(positions_list)

        if np.any(positions_list == None):
            return np.array([None, None, None, None])
        
        motor_encoder_positions = positions_list - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = self.initial_cable_lengths - motor_encoder_rotations * self.get_spool_circumference()
        return current_cable_lengths

    def get_present_current(self):
        currents_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_CURRENT)
        currents_list = np.array(currents_list)

        if np.any(currents_list == None):
            self.motors.disable_torque(motors=[1,2,3,4])
            print("ERROR: Could not read currents!")

        return currents_list

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

    def current_to_motor_input(self, current_vector):
        return [int(max(-2047, min(2047, v/0.00269))) for v in current_vector]
    
    def motor_current_units_to_force(self, motor_units_vector):
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
            l = cable_lenghts.copy()
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
            tol=1e-9,
            options={'ftol': 1e-9}
        )
        return result.x
    
    def compute_structure_matrix(self, cable_vectors, theta):
        S = np.zeros((3,4))
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta), np.cos(theta)]])
        Rq = np.array([R @ self.q1, R @ self.q2, R @ self.q3, R @ self.q4]).T
        for i in range(4):
            l = cable_vectors[i]
            l_norm = np.linalg.norm(l)
            u = - l / l_norm
            S[0,i] = u[0]
            S[1,i] = u[1]
            S[2,i] = (Rq[0,i]*u[1] - Rq[1,i]*u[0])
        torques = S[2, :]
        signs = np.sign(torques)
        # print(f"Torque Signs: {signs}")  # Uncomment to debug
        return S

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
            # Stop homing procedure
            if self.homing_active:
                self.homing_active = False
                self.motors.write(
                    motors=[1,2,3,4],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[0,0,0,0]
                )
                self.motors.disable_torque(motors=[1,2,3,4])
                self.homing_loop_counter = 0
                self.get_logger().info('Homing loop STOPPED.')

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
            pass

        self.last_buttons_state = current_buttons


    def init_visualisation(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        margin = 0.2
        self.ax.set_xlim(-margin, self.CDPR_width + margin)
        self.ax.set_ylim(-margin, self.CDPR_height + margin)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title("CDPR Real-Time State")
        self.pose_text = self.ax.text(0.05, 0.95, "", transform=self.ax.transAxes, 
                                      fontsize=10, verticalalignment='top', 
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        frame_x = [self.B1[0], self.B2[0], self.B3[0], self.B4[0], self.B1[0]]
        frame_y = [self.B1[1], self.B2[1], self.B3[1], self.B4[1], self.B1[1]]
        self.ax.plot(frame_x, frame_y, 'k--', linewidth=2, label='Frame')
        self.ee_line, = self.ax.plot([], [], 'b-', linewidth=2, label='End Effector')
        self.center_dot, = self.ax.plot([], [], 'ro')
        self.cable_lines = []
        colors = ['g', 'g', 'g', 'g']
        for i in range(4):
            line, = self.ax.plot([], [], color=colors[i], linewidth=1)
            self.cable_lines.append(line)
        plt.legend(loc='upper right')
    
    def visualisation(self):
        x, y, theta = self.pose
        theta_deg = np.degrees(theta)
        
        # 2. Update Text
        status_str = (
            f"X: {x:.4f} m\n"
            f"Y: {y:.4f} m\n"
            f"Θ: {theta:.4f} rad ({theta_deg:.1f}°)\n"
            f"T_min: {1.0} N" # Or whatever variable you use
        )
        self.pose_text.set_text(status_str)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        qs = [self.q1, self.q2, self.q3, self.q4]
        corners_world = []
        for q_local in qs:
            q_rotated = R @ q_local
            corner_world = np.array([x, y]) + q_rotated
            corners_world.append(corner_world)
        ee_x = [c[0] for c in corners_world] + [corners_world[0][0]]
        ee_y = [c[1] for c in corners_world] + [corners_world[0][1]]
        self.ee_line.set_data(ee_x, ee_y)
        self.center_dot.set_data([x], [y])
        Bs = [self.B1, self.B2, self.B3, self.B4]
        for i in range(4):
            self.cable_lines[i].set_data([Bs[i][0], corners_world[i][0]], [Bs[i][1], corners_world[i][1]])

        if hasattr(self, 'corner_labels'):
            for t in self.corner_labels: t.remove()
        self.corner_labels = []

        # ADD new text labels "1", "2", "3", "4" at the cable attachment points
        for i in range(4):
            # corner_world[i] is the [x,y] of the attachment point on the body
            lbl = self.ax.text(corners_world[i][0], corners_world[i][1], f"{i+1}", 
                               color='red', fontsize=12, fontweight='bold')
            self.corner_labels.append(lbl)
            
            # Also label the Anchors (B1..B4)
            lbl_b = self.ax.text(Bs[i][0], Bs[i][1], f"B{i+1}", 
                                 color='blue', fontsize=10)
            self.corner_labels.append(lbl_b)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
