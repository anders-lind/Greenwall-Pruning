#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
from cdpr_control_pkg.DynamixelSyncDummy import DynamixelSyncDummy
from plantwall_custom_interfaces.msg import CdprPose, MotorCmd, MotorState
import scipy
import matplotlib.pyplot as plt
import time
import sys


class CDPRBaseControlNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        print("CDPRBaseControlNode constructor")

        self.log_period = 0.5

        # Node state variables
        self.homing_active = False
        self.last_buttons_state = None
        self.control_loop_period = 0.05 # 20 Hz

        self.motor_IDs = [1,2,3,4]

        self.fast_callback_is_initialized = False
        self.slow_callback_is_initialized = False
        self.is_initialized = False

        self.auto_tighten_active = False
        self.cable_lengths_pre_tightened = None

        self.cable_errors = np.array([0.0, 0.0, 0.0, 0.0])

        # Motor state variables
        self.motor_id = None
        self.operating_mode = None
        self.homing_offset = None
        self.current_limit = None
        self.velocity_limit = None
        self.torque_enable = None
        self.goal_current = None
        self.goal_velocity = None
        self.profile_velocity = None
        self.goal_position = None
        self.moving = None
        self.present_load = None
        self.present_current = None
        self.present_velocity = None
        self.present_position = None

        # Homing parameters
        self.homing_speed = int(10) # Motor units [0.229 RPM]
        self.home_tension = 15 # Newton
        self.force_tension_time = 0.0 # S Old=0.1
        self.homing_loop_period = self.control_loop_period
        self.auto_tighten_loop_period = self.control_loop_period
        self.auto_tighten_activate_loop_period = 30.0 # S

        self.auto_tighten_tension = self.home_tension
        
        # Tension safety check parameters
        self.max_tension_threshold = 60.0 # Newton (without spring: 40)

        # CDPR parameters
        self.spool_radius = 0.0115 - 0.001 # spool outer radius minus cable radius
        self.spool_circumference = 2 * self.spool_radius * np.pi
        self.spool_pitch = 0.003
        self.cable_length_per_rot = np.sqrt(self.spool_circumference**2 + self.spool_pitch**2) # helix length
        self.effective_circumferences = np.array([
            self.cable_length_per_rot - self.spool_pitch, # Motor 1 (Top Left)
            self.cable_length_per_rot - self.spool_pitch, # Motor 2 (Top Right)
            self.cable_length_per_rot + self.spool_pitch, # Motor 3 (Bottom Right)
            self.cable_length_per_rot + self.spool_pitch  # Motor 4 (Bottom Left)
        ])

        self.pulley_radius = 0.01
        
        # self.CDPR_height_old = 0.944
        # self.CDPR_width_old = 0.908

        self.CDPR_height = 1.0
        self.CDPR_width = 1.0
        self.end_effector_height = 0.160
        self.end_effector_width = 0.107
        # Old
        # self.end_effector_height = 0.03916
        # self.end_effector_width = 0.09322

        # Old B point measured as the average release point of the pulley
        # self.B1 = np.array([0, self.CDPR_height])
        # self.B2 = np.array([self.CDPR_width, self.CDPR_height])
        # self.B3 = np.array([self.CDPR_width, 0])
        # self.B4 = np.array([0, 0])


        anchor_point_offset_from_frame = 0.032

        # self.B1_old = np.array([0, self.CDPR_height])
        # self.B2_old = np.array([self.CDPR_width, self.CDPR_height])
        # self.B3_old = np.array([self.CDPR_width, 0])
        # self.B4_old = np.array([0, 0])

        self.C1 = np.array([anchor_point_offset_from_frame, self.CDPR_height - anchor_point_offset_from_frame])
        self.C2 = np.array([self.CDPR_width - anchor_point_offset_from_frame, self.CDPR_height - anchor_point_offset_from_frame])
        self.C3 = np.array([self.CDPR_width - anchor_point_offset_from_frame, anchor_point_offset_from_frame])
        self.C4 = np.array([anchor_point_offset_from_frame, anchor_point_offset_from_frame])

        self.B1 = np.array([0.045, self.CDPR_height - 0.03])
        self.B2 = np.array([self.CDPR_width - 0.045, self.CDPR_height - 0.03])
        self.B3 = np.array([self.CDPR_width - 0.045, 0.03])
        self.B4 = np.array([0.045, 0.03])

        # New EE with metal springs 10 N
        # self.q1 = np.array([-0.125,  0.04]) 
        # self.q2 = np.array([ 0.125,  0.04])
        # self.q3 = np.array([ 0.125, -0.14])
        # self.q4 = np.array([-0.125, -0.14])

        # New EE with metal springs 15 N
        # self.q1 = np.array([-0.15,  0.06]) 
        # self.q2 = np.array([ 0.15,  0.06])
        # self.q3 = np.array([ 0.15, -0.15])
        # self.q4 = np.array([-0.15, -0.15])

        #New EE measured to spring mounting point
        self.q1 = np.array([-0.075,  -0.025]) 
        self.q2 = np.array([ 0.075,  -0.025])
        self.q3 = np.array([ 0.075, -0.08])
        self.q4 = np.array([-0.075, -0.08])

        # Old EE
        # self.q1 = np.array([-self.end_effector_width/2, self.end_effector_height/2])
        # self.q2 = np.array([self.end_effector_width/2, self.end_effector_height/2])
        # self.q3 = np.array([self.end_effector_width/2, -self.end_effector_height/2])
        # self.q4 = np.array([-self.end_effector_width/2, -self.end_effector_height/2])

        # CONTROLLER GAINS
        self.movement_speed = 0.02 # m/s
        self.rotation_speed = 0.1 # rad/s
        self.stopping_radius_pos = 1e-3 # m
        self.slowdown_radius_pos = 0.01 # m
        self.stopping_rot = 1e-3 # rad

        # Motor encoder filter variables
        self.filter_alpha = 1.0 # Smoothing factor for encoders
        self.motor_feedback_period = 0.01 # 100 Hz
        
        # Old initial_pose
        # self.initial_pose = np.array([self.CDPR_width/2, 0.52-0.03, 0.0]) # (x, y, theta)


        # State variables
        self.initial_pose = np.array([0.52, 0.545, 0.0])
        self.input = np.array([0.0, 0.0, 0.0]) # Joystick input (u)
        self.pose = self.initial_pose.copy()
        self.target_pose = self.initial_pose.copy() # (x, y, theta)
        self.previous_pose = self.initial_pose.copy() # (x, y, theta)
        
        # Initialize Logic
        # self.initialize_state()

        # ROS publishers
        self.current_pose_publisher = self.create_publisher(CdprPose, '/cdpr/current_pose', 10)

        delivery_guarantee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL
        )
        self.motor_write_publisher = self.create_publisher(MotorCmd, '/dynamixel_driver/motor_cmd', delivery_guarantee_qos)
        self.motor_write_continous_publisher = self.create_publisher(MotorCmd, '/dynamixel_driver/motor_cmd_continous', 10)

        # ROS subscribers
        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.goto_pose_subscriber = self.create_subscription(CdprPose, '/cdpr/goto_pose', self.goto_pose_callback, 10)
        
        self.motor_state_subscriber_slow = self.create_subscription(MotorState, '/dynamixel_driver/motor_state_slow', self.motor_state_slow_callback, 10)
        self.motor_state_subscriber_fast = self.create_subscription(MotorState, '/dynamixel_driver/motor_state_fast', self.motor_state_fast_callback, 10)
        
        self.control_timer = self.create_timer(self.control_loop_period, self.command_robot)
        self.control_timer.cancel()
        self.homing_timer = self.create_timer(self.homing_loop_period, self.homing)
        self.auto_tighten_timer = self.create_timer(self.auto_tighten_loop_period, self.auto_tighten_cables)
        self.auto_tighten_activate_timer = self.create_timer(self.auto_tighten_activate_loop_period, self.auto_tighten_cables_activate)
        self.auto_tighten_activate_timer.cancel()
        self.motor_feedback_timer = self.create_timer(self.motor_feedback_period, self.filtered_cable_lengths)

        # Crash behavior
        # sys.excepthook = self.myexcepthook

        self.get_logger().info(f"{node_name} Node has been started.")
    
    def __del__(self):
        print("CDPRBaseControlNode destructor")
    
    # def myexcepthook(self, type, value, tb):
    #     print("CRASH BEHAVIOR BEGUN")
    #     self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
    #     print("CRASH BEHAVIOR DONE")

    def initialize_state(self):
        # Motor initialization
        # Torque off
        self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))

        # Motor setting
        self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.OPERATING_MODE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[128,128,128,128], control_type_address=CONTROL_TABLE.VELOCITY_LIMIT.value[0]))
        a_max = 1000 

        self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[a_max], control_type_address=CONTROL_TABLE.PROFILE_ACCELERATION.value[0]))
        # Torque back on
        self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))

        # Assume robot starts at center or user manually centered it
        self.pose = self.initial_pose.copy()
        self.previous_pose = self.initial_pose.copy()

        # Compute release points after initial pose is set and before kinematics
        self.compute_release_points()
        
        # Calculate theoretical lengths for the center
        self.initial_cable_vectors = self.inverse_kinematics(self.initial_pose[0:2], self.initial_pose[2])
        self.initial_cable_lengths = [np.linalg.norm(l) for l in self.initial_cable_vectors]
        
        # Initialize filter with theoretical values
        self.cable_lengths = np.array(self.initial_cable_lengths)
        self.previous_cable_lengths = np.array(self.initial_cable_lengths)
        
        # Zero the encoders at this position
        self.zero_offsets = self.present_position.copy()

        self.is_initialized = True
        self.get_logger().info("CDPR State Initialized and ready for commands.")
    
    def joy_callback(self, msg: Joy):
        self.handle_button_events(msg.buttons)
        # Xbox controller mapping
        x = -msg.axes[3]
        y = msg.axes[4]
        if (abs(x) < 0.1): x = 0
        if (abs(y) < 0.1): y = 0
        # D-pad overrides
        if abs(msg.axes[6]) > 0: x = -msg.axes[6]
        if abs(msg.axes[7]) > 0: y = msg.axes[7]
        theta = (msg.axes[5] - msg.axes[2]) / 2
        self.input = np.array([x,y,theta])

    def goto_pose_callback(self, msg: Joy):
        target_pos = msg.position
        target_ori = msg.orientation
        self.target_pose = np.array([target_pos[0], target_pos[1], target_ori])

    def motor_state_slow_callback(self, msg: MotorState):
        motor_ids_full = np.array(msg.motor_id)
        indices = [np.where(motor_ids_full == id)[0][0] for id in self.motor_IDs]

        self.motor_ids = motor_ids_full[indices]
        self.operating_mode = np.array(msg.operating_mode)[indices]
        self.homing_offset = np.array(msg.homing_offset)[indices]
        self.current_limit = np.array(msg.current_limit)[indices]
        self.velocity_limit = np.array(msg.velocity_limit)[indices]
        self.goal_current = np.array(msg.goal_current)[indices]
        self.torque_enable = np.array(msg.torque_enable)[indices]
        self.goal_velocity = np.array(msg.goal_velocity)[indices]
        self.profile_velocity = np.array(msg.profile_velocity)[indices]
        self.goal_position = np.array(msg.goal_position)[indices]
        self.moving = np.array(msg.moving)[indices]

        if not self.slow_callback_is_initialized:
            self.slow_callback_is_initialized = True

    def motor_state_fast_callback(self, motor_state_msg: MotorState):
        motor_ids_full = np.array(motor_state_msg.motor_id)
        indices = [np.where(motor_ids_full == id)[0][0] for id in self.motor_IDs]

        self.present_current = np.array(motor_state_msg.present_current)[indices]
        self.present_position = np.array(motor_state_msg.present_position)[indices]

        if self.is_initialized:
            return

        if not self.fast_callback_is_initialized:
            self.fast_callback_is_initialized = True
        
        if self.fast_callback_is_initialized and self.slow_callback_is_initialized:
            self.initialize_state()
        
    def command_robot(self):
        print("Base Command_robot() call")

    def homing(self):
        if self.homing_active:
            # Begin tightening tensions
            present_current_list = self.present_current
            force_list = self.motor_current_units_to_force(present_current_list)
            tigthen_array = [0,0,0,0]
            not_tightened = False
            for i in range(4):
                if (force_list[i] < self.home_tension):
                    tigthen_array[i] = self.homing_speed
                    not_tightened = True
                else:
                    tigthen_array[i] = 0
                self.get_logger().info(f"force {i+1} is {force_list[i]:.2f}N, < desired {self.home_tension}N,  {"tightening..." if tigthen_array[i] != 0 else ""}")
                # print(f"force {i+1} is {force_list[i]:.2f}N, < desired {self.home_tension}N,  {"tightening..." if tigthen_array[i] != 0 else ""}")
            self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
            self.motor_write_continous_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=tigthen_array, control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # When all cables reach the desired tension, stop tightening and reinitialize state
            if not not_tightened:
                self.homing_active = False
                self.motor_write_continous_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))
                self.initialize_state()
                self.get_logger().info('Homing complete.')

            return
        
    def auto_tighten_cables_activate(self):
        if self.homing_active:
            self.get_logger().info("Blocked auto tighten request since system is homing")
            return
        
        if self.control_timer.is_canceled():
            return

        self.auto_tighten_active = True
        self.get_logger().info("Auto tighten started")
        self.cable_lengths_pre_tightened = self.cable_lengths.copy()
        self.auto_tighten_activate_timer.cancel()

    def auto_tighten_cables(self):

        if self.auto_tighten_active:
            if not self.control_timer.is_canceled():
                self.control_timer.cancel()
            current_forces = self.motor_current_units_to_force(self.present_current)            
            # Begin tightening tensions
            tigthen_array = [0,0,0,0]
            not_tightened = False
            for i in range(4):
                if (current_forces[i] < self.auto_tighten_tension):
                    tigthen_array[i] = self.homing_speed
                    not_tightened = True
                else:
                    tigthen_array[i] = 0
                # self.get_logger().info(f"force {i+1} is {current_forces[i]:.2f}N, < desired {self.auto_tighten_tension}N,  {"tightening..." if tigthen_array[i] != 0 else ""}", throttle_duration_sec=0.2)
                # print(f"force {i+1} is {current_forces[i]:.2f}N, < desired {self.auto_tighten_tension}N,  {"tightening..." if tigthen_array[i] != 0 else ""}")
            # self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
            self.get_logger().info(f"Current forces: {current_forces}, desired tension: {self.auto_tighten_tension}N", throttle_duration_sec=0.2)
            self.motor_write_continous_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=tigthen_array, control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # When all cables reach the desired tension, stop tightening
            if not not_tightened:
                self.get_logger().info('Auto tightening complete.')
                self.auto_tighten_active = False

                # Adjusted cable length logic
                self.initial_cable_lengths -= self.cable_lengths - self.cable_lengths_pre_tightened



                if self.control_timer.is_canceled():
                    self.control_timer.reset()
                    # self.get_logger().info("Control loop restarted")
                if self.auto_tighten_activate_timer.is_canceled():
                    self.auto_tighten_activate_timer.reset()
                    # self.get_logger().info("Auto tighten activate timer restarted")
            return

    def tension_safety_check(self):
        current_forces = self.motor_current_units_to_force(self.present_current)
        if (np.any(current_forces > self.max_tension_threshold)):

            self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
            self.motor_write_continous_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            self.control_timer.cancel()
            self.auto_tighten_activate_timer.cancel()
            self.auto_tighten_active = False

            self.get_logger().warn(f"Tension threshold ({self.max_tension_threshold} N) exceeded! Current forces: {current_forces}")
            return False
        
        return True

    def filtered_cable_lengths(self):   
        if not self.is_initialized:
            return

        raw_lengths = self.get_current_cable_lengths()
        
        if np.any(raw_lengths == None):
            return
        
        # Apply Exponential Moving Average (first order low pass)
        self.cable_lengths = (
            self.filter_alpha * raw_lengths + 
            (1.0 - self.filter_alpha) * self.cable_lengths
        )

    def get_current_cable_lengths(self):
        positions = self.present_position.copy()
        positions = np.array(positions)

        if np.any(positions == None):
            return np.array([None, None, None, None])
        
        motor_encoder_positions = positions - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = self.initial_cable_lengths - motor_encoder_rotations * self.cable_length_per_rot
        current_cable_lengths[0:2] = current_cable_lengths[0:2] + motor_encoder_rotations[0:2] * self.spool_pitch # Negative correction for winch box vertical travel
        current_cable_lengths[2:4] = current_cable_lengths[2:4] - motor_encoder_rotations[2:4] * self.spool_pitch # Positive corrention for winch box vertical travel
        return current_cable_lengths
    
    def compute_release_points(self):
        return
        theta = self.pose[2] # Pose orientation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        z_rot = np.array([[cos_t, -sin_t], 
                        [sin_t,  cos_t]])
        q_local_points = [self.q1, self.q2, self.q3, self.q4]
        pulley_centers = [self.C1, self.C2, self.C3, self.C4]
        
        release_points = []

        for i in range(4):
            # Transform q to world frame
            q_world = z_rot @ q_local_points[i] + self.pose[0:2]
            # Vector from pulley center (B) to pulling point (q)
            C = pulley_centers[i]
            diff = q_world - C
            d = np.linalg.norm(diff)
            phi = np.arctan2(diff[1], diff[0])
            alpha = np.arccos(self.pulley_radius / d)
            self.get_logger().info(f"Phi for pulley {i+1}: {np.degrees(phi):.2f} degrees, alpha: {np.degrees(alpha):.2f} degrees")

            if i % 2 == 0:
                beta = phi + alpha
            else:
                beta = phi - alpha
                
            # Calculate final Global Frame coordinates
            rx = C[0] + self.pulley_radius * np.cos(beta)
            ry = C[1] + self.pulley_radius * np.sin(beta)
            release_points.append(np.array([rx, ry]))

        self.B1, self.B2, self.B3, self.B4 = release_points

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

        # # Spring elongation from hooks law
        # l1_spring = self.present_current[0]/self.spring_k
        # l2_spring = self.present_current[1]/self.spring_k
        # l3_spring = self.present_current[2]/self.spring_k
        # l4_spring = self.present_current[3]/self.spring_k

        # # Total cable vector is now li + spring_elongation in the same direction as li
        # x1 = l1 + l1_spring*(l1/np.linalg.norm(l1))
        # x2 = l2 + l2_spring*(l2/np.linalg.norm(l2))
        # x3 = l3 + l3_spring*(l3/np.linalg.norm(l3))
        # x4 = l4 + l4_spring*(l4/np.linalg.norm(l4))

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
        
        margin = 0.03
        limit_theta = 1.0  # 0.3 # ~17 degrees
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
        
        residual = constraint_equations(result.x)
        self.get_logger().info(f"FK optimization residual: {residual:.20f}", throttle_duration_sec=self.log_period)
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
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
                self.get_logger().info('Control loop STARTED.')
            # if self.auto_tighten_activate_timer.is_canceled():
            #     self.auto_tighten_activate_timer.reset()


        # Button B (rising edge)
        if current_buttons[1] == 1 and self.last_buttons_state[1] == 0:
            # Deactivate CDPR
            if not self.control_timer.is_canceled():
                self.control_timer.cancel()
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
                self.get_logger().info('Control loop STOPPED.')
            # Stop auto tighten system
            if not self.auto_tighten_activate_timer.is_canceled():
                self.auto_tighten_activate_timer.cancel()
            # Stop homing procedure
            if self.homing_active:
                self.homing_active = False
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
                self.get_logger().info('Homing loop STOPPED.')

        # Button X (rising edge)
        if current_buttons[2] == 1 and self.last_buttons_state[2] == 0:
            # Release cables for homing (only when cdpr control loop is inactive)
            if self.control_timer.is_canceled():
                release_speed = -self.homing_speed
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1,1,1,1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[release_speed], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))



        # Button X (falling edge)
        if current_buttons[2] == 0 and self.last_buttons_state[2] == 1:
            # Stop cable release
            self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0,0,0,0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

        # Button Y (rising edge)
        if current_buttons[3] == 1 and self.last_buttons_state[3] == 0:
            # Start homing procedure (only when cdpr control loop is inactive)
            if self.control_timer.is_canceled():
                if not self.auto_tighten_active:
                    self.homing_active = True
                    self.get_logger().info('Homing started.')
                else:
                    self.get_logger().info('Can not home since system has auto tighten active')

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
