#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
import scipy
import matplotlib.pyplot as plt
import time


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_force_control')

        # Node state variables
        self.last_buttons_state = None

        # CDPR parameters
        self.spool_radius = 0.0115 - 0.001 # spool outer radius minus cable radius
        self.spool_circumference = 2 * self.spool_radius * np.pi
        
        self.CDPR_height = 0.944
        self.CDPR_width = 0.908
        self.end_effector_height = 0.03916
        self.end_effector_width = 0.09322

        # --- ANCHORS (Fixed Frame) ---
        # self.B1 = np.array([0, 0])
        # self.B2 = np.array([0, self.CDPR_height])
        # self.B3 = np.array([self.CDPR_width, self.CDPR_height])
        # self.B4 = np.array([self.CDPR_width, 0])

        self.B1 = np.array([0, self.CDPR_height])
        self.B2 = np.array([self.CDPR_width, self.CDPR_height])
        self.B3 = np.array([self.CDPR_width, 0])
        self.B4 = np.array([0, 0])

        # --- BODY ATTACHMENT POINTS (Local Frame) ---
        # self.q1 = np.array([-0.0425, -0.02])
        # self.q2 = np.array([-0.0425, 0.02])
        # self.q3 = np.array([0.0425, 0.02])
        # self.q4 = np.array([0.0425, -0.02])

        self.q1 = np.array([-self.end_effector_width/2, self.end_effector_height/2])  
        self.q2 = np.array([self.end_effector_width/2, self.end_effector_height/2])
        self.q3 = np.array([self.end_effector_width/2, -self.end_effector_height/2])
        self.q4 = np.array([-self.end_effector_width/2, -self.end_effector_height/2])

        # CONTROLLER GAINS
        self.joystick_sensitivity = np.array([40.0, 40.0, 1]) # Joystick Sensitivity (x, y, theta) [N, N, Nm]

        # Tunable parameters
        self.t_min = -3.0

        # Damping Gains (Newtons per m/s)
        self.Kd = np.array([15.0, 15.0, 1.0]) # (Damp_x, Damp_y, Damp_theta)

        self.control_loop_period = 0.005  # 200 Hz
        self.motor_feedback_period = 0.005
        self.filter_alpha = 0.4 # Smoothing factor for encoders
        
        # State variables
        self.input = np.array([0.0, 0.0, 0.0]) # Joystick input (u)
        self.initial_pose = np.array([self.CDPR_width/2, self.CDPR_height/2, 0.0]) # (x, y, theta)
        self.pose = self.initial_pose
        self.previous_pose = self.initial_pose # For numerical differentiation

        # Motor initialization
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=[1,2,3,4], directions=[-1,-1,1,1])
        self.motors.write(motors=[1,2,3,4], values=0, control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])
        
        # Initialize Logic
        self.initialize_state()

        # ROS Infrastructure
        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.control_timer = self.create_timer(self.control_loop_period, self.command_robot)
        self.motor_feedback_timer = self.create_timer(self.motor_feedback_period, self.motor_feedback)
        
        # Visualization
        # self.init_visualisation()
        # self.visualisation_timer = self.create_timer(2, self.visualisation)

        self.get_logger().info("CDPR Control Started")

    def initialize_state(self):
        # Assume robot starts at center or user manually centered it
        self.pose = self.initial_pose
        self.previous_pose = self.initial_pose
        
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
        # 1. START TOTAL TIMER
        # t_start = time.perf_counter()

        # # Measure FK Time
        # t_fk_start = time.perf_counter()
        
        # 1. Update Pose via FK (THE SUSPECT)
        raw_pose = self.forward_kinematics(self.cable_lengths, self.pose[0:2], self.pose[2])
        
        # t_fk_end = time.perf_counter()

        # SAFETY CLAMP
        margin = 0.05
        clamped_x = np.clip(raw_pose[0], margin, self.CDPR_width - margin)
        clamped_y = np.clip(raw_pose[1], margin, self.CDPR_height - margin)
        clamped_th = np.clip(raw_pose[2], -0.3, 0.3) 
        self.pose = np.array([clamped_x, clamped_y, clamped_th])

        # 2. Calculate velocity
        velocity = (self.pose - self.previous_pose) / self.control_loop_period
        self.previous_pose = self.pose

        # 3. Calculate Damping Force
        u_damping = -self.Kd * velocity

        # 4. Total Virtual Wrench
        u_joystick = self.joystick_sensitivity * self.input
        u_total = u_joystick + u_damping

        # 5. Compute Matrices
        cable_vectors = self.inverse_kinematics(self.pose[0:2], self.pose[2])
        S = self.compute_structure_matrix(cable_vectors, self.pose[2])

        # 6. Null Space Optimization
        T_move = np.linalg.pinv(S) @ u_total
        S_nullspace = scipy.linalg.null_space(S).flatten()

        # 7. Check for Validity
        valid_null_space = False
        T_final = None
        if S_nullspace.size > 0:
            S_nullspace = S_nullspace / np.linalg.norm(S_nullspace) # Normalize nullvector
            if np.sum(S_nullspace) < 0:
                S_nullspace = -S_nullspace # Allign nullvector
            if np.all(S_nullspace > -0.001): # Check for valid null vector
                valid_null_space = True
        
        # 8. Branching Logic
        if valid_null_space:
            ns_safe = np.maximum(S_nullspace, 1e-6)
            lambda_vec = (self.t_min - T_move) / ns_safe
            lambda_optimal = np.max(lambda_vec)
            T_final = T_move + lambda_optimal * S_nullspace
        else:
            T_final = np.maximum(T_move, self.t_min)
    
        # # 9. Send to Motors
        # t_motor_start = time.perf_counter()
        
        desired_currents = self.force_to_current(T_final)
        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_CURRENT,
            values=self.current_to_motor_input(desired_currents)
        )
        
        # t_motor_end = time.perf_counter()

        # # --- STOP TOTAL TIMER ---
        # t_end = time.perf_counter()

        # # Calculate durations in milliseconds
        # total_ms = (t_end - t_start) * 1000.0
        # fk_ms = (t_fk_end - t_fk_start) * 1000.0
        # motor_ms = (t_motor_end - t_motor_start) * 1000.0
        # calc_ms = total_ms - fk_ms - motor_ms

        # Print breakdown
        # print(f"Total: {total_ms:.2f}ms | FK: {fk_ms:.2f}ms | Math: {calc_ms:.2f}ms | Motors: {motor_ms:.2f}ms")

    def motor_feedback(self):   
        raw_lengths = self.get_current_cable_lengths()
        # Apply Exponential Moving Average
        self.cable_lengths = (
            self.filter_alpha * raw_lengths + 
            (1.0 - self.filter_alpha) * self.cable_lengths
        )

    # --- VISUALIZATION ---
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

    # --- HELPERS ---
    def get_current_cable_lengths(self):
        positions_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        motor_encoder_positions = np.array(positions_list) - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = self.initial_cable_lengths - motor_encoder_rotations * self.get_spool_circumference()
        return current_cable_lengths
    
    def get_spool_circumference(self):
        return self.spool_circumference

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
    
    def force_to_current(self, force_vector):
        torque_vector = force_vector * self.spool_radius
        current_vector = torque_vector * 0.625 
        return current_vector

    def current_to_motor_input(self, current_vector):
        return [int(max(-2047, min(2047, v/0.00269))) for v in current_vector]

    def handle_button_events(self, current_buttons):
        if self.last_buttons_state is None:
            self.last_buttons_state = current_buttons
            return
        if current_buttons[0] == 1 and self.last_buttons_state[0] == 0:
            if self.control_timer.is_canceled():
                self.control_timer.reset()
                self.motors.enable_torque(motors=[1,2,3,4])
                self.get_logger().info('Control loop STARTED.')
        if current_buttons[1] == 1 and self.last_buttons_state[1] == 0:
            if not self.control_timer.is_canceled():
                self.control_timer.cancel()
                self.motors.disable_torque(motors=[1,2,3,4])
                self.get_logger().info('Control loop STOPPED.')
        if current_buttons[3] == 1 and self.last_buttons_state[3] == 0:
            self.initialize_state()
            self.get_logger().info('New home set')
        self.last_buttons_state = current_buttons

def main(args=None):
    rclpy.init(args=args)
    node = CDPRControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()