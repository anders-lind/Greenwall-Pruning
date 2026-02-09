#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
import scipy
import matplotlib.pyplot as plt


class CDPRControlNode(Node):
    def __init__(self):
        super().__init__('cdpr_control_joystick_FK2')

        # Node state variables
        self.last_buttons_state = None

        # CDPR parameters
        self.spool_radius = 0.027
        self.spool_circumference = 2 * self.spool_radius * np.pi
        
        self.CDPR_height = 0.9600
        self.CDPR_width = 0.9325

        # --- ANCHORS (Fixed Frame) ---
        self.B1 = np.array([0, 0])
        self.B2 = np.array([0, self.CDPR_height])
        self.B3 = np.array([self.CDPR_width, self.CDPR_height])
        self.B4 = np.array([self.CDPR_width, 0])

        # --- BODY ATTACHMENT POINTS (Local Frame) ---
        self.q1 = np.array([-0.0425, -0.02])
        self.q2 = np.array([-0.0425, 0.02])
        self.q3 = np.array([0.0425, 0.02])
        self.q4 = np.array([0.0425, -0.02])

        # --- CONTROLLER GAINS ---
        self.joystick_sensitivity = np.array([20.0, 20.0, 0.1]) # Joystick Sensitivity (Newtons)

        # Damping Gains (Newtons per m/s)
        self.Kd = np.array([15.0, 15.0, 1.0]) # [Damp_X, Damp_Y, Damp_Theta]

        self.control_loop_period = 0.005  # 200 Hz
        self.motor_feedback_period = 0.005
        self.filter_alpha = 0.4 # Smoothing factor for encoders
        
        # State variables
        self.input = np.array([0.0, 0.0, 0.0]) # Joystick input
        self.initial_pose = np.array([self.CDPR_width/2, self.CDPR_height/2, 0.0]) # x, y, theta
        self.pose = self.initial_pose
        self.previous_pose = self.initial_pose # For velocity calculation

        # Motor initialization
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=[1,2,3,4], directions=[-1,-1,-1,-1])
        self.motors.write(motors=[1,2,3,4], values=0, control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])
        
        # Initialize Logic
        self.initialize_state()

        # ROS Infrastructure
        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.control_timer = self.create_timer(self.control_loop_period, self.command_robot)
        self.motor_feedback_timer = self.create_timer(self.motor_feedback_period, self.motor_feedback)
        
        # Visualization
        self.init_visualisation()
        self.visualisation_timer = self.create_timer(0.1, self.visualisation)

        self.get_logger().info("CDPR Control Started. Damping Active.")

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
        y = -msg.axes[4]
        # D-pad overrides
        if abs(msg.axes[6]) > 0: x = -msg.axes[6]
        if abs(msg.axes[7]) > 0: y = -msg.axes[7]
        theta = msg.axes[2] - msg.axes[5] 
        self.input = np.array([x,y,theta])
        
    def command_robot(self):
        # 1. Update Pose via FK
        # Note: self.cable_lengths is already filtered in motor_feedback loop
        raw_pose = self.forward_kinematics(self.cable_lengths, self.pose[0:2], self.pose[2])
        
        # --- SAFETY CLAMP (Prevents 10cm drift / Hallucinations) ---
        margin = 0.05
        clamped_x = np.clip(raw_pose[0], margin, self.CDPR_width - margin)
        clamped_y = np.clip(raw_pose[1], margin, self.CDPR_height - margin)
        clamped_th = np.clip(raw_pose[2], -0.707, 0.707) # +/- 45 degrees
        self.pose = np.array([clamped_x, clamped_y, clamped_th])

        # 2. Calculate Cartesian Velocity (Finite Difference)
        # v = dx / dt
        velocity = (self.pose - self.previous_pose) / self.control_loop_period
        self.previous_pose = self.pose

        # 3. Calculate Damping Force
        # F_damping = -Kd * v
        u_damping = -self.Kd * velocity

        # 4. Total Virtual Wrench
        # u_joystick = Sensitivity * input
        u_joystick = self.joystick_sensitivity * self.input
        u_total = u_joystick + u_damping

        # 5. Compute Matrices
        cable_vectors = self.inverse_kinematics(self.pose[0:2], self.pose[2])
        S = self.compute_structure_matrix(cable_vectors, self.pose[2])

        # 6. Null Space Optimization
        T_move = np.linalg.pinv(S) @ u_total
        
        S_nullspace = scipy.linalg.null_space(S).flatten()
        S_nullspace = S_nullspace / np.linalg.norm(S_nullspace)
        
        # Ensure consistent sign (All Pulling)
        if np.sum(S_nullspace) < 0:
            S_nullspace = -S_nullspace

        if np.any(S_nullspace < -0.01):
            # Only warn if significantly negative
            self.get_logger().warn("Mixed signs in Null Vector")
            pass

        # 7. Calculate Lambda (Tension Knob)
        t_min = 3.0 # Min tension in Newtons
        
        # We need T_move + lambda * N >= t_min
        # lambda >= (t_min - T_move) / N
        # We use a small epsilon to avoid division by zero
        lambda_vec = (t_min - T_move) / (S_nullspace + 1e-6)
        lambda_optimal = np.max(lambda_vec)

        # 8. Final Tensions
        T_final = T_move + lambda_optimal * S_nullspace
        
        # Safety: Ensure no negative values (float errors)
        T_final = np.maximum(T_final, 0.5) 

        # 9. Send to Motors
        desired_currents = self.force_to_current(T_final)
        self.motors.write(
            motors=[1,2,3,4],
            control_type=CONTROL_TABLE.GOAL_CURRENT,
            values=self.current_to_motor_input(desired_currents)
        )

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
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # --- HELPERS ---
    def get_current_cable_lengths(self):
        positions_list = self.motors.read(motors=[1,2,3,4], control_type=CONTROL_TABLE.PRESENT_POSITION)
        motor_encoder_positions = np.array(positions_list) - self.zero_offsets
        motor_encoder_rotations = motor_encoder_positions / 4096
        current_cable_lengths = self.initial_cable_lengths + motor_encoder_rotations * self.get_spool_circumference()
        return current_cable_lengths
    
    def get_spool_circumference(self):
        return self.spool_circumference

    def inverse_kinematics(self, position, orientation):
        p = np.array(position)
        theta = orientation
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
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
        result = scipy.optimize.minimize(constraint_equations, [p0[0], p0[1], theta0], tol=1e-4) # Added tolerance
        return result.x
    
    def compute_structure_matrix(self, cable_vectors, theta):
        S = np.zeros((3,4))
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        Rq = np.array([R @ self.q1, R @ self.q2, R @ self.q3, R @ self.q4]).T
        for i in range(4):
            l = cable_vectors[i]
            l_norm = np.linalg.norm(l)
            u = - l / l_norm
            S[0,i] = u[0]
            S[1,i] = u[1]
            S[2,i] = (Rq[0,i]*u[1] - Rq[1,i]*u[0])
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