#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from plantwall_custom_interfaces.msg import CdprPose
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CDPRVisualizer(Node):
    def __init__(self):
        super().__init__('cdpr_visualizer')

        self.cdpr_pose = [0.0, 0.0, 0.0]

        self.CDPR_height = 1.0
        self.CDPR_width = 1.0
        self.end_effector_height = 0.160
        self.end_effector_width = 0.107
        
        self.pose_subscriber = self.create_subscription(msg_type=CdprPose, topic='/cdpr/current_pose', callback=self.updatePose, qos_profile=0)
        self.goto_pose_subscriber = self.create_subscription(CdprPose, '/cdpr/goto_pose', self.goto_pose_callback, 10)
        self.visualize_timer = self.create_timer(timer_period_sec=0.1, callback=self.updateVisualization)

        w = self.CDPR_width
        h = self.CDPR_height
        
        # Define the anchor points on the CDPR frame with Bottom-Left at (0,0)
        self.anchors = np.array([
            [0, h],  # Top-Left
            [w, h],  # Top-Right
            [w, 0],  # Bottom-Right
            [0, 0]   # Bottom-Left
        ])

        # --- Plot Initialization ---
        plt.ion() # Turn on interactive mode for live updating
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        # Set axes limits with a small buffer, anchored around (0,0)
        buffer = 0.1
        self.ax.set_xlim(-buffer, w + buffer)
        self.ax.set_ylim(-buffer, h + buffer)
        self.ax.set_aspect('equal') # Prevents distortion of the robot
        self.ax.set_title("Live CDPR Visualization (Origin: Bottom-Left)")
        self.ax.grid(True, linestyle='--', alpha=0.6)

        # Draw the static outer frame starting from (0,0)
        self.frame = patches.Rectangle(
            (0, 0), w, h,
            linewidth=2, edgecolor='black', facecolor='none'
        )
        self.ax.add_patch(self.frame)

        # Initialize the end effector (starts as empty polygon)
        self.ee_patch = patches.Polygon(
            np.zeros((4, 2)), closed=True, facecolor='royalblue', edgecolor='black', zorder=3
        )
        self.ax.add_patch(self.ee_patch)

        # Initialize the target end effector (green, lower zorder)
        self.target_ee_patch = patches.Polygon(
            np.zeros((4, 2)), closed=True, facecolor='green', edgecolor='black', zorder=1
        )
        self.ax.add_patch(self.target_ee_patch)

        # Initialize 4 cables (red lines)
        self.cables = [self.ax.plot([], [], 'r-', linewidth=1.5, zorder=2)[0] for _ in range(4)]


    def updatePose(self, msg: CdprPose):
        self.cdpr_pose = [msg.position[0], msg.position[1], msg.orientation]
    

    def goto_pose_callback(self, msg: CdprPose):
        target_pos = msg.position
        target_ori = msg.orientation
        self.target_pose = np.array([target_pos[0], target_pos[1], target_ori])

    def updateVisualization(self):
        """Updates the visualizer with the current self.cdpr_pose."""
        x, y, theta = self.cdpr_pose
        
        # 1. Update End Effector Position and Rotation
        ee_corners = self._get_end_effector_corners(x, y, theta)
        self.ee_patch.set_xy(ee_corners)

        # 2. Update Target End Effector if available
        if hasattr(self, 'target_pose'):
            tx, ty, ttheta = self.target_pose
            target_corners = self._get_end_effector_corners(tx, ty, ttheta)
            self.target_ee_patch.set_xy(target_corners)

        # 3. Update Cable Lines
        for i in range(4):
            # Connect anchor 'i' to end-effector corner 'i'
            self.cables[i].set_data(
                [self.anchors[i, 0], ee_corners[i, 0]],
                [self.anchors[i, 1], ee_corners[i, 1]]
            )

        # 4. Draw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

    def _get_end_effector_corners(self, x, y, theta):
        """Calculates the global coordinates of the end effector's corners based on pose."""
        ew2 = self.end_effector_width / 2
        eh2 = self.end_effector_height / 2
        
        # Local corners relative to the end effector's center
        local_corners = np.array([
            [-ew2, eh2],  # Top-Left
            [ew2, eh2],   # Top-Right
            [ew2, -eh2],  # Bottom-Right
            [-ew2, -eh2]  # Bottom-Left
        ])
        
        # 2D Rotation Matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        # Rotate and translate
        global_corners = (R @ local_corners.T).T + np.array([x, y])
        return global_corners



def main(args = None):
    rclpy.init(args=args)
    node = CDPRVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()