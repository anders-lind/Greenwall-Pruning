#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from plantwall_custom_interfaces.msg import CdprPose
from std_srvs.srv import SetBool
from plantwall_custom_interfaces.srv import CdprPos3D
import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class LeafDetectionNode(Node):
    def __init__(self):
        super().__init__('leaf_detection')
        self.get_logger().info("Leaf Detection STUB Node has been started.")

        self.current_pose = None
        self.leaf_detector_running = False
        self.iterator = 0
        self.iterator_threshold = 100

        self.control_loop_period = 0.1 # 10 Hz
        self.ai_loop_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_timer = self.create_timer(self.control_loop_period, self.leaf_detector, callback_group=self.ai_loop_cb_group)

        self.cv_bridge = CvBridge()
        self.color_image: np.ndarray = np.zeros(0)
        self.depth_image: np.ndarray = np.zeros(0)

        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_image_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_image_callback, 10)

        # Subscribe to current_pose
        self.current_pose_subscriber = self.create_subscription(
            CdprPose, 
            '/cdpr/current_pose', 
            self.current_pose_callback, 
            10)

        # Ros service servers
        self.toggle_srv = self.create_service(
            SetBool, 
            '/leaf_detection/toggle', 
            self.change_state_service_callback
        )

        # Ros service clients
        self.trigger_pruning_sequence_client = self.create_client(
            CdprPos3D, 
            '/greenwall_pruning/trigger_pruning_sequence'
        )
    
    def current_pose_callback(self, msg: CdprPose):
        current_pos = msg.position
        current_ori = msg.orientation
        self.current_pose = np.array([current_pos[0], current_pos[1], current_ori])

    def change_state_service_callback(self, request, response):
        self.leaf_detector_running = request.data
        response.success = True
        status = "ENABLED" if self.leaf_detector_running else "DISABLED"
        response.message = f"Leaf Detection {status}."
        self.get_logger().info(response.message)
        return response
    
    def depth_image_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        # self.get_logger().info(f"Pixel 300, 300: {self.depth_image[300, 300]}")


    def color_image_callback(self, msg):    
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    async def leaf_detector(self):
        if not self.leaf_detector_running:
            return
        
        img_depth = self.depth_image.copy()
        print("Shape of img_depth:", img_depth.shape)
        # img_depth = np.ones((480, 640), dtype=np.uint8)*300
        img_pose = self.current_pose.copy()

        # Make dummy segment mask for isolated testing
        # Parametrized circle at a specific location and radius
        radius = 10
        u,v = (475,85)
        sam2_segment_mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(sam2_segment_mask, (u,v), radius, 255, -1)

        # Compute 3D Pointcloud
        leaf_pc = self.get_leaf_pointcloud(sam2_segment_mask, img_depth)
        if len(leaf_pc) == 0:
            return
        # self.publish_pointcloud(leaf_pc)
        self.get_logger().info(f"Generated and published pointcloud with {len(leaf_pc)} points.")

        # Find grasp point (TODO: currently as median of pointcloud, perhaps find better way)
        grasp_target = np.median(leaf_pc, axis=0)

        self.get_logger().info(f"Target found. Seed point: {grasp_target}")

        # Transform grasp point from camera frame to delta in end-effector frame
        dx, dy, dz = self.transform_cam_to_ee(grasp_target[0], grasp_target[1], grasp_target[2])
        self.get_logger().info(f"Relative grapsh point: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")

        if self.trigger_pruning_sequence_client.service_is_ready():
            req = CdprPos3D.Request()
            
            req.x = float(img_pose[0] + dx)
            req.y = float(img_pose[1] + dy)
            req.z = float(dz) 
            self.get_logger().info(f"Sending leaf pruning coordinates: x={req.x:.3f}, y={req.y:.3f}, z={req.z:.3f}")
            
            await self.trigger_pruning_sequence_client.call_async(req)
            self.get_logger().info("Pruning sequence completed. Resuming perception.")
            self.leaf_detector_running = False
        else:
            self.get_logger().error("Greenwall Pruning service is not available.")

        # self.iterator += 1
        # self.get_logger().info(f"Iterator: {self.iterator}", throttle_duration_sec=0.1)
        # if self.iterator > self.iterator_threshold: # Activate after 10 seconds
        #     self.iterator = 0
        #     if self.trigger_pruning_sequence_client.service_is_ready():
        #         req = CdprPos3D.Request()

        #         req.x = float(self.current_pose[0] + np.random.uniform(-0.05, 0.05))
        #         req.y = float(self.current_pose[1] + np.random.uniform(-0.05, 0.05))
        #         req.z = np.random.uniform(0.1, 0.3)
                
        #         self.get_logger().info(f"Sending leaf coordinates to pruning node: x={req.x:.3f}, y={req.y:.3f}, z={req.z:.3f}")
        #         # The AI timer pauses here while the robot physically performs the pruning sequence
        #         await self.trigger_pruning_sequence_client.call_async(req)
        #         self.get_logger().info("Pruning sequence completed. Resuming perception.")
                
        #     else:
        #         self.get_logger().error("Greenwall Pruning service is not available. Cannot trigger pruning!")


    def get_leaf_pointcloud(self, mask, img_depth):
        v_coords, u_coords = np.where(mask > 0)
        z_values = img_depth[v_coords, u_coords].astype(float)
        
        valid_indices = z_values > 0
        u = u_coords[valid_indices]
        v = v_coords[valid_indices]
        Z = z_values[valid_indices] / 1000.0
        
        # Compute X and Y colors from pinhole model
        X = (u - 316.615) * Z / 611.560
        Y = (v - 247.956) * Z / 611.605
        
        # Stack into n x 3 array of [X, Y, Z]
        return np.column_stack((X, Y, Z)).astype(np.float32)
    
    def transform_cam_to_ee(self, cam_x, cam_y, cam_z):
        # [x_offset, y_offset, z_offset - offset_from_ext_glass] measured in TCP frame
        ee_to_cam = np.array([-0.026, -0.088, -0.028 - 0.0042])

        # Axis Remapping
        ee_x = -cam_y
        ee_y = cam_x
        ee_z = cam_z

        # Apply transformation
        delta_x = ee_x + ee_to_cam[0]
        delta_y = ee_y + ee_to_cam[1]
        delta_z = ee_z + ee_to_cam[2]

        return (delta_x, delta_y, delta_z)


def main(args=None):
    rclpy.init(args=args)
    node = LeafDetectionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
