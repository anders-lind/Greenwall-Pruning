#!/usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, CameraInfo # Import the Image message type
from plantwall_custom_interfaces.msg import CdprPose
from std_srvs.srv import SetBool
from plantwall_custom_interfaces.srv import CdprPos3D
import torch
import numpy as np
from cv_bridge import CvBridge
import os
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class LeafDetectionNode(Node):
    def __init__(self):
        super().__init__('leaf_detection')
        self.get_logger().info("Leaf Detection Node has been started.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using {self.device} for leaf detection.")

        self.cv_bridge = CvBridge()
        self.color_image: np.ndarray = np.zeros(0)
        self.depth_image: np.ndarray = np.zeros(0)

        self.info_received = False
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_height = None
        self.camerea_width = None

        self.current_pose = None

        self.leaf_detector_running = False

        self.mahalanobis_contour_area_threshold = 0.001 # ratio of entire image area
        self.sam2_contour_area_threshold = 0.01 # ratio of entire image area

        self.control_loop_period = 0.1 # 10 Hz
        self.ai_loop_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_timer = self.create_timer(self.control_loop_period, self.leaf_detector, callback_group=self.ai_loop_cb_group)

        self.iterator = 0

        # Load training
        # mahal_data_path = "/home/anders/workspace/masters_thesis/Greenwall-Pruning/test_scripts/perception/perception_stats_cielab.npy"
        mahal_data_path = "/home/alex/Thesis/Greenwall-Pruning/test_scripts/perception/perception_stats.npy"
        if not os.path.exists(mahal_data_path):
            print(f"Error: {mahal_data_path} not found. Please run your training script first.")
            exit()
        self.mahal_data = np.load(mahal_data_path, allow_pickle=True).item()

        home = os.path.expanduser("~")
        sam2_checkpoint = os.path.join(home, "/home/alex/Thesis/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
        model_cfg = "sam2_hiera_b+.yaml"
        sam2_model = build_sam2(model_cfg, ckpt_path=None, device=self.device)
        sd = torch.load(sam2_checkpoint, map_location=self.device, weights_only=True)["model"]
        sam2_model.load_state_dict(sd, strict=False)
        self.predictor = SAM2ImagePredictor(sam2_model)

        self.pc_publisher = self.create_publisher(PointCloud2, '/leaf_detection/pointcloud', 10)

        # Subscriber for Color Image
        self.realsense_color_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_image_callback, 
            10)

        # Subscriber for Depth Image
        self.realsense_depth_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/aligned_depth_to_color/image_raw', 
            self.depth_image_callback, 
            10)
        
        # Subscribe to camera intrinsics
        self.realsense_aligned_intrinsics_subscriber = self.create_subscription(
            CameraInfo, 
            '/camera/camera/aligned_depth_to_color/camera_info',
            self.camera_info_callback, 
            10)
        
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

    def color_image_callback(self, msg):    
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.get_logger().info("Received color image",throttle_duration_sec=0.1)

    def depth_image_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.get_logger().info("Received depth image",throttle_duration_sec=0.1)

    def current_pose_callback(self, msg: CdprPose):
        current_pos = msg.position
        current_ori = msg.orientation
        self.current_pose = np.array([current_pos[0], current_pos[1], current_ori])

    def camera_info_callback(self, msg):
        if not self.info_received:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_height = msg.height
            self.camerea_width = msg.width
            self.info_received = True

    def change_state_service_callback(self, request, response):
        self.leaf_detector_running = request.data
        response.success = True
        status = "ENABLED" if self.leaf_detector_running else "DISABLED"
        response.message = f"Leaf Detection {status}."
        self.get_logger().info(response.message)
        return response

    async def leaf_detector(self):
        if not self.leaf_detector_running:
            return
        
        if (self.color_image.size == 0) or (self.depth_image.size == 0):
            self.get_logger().info("Color or depth image is not ready!")
            return

        # # Check image using fast mahalanobis
        # leaf_coordinate = self.mahalanobis_check()

        # if leaf_coordinate is None:
        #     self.get_logger().info(f"No leaf candidate found by Mahalanobis")
        #     return

        # self.get_logger().info(f"Mahalanobis candidate leaf found at coordinate: {leaf_coordinate}")

        # # If leaf candidate is found, use SAM2 on leaf coordinate
        # sam2_segment_mask = self.run_sam2(leaf_coordinate)
        # if sam2_segment_mask is None:
        #     return
        
        # sam2_segment_mask_u8 = (sam2_segment_mask.astype(np.uint8)) * 255

        # # Verify Area of sam2 leaf segmentation is large enough
        # sam2_contours,_ = cv2.findContours(sam2_segment_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if not sam2_contours:
        #     return
        # sam2_largest_contour = max(sam2_contours, key=cv2.contourArea)

        # if cv2.contourArea(sam2_largest_contour) < self.sam2_contour_area_threshold * (sam2_segment_mask_u8.shape[0] * sam2_segment_mask_u8.shape[1]):
        #     self.get_logger().info("SAM2 found leaf too small for picking")
        #     return
        
        # leaf_pc = self.get_leaf_pointcloud(sam2_segment_mask)
        # # self.save_3d_plot(leaf_pc)
        # self.publish_pointcloud(leaf_pc)

        # n_points = len(leaf_pc)
        # self.get_logger().info(f"Generated pointcloud with {n_points} points.")

        # Placeholder: Currently we use point cloud median as grasping point
        # if n_points > 0:
        self.iterator += 1
        self.get_logger().info(f"Iterator: {self.iterator}", throttle_duration_sec=0.1)
        if self.iterator > 100: # Activate after 10 seconds
            # grasp_target = np.median(leaf_pc, axis=0)
            # dx, dy, dz = self.transform_cam_to_ee(grasp_target[0], grasp_target[1], grasp_target[2])


            self.iterator = 0
            if self.trigger_pruning_sequence_client.service_is_ready():
                req = CdprPos3D.Request()
                
                # Cast to standard Python floats to avoid Numpy errors
                # req.x = float(dx)
                # req.y = float(dy)
                # req.z = float(dz)

                # Make request with dummy coordinates as random offsets from current pose. Replace with real coordinates when perception is working.
                req.x = float(self.current_pose[0] + np.random.uniform(-0.05, 0.05))
                req.y = float(self.current_pose[1] + np.random.uniform(-0.05, 0.05))
                req.z = np.random.uniform(0.1, 0.3)
                
                self.get_logger().info(f"Sending leaf coordinates to pruning node: x={req.x:.3f}, y={req.y:.3f}, z={req.z:.3f}")
                # The AI timer pauses here while the robot physically performs the pruning sequence
                await self.trigger_pruning_sequence_client.call_async(req)
                self.get_logger().info("Pruning sequence completed. Resuming perception.")
                
            else:
                self.get_logger().error("Greenwall Pruning service is not available. Cannot trigger pruning!")

    def mahalanobis_check(self) -> np.ndarray|None:
        class_configs = {
            "yellow": {
                "threshold": 5.5,       # Sensitivity: Lower = stricter
                "kernel_size": (5, 5)   # Morphological cleaning
            },
            "brown": {
                "threshold": 4.0,       # Brown often needs a wider threshold
                "kernel_size": (5, 5)  # Larger kernel for "crunchy" textures
            }
        }

        # Get copy of image
        img = self.color_image.copy()

        # Get image information
        rows, cols, _ = img.shape
        pixels = img.reshape(-1, 3)
        combined_morphed = np.zeros((rows, cols), dtype=np.uint8)

        # Created Mahalanobis masks for each leaf class
        for i, (cls, config) in enumerate(class_configs.items()):
            mu = self.mahal_data[cls]["mean"]
            inv_cov = self.mahal_data[cls]["inv_cov"]
            
            diff = pixels - mu
            dist = np.sum(diff * (diff @ inv_cov), axis=1).reshape(rows, cols)
            
            mask = (dist < config["threshold"]).astype(np.uint8) * 255
            kernel = np.ones(config["kernel_size"], np.uint8)
            morphed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
            
            # Combine morphed images
            combined_morphed = cv2.bitwise_or(combined_morphed, morphed)
        
        # Find largest contour
        contours,_ = cv2.findContours(combined_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().info(f"No contours found!")
            return None
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if largest contour is big enough
        if cv2.contourArea(largest_contour) < self.mahalanobis_contour_area_threshold * (combined_morphed.shape[0] * combined_morphed.shape[1]):
            self.get_logger().info("Mahal found no leaf big enough")
            return None
        
        biggest_leaf_mask = np.zeros_like(combined_morphed)
        cv2.drawContours(biggest_leaf_mask, [largest_contour], -1, 255, -1)
    
        dist_trans = cv2.distanceTransform(biggest_leaf_mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_trans)
        seed_point = np.array([[max_loc[0], max_loc[1]]])
        self.get_logger().info(f"Seed point: {seed_point}")

        return seed_point


    def run_sam2(self, leaf_coordinate):
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            self.predictor.set_image(self.color_image)
            masks, scores, _ = self.predictor.predict(
                point_coords=leaf_coordinate,
                point_labels=np.array([1]), 
                multimask_output=True,
            )
        
        best_mask = masks[np.argmax(scores)]
        return best_mask
    
    def get_leaf_pointcloud(self, mask):
        v_coords, u_coords = np.where(mask > 0)
        
        z_values = self.depth_image[v_coords, u_coords].astype(float)
        
        # Filter out invalid depths and convert to meters
        valid_indices = z_values > 0
        u = u_coords[valid_indices]
        v = v_coords[valid_indices]
        Z = z_values[valid_indices] / 1000.0
        
        # Vectorized Pinhole Projection
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        
        # Stack into a (N, 3) array: [[x, y, z], [x, y, z], ...]
        point_cloud = np.column_stack((X, Y, Z)).astype(np.float32)
        
        return point_cloud
    
    def transform_cam_to_ee(self, cam_x, cam_y, cam_z):
        # Mounting Offset (Vector from EE center to Camera Lens)
        ee_to_cam = np.array([0.05, 0.02, 0.05]) # [x_offset, y_offset, z_offset]

        # Axis Remapping
        ee_x = -cam_x
        ee_y = -cam_x
        ee_z = cam_z

        # Apply transformation
        delta_x = ee_x + ee_to_cam[0]
        delta_y = ee_y + ee_to_cam[1]
        delta_z = ee_z + ee_to_cam[2]

        return (delta_x, delta_y, delta_z)

    def publish_pointcloud(self, pc_array):
        if pc_array.size == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_color_optical_frame'

        # The PointField MUST match the datatype of the pc_array
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Use the sensor_msgs_py helper to pack the cloud
        pc_msg = pc2.create_cloud(header, fields, pc_array)
        self.pc_publisher.publish(pc_msg)

    def save_3d_plot(self, pc, filename="leaf_plot.png"):
        """
        Saves a 3D scatter plot of the leaf pointcloud.
        pc: np.array of shape (N, 3)
        """
        if pc.size == 0:
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract X, Y, Z
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]

        # Create Scatter Plot
        # c=z colors the points by their depth (useful for visual depth perception)
        img = ax.scatter(x, y, z, c=z, cmap='viridis', s=2)
        
        # Add a color bar
        fig.colorbar(img, ax=ax, label='Depth (m)')

        # Labels (Important for your Thesis figures!)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Segmented Leaf 3D Reconstruction')

        # Force the axes to be equal (so the leaf isn't stretched)
        # This is a common issue in Matplotlib 3D
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Save to disk
        save_path = os.path.join(os.path.expanduser("~"), "Thesis/plots", filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig) # Close to free up memory
        self.get_logger().info(f"3D Plot saved to {save_path}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = LeafDetectionNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

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
