#!/usr/bin/env python3

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from cv_bridge import CvBridge

from std_msgs.msg import Header
from std_srvs.srv import SetBool
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo, Joy
import sensor_msgs_py.point_cloud2 as pc2
from plantwall_custom_interfaces.msg import CdprPose
from plantwall_custom_interfaces.srv import CdprPos3D

import csv
from datetime import datetime
import time

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class LeafDetectionNode(Node):
    def __init__(self):
        super().__init__('leaf_detection')
        self.get_logger().info("Leaf Detection Node has been started.")

        # Creates a folder like ~/Thesis/experiment_logs/20260512_1030/
        home = os.path.expanduser("~")
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(home, "Thesis/experiment_logs", self.session_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.log_dir, "experiment_results.csv")
        self.init_csv()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using {self.device} for leaf detection.")

        self.button_array = None

        self.cv_bridge = CvBridge()
        self.color_image: np.ndarray = np.zeros(0)
        self.depth_image: np.ndarray = np.zeros(0)

        self.info_received = False
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_height = self.camera_width = None
        self.current_pose = None
        self.leaf_detector_running = True

        # Perception system hyperparameters
        # self.area_min_ratio = 0.0015 
        # self.area_max_ratio = 0.15     
        # self.pre_sam_thresh = 16.0      
        # self.post_sam_thresh = 3.5     
        # self.percentile = 15.0         
        # self.morph_kernel = np.ones((5, 5), np.uint8)
        # self.adaptive_alpha = 0.00 # online reference color adaption filter coefficient
        # self.max_drift_distance = 15.0 # max distance from anchor reference color

        self.area_min_ratio = 0.0015 
        self.area_max_ratio = 0.15     
        self.pre_sam_thresh = 4.0
        self.post_sam_thresh = 4.0     
        self.percentile = 15.0         
        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.adaptive_alpha = 0.00 # online reference color adaption filter coefficient
        self.max_drift_distance = 15.0 # max distance from anchor reference color

        # Load Mahalanobis learned reference data
        # mahal_data_path = "/home/alex/Thesis/Greenwall-Pruning/perception_stats_cielab_new.npy"

        mahal_data_path = "/home/alex/Thesis/perception_stats_cielab.npy"

        if not os.path.exists(mahal_data_path):
            self.get_logger().error(f"{mahal_data_path} not found. Exiting.")
            exit()
        mahal_data = np.load(mahal_data_path, allow_pickle=True).item()
        self.anchor_mu = mahal_data["yellow"]["mean"]
        self.inv_cov = mahal_data["yellow"]["inv_cov"]
        self.current_mu = np.copy(self.anchor_mu)

        # Load SAM 2
        home = os.path.expanduser("~")
        sam2_checkpoint = os.path.join(home, "Thesis/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
        model_cfg = "sam2_hiera_b+.yaml"
        sam2_model = build_sam2(model_cfg, ckpt_path=None, device=self.device)
        sd = torch.load(sam2_checkpoint, map_location=self.device, weights_only=True)["model"]
        sam2_model.load_state_dict(sd, strict=False)
        self.predictor = SAM2ImagePredictor(sam2_model)

        # ROS publishers
        self.pc_publisher = self.create_publisher(PointCloud2, '/leaf_detection/pointcloud', 10)

        self.pub_mahal_raw = self.create_publisher(Image, '/leaf_detection/mahal_raw', 10)
        self.pub_mahal_morphed = self.create_publisher(Image, '/leaf_detection/mahal_morphed', 10)
        self.pub_sam2_overlay = self.create_publisher(Image, '/leaf_detection/sam2_overlay', 10)

        # ROS subscribers
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_image_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(CdprPose, '/cdpr/current_pose', self.current_pose_callback, 10)
        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)


        # ROS Service servers
        self.create_service(SetBool, '/leaf_detection/toggle', self.change_state_service_callback)

        # ROS Service client
        self.trigger_pruning_sequence_client = self.create_client(CdprPos3D, '/greenwall_pruning/trigger_pruning_sequence')

        # Main loop
        self.control_loop_period = 0.1 # 10 Hz
        self.ai_loop_cb_group = MutuallyExclusiveCallbackGroup()
        self.control_timer = self.create_timer(self.control_loop_period, self.leaf_detector, callback_group=self.ai_loop_cb_group)

    def color_image_callback(self, msg):    
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def depth_image_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

    def current_pose_callback(self, msg: CdprPose):
        self.current_pose = np.array([msg.position[0], msg.position[1], msg.orientation])

    def camera_info_callback(self, msg):
        if not self.info_received:
            self.fx, self.cx = msg.k[0], msg.k[2]
            self.fy, self.cy = msg.k[4], msg.k[5]
            self.camera_height, self.camera_width = msg.height, msg.width
            self.info_received = True

    def change_state_service_callback(self, request, response):
        self.leaf_detector_running = request.data
        status = "ENABLED" if self.leaf_detector_running else "DISABLED"
        self.get_logger().info(f"Leaf Detection {status}.")
        response.success = True
        return response

    def joy_callback(self, msg: Joy):
        self.button_array = msg.buttons

    def init_csv(self):
        """Initializes the CSV file with updated headers."""
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 
                'image_file', 
                'dx', 'dy', 'dz', 
                'robot_x', 'robot_y', 'robot_yaw', 
                'u_seed', 'v_seed',             
                'verif_percentile',             
                'verif_dist',                   
                'user_accepted', 
                'pruning_success'
            ])
    def save_experiment_data(self, image, dx, dy, dz, pose, accepted, u, v, p_dist):
        """Saves image and logs all metrics to CSV."""
        timestamp = datetime.now().strftime("%H-%M-%S_%f")
        img_filename = f"capture_{timestamp}.png"
        img_path = os.path.join(self.log_dir, img_filename)

        # Save Image (RGB -> BGR for OpenCV)
        try:
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {e}")

        # Log to CSV
        try:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    img_filename,
                    f"{dx:.4f}", f"{dy:.4f}", f"{dz:.4f}",
                    f"{pose[0]:.4f}", f"{pose[1]:.4f}", f"{pose[2]:.4f}",
                    u, v,                            # Seed point coordinates
                    self.percentile,                 # Verification percentile
                    f"{p_dist:.4f}",                 # Mahalanobis distance at percentile
                    1 if accepted else 0,
                    ""                               # Manual success entry
                ])
        except Exception as e:
            self.get_logger().error(f"Failed to log CSV data: {e}")
        
    async def leaf_detector(self):
        if not self.leaf_detector_running:
            self.get_logger().info("leaf_detector_running is False.", throttle_duration_sec=1.0)
            return
        if self.color_image.size == 0:
            self.get_logger().info("color_image is empty.", throttle_duration_sec=1.0)
            return
        if self.depth_image.size == 0:
            self.get_logger().info("depth_image is empty.", throttle_duration_sec=1.0)
            return
        if self.current_pose is None:
            self.get_logger().info("current_pose is None.", throttle_duration_sec=1.0)
            return
        if not self.info_received:
            self.get_logger().info("info_received is False.", throttle_duration_sec=1.0)
            return

        # Convert RGB image from Realsense to CIELAB
        img_rgb = self.color_image.copy()
        img_cielab = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2LAB)
        img_depth = self.depth_image.copy()
        img_pose = self.current_pose.copy()

        # Get Seed Point
        seed_point = self.get_seed_point(img_cielab)
        if seed_point is None:
            return
        self.get_logger().info(f"Target found. Seed point: {seed_point}")
        
        # Draw seedpoint on RGB image for visualization
        seed_viz = img_rgb.copy()
        cv2.circle(seed_viz, (seed_point[0, 0], seed_point[0, 1]), 10, (0, 0, 255), 2)

        u_log = int(seed_point[0, 0])
        v_log = int(seed_point[0, 1])

        # Check if seedpoint is a minimum amount of pixels from the border of the image, if so reject it
        min_border_dist = 100
        if np.any(seed_point < min_border_dist) or np.any(seed_point > self.camera_width - min_border_dist):
            self.get_logger().info("Seed point too close to image border. Rejected.")
            return

        # SAM 2 Segmentation around seed point (if found)
        sam2_segment_mask = self.run_sam2(seed_point, img_rgb)
        if sam2_segment_mask is None:
            self.get_logger().info("SAM 2 failed to generate a mask. Rejected.")
            return
        
        mask_bool = (sam2_segment_mask > 0).astype(np.uint8)
        sam2_viz = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_bool)
        self.pub_sam2_overlay.publish(self.cv_bridge.cv2_to_imgmsg(sam2_viz, encoding='rgb8'))
        
        leaf_pc = self.get_leaf_pointcloud(sam2_segment_mask, img_depth)
        # SAM 2 mask verification 

        
        is_valid, dist_val = self.verify_mask(sam2_segment_mask, img_cielab)
        
        # Even if verification fails, you might want to log it as a system-rejected FP
        if not is_valid:
            self.get_logger().info(f"SAM 2 mask rejected (Dist: {dist_val:.2f})")
            return
        
        # Online daptive reference color
        if self.adaptive_alpha > 0.0:
            self.update_adaptive_mean(sam2_segment_mask, img_cielab)
        

        # Make dummy segment mask for isolated testing
        # The dummy should be a parametrized circle at a specific location and radius
        # radius = 50
        # u,v = (100,100)
        # sam2_segment_mask = np.zeros((self.camera_height, self.camera_width), dtype=np.uint8)
        # cv2.circle(sam2_segment_mask, (u,v), radius, 255, -1)

        # Compute 3D Pointcloud
        if len(leaf_pc) == 0:
            self.get_logger().info("No valid points in leaf pointcloud. Rejected.")
            return
        self.publish_pointcloud(leaf_pc)
        self.get_logger().info(f"Generated and published pointcloud with {len(leaf_pc)} points.")

        # Find grasp point (TODO: currently as median of pointcloud, perhaps find better way)
        grasp_target = np.median(leaf_pc, axis=0)

    
        # Transform grasp point from camera frame to delta in end-effector frame
        dx, dy, dz = self.transform_cam_to_ee(grasp_target[0], grasp_target[1], grasp_target[2])

        # Compensate for end effector sagging
        # Δy = -0.0546 + (0.2006 * |y - 0.5|) + (0.4081 * z)
        y = img_pose[1]
        z = dz
        offset_y = -0.0546 + (0.1 * abs(y - 0.5)) + (0.4081 * z)

        x = img_pose[0]
        offset_x = (0.1 * (x - 0.5)) 

        # offset_y = -0.04 + (0.2006 * abs(y - 0.5)) + (0.4081 * z)

        dy = dy + offset_y
        dx = dx + offset_x


        # --- USER CONFIRMATION ---
        leaf_detection_accepted = False
        self.get_logger().info("Leaf found. Press RB to Accept, LB to Reject.")

        # Your original logic works because of the MultiThreadedExecutor
        while leaf_detection_accepted == False:
            if self.button_array is not None:
                if self.button_array[5] == 1: # RB
                    leaf_detection_accepted = True
                    self.get_logger().info("Detection Accepted.")
                elif self.button_array[4] == 1: # LB
                    self.get_logger().info("Detection Rejected.")
                    # Log the rejection before returning
                    # self.save_experiment_data(img_rgb, dx, dy, dz, img_pose, 
                    #               accepted=False, u=u_log, v=v_log, 
                    #               p_dist=dist_val)
                    return
            
            # Keep the while loop from consuming 100% CPU on its thread
            time.sleep(0.01)

        # Log the acceptance

        self.save_experiment_data(seed_viz, dx, dy, dz, img_pose, 
                                  accepted=True, u=u_log, v=v_log, 
                                  p_dist=dist_val)

        if self.trigger_pruning_sequence_client.service_is_ready():
            req = CdprPos3D.Request()
            
            req.x = float(img_pose[0] + dx)
            req.y = float(img_pose[1] + dy)
            req.z = float(dz) 
            self.get_logger().info(f"Sending leaf pruning coordinates: x={req.x:.3f}, y={req.y:.3f}, z={req.z:.3f}")
            
            await self.trigger_pruning_sequence_client.call_async(req)
            self.get_logger().info("Pruning sequence completed. Resuming perception.")
        else:
            self.get_logger().error("Greenwall Pruning service is not available.")


    def get_seed_point(self, img_lab) -> np.ndarray|None:
        rows, cols, _ = img_lab.shape
        pixels = img_lab.reshape(-1, 3).astype(np.float32)

        # Mahalanobis filter
        diff = pixels - self.current_mu
        dist = np.sum(diff * (diff @ self.inv_cov), axis=1).reshape(rows, cols)
        
        # Thresholding and morphological cleaning
        mask = (dist < self.pre_sam_thresh).astype(np.uint8) * 255
        self.pub_mahal_raw.publish(self.cv_bridge.cv2_to_imgmsg(mask, encoding='mono8')) # Publish
        morphed = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel), cv2.MORPH_CLOSE, self.morph_kernel)
        self.pub_mahal_morphed.publish(self.cv_bridge.cv2_to_imgmsg(morphed, encoding='mono8')) # Publish
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter contours by min/max area
        total_area = rows * cols
        valid_contours = [c for c in contours if (total_area * self.area_max_ratio) > cv2.contourArea(c) > (total_area * self.area_min_ratio)]
        if not valid_contours:
            return None
        
        # Return CoM of largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        biggest_leaf_mask = np.zeros_like(morphed)
        cv2.drawContours(biggest_leaf_mask, [largest_contour], -1, 255, -1)
        dist_trans = cv2.distanceTransform(biggest_leaf_mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_trans)
        return np.array([[max_loc[0], max_loc[1]]])

    def run_sam2(self, leaf_coordinate, image):
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=leaf_coordinate,
                point_labels=np.array([1]), 
                multimask_output=True,
            )
        return masks[np.argmax(scores)]

    def verify_mask(self, mask, img_lab) -> bool:
        mask_bool = mask > 0
        pixels = img_lab[mask_bool].astype(np.float32)
        if len(pixels) == 0:
            return False

        # diff = pixels - self.anchor_mu
        diff = pixels - self.current_mu
        dists = np.sum(diff * (diff @ self.inv_cov), axis=1)
        percentile_dist = np.percentile(dists, self.percentile)
        return (percentile_dist <= self.post_sam_thresh), percentile_dist

    def update_adaptive_mean(self, mask, img_lab):
        mask_bool = mask > 0
        pixels = img_lab[mask_bool].astype(np.float32)
        if len(pixels) == 0:
            return

        # Update mean color with Exponential Moving Average (low-pass filter)
        mask_mean = np.mean(pixels, axis=0)
        new_mu = (self.adaptive_alpha * mask_mean) + ((1.0 - self.adaptive_alpha) * self.current_mu)
        
        # Ensure the mean color does not drift too far from the anchor reference color
        drift_dist = np.linalg.norm(new_mu - self.anchor_mu)
        if drift_dist <= self.max_drift_distance:
            self.current_mu = new_mu
            self.get_logger().info(f"Adapted Mu. Current drift distance: {drift_dist:.2f}")
        else:
            self.get_logger().info(f"Drift limit reached ({drift_dist:.2f}). Ignored update.")

    def get_leaf_pointcloud(self, mask, img_depth):
        v_coords, u_coords = np.where(mask > 0)
        z_values = img_depth[v_coords, u_coords].astype(float)
        
        valid_indices = z_values > 0
        u = u_coords[valid_indices]
        v = v_coords[valid_indices]
        Z = z_values[valid_indices] / 1000.0
        
        # Compute X and Y colors from pinhole model
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        
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

    def publish_pointcloud(self, pc_array):
        if pc_array.size == 0:
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_color_optical_frame'

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pc_msg = pc2.create_cloud(header, fields, pc_array)
        self.pc_publisher.publish(pc_msg)


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