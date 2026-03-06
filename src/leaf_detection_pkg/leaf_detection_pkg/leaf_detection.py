#!/usr/bin/env python3

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image # Import the Image message type
from plantwall_custom_interfaces.msg import CdprPose
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

        self.contour_area_threshold = 100 # pixels

        self.control_loop_period = 0.1 # 10 Hz
        self.control_timer = self.create_timer(self.control_loop_period, self.leaf_detector)

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


    def color_image_callback(self, msg):    
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')        


    def depth_image_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        

    def leaf_detector(self):
        if (self.color_image.size == 0) or (self.depth_image.size == 0):
            self.get_logger().info("Color or depth image is not ready!")
            return

        # Check image using fast mahalanobis
        leaf_coordinate = self.mahalanobis_check()

        if leaf_coordinate is None:
            print("no leaf coordinate found")
            return

        self.get_logger().info(f"Mahalanobis candidate leaf found at coordinate: {leaf_coordinate}")

        # If possible leaf found, use SAM2 on point
        segment_mask = self.run_sam2(leaf_coordinate)

        # Apply mask on self.color_image for visualization
        masked_img = cv2.bitwise_and(self.color_image, self.color_image, mask=segment_mask.astype(np.uint8))
        # # show masked image
        # plt.imshow(masked_img)
        # plt.show()

        

    def mahalanobis_check(self) -> np.ndarray|None:
        class_configs = {
            "yellow": {
                "threshold": 20.0,       # Sensitivity: Lower = stricter
                "kernel_size": (5, 5)   # Morphological cleaning
            },
            "brown": {
                "threshold": 5.0,       # Brown often needs a wider threshold
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
            print("No contours found!")
            return None
        largest_contour = max(contours, key=cv2.contourArea)

        # Check if largest contour is big enough
        if cv2.contourArea(largest_contour) < self.contour_area_threshold:
            self.get_logger().info("Mahal found no leaf big enough")
            return None
        
        biggest_leaf_mask = np.zeros_like(combined_morphed)
        cv2.drawContours(biggest_leaf_mask, [largest_contour], -1, 255, -1)
    
        dist_trans = cv2.distanceTransform(biggest_leaf_mask, cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist_trans)
        seed_point = np.array([[max_loc[0], max_loc[1]]])
        print(f"Seed point: {seed_point}")

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

def main(args=None):
    rclpy.init(args=args)
    node = LeafDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
