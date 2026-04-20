#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from plantwall_custom_interfaces.msg import CdprPose
from std_srvs.srv import SetBool
from plantwall_custom_interfaces.srv import CdprPos3D
import numpy as np


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

    async def leaf_detector(self):
        if not self.leaf_detector_running:
            return
        
        self.iterator += 1
        self.get_logger().info(f"Iterator: {self.iterator}", throttle_duration_sec=0.1)
        if self.iterator > self.iterator_threshold: # Activate after 10 seconds
            self.iterator = 0
            if self.trigger_pruning_sequence_client.service_is_ready():
                req = CdprPos3D.Request()

                req.x = float(self.current_pose[0] + np.random.uniform(-0.05, 0.05))
                req.y = float(self.current_pose[1] + np.random.uniform(-0.05, 0.05))
                req.z = np.random.uniform(0.1, 0.3)
                
                self.get_logger().info(f"Sending leaf coordinates to pruning node: x={req.x:.3f}, y={req.y:.3f}, z={req.z:.3f}")
                # The AI timer pauses here while the robot physically performs the pruning sequence
                await self.trigger_pruning_sequence_client.call_async(req)
                self.get_logger().info("Pruning sequence completed. Resuming perception.")
                
            else:
                self.get_logger().error("Greenwall Pruning service is not available. Cannot trigger pruning!")


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
