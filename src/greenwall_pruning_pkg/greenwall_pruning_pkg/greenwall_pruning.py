#!/usr/bin/env python3

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from std_srvs.srv import SetBool

from plantwall_custom_interfaces.srv import Float64 as Float64Srv
from plantwall_custom_interfaces.srv import CdprPose as CdprPoseSrv
from plantwall_custom_interfaces.srv import CdprPos3D as CdprPos3DSrv

import numpy as np

class GreenwallPruningNode(Node):
    def __init__(self):
        super().__init__('greenwall_pruning')
        self.get_logger().info("Greenwall Pruning Node has been started.")

        self.leaf_pos3D = None
        
        self.pruning_active = False
        self.executing_sequence = False # NEW: Lock to prevent timer re-entry

        self.node_frequency = 10.0 # [Hz]
        self.node_loop_period = 1.0 / self.node_frequency

        self.cdpr_prepicking_offset = 0.05 # [m], vertical distance underneath detected leaf
        self.cdpr_picking_offset = 0.1 # [m], vertical distance to move down after gripping leaf to pluck it
        self.ee_prepicking_finger_gap = 0.02 # [m], distance between gripper fingers for pruning
        self.ee_release_finger_gap = 0.05 # [m], distance between gripper fingers for releasing leaf after pruning


        # Service servers
        self.leaf_found_srv = self.create_service(CdprPos3DSrv, '/greenwall_pruning/leaf_found', self.leaf_found_callback)

        # Service clients
        self.cdpr_pathplanner_toggle_client = self.create_client(SetBool, '/cdpr_pathplanner/toggle_search_state')
        self.cdpr_pathplanner_goto_pose_client = self.create_client(CdprPoseSrv, '/cdpr_pathplanner/goto_pose')

        self.perception_toggle_client = self.create_client(SetBool, '/leaf_detection/toggle')

        self.gripper_control_move_TCP_client = self.create_client(Float64Srv, '/gripper_control/move_TCP')
        self.gripper_control_set_finger_distance_client = self.create_client(Float64Srv, '/gripper_control/set_finger_distance')
        self.gripper_control_grip_client = self.create_client(Float64Srv, '/gripper_control/grip')
        
        # self.greenwall_pruning_timer = self.create_timer(self.node_loop_period, self.greenwall_pruning_loop)
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.greenwall_pruning_timer = self.create_timer(
            self.node_loop_period, 
            self.greenwall_pruning_loop,
            callback_group=self.timer_cb_group
        )

    def leaf_found_callback(self, request, response):
        self.leaf_pos3D = np.array([request.x, request.y, request.z])
        self.get_logger().info(f"Received leaf position: {self.leaf_pos3D}")
        self.pruning_active = True
        return response

    async def greenwall_pruning_loop(self):
        # Dont start if pruning inactive or if already in the middle of a sequence
        if not self.pruning_active or self.executing_sequence:
            return
        
        self.executing_sequence = True # Lock the sequence
        
        try:
            # 1. Toggle off perception system 
            if self.perception_toggle_client.wait_for_service(timeout_sec=1.0):
                toggle_req = SetBool.Request()
                toggle_req.data = False
                # NEW: await the async call
                await self.perception_toggle_client.call_async(toggle_req)
                self.get_logger().info("Toggled perception to inactive.")
            else:
                self.get_logger().error("Failed to call perception toggle service.")
                return

            # 2. Toggle off pathplanner search mode
            if self.cdpr_pathplanner_toggle_client.wait_for_service(timeout_sec=1.0):
                toggle_req = SetBool.Request()
                toggle_req.data = False
                # NEW: await the async call
                await self.cdpr_pathplanner_toggle_client.call_async(toggle_req)
                self.get_logger().info("Toggled pathplanner search state to inactive.")
            else:
                self.get_logger().error("Failed to call pathplanner search state service.")
                return
            
            # 3. Move CDPR to pre-picking pose
            if self.cdpr_pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
                goto_req = CdprPoseSrv.Request()
                goto_req.position[0] = self.leaf_pos3D[0]
                goto_req.position[1] = self.leaf_pos3D[1] - self.cdpr_prepicking_offset
                goto_req.orientation = 0.0
                # NEW: await the async call
                await self.cdpr_pathplanner_goto_pose_client.call_async(goto_req)
                self.get_logger().info("Sent goto_pose request to pathplanner.")
            else:
                self.get_logger().error("Failed to call pathplanner goto_pose service.")
                return
            
           # 4. Move TCP to correct match depth of detected leaf
            if self.gripper_control_move_TCP_client.wait_for_service(timeout_sec=1.0):
                move_TCP_req = Float64Srv.Request()
                move_TCP_req.value = float(self.leaf_pos3D[2]) # Assuming z coordinate corresponds to depth
                await self.gripper_control_move_TCP_client.call_async(move_TCP_req)
                self.get_logger().info("Sent move_TCP request to end effector.")
            else:
                self.get_logger().error("Failed to call end effector move_TCP service.")
                return
            
            # 5. Open gripper fingers to prepare for pruning
            if self.gripper_control_set_finger_distance_client.wait_for_service(timeout_sec=1.0):
                move_finger_req = Float64Srv.Request()
                move_finger_req.value = float(self.ee_prepicking_finger_gap)
                await self.gripper_control_set_finger_distance_client.call_async(move_finger_req)
                self.get_logger().info("Sent move_finger request to end effector to set pre-picking finger gap.")
            else:
                self.get_logger().error("Failed to call end effector move_finger service to set pre-picking finger gap.")
                return
            
            # 6. Move CDPR to match leaf_pose
            if self.cdpr_pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
                goto_req = CdprPoseSrv.Request()
                goto_req.position[0] = float(self.leaf_pos3D[0])
                goto_req.position[1] = float(self.leaf_pos3D[1])
                goto_req.orientation = 0.0
                await self.cdpr_pathplanner_goto_pose_client.call_async(goto_req)
                self.get_logger().info("Sent goto_pose request to pathplanner for final pruning pose.")
            else:
                self.get_logger().error("Failed to call pathplanner goto_pose service for final pruning pose.")
                return
            
            # 7. Close gripper fingers to prune the leaf
            if self.gripper_control_grip_client.wait_for_service(timeout_sec=1.0):
                move_finger_req = Float64Srv.Request()
                move_finger_req.value = 0.0 # Assuming 0.0 corresponds to fully closed fingers for gripping
                await self.gripper_control_grip_client.call_async(move_finger_req)
                self.get_logger().info("Sent grip request to end effector to prune the leaf.")
            else:
                self.get_logger().error("Failed to call end effector grip service to prune the leaf.")
                return
            
            # 8. Move CDPR down and away from leaf to pluck the leaf
            if self.cdpr_pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
                goto_req = CdprPoseSrv.Request()
                goto_req.position[0] = float(self.leaf_pos3D[0])
                goto_req.position[1] = float(self.leaf_pos3D[1] - self.cdpr_picking_offset)
                goto_req.orientation = 0.0
                await self.cdpr_pathplanner_goto_pose_client.call_async(goto_req)
                self.get_logger().info("Sent goto_pose request to pathplanner to pluck the leaf.")
            else:
                self.get_logger().error("Failed to call pathplanner goto_pose service to pluck the leaf.")
                return
            
            # 9. Move TCP back into zero position
            if self.gripper_control_move_TCP_client.wait_for_service(timeout_sec=1.0):
                move_TCP_req = Float64Srv.Request()
                move_TCP_req.value = 0.0 # Move back to default TCP position
                await self.gripper_control_move_TCP_client.call_async(move_TCP_req)
                self.get_logger().info("Sent move_TCP request to end effector to reset TCP position.")
            else:
                self.get_logger().error("Failed to call end effector move_TCP service to reset TCP position.")
                return
            
            # 10. Release leaf by opening gripper fingers
            if self.gripper_control_set_finger_distance_client.wait_for_service(timeout_sec=1.0):
                move_finger_req = Float64Srv.Request()
                move_finger_req.value = float(self.ee_release_finger_gap)
                await self.gripper_control_set_finger_distance_client.call_async(move_finger_req)
                self.get_logger().info("Sent move_finger request to end effector to release the leaf.")
            else:
                self.get_logger().error("Failed to call end effector move_finger service to release the leaf.")
                return
            
            # 11. Toggle perception system back on
            if self.perception_toggle_client.wait_for_service(timeout_sec=1.0):
                toggle_req = SetBool.Request()
                toggle_req.data = True
                await self.perception_toggle_client.call_async(toggle_req)
                self.get_logger().info("Toggled perception back to active.")
            else:
                self.get_logger().error("Failed to call perception toggle service to reactivate.")
                return
            
            # 12. Toggle pathplanner search mode back on
            if self.cdpr_pathplanner_toggle_client.wait_for_service(timeout_sec=1.0):
                toggle_req = SetBool.Request()
                toggle_req.data = True
                await self.cdpr_pathplanner_toggle_client.call_async(toggle_req)
                self.get_logger().info("Toggled pathplanner search state back to active.")
            else:
                self.get_logger().error("Failed to call pathplanner search state service to reactivate.")
                return

        finally:
            self.pruning_active = False
            self.executing_sequence = False
            self.get_logger().info("Pruning sequence finished.")
        
# def main(args=None):
#     rclpy.init(args=args)
#     node = GreenwallPruningNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = GreenwallPruningNode()
    # MultiThreadedExecutor so the node can listen for service responses while the timer callback awaits
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
