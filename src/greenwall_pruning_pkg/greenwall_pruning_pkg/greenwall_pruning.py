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

        self.executing_sequence = False # Lock to prevent simultaneous sequences

        self.cdpr_prepicking_offset = 0.05 # [m]
        self.cdpr_picking_offset = 0.1 # [m]
        self.ee_prepicking_finger_gap = 0.02 # [m]
        self.ee_release_finger_gap = 0.05 # [m]

        # --- THE THREADING ARCHITECTURE ---
        # Create a dedicated callback group for the Orchestrator Service
        self.orchestrator_cb_group = MutuallyExclusiveCallbackGroup()

        # Service server (Assign to the orchestrator group!)
        self.trigger_pruning_sequence_srv = self.create_service(
            CdprPos3DSrv, 
            '/greenwall_pruning/trigger_pruning_sequence', 
            self.trigger_pruning_callback,
            callback_group=self.orchestrator_cb_group
        )

        # Service clients (Leave in the default group!)
        self.cdpr_pathplanner_toggle_client = self.create_client(SetBool, '/cdpr_pathplanner/toggle')
        self.cdpr_pathplanner_goto_pose_client = self.create_client(CdprPoseSrv, '/cdpr_pathplanner/goto_pose')
        self.perception_toggle_client = self.create_client(SetBool, '/leaf_detection/toggle')
        self.gripper_control_move_TCP_client = self.create_client(Float64Srv, '/gripper_control/move_TCP')
        self.gripper_control_set_finger_distance_client = self.create_client(Float64Srv, '/gripper_control/set_finger_distance')
        self.gripper_control_grip_client = self.create_client(Float64Srv, '/gripper_control/grip')

    async def trigger_pruning_callback(self, request, response):
        
        # Don't start if we are already in the middle of a sequence
        if self.executing_sequence:
            self.get_logger().warn("Sequence already executing. Rejecting request.")
            return response
        
        self.executing_sequence = True # Lock the sequence
        
        # Extract coordinates from the request
        leaf_pos3D = np.array([request.x, request.y, request.z])
        self.get_logger().info(f"Received pruning request for leaf at: {leaf_pos3D}")
        
        try:
            # # 1. Toggle off perception system 
            # if self.perception_toggle_client.wait_for_service(timeout_sec=1.0):
            #     toggle_req = SetBool.Request()
            #     toggle_req.data = False
            #     await self.perception_toggle_client.call_async(toggle_req)
            #     self.get_logger().info("Toggled perception to inactive.")
            # else:
            #     self.get_logger().error("Failed to call perception toggle service.")
            #     return response

            # 2. Toggle off pathplanner search mode
            if self.cdpr_pathplanner_toggle_client.wait_for_service(timeout_sec=1.0):
                toggle_req = SetBool.Request()
                toggle_req.data = False
                await self.cdpr_pathplanner_toggle_client.call_async(toggle_req)
                self.get_logger().info("Toggled pathplanner search state to inactive.")
            else:
                self.get_logger().error("Failed to call pathplanner search state service.")
                return response
            
            # 3. Move CDPR to pre-picking pose
            if self.cdpr_pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
                goto_req = CdprPoseSrv.Request()
                goto_req.position[0] = float(leaf_pos3D[0])
                goto_req.position[1] = float(leaf_pos3D[1] - self.cdpr_prepicking_offset)
                goto_req.orientation = 0.0
                await self.cdpr_pathplanner_goto_pose_client.call_async(goto_req)
                self.get_logger().info("Sent goto_pose request to pathplanner.")
            else:
                self.get_logger().error("Failed to call pathplanner goto_pose service.")
                return response
            
            # 4. Move TCP to correct match depth of detected leaf
            if self.gripper_control_move_TCP_client.wait_for_service(timeout_sec=1.0):
                move_TCP_req = Float64Srv.Request()
                move_TCP_req.value = float(leaf_pos3D[2]) 
                await self.gripper_control_move_TCP_client.call_async(move_TCP_req)
                self.get_logger().info("Sent move_TCP request to end effector.")
            else:
                self.get_logger().error("Failed to call end effector move_TCP service.")
                return response
            
            # 5. Open gripper fingers to prepare for pruning
            if self.gripper_control_set_finger_distance_client.wait_for_service(timeout_sec=1.0):
                move_finger_req = Float64Srv.Request()
                move_finger_req.value = float(self.ee_prepicking_finger_gap)
                await self.gripper_control_set_finger_distance_client.call_async(move_finger_req)
                self.get_logger().info("Sent move_finger request to end effector to set pre-picking finger gap.")
            else:
                self.get_logger().error("Failed to call end effector move_finger service to set pre-picking finger gap.")
                return response
            
            # 6. Move CDPR to match leaf_pose
            if self.cdpr_pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
                goto_req = CdprPoseSrv.Request()
                goto_req.position[0] = float(leaf_pos3D[0])
                goto_req.position[1] = float(leaf_pos3D[1])
                goto_req.orientation = 0.0
                await self.cdpr_pathplanner_goto_pose_client.call_async(goto_req)
                self.get_logger().info("Sent goto_pose request to pathplanner for final pruning pose.")
            else:
                self.get_logger().error("Failed to call pathplanner goto_pose service for final pruning pose.")
                return response
            
            # 7. Close gripper fingers to prune the leaf
            if self.gripper_control_grip_client.wait_for_service(timeout_sec=1.0):
                move_finger_req = Float64Srv.Request()
                move_finger_req.value = 0.0 # fully closed fingers
                await self.gripper_control_grip_client.call_async(move_finger_req)
                self.get_logger().info("Sent grip request to end effector to prune the leaf.")
            else:
                self.get_logger().error("Failed to call end effector grip service to prune the leaf.")
                return response
            
            # 8. Move CDPR down and away from leaf to pluck the leaf
            if self.cdpr_pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
                goto_req = CdprPoseSrv.Request()
                goto_req.position[0] = float(leaf_pos3D[0])
                goto_req.position[1] = float(leaf_pos3D[1] - self.cdpr_picking_offset)
                goto_req.orientation = 0.0
                await self.cdpr_pathplanner_goto_pose_client.call_async(goto_req)
                self.get_logger().info("Sent goto_pose request to pathplanner to pluck the leaf.")
            else:
                self.get_logger().error("Failed to call pathplanner goto_pose service to pluck the leaf.")
                return response
            
            # 9. Move TCP back into zero position
            if self.gripper_control_move_TCP_client.wait_for_service(timeout_sec=1.0):
                move_TCP_req = Float64Srv.Request()
                move_TCP_req.value = 0.0 
                await self.gripper_control_move_TCP_client.call_async(move_TCP_req)
                self.get_logger().info("Sent move_TCP request to end effector to reset TCP position.")
            else:
                self.get_logger().error("Failed to call end effector move_TCP service to reset TCP position.")
                return response
            
            # 10. Release leaf by opening gripper fingers
            if self.gripper_control_set_finger_distance_client.wait_for_service(timeout_sec=1.0):
                move_finger_req = Float64Srv.Request()
                move_finger_req.value = float(self.ee_release_finger_gap)
                await self.gripper_control_set_finger_distance_client.call_async(move_finger_req)
                self.get_logger().info("Sent move_finger request to end effector to release the leaf.")
            else:
                self.get_logger().error("Failed to call end effector move_finger service to release the leaf.")
                return response
            
            # 11. Toggle perception system back on
            if self.perception_toggle_client.wait_for_service(timeout_sec=1.0):
                toggle_req = SetBool.Request()
                toggle_req.data = True
                await self.perception_toggle_client.call_async(toggle_req)
                self.get_logger().info("Toggled perception back to active.")
            else:
                self.get_logger().error("Failed to call perception toggle service to reactivate.")
                return response
            
            # 12. Toggle pathplanner search mode back on
            if self.cdpr_pathplanner_toggle_client.wait_for_service(timeout_sec=1.0):
                toggle_req = SetBool.Request()
                toggle_req.data = True
                await self.cdpr_pathplanner_toggle_client.call_async(toggle_req)
                self.get_logger().info("Toggled pathplanner search state back to active.")
            else:
                self.get_logger().error("Failed to call pathplanner search state service to reactivate.")
                return response

        finally:
            self.executing_sequence = False
            self.get_logger().info("Pruning sequence finished.")
            
        # Return the response to the Leaf Detection Node so it knows we finished!
        return response
        
def main(args=None):
    rclpy.init(args=args)
    node = GreenwallPruningNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()