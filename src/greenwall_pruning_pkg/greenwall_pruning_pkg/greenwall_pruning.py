#!/usr/bin/env python3

import rclpy
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

        self.pruning_active = False
        self.node_frequency = 10.0 # [Hz]
        self.node_loop_period = 1.0 / self.node_frequency

        self.cdpr_prepicking_offset = 0.05 # [m], vertical distance underneath detected leaf
        self.cdpr_picking_offset = 0.1 # [m], vertical distance to move down after gripping leaf to pluck it
        self.ee_prepicking_finger_gap = 0.02 # [m], distance between gripper fingers for pruning
        self.ee_release_finger_gap = 0.05 # [m], distance between gripper fingers for releasing leaf after pruning

        # Service servers
        self.leaf_found_srv = self.create_service(CdprPos3DSrv, '/greenwall_pruning/leaf_found', self.leaf_found_callback)

        # Service clients
        self.pathplanner_toggle_client = self.create_client(SetBool, '/pathplanner/search_state')
        self.pathplanner_goto_pose_client = self.create_client(CdprPoseSrv, '/pathplanner/goto_pose')

        self.perception_toggle_client = self.create_client(SetBool, '/perception/toggle')

        self.end_effector_move_TCP_client = self.create_client(Float64Srv, '/end_effector/move_TCP')
        self.end_effector_set_finger_distance_client = self.create_client(Float64Srv, '/end_effector/set_finger_distance')
        self.end_effector_grip_client = self.create_client(SetBool, '/end_effector/grip')
        
        self.greenwall_pruning_timer = self.create_timer(self.node_loop_period, self.greenwall_pruning_loop)

    def leaf_found_callback(self, request, response):
        leaf_pos = np.array([request.x, request.y, request.z])
        self.get_logger().info(f"Received leaf position: {leaf_pos}")
        self.pruning_active = True
        return response

    def greenwall_pruning_loop(self):
        if not self.pruning_active:
            return
        
        # 1. Toggle off perception system 
        if self.perception_toggle_client.wait_for_service(timeout_sec=1.0):
            toggle_req = SetBool.Request()
            toggle_req.data = False
            self.perception_toggle_client.call_async(toggle_req)
            self.get_logger().info("Toggled perception to inactive.")
        else:
            self.get_logger().error("Failed to call perception toggle service.")
            return

        # 2. Toggle off pathplanner search mode
        if self.pathplanner_toggle_client.wait_for_service(timeout_sec=1.0):
            toggle_req = SetBool.Request()
            toggle_req.data = False
            self.pathplanner_toggle_client.call_async(toggle_req)
            self.get_logger().info("Toggled pathplanner search state to inactive.")
        else:
            self.get_logger().error("Failed to call pathplanner search state service.")
            return
        
        # 3. Move CDPR to pre-picking pose underneath the detected leaf
        if self.pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
            goto_req = CdprPoseSrv.Request()
            goto_req.position.x = self.leaf_pos[0]
            goto_req.position.y = self.leaf_pos[1] - self.cdpr_ee_prepicking_offset
            goto_req.orientation = 0.0
            self.pathplanner_goto_pose_client.call_async(goto_req)
            self.get_logger().info("Sent goto_pose request to pathplanner.")
        else:
            self.get_logger().error("Failed to call pathplanner goto_pose service.")
            return
        
        # 4. Move TCP to correct match depth of detected leaf
        if self.end_effector_move_TCP_client.wait_for_service(timeout_sec=1.0):
            move_TCP_req = Float64Srv.Request()
            move_TCP_req.value = self.leaf_pos[2] # Assuming z coordinate of leaf position corresponds to depth
            self.end_effector_move_TCP_client.call_async(move_TCP_req)
            self.get_logger().info("Sent move_TCP request to end effector.")
        else:
            self.get_logger().error("Failed to call end effector move_TCP service.")
            return
        
        # 5. Open gripper fingers to prepare for pruning
        if self.end_effector_grip_client.wait_for_service(timeout_sec=1.0):
            move_finger_req = Float64Srv.Request()
            move_finger_req.value = self.ee_prepicking_finger_gap
            self.end_effector_grip_client.call_async(move_finger_req)
            self.get_logger().info("Sent move_finger request to end effector.")
        else:
            self.get_logger().error("Failed to call end effector move_finger service.")
            return
        
        # 6. Move CDPR to match leaf_pose
        if self.pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
            goto_req = CdprPoseSrv.Request()
            goto_req.position.x = self.leaf_pos[0]
            goto_req.position.y = self.leaf_pos[1]
            goto_req.orientation = 0.0
            self.pathplanner_goto_pose_client.call_async(goto_req)
            self.get_logger().info("Sent goto_pose request to pathplanner for final pruning pose.")
        else:
            self.get_logger().error("Failed to call pathplanner goto_pose service for final pruning pose.")
            return
        
        # 7. Close gripper fingers to prune the leaf
        if self.end_effector_grip_client.wait_for_service(timeout_sec=1.0):
            grip_req = SetBool.Request()
            grip_req.data = True
            self.end_effector_grip_client.call_async(grip_req)
            self.get_logger().info("Sent grip request to end effector to prune the leaf.")
        else:
            self.get_logger().error("Failed to call end effector grip service.")
            return
        
        # 8. Move CDPR down and away from leaf cdpr to pluck the leaf
        if self.pathplanner_goto_pose_client.wait_for_service(timeout_sec=1.0):
            goto_req = CdprPoseSrv.Request()
            goto_req.position.x = self.leaf_pos[0]
            goto_req.position.y = self.leaf_pos[1] - self.cdpr_picking_offset
            goto_req.orientation = 0.0
            self.pathplanner_goto_pose_client.call_async(goto_req)
            self.get_logger().info("Sent goto_pose request to pathplanner to pluck the leaf.")
        else:
            self.get_logger().error("Failed to call pathplanner goto_pose service to pluck the leaf.")
            return
        
        # 9. Move TCP back into zero position
        if self.end_effector_move_TCP_client.wait_for_service(timeout_sec=1.0):
            move_TCP_req = Float64Srv.Request()
            move_TCP_req.value = 0.0 # Move back to default TCP position
            self.end_effector_move_TCP_client.call_async(move_TCP_req)
            self.get_logger().info("Sent move_TCP request to end effector to reset TCP position.")
        else:
            self.get_logger().error("Failed to call end effector move_TCP service to reset TCP position.")
            return
        
        # 10. Release leaf by opening gripper fingers
        if self.end_effector_grip_client.wait_for_service(timeout_sec=1.0):
            move_finger_req = Float64Srv.Request()
            move_finger_req.value = self.ee_release_finger_gap
            self.end_effector_grip_client.call_async(move_finger_req)
            self.get_logger().info("Sent move_finger request to end effector to release the leaf.")
        else:
            self.get_logger().error("Failed to call end effector move_finger service to release the leaf.")
            return
        
        # 11. Toggle perception system back on
        if self.perception_toggle_client.wait_for_service(timeout_sec=1.0):
            toggle_req = SetBool.Request()
            toggle_req.data = True
            self.perception_toggle_client.call_async(toggle_req)
            self.get_logger().info("Toggled perception back to active.")
        else:
            self.get_logger().error("Failed to call perception toggle service to reactivate.")
            return
        
        # 12. Toggle pathplanner search mode back on
        if self.pathplanner_toggle_client.wait_for_service(timeout_sec=1.0):
            toggle_req = SetBool.Request()
            toggle_req.data = True
            self.pathplanner_toggle_client.call_async(toggle_req)
            self.get_logger().info("Toggled pathplanner search state back to active.")
        else:
            self.get_logger().error("Failed to call pathplanner search state service to reactivate.")
            return
        
        self.pruning_active = False
        self.get_logger().info("Pruning completed.")

        return
        
def main(args=None):
    rclpy.init(args=args)
    node = GreenwallPruningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
