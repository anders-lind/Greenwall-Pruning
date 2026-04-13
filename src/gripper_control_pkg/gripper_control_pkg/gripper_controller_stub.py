import sys
import rclpy
import numpy as np
from rclpy.node import Node
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from plantwall_custom_interfaces.srv import Float64 as Float64Srv
import time

 
class GripperController(Node):
    def __init__(self, node_name = "gripper_controller_stub"):
        super().__init__(node_name)

        # Service servers
        self.grip_srv = self.create_service(Float64Srv, '/gripper_control/grip', self.grip)
        self.set_finger_distance_srv = self.create_service(Float64Srv, '/gripper_control/set_finger_distance', self.finger_distance)
        self.move_TCP_srv = self.create_service(Float64Srv, '/gripper_control/move_TCP', self.move_tcp)

        self.get_logger().info(f"{node_name} Node has been started.")
    

    def grip(self, request, response):
        grip_thickness = request.value
        print("GRIP: grip_thickness =", grip_thickness)
        time.sleep(1)
        return response
    
    def finger_distance(self, request, response):
        finger_distance = request.value
        print("FINGER DISTANCE: finger_distance =", finger_distance)
        time.sleep(1)
        return response

    def move_tcp(self, request, response):
        tcp_depth = request.value
        print("tcp_depth =", tcp_depth)
        time.sleep(1)
        return response



def main(args=None):
    rclpy.init(args=args)
    node = GripperController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()