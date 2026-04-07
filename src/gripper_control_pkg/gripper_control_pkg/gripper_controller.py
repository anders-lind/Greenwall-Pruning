import sys
import rclpy
import numpy as np
from rclpy.node import Node
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from plantwall_custom_interfaces.srv import FingerDistance, Grip, MoveTCP

 
class GripperController(Node):
    def __init__(self, node_name = "gripper_controller"):
        super().__init__(node_name)

        # Behavior when program crashes
        sys.excepthook = self.myexcepthook

        # Initialize motors
        self.motor_IDs = [11,12]
        self.motor_directions = [1,-1]
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=self.motor_IDs, directions=self.motor_directions)
        self.motors.disable_torque(motors=self.motor_IDs)
        self.motors.write(self.motor_IDs, OPERATING_MODES.EXTENDED_POSITION_CONTROL_MODE, CONTROL_TABLE.OPERATING_MODE)
        self.motors.write(self.motor_IDs, 100, CONTROL_TABLE.PROFILE_VELOCITY)
        self.motors.enable_torque(self.motor_IDs)
        
        # Get Initial position
        self.initial_motor_pos = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_POSITION)

        # publish services
        self.grip_srv = self.create_service(Grip, "grip", self.grip)
        self.finger_distance_srv = self.create_service(FingerDistance, "finger_distance", self.finger_distance)
        self.move_tcp_srv = self.create_service(MoveTCP, "move_tcp", self.move_tcp)

        self.get_logger().info(f"{node_name} Node has been started.")
    

    def grip(self, request, response):
        print("GRIP")
        
        response.success = True

        return response
    
    def finger_distance(self, request, response):
        print("FINGER DISTANCE: finger_dist =", request.finger_distance)

        response.success = True

        return response
    

    def move_tcp(self, request, response):
        print("MOVE TCP: z =", request.z)

        goal_pos = [self.initial_motor_pos[0]+z, self.initial_motor_pos[1]+z]
        print("goal: ", goal_pos)

        self.motors.write(self.motor_IDs, goal_pos, CONTROL_TABLE.GOAL_POSITION)

        actual_goal = self.motors.read(self.motor_IDs, CONTROL_TABLE.GOAL_POSITION)
        actual_pos = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_POSITION)
        print("actual_goal:", actual_goal)
        print("actual_pos:", actual_pos)

        response.success = True

        return response




    def myexcepthook(self, type, value, tb):
        print("CRASH BEHAVIOR BEGUN")
        print("CRASH BEHAVIOR DONE")



def main(args=None):
    rclpy.init(args=args)
    node = GripperController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()