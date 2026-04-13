import sys
import rclpy
import numpy as np
from rclpy.node import Node
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from plantwall_custom_interfaces.srv import Float64 as Float64Srv
import time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


 
class GripperController(Node):
    def __init__(self, node_name = "gripper_controller"):
        super().__init__(node_name)

        # Member variables
        self.goal_distance_threshold = 100.0 # The allowed maximum deviation from the exact goal position

        # Physical properties
        self.gear_radius = 0.015
        self.gear_circumference = 2*3.1415*self.gear_radius
        self.motor_values_per_rotation = 4095
        self.dist_to_motor_value = self.motor_values_per_rotation / self.gear_circumference
        self.max_top_pos = 0.05
        self.max_bot_pos = 0.05
        self.min_top_pos = 0
        self.min_bot_pos = 0

        # Zero inits
        self.tcp_pos = 0.0
        self.finger_distance = 0.0

        # Behavior when program crashes
        # sys.excepthook = self.myexcepthook

        # Initialize motors
        self.motor_IDs = [12,11]
        self.motor_directions = [-1,1]
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=self.motor_IDs, directions=self.motor_directions)
        self.motors.disable_torque(motors=self.motor_IDs)
        self.motors.write(self.motor_IDs, OPERATING_MODES.EXTENDED_POSITION_CONTROL_MODE, CONTROL_TABLE.OPERATING_MODE)
        self.motors.write(self.motor_IDs, 100, CONTROL_TABLE.PROFILE_VELOCITY)        
        initial_motor_pos = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_POSITION)
        initial_homing_offset = self.motors.read(self.motor_IDs, CONTROL_TABLE.HOMING_OFFSET)
        self.motors.write(self.motor_IDs, [initial_homing_offset[0]-initial_motor_pos[0], initial_homing_offset[1]-initial_motor_pos[1]], CONTROL_TABLE.HOMING_OFFSET)
        self.motors.enable_torque(self.motor_IDs)

        blocking_callback_group = MutuallyExclusiveCallbackGroup()

        # Service servers
        self.grip_srv = self.create_service(Float64Srv, '/gripper_control/grip', self.grip_callback, callback_group=blocking_callback_group)
        self.set_finger_distance_srv = self.create_service(Float64Srv, '/gripper_control/set_finger_distance', self.set_finger_distance_callback, callback_group=blocking_callback_group)
        self.move_TCP_srv = self.create_service(Float64Srv, '/gripper_control/move_TCP', self.move_tcp_callback, callback_group=blocking_callback_group)

        print("Created service: \"/gripper_control/grip\"")
        print("Created service: \"/gripper_control/set_finger_distance\"")
        print("Created service: \"/gripper_control/move_TCP\"")

        self.get_logger().info(f"{node_name} Node has been started!.")
    

    def move_to_desired(self):
        print(f"Desired tcp: {self.tcp_pos}")
        print(f"Desired finger dist: {self.finger_distance}")

        # Find finger positions
        desired_top_finger_pos = self.tcp_pos - 0.5*self.finger_distance
        desired_bot_finger_pos = self.tcp_pos + 0.5*self.finger_distance

        # Clip finger positions
        desired_top_finger_pos = np.clip(desired_top_finger_pos, self.min_top_pos, self.max_top_pos)
        desired_bot_finger_pos = np.clip(desired_bot_finger_pos, self.min_bot_pos, self.max_bot_pos)

        # Send positions to motors
        motor_positions = [int(desired_top_finger_pos*self.dist_to_motor_value), int(desired_bot_finger_pos*self.dist_to_motor_value)]
        self.motors.write(self.motor_IDs, motor_positions, CONTROL_TABLE.GOAL_POSITION)

        # Wait for motors to reach the desired positions and stop
        # while True:
        #     moving_top, moving_bot = self.motors.read(self.motor_IDs, CONTROL_TABLE.MOVING)
        #     if (not moving_top and not moving_bot):
        #         print("Not moving")
        #         present_position_top, present_position_bot = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_POSITION)
        #         if abs(present_position_top - desired_top_finger_pos) < self.goal_distance_threshold:
        #             print("Top close enough")
        #             if abs(present_position_bot - desired_bot_finger_pos) < self.goal_distance_threshold:
        #                 print("Bot close enough")
        #     break
        #     # Delay to not overwork CPU
        #     time.sleep(0.1)
    

    def grip_callback(self, request, response):
        print("grip_callback:", request.value)
        
        grip_thickness = request.value
        
        # GO TO FINGER DISTANCE LOW

        # Begin grasp
        self.motors.disable_torque(self.motor_IDs)
        self.motors.write(self.motor_IDs, OPERATING_MODES.VELOCITY_CONTROL_MODE, CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(self.motor_IDs)
        self.motors.write(self.motor_IDs, [5,-5], CONTROL_TABLE.GOAL_VELOCITY)
        
        # WAIT FOR PRESENT_LOAD = (126, 2, True) HIGH (FILTER PRESENT LOAD?)
        self.motors.write(self.motor_IDs, [0,0], CONTROL_TABLE.GOAL_VELOCITY)


        print("OPERATING_MODE:", self.motors.read(self.motor_IDs, CONTROL_TABLE.OPERATING_MODE))
        print("GOAL_CURRENT:", self.motors.read(self.motor_IDs, CONTROL_TABLE.GOAL_CURRENT))

        
        return response
    
    
    def set_finger_distance_callback(self, request, response):
        print("set_finger_distance_callback:", request.value)
        
        self.finger_distance = request.value
        self.move_to_desired()

        return response
            

    def move_tcp_callback(self, request, response):
        print("move_tcp_callback:", request.value)

        self.tcp_pos = request.value
        self.move_to_desired()

        return response


    def myexcepthook(self, type, value, tb):
        print("CRASH BEHAVIOR BEGUN")
        print("CRASH BEHAVIOR DONE")



def main(args=None):
    rclpy.init(args=args)
    node = GripperController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()