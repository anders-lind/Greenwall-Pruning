import sys
import rclpy
import numpy as np
from rclpy.node import Node
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from plantwall_custom_interfaces.srv import Float64 as Float64Srv
from example_interfaces.srv import SetBool
import time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


 
class GripperController(Node):
    def __init__(self, node_name = "gripper_controller"):
        super().__init__(node_name)

        # Member variables
        self.safety_check_load_threshold = 300.0 # unit is 0.1% of motor max torque
        self.grasp_speed = 5
        self.grasp_force = 200
        self.is_grasping = False
        self.speed_profile = 60

        # Physical properties
        self.gear_radius = 0.015
        self.gear_circumference = 2*3.1415*self.gear_radius
        self.motor_values_per_rotation = 4095
        self.dist_to_motor_value = self.motor_values_per_rotation / self.gear_circumference
        self.max_top_pos = 0.30
        self.max_bot_pos = 0.30
        self.min_top_pos = 0
        self.min_bot_pos = 0

        # Zero inits
        self.tcp_pos = 0.0
        self.finger_distance = 0.0

        # Behavior when program crashes
        sys.excepthook = self.myexcepthook

        # Initialize motors
        top_motor_ID = 12
        bot_motor_ID = 11
        self.motor_IDs = [bot_motor_ID, top_motor_ID]
        self.motor_directions = [1,-1]
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=self.motor_IDs, directions=self.motor_directions)
        self.motors.disable_torque(motors=self.motor_IDs)
        self.motors.write(self.motor_IDs, OPERATING_MODES.EXTENDED_POSITION_CONTROL_MODE, CONTROL_TABLE.OPERATING_MODE)
        self.motors.write(self.motor_IDs, self.speed_profile, CONTROL_TABLE.PROFILE_VELOCITY)
        initial_motor_pos = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_POSITION)
        initial_homing_offset = self.motors.read(self.motor_IDs, CONTROL_TABLE.HOMING_OFFSET)
        self.motors.write(self.motor_IDs, [initial_homing_offset[0]-initial_motor_pos[0], initial_homing_offset[1]-initial_motor_pos[1]], CONTROL_TABLE.HOMING_OFFSET)
        self.motors.enable_torque(self.motor_IDs)

        blocking_callback_group = MutuallyExclusiveCallbackGroup()

        # Service servers
        self.grip_srv = self.create_service(SetBool, '/gripper_control/grip', self.grip_callback, callback_group=blocking_callback_group)
        self.set_finger_distance_srv = self.create_service(Float64Srv, '/gripper_control/set_finger_distance', self.set_finger_distance_callback, callback_group=blocking_callback_group)
        self.move_TCP_srv = self.create_service(Float64Srv, '/gripper_control/move_TCP', self.move_tcp_callback, callback_group=blocking_callback_group)

        self.background_loop_period = 0.1 # 100 Hz
        # self.background_loop = self.create_timer(self.background_loop_period, self.background_loop)


        self.get_logger().debug("Created service: \"/gripper_control/grip\"")
        self.get_logger().debug("Created service: \"/gripper_control/set_finger_distance\"")
        self.get_logger().debug("Created service: \"/gripper_control/move_TCP\"")

        self.get_logger().info(f"{node_name} Node has been started!.")


    def background_loop(self):
        # Check if present load is above threshold and stop if so
        self.get_logger().debug(self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_LOAD))
        present_load = np.array(self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_LOAD))
        if present_load[0] == None or present_load[1] == None:
            return
        if np.any(present_load > self.safety_check_load_threshold):
            self.get_logger().warn("Load above threshold! Stopping motors.")
            self.motors.disable_torque(self.motor_IDs)


    def loosen_grip(self):
        self.get_logger().debug("Loosening grip")
        self.motors.disable_torque(self.motor_IDs)
        self.motors.write(self.motor_IDs, OPERATING_MODES.EXTENDED_POSITION_CONTROL_MODE, CONTROL_TABLE.OPERATING_MODE)
        self.motors.write(self.motor_IDs, self.speed_profile, CONTROL_TABLE.PROFILE_VELOCITY)
        self.motors.enable_torque(self.motor_IDs)
        self.is_grasping = False


    def move_to_desired(self):
        self.get_logger().debug(f"Move to desired tcp: {self.tcp_pos}, and finger distance: {self.finger_distance}")

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
        time.sleep(0.1) # Wait for the motors to start the movement
        moving_top, moving_bot = self.motors.read(self.motor_IDs, CONTROL_TABLE.MOVING)
        while (moving_top or moving_bot):
            moving_top, moving_bot = self.motors.read(self.motor_IDs, CONTROL_TABLE.MOVING)
    

    def grip_callback(self, request, response):
        self.get_logger().debug(f"grip {request.data}")

        stop_gripping = not request.data
        
        # Stop gripping if requested
        if stop_gripping:
            self.loosen_grip()
            response.success = True
            return response
                
        # GO TO FINGER DISTANCE LOW
        self.finger_distance = 0.01
        self.move_to_desired()
        self.finger_distance = 0.0

        # Initialize and begin grasp
        self.is_grasping = True
        self.motors.disable_torque(self.motor_IDs)
        self.motors.write(self.motor_IDs, OPERATING_MODES.VELOCITY_CONTROL_MODE, CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(self.motor_IDs)
        self.motors.write(self.motor_IDs, [self.grasp_speed, -self.grasp_speed], CONTROL_TABLE.GOAL_VELOCITY)
        
        # Tighten grasp until grasp force is reached on both motors
        grasp_force_exceeded = False
        while (not grasp_force_exceeded):
            # get present load
            present_load = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_LOAD)

            # Stop each motor as they reach desired grasp force
            move_top_motor = False
            move_bot_motor = False
            if abs(present_load[0]) < self.grasp_force:
                move_top_motor = True
            if abs(present_load[1]) < self.grasp_force:
                move_bot_motor = True
            actual_grasp_speed = [self.grasp_speed*move_top_motor, -self.grasp_speed*move_bot_motor]

            # Send motor commands
            self.motors.write(self.motor_IDs, actual_grasp_speed, CONTROL_TABLE.GOAL_VELOCITY)
            
            # Stop while-loop if both motors have reached desired grasp force
            if not move_bot_motor and not move_top_motor:
                grasp_force_exceeded = True
                break
            else:
                time.sleep(0.01)

        response.success = True
        return response
    
    
    def set_finger_distance_callback(self, request, response):
        self.get_logger().debug(f"set finger distance to: {request.value}")

        if self.is_grasping:
            self.loosen_grip()
        
        self.finger_distance = request.value
        self.move_to_desired()

        return response
            

    def move_tcp_callback(self, request, response):
        self.get_logger().debug("move tcp to: {request.value}")

        if self.is_grasping:
            self.loosen_grip()

        self.tcp_pos = request.value
        self.move_to_desired()

        return response


    def myexcepthook(self, type, value, tb):
        self.get_logger().error("CRASH BEHAVIOR BEGUN")
        self.get_logger().error("CRASH BEHAVIOR DONE")



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