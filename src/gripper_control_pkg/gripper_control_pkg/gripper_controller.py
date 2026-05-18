import sys
import rclpy
import numpy as np
from rclpy.node import Node
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from plantwall_custom_interfaces.srv import Float64 as Float64Srv
from plantwall_custom_interfaces.msg import MotorState, MotorCmd
from example_interfaces.srv import SetBool
import time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


 
class GripperController(Node):
    def __init__(self, node_name = "gripper_controller"):
        super().__init__(node_name)

        # Member variables
        self.safety_check_load_threshold = 300.0 # unit is 0.1% of motor max torque
        self.grasp_speed = 5
        self.grasp_force = 250
        self.is_grasping = False
        self.speed_profile = 120 # 0.229 [rev/min]
        self.acc_profile = 10 # 214.577 [rev/min2]
        self.is_initialized = False
        self.fast_callback_is_initialized = False
        self.slow_callback_is_initialized = False

        # Motor variables
        bot_motor_ID = 12
        top_motor_ID = 11
        self.motor_IDs = [bot_motor_ID, top_motor_ID]
        # self.motor_directions = [1,-1]

        # Physical properties
        self.gear_radius = 0.015
        self.gear_circumference = 2*3.1415*self.gear_radius
        self.motor_values_per_rotation = 4095
        self.dist_to_motor_value = self.motor_values_per_rotation / self.gear_circumference
        self.max_top_pos = 0.40
        self.max_bot_pos = 0.40
        self.min_top_pos = 0
        self.min_bot_pos = 0

        # Zero inits
        self.tcp_pos = 0.0
        self.finger_distance = 0.0

        # Behavior when program crashes
        # sys.excepthook = self.myexcepthook


        blocking_callback_group = MutuallyExclusiveCallbackGroup()

        # Motor Command Publishers
        delivery_guarantee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL
        )
        self.motor_write_publisher = self.create_publisher(MotorCmd, '/dynamixel_driver/motor_cmd', 10)
        self.motor_write_continous_publisher = self.create_publisher(MotorCmd, '/dynamixel_driver/motor_cmd_continous', delivery_guarantee_qos)

        # Motor Driver Subscribers
        self.slow_subscriber = self.create_subscription(MotorState, '/dynamixel_driver/motor_state_slow', self.motor_state_slow_callback, 10)
        self.fast_subscriber = self.create_subscription(MotorState, '/dynamixel_driver/motor_state_fast', self.motor_state_fast_callback, 10)
        # Subscriber variables
        self.motor_ids = None
        self.operating_mode = None
        self.homing_offset = None
        self.current_limit = None
        self.velocity_limit = None
        self.torque_enable = None
        self.goal_current = None
        self.goal_velocity = None
        self.profile_velocity = None
        self.goal_position = None
        self.moving = None
        self.present_load = None
        self.present_position = None

        # Service servers
        self.grip_srv = self.create_service(SetBool, '/gripper_control/grip', self.grip_callback, callback_group=blocking_callback_group)
        self.set_finger_distance_srv = self.create_service(Float64Srv, '/gripper_control/set_finger_distance', self.set_finger_distance_callback, callback_group=blocking_callback_group)
        self.move_TCP_srv = self.create_service(Float64Srv, '/gripper_control/move_TCP', self.move_tcp_callback, callback_group=blocking_callback_group)

        self.get_logger().debug("Created service: \"/gripper_control/grip\"")
        self.get_logger().debug("Created service: \"/gripper_control/set_finger_distance\"")
        self.get_logger().debug("Created service: \"/gripper_control/move_TCP\"")

        self.get_logger().info(f"{node_name} Node has been started!.")
    

    def motor_state_slow_callback(self, motor_state_msg: MotorState):
        motor_ids_full = np.array(motor_state_msg.motor_id)
        indices = [np.where(motor_ids_full == id)[0][0] for id in self.motor_IDs]

        self.motor_ids = motor_ids_full[indices]
        self.operating_mode = np.array(motor_state_msg.operating_mode)[indices]
        self.homing_offset = np.array(motor_state_msg.homing_offset)[indices]
        self.current_limit = np.array(motor_state_msg.current_limit)[indices]
        self.velocity_limit = np.array(motor_state_msg.velocity_limit)[indices]
        self.goal_current = np.array(motor_state_msg.goal_current)[indices]
        self.torque_enable = np.array(motor_state_msg.torque_enable)[indices]
        self.goal_velocity = np.array(motor_state_msg.goal_velocity)[indices]
        self.profile_velocity = np.array(motor_state_msg.profile_velocity)[indices]
        self.goal_position = np.array(motor_state_msg.goal_position)[indices]
        self.moving = np.array(motor_state_msg.moving)[indices]
        
        if not self.slow_callback_is_initialized:
            self.slow_callback_is_initialized = True
        

    def motor_state_fast_callback(self, motor_state_msg: MotorState):
        motor_ids_full = np.array(motor_state_msg.motor_id)
        indices = [np.where(motor_ids_full == id)[0][0] for id in self.motor_IDs]

        self.present_load = np.array(motor_state_msg.present_current)[indices]
        self.present_position = np.array(motor_state_msg.present_position)[indices]

        if self.is_initialized:
            return

        if not self.fast_callback_is_initialized:
            self.fast_callback_is_initialized = True

        if self.slow_callback_is_initialized and self.fast_callback_is_initialized:
            self.initialize()
    

    def initialize(self):
        # Initialize motors
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[OPERATING_MODES.EXTENDED_POSITION_CONTROL_MODE], control_type_address=CONTROL_TABLE.OPERATING_MODE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[self.speed_profile], control_type_address=CONTROL_TABLE.PROFILE_VELOCITY.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[self.acc_profile], control_type_address=CONTROL_TABLE.PROFILE_ACCELERATION.value[0]))
        initial_motor_pos = self.present_position.copy()
        initial_homing_offset = self.homing_offset.copy()
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[initial_homing_offset[0]-initial_motor_pos[0], initial_homing_offset[1]-initial_motor_pos[1]], control_type_address=CONTROL_TABLE.HOMING_OFFSET.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))

        self.get_logger().info("Node is initialized")
        self.is_initialized = True



    def background_loop(self):
        if self.present_load == None:
            time.sleep(0.1)
            return

        # Check if present load is above threshold and stop if so
        self.get_logger().debug(f"Present load: {self.present_load}")
        if self.present_load[0] == None or self.present_load[1] == None:
            return
        if np.any(self.present_load > self.safety_check_load_threshold):
            self.get_logger().warn("Load above threshold! Stopping motors.")
            self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))


    def loosen_grip(self):
        self.get_logger().debug("Loosening grip")
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[OPERATING_MODES.EXTENDED_POSITION_CONTROL_MODE], control_type_address=CONTROL_TABLE.OPERATING_MODE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[self.speed_profile], control_type_address=CONTROL_TABLE.PROFILE_VELOCITY.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
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
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=motor_positions, control_type_address=CONTROL_TABLE.GOAL_POSITION.value[0]))


        # Wait for motors to reach the desired positions and stop
        present_load = self.present_load.copy()
        time.sleep(0.4) # Wait for the motors to start the movement
        moving_top, moving_bot = self.moving
        while (moving_top or moving_bot):
            present_load = self.present_load.copy()
            moving_top, moving_bot = self.moving
            time.sleep(0.01) # Do not burn the CPU
    

    def grip_callback(self, request, response):
        self.get_logger().debug(f"grip {request.data}")
        
        if not self.is_initialized:
            self.get_logger().info("Not initialized yet")
            return

        stop_gripping = not request.data
        
        # Stop gripping if requested
        if stop_gripping:
            self.loosen_grip()
            response.success = True
            return response
                
        # Go to finger distance low
        self.finger_distance = 0.01
        self.move_to_desired()
        self.finger_distance = 0.0

        # Initialize and begin grasp
        self.is_grasping = True

        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[0]                                      , control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[OPERATING_MODES.VELOCITY_CONTROL_MODE]  , control_type_address=CONTROL_TABLE.OPERATING_MODE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[1]                                      , control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
        self.motor_write_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=[self.grasp_speed, -self.grasp_speed]    , control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))
        
        # Tighten grasp until grasp force is reached on both motors
        grasp_force_exceeded = False
        while (not grasp_force_exceeded):
            # get present load
            present_load = self.present_load.copy()

            # Stop each motor as they reach desired grasp force
            move_top_motor = False
            move_bot_motor = False
            if abs(present_load[0]) < self.grasp_force:
                move_top_motor = True
            if abs(present_load[1]) < self.grasp_force:
                move_bot_motor = True
            actual_grasp_speed = [self.grasp_speed*move_top_motor, -self.grasp_speed*move_bot_motor]

            # Send motor commands
            self.motor_write_continous_publisher.publish(MotorCmd(motor_id=self.motor_IDs, value=actual_grasp_speed, control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))
            
            # Stop while-loop if both motors have reached desired grasp force
            if not move_bot_motor and not move_top_motor:
                grasp_force_exceeded = True
                break
            else:
                time.sleep(0.05)

        self.get_logger().info("Finished gripping.")

        response.success = True
        return response
    
    
    def set_finger_distance_callback(self, request, response):
        self.get_logger().debug(f"set finger distance to: {request.value}")

        if not self.is_initialized:
            self.get_logger().info("Not initialized yet")
            return

        if self.is_grasping:
            self.loosen_grip()
        
        self.finger_distance = request.value
        self.move_to_desired()

        self.get_logger().info(f"Finished setting finger distance to: {request.value:.4f} ")

        return response
            

    def move_tcp_callback(self, request, response):
        self.get_logger().debug("move tcp to: {request.value}")

        if not self.is_initialized:
            self.get_logger().info("Not initialized yet")
            return

        if self.is_grasping:
            self.loosen_grip()

        self.tcp_pos = request.value
        self.move_to_desired()

        self.get_logger().info(f"Finished moving tcp to: {request.value:.4f}")

        return response


    def myexcepthook(self, type, value, tb):
        self.get_logger().error("CRASH BEHAVIOR BEGUN")
        print("type:",type)
        print("value:",value)
        print("tb:",tb)
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