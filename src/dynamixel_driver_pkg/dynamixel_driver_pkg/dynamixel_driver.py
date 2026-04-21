#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from dynamixel_driver_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from dynamixel_driver_pkg.DynamixelSyncDummy import DynamixelSyncDummy
from plantwall_custom_interfaces.msg import MotorCmd, MotorState
import numpy as np

class DynamixelDriverNode(Node):
    def __init__(self, node_name = 'dynamixel_driver'):
        super().__init__(node_name)

        # Motor initialization
        self.motors = DynamixelSync()

        self.motor_IDs =[1, 2, 3, 4, 11, 12]
        self.motors.setTurningDirection(motors=self.motor_IDs, directions=[-1,-1,1,1,1,-1])

        delivery_guarantee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL
        )

        self.create_subscription(MotorCmd, '/dynamixel_driver/motor_cmd_continous', self.motor_write_continous, 10)
        self.create_subscription(MotorCmd, '/dynamixel_driver/motor_cmd', self.motor_write, delivery_guarantee_qos)
        self.motor_state_publisher_slow = self.create_publisher(MotorState, '/dynamixel_driver/motor_state_slow', 10)
        self.motor_state_publisher_fast = self.create_publisher(MotorState, '/dynamixel_driver/motor_state_fast', 10)

        # Main loop
        self.slow_read_period = 0.25 # 4 Hz
        self.fast_read_period = 0.05 # 20 Hz
        self.control_timer = self.create_timer(self.slow_read_period, self.slow_read)
        self.control_timer = self.create_timer(self.fast_read_period, self.fast_read)

        self.get_logger().info(f"{node_name} Node has been started.")

    def slow_read(self):
        try:
            motor_state_msg = MotorState()
            motor_state_msg.motor_id = self.motor_IDs
            motor_state_msg.operating_mode = self.motors.read(self.motor_IDs, CONTROL_TABLE.OPERATING_MODE)
            motor_state_msg.homing_offset = self.motors.read(self.motor_IDs, CONTROL_TABLE.HOMING_OFFSET)
            motor_state_msg.current_limit = self.motors.read(self.motor_IDs, CONTROL_TABLE.CURRENT_LIMIT)
            motor_state_msg.velocity_limit = self.motors.read(self.motor_IDs, CONTROL_TABLE.VELOCITY_LIMIT)
            motor_state_msg.torque_enable = self.motors.read(self.motor_IDs, CONTROL_TABLE.TORQUE_ENABLE)
            motor_state_msg.goal_current = self.motors.read(self.motor_IDs, CONTROL_TABLE.GOAL_CURRENT)
            motor_state_msg.goal_velocity = self.motors.read(self.motor_IDs, CONTROL_TABLE.GOAL_VELOCITY)
            motor_state_msg.profile_velocity = self.motors.read(self.motor_IDs, CONTROL_TABLE.PROFILE_VELOCITY)
            motor_state_msg.goal_position = self.motors.read(self.motor_IDs, CONTROL_TABLE.GOAL_POSITION)
            motor_state_msg.moving = self.motors.read(self.motor_IDs, CONTROL_TABLE.MOVING)

            self.motor_state_publisher_slow.publish(motor_state_msg)
        except:
            self.get_logger().warn("Could not slow read motor states")

    def fast_read(self):
        try:
            motor_state_msg = MotorState()
            motor_state_msg.motor_id = self.motor_IDs
            motor_state_msg.present_current = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_CURRENT)
            motor_state_msg.present_position = self.motors.read(self.motor_IDs, CONTROL_TABLE.PRESENT_POSITION)

            self.motor_state_publisher_fast.publish(motor_state_msg)
        except:
            self.get_logger().warn("Could not fast read motor states")

    def motor_write(self, msg):
        if len(msg.value) == 1:
            values = np.ones(len(msg.motor_id))*msg.value[0]
            self.motors.write(list(msg.motor_id), values.astype(int).tolist(), self.get_control_table_by_address(msg.control_type_address))
            return
        self.motors.write(list(msg.motor_id), list(msg.value), self.get_control_table_by_address(msg.control_type_address))

    def motor_write_continous(self, msg):
        if len(msg.value) == 1:
            values = np.ones(len(msg.motor_id))*msg.value[0]
            self.motors.write(list(msg.motor_id), values.astype(int).tolist(), self.get_control_table_by_address(msg.control_type_address))
            return
        self.motors.write(list(msg.motor_id), list(msg.value), self.get_control_table_by_address(msg.control_type_address))
    
    def get_control_table_by_address(self, address: int) -> CONTROL_TABLE:      
        for control_type in CONTROL_TABLE:
            if control_type.value[0] == address:
                return control_type
        self.motors.write(self.motor_IDs, [0,0,0,0,0,0], CONTROL_TABLE.TORQUE_ENABLE)
        raise ValueError(f"No member in control table with address {address}")

def main(args=None):
    rclpy.init(args=args)
    node = DynamixelDriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()