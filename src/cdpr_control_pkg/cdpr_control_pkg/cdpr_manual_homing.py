import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Joy
import scipy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from plantwall_custom_interfaces.msg import MotorCmd, MotorState


class CDPRManualHomingNode(Node):
    def __init__(self):
        super().__init__('cdpr_manual_homing')

        self.velocity = 0
        self.last_buttons_state = None
        self.is_initialized = False

        delivery_guarantee_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL
        )
        self.motor_write_publisher = self.create_publisher(MotorCmd, '/dynamixel_driver/motor_cmd', delivery_guarantee_qos)
        self.motor_write_continous_publisher = self.create_publisher(MotorCmd, '/dynamixel_driver/motor_cmd_continous', 10)
        self.control_timer = self.create_timer(1, self.background_tasks)
        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.motor_state_subscriber_fast = self.create_subscription(MotorState, '/dynamixel_driver/motor_state_fast', self.motor_state_fast_callback, 10)


    def background_tasks(self):
        if not self.is_initialized:
            # Motor initialization
            self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
            self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[OPERATING_MODES.VELOCITY_CONTROL_MODE], control_type_address=CONTROL_TABLE.OPERATING_MODE.value[0]))
            self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[1], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))
            self.get_logger().info("CDPR State Initialized and ready for commands.")
            self.is_initialized = True

    def motor_state_fast_callback(self, motor_state_msg: MotorState):
        motor_ids_full = np.array(motor_state_msg.motor_id)
        indices = [np.where(motor_ids_full == id)[0][0] for id in [1,2,3,4]]

        # present_current = np.array(motor_state_msg.present_current)[indices]
        # self.get_logger().info(f"Present current: {present_current}")
        # present_position = np.array(motor_state_msg.present_position)[indices]

    def joy_callback(self, msg: Joy):
        self.handle_button_events(msg.buttons)
        # Xbox controller mapping
        x = -msg.axes[3]
        y = msg.axes[4]
        # D-pad overrides
        if abs(msg.axes[6]) > 0: x = -msg.axes[6]
        if abs(msg.axes[7]) > 0: y = msg.axes[7]
        theta = (msg.axes[5] - msg.axes[2]) / 2 
        self.input = np.array([x,y,theta])

    def handle_button_events(self, current_buttons):
            if self.last_buttons_state is None:
                self.last_buttons_state = current_buttons
                return
            
            # Button A (rising edge)
            if current_buttons[0] == 1 and self.last_buttons_state[0] == 0:
                self.get_logger().info("Button A pressed")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[4], value=[self.velocity], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # Button A (falling edge)
            if current_buttons[0] == 0 and self.last_buttons_state[0] == 1:
                self.get_logger().info("Button A released")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[4], value=[0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # Button B (rising edge)
            if current_buttons[1] == 1 and self.last_buttons_state[1] == 0:
                self.get_logger().info("Button B pressed")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[3], value=[self.velocity], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))
            
            # Button B (falling edge)
            if current_buttons[1] == 0 and self.last_buttons_state[1] == 1:
                self.get_logger().info("Button B released")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[3], value=[0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # Button X (rising edge)
            if current_buttons[2] == 1 and self.last_buttons_state[2] == 0:
                self.get_logger().info("Button X pressed")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1], value=[self.velocity], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # Button X (falling edge)
            if current_buttons[2] == 0 and self.last_buttons_state[2] == 1:
                self.get_logger().info("Button X released")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1], value=[0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # Button Y (rising edge)
            if current_buttons[3] == 1 and self.last_buttons_state[3] == 0:
                self.get_logger().info("Button Y pressed")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[2], value=[self.velocity], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # Button Y (falling edge)
            if current_buttons[3] == 0 and self.last_buttons_state[3] == 1:
                self.get_logger().info("Button Y released")
                self.motor_write_publisher.publish(MotorCmd(motor_id=[2], value=[0], control_type_address=CONTROL_TABLE.GOAL_VELOCITY.value[0]))

            # Button LB (rising edge)
            if current_buttons[4] == 1 and self.last_buttons_state[4] == 0:
                self.velocity -= 10
                if self.velocity < -128:
                    self.velocity = -128
                self.get_logger().info(f"self.velocity = {self.velocity}")

            # Button RB (rising edge)
            if current_buttons[5] == 1 and self.last_buttons_state[5] == 0:
                self.velocity += 10
                if self.velocity > 128:
                    self.velocity = 128
                self.get_logger().info(f"self.velocity = {self.velocity}")

            # Button Back
            if current_buttons[6] == 1 and self.last_buttons_state[6] == 0:
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))

            # Button Start
            if current_buttons[7] == 1 and self.last_buttons_state[7] == 0:
                self.motor_write_publisher.publish(MotorCmd(motor_id=[1,2,3,4], value=[0], control_type_address=CONTROL_TABLE.TORQUE_ENABLE.value[0]))

            self.last_buttons_state = current_buttons


def main(args=None):
    rclpy.init(args=args)
    node = CDPRManualHomingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()