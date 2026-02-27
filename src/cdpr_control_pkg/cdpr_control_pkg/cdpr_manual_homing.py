import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
import scipy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE




class CDPRManualHomingNode(Node):
    def __init__(self):
        super().__init__('cdpr_manual_homing')

        self.velocity = 0
        self.last_buttons_state = None

        # Motor initialization
        self.motors = DynamixelSync()
        self.motors.setTurningDirection(motors=[1,2,3,4], directions=[-1,-1,1,1])

        self.motors.disable_torque(motors=[1,2,3,4])
        self.motors.write(motors=[1,2,3,4], values=1, control_type=CONTROL_TABLE.OPERATING_MODE)
        self.motors.enable_torque(motors=[1,2,3,4])

        self.joy_subscriber = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

    
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
            # print(f"1: {self.motors.read(motors=[1], control_type=CONTROL_TABLE.PRESENT_POSITION)}")
            # print(f"2: {self.motors.read(motors=[2], control_type=CONTROL_TABLE.PRESENT_POSITION)}")
            # print(f"3: {self.motors.read(motors=[3], control_type=CONTROL_TABLE.PRESENT_POSITION)}")
            # print(f"4: {self.motors.read(motors=[4], control_type=CONTROL_TABLE.PRESENT_POSITION)}")



            if self.last_buttons_state is None:
                self.last_buttons_state = current_buttons
                return
            
            # Button A (rising edge)
            if current_buttons[0] == 1 and self.last_buttons_state[0] == 0:
                self.motors.write(
                    motors=[4],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[self.velocity]
                )
            
            # Button A (falling edge)
            if current_buttons[0] == 0 and self.last_buttons_state[0] == 1:
                self.motors.write(
                    motors=[4],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[0]
                )

            # Button B (rising edge)
            if current_buttons[1] == 1 and self.last_buttons_state[1] == 0:
                self.motors.enable_torque(motors=[3])
                self.motors.write(
                    motors=[3],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[self.velocity]
                )
            
            # Button B (falling edge)
            if current_buttons[1] == 0 and self.last_buttons_state[1] == 1:
                self.motors.write(
                    motors=[3],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[0]
                )

            # Button X (rising edge)
            if current_buttons[2] == 1 and self.last_buttons_state[2] == 0:
                self.motors.write(
                    motors=[1],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[self.velocity]
                )
            # Button X (falling edge)
            if current_buttons[2] == 0 and self.last_buttons_state[2] == 1:
                self.motors.write(
                    motors=[1],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[0]
                )

            # Button Y (rising edge)
            if current_buttons[3] == 1 and self.last_buttons_state[3] == 0:
                self.motors.write(
                    motors=[2],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[self.velocity]
                )
            # Button Y (falling edge)
            if current_buttons[3] == 0 and self.last_buttons_state[3] == 1:
                self.motors.write(
                    motors=[2],
                    control_type=CONTROL_TABLE.GOAL_VELOCITY,
                    values=[0]
                )

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
                self.motors.disable_torque([1,2,3,4])
            
            # Button Start
            if current_buttons[7] == 1 and self.last_buttons_state[7] == 0:
                self.motors.disable_torque([1,2,3,4])

            self.last_buttons_state = current_buttons


def main(args=None):
    rclpy.init(args=args)
    node = CDPRManualHomingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()