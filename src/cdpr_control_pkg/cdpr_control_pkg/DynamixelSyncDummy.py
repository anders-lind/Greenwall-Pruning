from enum import Enum
from cdpr_control_pkg.DynamixelSync import CONTROL_TABLE, OPERATING_MODES

class DynamixelSyncDummy:
    '''
    This is a dummy version of the Dynamixel library which requires no connection to any motors.
    Reads always return 1 or the turning direction of the motor for signed values.
    '''
    def __init__(self, port_name: str = "/dev/ttyUSB0", buad_rate: int = 115200, protocol_version: float = 2.0) -> None:
        self.motorDirections = {1:1, 2:1, 3:1, 4:1}
    
    def __del__(self):
        print("Dynamixel dummy destructor")
        self.disable_torque(motors=[1,2,3,4])
    

    def setTurningDirection(self, motors: list[int], directions: list[int]) -> None:
        # Check if direction contain invalid values (anything other than -1 and 1)
        for i in range(len(directions)):
            if not (directions[i] == -1 or directions[i] == 1):
                print(f"Warning: Turning direction contains the invalid number \"{directions[i]}\"! Only -1 and 1 allowed.")
                if directions[i] >= 0:
                    directions[i] = 1
                    print(f"Warning: Turning direction have been automatically set to 1")
                else:
                    directions[i] = -1
                    print(f"Warning: Turning direction have been automatically set to -1")

        for i in range(len(motors)):
            self.motorDirections[motors[i]] = directions[i]


    def write(self, motors: list[int], values: int|list[int]|OPERATING_MODES, control_type: CONTROL_TABLE) -> None:
        pass
    

    def read(self, motors: list[int], control_type: CONTROL_TABLE) -> list:
        address = control_type.value[0]
        data_length = control_type.value[1]
        is_signed = control_type.value[2]

        values = []

        for i in range(len(motors)):
            motor_id = self.motor_name_to_motor_id(motors[i])
            value = 1
            
            if is_signed:
                bit_length = data_length * 8
                sign_bit_mask = 1 << (bit_length - 1)
                if value & sign_bit_mask:
                    value = value - (1 << bit_length)

            values.append(value)
        
        # If signed, adjust values based on motor direction
        if is_signed:
            for i in range(len(values)):
                values[i] = values[i] * self.motorDirections[motors[i]]

        return values



    def enable_torque(self, motors) -> None:
        values = []
        for i in range(len(motors)):
            values.append(1)

        self.write(motors=motors, values=values, control_type=CONTROL_TABLE.TORQUE_ENABLE)


    def disable_torque(self, motors) -> None:
        values = []
        for i in range(len(motors)):
            values.append(0)

        self.write(motors=motors, values=values, control_type=CONTROL_TABLE.TORQUE_ENABLE)


    def motor_name_to_motor_id(self, motor_num: int) -> int:
        if motor_num == 1:
            return 1
        if motor_num == 2:
            return 2
        if motor_num == 3:
            return 3
        if motor_num == 4:
            return 4
        
        print("ERROR: Motor number has no accociated ID")
        raise ValueError


if __name__  == "__main__":
    dmx = DynamixelSyncDummy()
    motors = [2,1]
    dmx.setTurningDirection(motors,[1,-11])
    
    dmx.enable_torque(motors)
    # dmx.disable_torque(motors)
    dmx.write(motors, [-10], CONTROL_TABLE.GOAL_VELOCITY)

    # for i in range(1000):
    values = dmx.read(motors, CONTROL_TABLE.GOAL_VELOCITY)
    print(values)