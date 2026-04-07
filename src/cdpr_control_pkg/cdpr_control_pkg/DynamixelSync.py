from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead
from enum import Enum

class CONTROL_TABLE(Enum):
    # NAME = (ADDRESS, LEN, IS_SIGNED)
    DRIVE_MODE = (10, 1, False)
    OPERATING_MODE = (11, 1, False)
    PROTOCOL_TYPE = (13, 1, False)
    HOMING_OFFSET = (20, 4, True)
    MAX_VOLTAGE_LIMIT = (32, 2, False)
    MIN_VOLTAGE_LIMIT = (34, 2, False)
    CURRENT_LIMIT = (38, 2, False)
    VELOCITY_LIMIT = (44, 4, False)
    TORQUE_ENABLE = (64, 1, False)
    LED = (65, 1, False)
    GOAL_CURRENT = (102, 2, True)
    GOAL_VELOCITY = (104, 4, True)
    PROFILE_ACCELERATION = (108, 4, False)
    PROFILE_VELOCITY = (112, 4, False)
    GOAL_POSITION = (116, 4, True)
    PRESENT_CURRENT = (126, 2, True)
    PRESENT_VELOCITY = (128, 4, True)
    PRESENT_POSITION = (132, 4, True)


class OPERATING_MODES(Enum):
    CURRENT_CONTROL_MODE = 0
    VELOCITY_CONTROL_MODE = 1
    POSITION_CONTROL_MODE = 3
    EXTENDED_POSITION_CONTROL_MODE = 4
    CURRENT_BASSED_POSITION_CONTROL_MODE = 5
    PWN_CONTROL_MODE = 16


class DynamixelSync:
    def __init__(self, port_name: str = "/dev/ttyUSB0", buad_rate: int = 115200, protocol_version: float = 2.0) -> None:
        self.port_handler = PortHandler(port_name=port_name)
        self.packet_handler = PacketHandler(protocol_version=protocol_version)
        self.port_handler.openPort()
        self.port_handler.setBaudRate(baudrate=buad_rate)
        self.motorDirections = {1:1, 2:1, 3:1, 4:1}

    
    def __del__(self):
        print("Dynamixel destructor")
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
        address = control_type.value[0]
        data_length = control_type.value[1]
        is_signed = control_type.value[2]

        group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, address, data_length)

        # If signed, adjust values based on motor direction
        if is_signed:
            if type(values) == int:
                values = values * self.motorDirections[motors[0]]
            elif type(values) == list:
                for i in range(len(values)):
                    values[i] = values[i] * self.motorDirections[motors[i]]

        for i in range(len(motors)):
            motor_id = self.motor_name_to_motor_id(motors[i])
            param = None
            if type(values) == int:
                try:
                    param = (values).to_bytes(data_length, 'little', signed=is_signed)
                except:
                    print("ERROR: (values, data_len)", values, ",", data_length)
            elif type(values) == list:
                try:
                    param = (values[i]).to_bytes(data_length, 'little', signed=is_signed)
                except:
                    print("ERROR: (values[i], data_len)", values[i], ",", data_length)
            elif type(values) == OPERATING_MODES:
                param = (values.value).to_bytes(data_length, 'little', signed=is_signed)
            else:
                print("ERROR: invalid values type. Got:", type(values))
            group_sync_write.addParam(motor_id, param)
        
        success = group_sync_write.txPacket()
        if success != 0:
            print("ERROR: Could not write! Got result:", success)
    

    def read(self, motors: list[int], control_type: CONTROL_TABLE) -> list:
        address = control_type.value[0]
        data_length = control_type.value[1]
        is_signed = control_type.value[2]

        group_sync_read = GroupSyncRead(self.port_handler, self.packet_handler, address, data_length)
        values = []

        for i in range(len(motors)):
            motor_id = self.motor_name_to_motor_id(motors[i])
            res = group_sync_read.addParam(motor_id)
            if res != True:
                print("ERROR: Invalid read param motor ID: " + str(motor_id))
                raise KeyError

        success = group_sync_read.txRxPacket()
        if success != 0:
            print("ERROR: Could not read. Got result:", success)
            for i in range(len(motors)):
                values.append(None)
            return values


        for i in range(len(motors)):
            motor_id = self.motor_name_to_motor_id(motors[i])
            value = group_sync_read.getData(motor_id, control_type.value[0], control_type.value[1])
            
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
        return motor_num
        
        print("ERROR: Motor number has no accociated ID")
        raise ValueError


if __name__  == "__main__":
    dmx = DynamixelSync()
    motors = [2]
    
    dmx.enable_torque(motors)
    # dmx.disable_torque(motors)
    dmx.write(motors, [-10], CONTROL_TABLE.GOAL_VELOCITY)

    # for i in range(1000):
    values = dmx.read(motors, CONTROL_TABLE.PRESENT_POSITION)
    print(values)