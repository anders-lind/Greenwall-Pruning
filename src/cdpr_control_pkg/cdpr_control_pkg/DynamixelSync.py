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
    TORQUE_ENABLE = (64, 1, False)
    LED = (65, 1, False)
    GOAL_CURRENT = (102, 2, True)
    GOAL_VELOCITY = (104, 4, True)
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

    
    def __del__(self):
        print("dynamixel desctuctor")
        self.disable_torque(motors=[1,2,3,4])


    def write(self, motors: list[int], values: int|list[int]|OPERATING_MODES, control_type: CONTROL_TABLE) -> None:
        address = control_type.value[0]
        data_length = control_type.value[1]
        is_signed = control_type.value[2]

        group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, address, data_length)

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
            motor_id = self.motor_name_to_motor_id(motors[i])
            value = group_sync_read.getData(motor_id, control_type.value[0], control_type.value[1])
            
            if is_signed:
                bit_length = data_length * 8
                sign_bit_mask = 1 << (bit_length - 1)
                if value & sign_bit_mask:
                    value = value - (1 << bit_length)

            values.append(value)

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
    dmx = DynamixelSync()
    motors = [2]
    
    dmx.enable_torque(motors)
    # dmx.disable_torque(motors)
    dmx.write(motors, [-10], CONTROL_TABLE.GOAL_VELOCITY)

    # for i in range(1000):
    values = dmx.read(motors, CONTROL_TABLE.PRESENT_POSITION)
    print(values)
    


    # test read
    # test disable torque