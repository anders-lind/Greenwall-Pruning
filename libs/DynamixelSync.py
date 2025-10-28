import dynamixel_sdk as dxl
from enum import Enum

class CONTROL_ADDRESS(Enum):
    # NAME = (ADDRESS, LEN)
    TORQUE_ENABLE = (64, 1)
    LED = (65, 1)
    GOAL_VELOCITY = (104, 4)
    PRESENT_POSITION = (132, 4)


class DynamixelSync:
    def __init__(self, port_name: str = "/dev/ttyUSB0", buad_rate: int = 115200, protocol_version: float = 2.0) -> None:
        self.port_handler = dxl.PortHandler(port_name=port_name)
        self.packet_handler = dxl.PacketHandler(protocol_version=protocol_version)
        self.port_handler.openPort()
        self.port_handler.setBaudRate(baudrate=buad_rate)
    

    def __del__(self):
        self.disable_torque([1,2,3,4])


    def write(self, motors: list[int], values: list, control_type: CONTROL_ADDRESS) -> None:
        if len(values) != len(motors):
            print("ERROR: Values list not same length as motors list.")
            raise ValueError
        
        group_sync_write = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, control_type.value[0], control_type.value[1])
        
        for i in range(len(motors)):
            motor_id = self.motor_name_to_motor_id(motors[i])
            param = (values[i]).to_bytes(control_type.value[1], 'little', signed=True)
            group_sync_write.addParam(motor_id, param)
        
        group_sync_write.txPacket()
    

    def read(self, motors: list[int], control_type: CONTROL_ADDRESS) -> list:
        group_sync_read = dxl.GroupSyncRead(self.port_handler, self.packet_handler, control_type.value[0], control_type.value[1])
        values = []

        for i in range(len(motors)):
            motor_id = self.motor_name_to_motor_id(motors[i])
            res = group_sync_read.addParam(motor_id)
            if res != True:
                print("ERROR: Invalid read param motor ID: " + str(motor_id))
                raise KeyError

        group_sync_read.txRxPacket()

        for i in range(len(motors)):
            motor_id = self.motor_name_to_motor_id(motors[i])
            value = group_sync_read.getData(motor_id, control_type.value[0], control_type.value[1])
            values.append(value)

        return values


    def enable_torque(self, motors) -> None:
        values = []
        for i in range(len(motors)):
            values.append(1)

        self.write(motors=motors, values=values, control_type=CONTROL_ADDRESS.TORQUE_ENABLE)


    def disable_torque(self, motors) -> None:
        values = []
        for i in range(len(motors)):
            values.append(0)

        self.write(motors=motors, values=values, control_type=CONTROL_ADDRESS.TORQUE_ENABLE)


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
    motors = [1, 2, 3, 4]
    
    # dmx.enable_torque(motors)
    # dmx.write(motors, [-10, -1 ], CONTROL_ADDRESS.GOAL_VELOCITY)

    # for i in range(1000):
    values = dmx.read(motors, CONTROL_ADDRESS.PRESENT_POSITION)
    print(values)