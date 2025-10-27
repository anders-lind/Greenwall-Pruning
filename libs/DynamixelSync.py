import dynamixel_sdk as dxl
from enum import Enum

class CONTROL_ADDRESS(Enum):
    # NAME = (ADDRESS, LEN)
    GOAL_VELOCITY = (104, 4)
    TORQUE_ENABLE = (64, 1)
    LED = (65, 1)


class DynamixelSync:
    def __init__(self, port_name: str = "/dev/ttyUSB0", buad_rate: int = 115200, protocol_version: float = 2.0) -> None:
        self.port_handler = dxl.PortHandler(port_name=port_name)
        self.packet_handler = dxl.PacketHandler(protocol_version=protocol_version)
        self.port_handler.openPort()
        self.port_handler.setBaudRate(baudrate=buad_rate)

        self.motor_num_to_id_list = [0,1,2,3]



    def write(self, motors: list[int], values: list, control_type: CONTROL_ADDRESS):
        groupSyncWrite = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, control_type.value[0], control_type.value[1])
        
        for i in range(len(motors)):
            motor_id = self.motor_to_motor_id(motors[i])
            param = (values[i]).to_bytes(control_type.value[1], 'little', signed=True)
            groupSyncWrite.addParam(motor_id, param)
        
        groupSyncWrite.txPacket()


    def enable_torque(self, motors):
        values = []
        for i in range(len(motors)):
            values.append(1)

        self.write(motors=motors, values=values, control_type=CONTROL_ADDRESS.TORQUE_ENABLE)


    def motor_to_motor_id(self, motor_num: int) -> int:
        return self.motor_num_to_id_list[motor_num]


if __name__  == "__main__":
    dmx = DynamixelSync()
    motors = [2]
    dmx.enable_torque(motors)
    dmx.write(motors, [000], CONTROL_ADDRESS.GOAL_VELOCITY)