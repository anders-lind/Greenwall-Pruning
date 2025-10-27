import dynamixel_sdk as dxl
import time

portHandler = dxl.PortHandler("/dev/ttyUSB0")  # dxl port name
packetHandler = dxl.PacketHandler(2.0)         # protocol version

portHandler.openPort()
portHandler.setBaudRate(115200)

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104

packetHandler.write1ByteTxRx(portHandler, 2, ADDR_TORQUE_ENABLE, 0)
packetHandler.write1ByteTxRx(portHandler, 2, ADDR_OPERATING_MODE, 1)
packetHandler.write1ByteTxRx(portHandler, 2, ADDR_TORQUE_ENABLE, 1)

# packetHandler.write1ByteTxRx(portHandler, 2, 65, 0)  # LED on

present_position,_,_ = packetHandler.read4ByteTxRx(portHandler, 2, 132)
print(present_position)
target_position = present_position+1000

packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_VELOCITY, 20)  # Goal speed

while True:
    present_position,_,_ = packetHandler.read4ByteTxRx(portHandler, 2, 132)
    print(present_position)
    if target_position-present_position<=10:
        break   

packetHandler.write4ByteTxRx(portHandler, 2, ADDR_GOAL_VELOCITY, 0)
