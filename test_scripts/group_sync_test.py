import dynamixel_sdk as dxl

# Setup
portHandler = dxl.PortHandler("/dev/ttyUSB0")
packetHandler = dxl.PacketHandler(2.0)
portHandler.openPort()
portHandler.setBaudRate(115200)

# Control table addresses
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_LED = 65
LEN_GOAL_VELOCITY = 4  # bytes for velocity

# Enable torque on both motors
for dxl_id in [1, 2]:
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)

# Create GroupSyncWrite instance
groupSyncWrite = dxl.GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY)

# Example goal velocities (4-byte little-endian values)
param_1 = (100).to_bytes(4, 'little', signed=True)
param_2 = (000).to_bytes(4, 'little', signed=True)
# param_1 = (True).to_bytes(1, 'little', signed=True)
# param_2 = (True).to_bytes(1, 'little', signed=True)

# Add parameters for each motor
groupSyncWrite.addParam(1, param_1)
groupSyncWrite.addParam(2, param_2)

# Transmit to both motors at once
groupSyncWrite.txPacket()

# Clear parameters
groupSyncWrite.clearParam()