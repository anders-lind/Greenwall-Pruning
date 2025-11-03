from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
import time

dmx = DynamixelSync()


motors = [4]

if __name__ == "__main__":
    dmx.disable_torque(motors)
    dmx.write(motors, OPERATING_MODES.VELOCITY_CONTROL_MODE, CONTROL_TABLE.OPERATING_MODE)
    dmx.enable_torque(motors)
    dmx.write(motors, -50, CONTROL_TABLE.GOAL_VELOCITY)
    
    print(dmx.read(motors, CONTROL_TABLE.GOAL_VELOCITY))
    time.sleep(2)
    print(dmx.read(motors, CONTROL_TABLE.PRESENT_VELOCITY ))
    time.sleep(0)
    
    dmx.disable_torque(motors)
