import numpy as np
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE
from matplotlib import pyplot as plt
from datetime import datetime, timedelta


motors = DynamixelSync()
motor_id = [1]

velocities = []
currents = []
positions = []
timestamps = []


VEL = 128
DURATION = 10


# Run motor test
motors.disable_torque(motor_id)
motors.write(motor_id, 1, CONTROL_TABLE.OPERATING_MODE)
motors.enable_torque(motor_id)
motors.write(motor_id, VEL, CONTROL_TABLE.GOAL_VELOCITY)

# Create timer
time_started = datetime.now()

# Collect data
while datetime.now() - time_started < timedelta(seconds=DURATION):
    # Save data
    positions.append(motors.read(motor_id, CONTROL_TABLE.PRESENT_POSITION))
    currents.append(motors.read(motor_id, CONTROL_TABLE.PRESENT_CURRENT))
    velocities.append(motors.read(motor_id, CONTROL_TABLE.PRESENT_VELOCITY))
    timestamps.append((datetime.now() - time_started).total_seconds())

motors.write(motor_id, 0, CONTROL_TABLE.GOAL_VELOCITY)
motors.disable_torque(motor_id)


# Save all readings as csv for plotting
csv_filename = 'motor_test_readings.csv'
with open(csv_filename, 'w') as f:
    f.write('timestamp,position,current,velocity\n')       
    # Zip aggregates the lists so you can iterate them simultaneously
    for t, p, c, v in zip(timestamps, positions, currents, velocities):
        f.write(f"{t},{p},{c},{v}\n")