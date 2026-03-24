#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Joy
from cdpr_control_pkg.DynamixelSync import DynamixelSync, CONTROL_TABLE, OPERATING_MODES
from cdpr_control_pkg.DynamixelSyncDummy import DynamixelSyncDummy
from cdpr_control_pkg.cdpr_force_control import CDPRForceControlNode
from plantwall_custom_interfaces.msg import CdprPose
import scipy
import matplotlib.pyplot as plt
import time


class CDPRForceControlNodeJoy(CDPRForceControlNode):
    def __init__(self):
        super().__init__('cdpr_force_control_joy')

        ## Input type
        self.USE_PATHPLANNER = False
        self.USE_JOY = True


def main(args=None):
    rclpy.init(args=args)
    node = CDPRForceControlNodeJoy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()