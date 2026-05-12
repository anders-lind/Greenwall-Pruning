import launch
import launch_ros.actions
from launch.actions import ExecuteProcess
from launch.substitutions import FindExecutable

def generate_launch_description():
    ld = launch.LaunchDescription([

        # Dynamixel Driver
        launch_ros.actions.Node(
            package='dynamixel_driver_pkg',
            executable='dynamixel_driver',
            name='dynamixel_driver'),
        
         # Joy
        launch_ros.actions.Node(
            package='joy',
            executable='joy_node',
            name='joy_node'),
        
        # CDPR Manual Homing
        launch_ros.actions.Node(
            package='cdpr_control_pkg',
            executable='cdpr_manual_homing',
            name='cdpr_manual_homing'),
        
    ])

    return ld