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

        # Realsense Camera
        launch_ros.actions.Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            arguments=[
                "--ros-args",
                "-p", "align_depth.enable:=true"]),
            
        # Gripper control
        launch_ros.actions.Node(
            package='gripper_control_pkg',
            executable='gripper_controller',
            name='gripper_controller'),
        
        # CDPR Speed Control Joy
        launch_ros.actions.Node(
            package='cdpr_control_pkg',
            executable='cdpr_speed_control_joy',
            name='cdpr_speed_control_joy'),
        
        # CDPR Visualizer
        launch_ros.actions.Node(
            package='cdpr_control_pkg',
            executable='cdpr_visualizer',
            name='cdpr_visualizer'),
        
    ])

    # Set camera parameters
    ld.add_action(
        ExecuteProcess(
            cmd=[
                [
                    "sleep 5; ",
                    FindExecutable(name="ros2"),
                    " param set ",
                    "/camera/camera ",
                    "rgb_camera.enable_auto_exposure ",
                    "false",
                ]
            ],
            shell=True,
        ))
    
    ld.add_action(
        ExecuteProcess(
            cmd=[
                [
                    "sleep 5; ",
                    FindExecutable(name="ros2"),
                    " param set ",
                    "/camera/camera ",
                    "rgb_camera.enable_auto_white_balance ",
                    "false",
                ]
            ],
            shell=True,
        ))
    
    ld.add_action(
        ExecuteProcess(
            cmd=[
                [
                    "sleep 5; ",
                    FindExecutable(name="ros2"),
                    " param set ",
                    "/camera/camera ",
                    "rgb_camera.exposure ",
                    "500",
                ]
            ],
            shell=True,
        ))
    
    ld.add_action(
        ExecuteProcess(
            cmd=[
                [
                    "sleep 5; ",
                    FindExecutable(name="ros2"),
                    " param set ",
                    "/camera/camera ",
                    "rgb_camera.white_balance ",
                    "3500.0",
                ]
            ],
            shell=True,
        ))


    return ld