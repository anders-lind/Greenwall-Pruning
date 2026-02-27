from setuptools import find_packages, setup

package_name = 'cdpr_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'dynamixel_sdk', 'rclpy', 'plantwall_custom_interfaces'],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='alexellegaard@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cdpr_speed_control = cdpr_control_pkg.cdpr_speed_control:main',
            'cdpr_speed_control_joy = cdpr_control_pkg.cdpr_speed_control_joy:main',
            'cdpr_speed_control_feedback = cdpr_control_pkg.cdpr_speed_control_feedback:main',
            'cdpr_position_and_tension_control = cdpr_control_pkg.cdpr_position_and_tension_control:main',
            'cdpr_force_control = cdpr_control_pkg.cdpr_force_control:main',
            'cdpr_manual_homing = cdpr_control_pkg.cdpr_manual_homing:main',
            'cdpr_pathplanner = cdpr_control_pkg.cdpr_pathplanner:main',
            'cdpr_visualizer = cdpr_control_pkg.cdpr_visualizer:main',
            'visualizer_tester = cdpr_control_pkg.visualizer_tester:main'
        ],
    },
)