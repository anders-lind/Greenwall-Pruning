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
    install_requires=['setuptools', 'dynamixel_sdk', 'rclpy'],
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
            'cdpr_control_joystick_speed_control = cdpr_control_pkg.cdpr_control_joystick_speed_control:main',
            'cdpr_control_joystick_simple = cdpr_control_pkg.cdpr_control_joystick_simple:main'
        ],
    },
)