from setuptools import find_packages, setup

package_name = 'gripper_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anders',
    maintainer_email='anders@lind-thomsen.dk',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gripper_controller = gripper_control_pkg.gripper_controller:main',
            'gripper_controller_stub = gripper_control_pkg.gripper_controller_stub:main',
        ],
    },
)
