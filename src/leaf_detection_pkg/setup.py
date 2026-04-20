from setuptools import find_packages, setup

package_name = 'leaf_detection_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'torch', 'torchvision'],
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
            'leaf_detection = leaf_detection_pkg.leaf_detection:main',
            'leaf_detection_stub = leaf_detection_pkg.leaf_detection_stub:main'
        ],
    },
)
