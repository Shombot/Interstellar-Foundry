from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'uav_detection'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team 7',
    maintainer_email='team7@interstellarfoundry.io',
    description='Modular UAV Detection System — Interstellar Foundry',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ROS2 nodes
            'radar_node        = uav_detection.radar_node:main',
            'camera_node       = uav_detection.camera_node:main',
            'fusion_node       = uav_detection.fusion_node:main',
            'detection_node    = uav_detection.detection_node:main',
            'dashboard_bridge  = uav_detection.dashboard_bridge:main',
            # Standalone scripts (run directly on Jetson without ROS2 launch)
            'radar_display     = uav_detection.radar_display:main',
            'radar_fusion      = uav_detection.radar_camera_fusion:main',
        ],
    },
)
