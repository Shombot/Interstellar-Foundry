"""
uav_detection.launch.py
Interstellar Foundry — Team 7

Launches the full UAV detection pipeline.

Hardware:
  Radar  : FM24-NP100 24GHz mmWave → /dev/ttyTHS1 @ 57600 baud
  Camera : Luxonis OAK-D Pro (depthai 3.5.0)
  Compute: Jetson Orin Nano · Ubuntu 22.04 · ROS2 Humble

Usage:
  ros2 launch uav_detection uav_detection.launch.py
  ros2 launch uav_detection uav_detection.launch.py sim_mode:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_arg = DeclareLaunchArgument(
        'sim_mode',
        default_value='false',
        description='Simulation mode — no hardware needed (good for Mac dev via Docker)'
    )
    sim_mode = LaunchConfiguration('sim_mode')

    return LaunchDescription([
        sim_arg,
        LogInfo(msg='=== Interstellar Foundry — UAV Detection Pipeline ==='),
        LogInfo(msg=['Radar: FM24-NP100 @ /dev/ttyTHS1 57600 baud | sim=', sim_mode]),

        Node(
            package='uav_detection',
            executable='radar_node',
            name='radar_node',
            output='screen',
            parameters=[{
                'serial_port':     '/dev/ttyTHS1',
                'baud_rate':       57600,
                'frame_id':        'radar_frame',
                'publish_rate_hz': 10.0,
                'sim_mode':        sim_mode,
            }]
        ),

        Node(
            package='uav_detection',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{
                'frame_id': 'camera_frame',
                'fps':      30,
                'sim_mode': sim_mode,
            }]
        ),

        Node(
            package='uav_detection',
            executable='fusion_node',
            name='fusion_node',
            output='screen',
            parameters=[{
                'depth_match_threshold_m': 2.0,
                'radar_stale_sec':         1.0,
                'min_peak_amp':            2.0,
            }]
        ),

        Node(
            package='uav_detection',
            executable='detection_node',
            name='detection_node',
            output='screen',
            parameters=[{
                'snr_threshold':        0.5,
                'depth_validated_bonus': 0.15,
                'alert_range_m':        20.0,
            }]
        ),

        Node(
            package='uav_detection',
            executable='dashboard_bridge',
            name='dashboard_bridge',
            output='screen',
            parameters=[{
                'ws_port':           9090,
                'broadcast_rate_hz': 5.0,
            }]
        ),
    ])
