"""
uav_detection.launch.py
Interstellar Foundry — Team 7

Launches all nodes in the UAV detection pipeline.
Run with:
    ros2 launch uav_detection uav_detection.launch.py
    ros2 launch uav_detection uav_detection.launch.py sim_mode:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_mode_arg = DeclareLaunchArgument(
        'sim_mode',
        default_value='false',
        description='Run in simulation mode (no hardware required, useful for Mac dev via Docker)'
    )
    sim_mode = LaunchConfiguration('sim_mode')

    return LaunchDescription([
        sim_mode_arg,

        LogInfo(msg='=== Interstellar Foundry — UAV Detection System ==='),
        LogInfo(msg=['Simulation mode: ', sim_mode]),

        # --- Radar Node ---
        Node(
            package='uav_detection',
            executable='radar_node',
            name='radar_node',
            output='screen',
            parameters=[{
                'serial_port': '/dev/ttyTHS1',
                'baud_rate': 115200,
                'frame_id': 'radar_frame',
                'publish_rate_hz': 10.0,
                'sim_mode': sim_mode,
            }]
        ),

        # --- Camera Node ---
        Node(
            package='uav_detection',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{
                'frame_id': 'camera_frame',
                'fps': 30,
                'width': 640,
                'height': 400,
                'sim_mode': sim_mode,
            }]
        ),

        # --- Fusion Node ---
        Node(
            package='uav_detection',
            executable='fusion_node',
            name='fusion_node',
            output='screen',
            parameters=[{
                'depth_match_threshold_m': 1.5,
                'radar_buffer_sec': 0.5,
                'min_snr': 0.4,
            }]
        ),

        # --- Detection / Classification Node ---
        Node(
            package='uav_detection',
            executable='detection_node',
            name='detection_node',
            output='screen',
            parameters=[{
                'snr_threshold': 0.5,
                'depth_validated_bonus': 0.15,
                'alert_range_m': 20.0,
            }]
        ),

        # --- Dashboard WebSocket Bridge ---
        Node(
            package='uav_detection',
            executable='dashboard_bridge',
            name='dashboard_bridge',
            output='screen',
            parameters=[{
                'ws_port': 9090,
                'broadcast_rate_hz': 5.0,
            }]
        ),
    ])
