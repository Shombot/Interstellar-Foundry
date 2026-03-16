#!/usr/bin/env python3
"""
radar_node.py
Interstellar Foundry — Team 7

Reads mmWave radar data from the DFRobot sensor over UART
and publishes point cloud and telemetry to ROS2 topics.

Hardware: DFRobot mmWave Radar → Jetson Orin Nano (UART)
Platform: ROS2 Humble · Ubuntu 22.04
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, Header
import serial
import struct
import json
import time
import threading


class RadarNode(Node):
    """
    Reads DFRobot mmWave radar frames over UART.
    Publishes:
        /radar/raw       (sensor_msgs/PointCloud2)
        /radar/telemetry (std_msgs/String JSON)
    """

    def __init__(self):
        super().__init__('radar_node')

        # --- Parameters ---
        self.declare_parameter('serial_port', '/dev/ttyTHS1')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('frame_id', 'radar_frame')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('sim_mode', False)  # Set True on Mac for dev

        self.port = self.get_parameter('serial_port').value
        self.baud = self.get_parameter('baud_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.rate_hz = self.get_parameter('publish_rate_hz').value
        self.sim_mode = self.get_parameter('sim_mode').value

        # --- Publishers ---
        self.pub_cloud = self.create_publisher(PointCloud2, '/radar/raw', 10)
        self.pub_telem = self.create_publisher(String, '/radar/telemetry', 10)

        # --- Serial connection (skip in sim mode) ---
        self.ser = None
        if not self.sim_mode:
            self._connect_serial()
        else:
            self.get_logger().warn('Radar running in SIMULATION MODE — no hardware required.')

        # --- Timer ---
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_callback)
        self.detection_count = 0

        self.get_logger().info(
            f'RadarNode started | port={self.port} | rate={self.rate_hz}Hz | sim={self.sim_mode}'
        )

    def _connect_serial(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1.0)
            self.get_logger().info(f'UART connected on {self.port} @ {self.baud} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {self.port}: {e}')
            self.get_logger().warn('Falling back to simulation mode.')
            self.sim_mode = True

    def timer_callback(self):
        if self.sim_mode:
            points = self._generate_sim_points()
        else:
            points = self._read_radar_frame()

        if points:
            cloud_msg = self._points_to_cloud(points)
            self.pub_cloud.publish(cloud_msg)

            telem = {
                'timestamp': time.time(),
                'point_count': len(points),
                'sim_mode': self.sim_mode,
            }
            telem_msg = String()
            telem_msg.data = json.dumps(telem)
            self.pub_telem.publish(telem_msg)

            self.detection_count += 1

    def _read_radar_frame(self):
        """
        Read and parse one frame from the DFRobot mmWave radar.
        The radar outputs frames as ASCII lines: 'x,y,z,snr'
        Returns list of (x, y, z, intensity) tuples in meters.
        """
        points = []
        if self.ser is None or not self.ser.is_open:
            return points
        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            for entry in line.split(';'):
                parts = entry.split(',')
                if len(parts) >= 4:
                    x, y, z, snr = map(float, parts[:4])
                    points.append((x, y, z, snr))
        except Exception as e:
            self.get_logger().warn(f'Radar parse error: {e}')
        return points

    def _generate_sim_points(self):
        """Return synthetic radar points for development/testing on Mac."""
        import math, random
        t = time.time()
        points = []
        # Simulate 2 moving UAVs
        for i, (base_angle, base_r) in enumerate([(0.8, 8.0), (2.4, 12.0)]):
            angle = base_angle + math.sin(t * 0.3 + i) * 0.3
            r = base_r + math.sin(t * 0.5 + i) * 1.5
            x = r * math.cos(angle) + random.gauss(0, 0.1)
            y = r * math.sin(angle) + random.gauss(0, 0.1)
            z = 3.0 + math.sin(t * 0.2 + i) * 0.5
            snr = 0.75 + random.gauss(0, 0.05)
            points.append((x, y, z, snr))
        return points

    def _points_to_cloud(self, points):
        """Pack (x, y, z, intensity) list into a sensor_msgs/PointCloud2."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 16
        data = b''.join(struct.pack('ffff', *p) for p in points)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(points)
        msg.data = data
        msg.is_dense = True
        return msg

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RadarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
