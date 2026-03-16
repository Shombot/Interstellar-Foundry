#!/usr/bin/env python3
"""
fusion_node.py
Interstellar Foundry — Team 7

Subscribes to radar point cloud and camera depth/RGB, fuses them
into unified detection candidates, and publishes to /detections.

Platform: ROS2 Humble · Ubuntu 22.04
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import String
import json
import time
import struct
import numpy as np


class FusionNode(Node):
    """
    Fuses mmWave radar + OAK-D camera data into UAV detection candidates.

    Strategy:
    1. Maintain a rolling buffer of radar points
    2. On each camera frame, project nearby radar points into image space
    3. If radar point + depth match within threshold → confirmed candidate
    4. Publish candidate list as JSON on /detections
    """

    def __init__(self):
        super().__init__('fusion_node')

        # --- Parameters ---
        self.declare_parameter('depth_match_threshold_m', 1.5)
        self.declare_parameter('radar_buffer_sec', 0.5)
        self.declare_parameter('min_snr', 0.4)

        self.depth_threshold = self.get_parameter('depth_match_threshold_m').value
        self.buffer_sec = self.get_parameter('radar_buffer_sec').value
        self.min_snr = self.get_parameter('min_snr').value

        # --- State ---
        self.radar_buffer = []   # list of (timestamp, x, y, z, snr)
        self.latest_depth = None

        # --- Subscribers ---
        self.sub_radar = self.create_subscription(
            PointCloud2, '/radar/raw', self.radar_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth', self.depth_callback, 10)

        # --- Publisher ---
        self.pub_detections = self.create_publisher(String, '/detections', 10)

        # --- Timer: fuse at 5 Hz ---
        self.timer = self.create_timer(0.2, self.fuse_and_publish)

        self.get_logger().info('FusionNode started.')

    def radar_callback(self, msg: PointCloud2):
        """Unpack radar points and add to rolling buffer."""
        now = time.time()
        # Prune old points
        self.radar_buffer = [(t, x, y, z, s) for t, x, y, z, s in self.radar_buffer
                             if now - t < self.buffer_sec]

        # Unpack PointCloud2 (XYZI, float32 x4)
        point_step = msg.point_step
        for i in range(msg.width):
            offset = i * point_step
            x, y, z, intensity = struct.unpack_from('ffff', msg.data, offset)
            if intensity >= self.min_snr:
                self.radar_buffer.append((now, x, y, z, intensity))

    def depth_callback(self, msg: Image):
        """Store the latest depth frame (uint16, millimeters)."""
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        self.latest_depth = arr

    def fuse_and_publish(self):
        """Combine radar candidates with depth data and publish."""
        if not self.radar_buffer:
            return

        candidates = []
        for t, x, y, z, snr in self.radar_buffer:
            # Simple fusion: if depth map is available, look up depth at
            # the projected pixel and cross-validate range.
            range_m = float(np.sqrt(x**2 + y**2 + z**2))
            depth_validated = False

            if self.latest_depth is not None:
                # Naive projection: use radar azimuth/elevation to estimate pixel
                h, w = self.latest_depth.shape
                # Horizontal FoV ~70° for OAK-D → pixels per radian ≈ w / (70 * π/180)
                fov_h_rad = np.deg2rad(70)
                px_per_rad = w / fov_h_rad
                azimuth = np.arctan2(y, x)
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
                cx = int(w / 2 + azimuth * px_per_rad)
                cy = int(h / 2 - elevation * px_per_rad)
                if 0 <= cx < w and 0 <= cy < h:
                    depth_px_m = self.latest_depth[cy, cx] / 1000.0  # mm → m
                    if abs(depth_px_m - range_m) < self.depth_threshold and depth_px_m > 0:
                        depth_validated = True

            candidates.append({
                'x': round(x, 3),
                'y': round(y, 3),
                'z': round(z, 3),
                'range_m': round(range_m, 2),
                'snr': round(snr, 3),
                'depth_validated': depth_validated,
                'timestamp': t,
            })

        # Deduplicate: cluster points within 1 m of each other
        candidates = self._cluster_candidates(candidates)

        msg = String()
        msg.data = json.dumps({
            'header': {
                'stamp': time.time(),
                'count': len(candidates),
            },
            'candidates': candidates,
        })
        self.pub_detections.publish(msg)

    def _cluster_candidates(self, candidates, cluster_radius=1.0):
        """Merge detections that are within cluster_radius meters of each other."""
        if not candidates:
            return []
        merged = []
        used = [False] * len(candidates)
        for i, a in enumerate(candidates):
            if used[i]:
                continue
            cluster = [a]
            for j, b in enumerate(candidates):
                if i == j or used[j]:
                    continue
                dist = np.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)
                if dist < cluster_radius:
                    cluster.append(b)
                    used[j] = True
            # Average the cluster
            merged.append({
                'x': round(np.mean([p['x'] for p in cluster]), 3),
                'y': round(np.mean([p['y'] for p in cluster]), 3),
                'z': round(np.mean([p['z'] for p in cluster]), 3),
                'range_m': round(np.mean([p['range_m'] for p in cluster]), 2),
                'snr': round(max(p['snr'] for p in cluster), 3),
                'depth_validated': any(p['depth_validated'] for p in cluster),
                'timestamp': max(p['timestamp'] for p in cluster),
            })
            used[i] = True
        return merged


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
