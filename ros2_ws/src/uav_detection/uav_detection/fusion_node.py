#!/usr/bin/env python3
"""
fusion_node.py
Interstellar Foundry — Team 7

Fuses FM24-NP100 radar telemetry with OAK-D depth frames.

The radar gives distance-only (no azimuth).  The camera gives a full
depth map.  This node cross-validates the radar range against the
median depth in a centre ROI and publishes unified detection candidates.

Platform : ROS2 Humble · Ubuntu 22.04
"""

import time, json, struct
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import String


class FusionNode(Node):
    """
    Subscribes:
        /radar/telemetry   std_msgs/String JSON  (from radar_node)
        /camera/depth      sensor_msgs/Image     (from camera_node)

    Publishes:
        /detections        std_msgs/String JSON  (candidates for detection_node)
    """

    def __init__(self):
        super().__init__('fusion_node')

        self.declare_parameter('depth_match_threshold_m', 2.0)   # radar vs camera range tolerance
        self.declare_parameter('radar_stale_sec',         1.0)   # ignore radar older than this
        self.declare_parameter('min_peak_amp',            2.0)   # minimum spectral peak to count

        self.depth_thr   = self.get_parameter('depth_match_threshold_m').value
        self.stale_sec   = self.get_parameter('radar_stale_sec').value
        self.min_peak    = self.get_parameter('min_peak_amp').value

        self.latest_radar = None   # most recent radar telemetry dict
        self.latest_depth = None   # most recent depth image (np uint16, mm)

        self.create_subscription(String, '/radar/telemetry', self._on_radar, 10)
        self.create_subscription(Image,  '/camera/depth',    self._on_depth, 10)

        self.pub = self.create_publisher(String, '/detections', 10)

        # Fuse at 5 Hz
        self.create_timer(0.2, self._fuse)
        self.get_logger().info('FusionNode ready.')

    # ------------------------------------------------------------------ #

    def _on_radar(self, msg: String):
        try:
            self.latest_radar = json.loads(msg.data)
        except Exception:
            pass

    def _on_depth(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
            self.latest_depth = arr
        except Exception:
            pass

    def _fuse(self):
        r = self.latest_radar
        if r is None or r.get('mode') is None:
            return

        # Drop stale radar frames
        if r.get('stale', True):
            return

        radar_dist_m = float(r['distance_m'])
        spectrum     = r.get('spectrum', [])
        peak_amp     = float(r.get('peak_amp', 0.0))

        if peak_amp < self.min_peak:
            return

        # Camera depth cross-validation: sample centre 60×60 px ROI
        depth_dist_m  = None
        depth_ok      = False
        if self.latest_depth is not None:
            d = self.latest_depth
            h, w = d.shape
            roi = d[h//2-30:h//2+30, w//2-30:w//2+30]
            nonzero = roi[roi > 0]
            if nonzero.size > 10:
                depth_dist_m = float(np.median(nonzero)) / 1000.0   # mm → m
                depth_ok     = abs(depth_dist_m - radar_dist_m) < self.depth_thr

        candidate = {
            'x':              radar_dist_m,   # best-effort: straight ahead
            'y':              0.0,
            'z':              0.0,
            'range_m':        round(radar_dist_m, 2),
            'snr':            round(min(peak_amp / 44.0, 1.0), 3),   # normalised 0-1
            'peak_bin':       r.get('peak_bin', -1),
            'peak_amp':       round(peak_amp, 1),
            'radar_mode':     r.get('mode'),
            'depth_m':        round(depth_dist_m, 2) if depth_dist_m else None,
            'depth_validated': depth_ok,
            'timestamp':      r.get('timestamp', time.time()),
        }

        out = String()
        out.data = json.dumps({
            'header':     {'stamp': time.time(), 'count': 1},
            'candidates': [candidate],
        })
        self.pub.publish(out)


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
