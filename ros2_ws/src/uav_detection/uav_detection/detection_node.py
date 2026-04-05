#!/usr/bin/env python3
"""
detection_node.py
Interstellar Foundry — Team 7

Receives fused detection candidates from /detections, classifies
them as Group 1 / Group 2 / Unknown, and publishes classified events
to /detections/classified.

Platform: ROS2 Humble · Ubuntu 22.04
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import math


# ---------------------------------------------------------------------------
# UAV Signature Database (placeholder — extend with real radar signatures)
# ---------------------------------------------------------------------------
UAV_SIGNATURES = {
    # name: (group, typical_range_m_min, typical_range_m_max, typical_alt_m, rcs_approx)
    'DJI Mini 3 Pro':    (1, 0, 40, 1, 20,  0.01),
    'DJI Phantom 4':     (2, 0, 40, 1, 30,  0.05),
    'DJI Mavic 3':       (1, 0, 40, 1, 25,  0.02),
    'Parrot Anafi':      (1, 0, 35, 1, 20,  0.01),
    'Autel EVO Nano':    (1, 0, 30, 1, 15,  0.008),
    'Skydio 2':          (1, 0, 40, 1, 20,  0.015),
}


class DetectionNode(Node):
    """
    Classifies UAV detections into Group 1 / Group 2 / Unknown.
    Assigns bearing, confidence, and alert level.

    Publishes:
        /detections/classified  (std_msgs/String JSON)
    """

    def __init__(self):
        super().__init__('detection_node')

        # --- Parameters ---
        self.declare_parameter('snr_threshold', 0.5)
        self.declare_parameter('depth_validated_bonus', 0.15)
        self.declare_parameter('alert_range_m', 20.0)

        self.snr_threshold = self.get_parameter('snr_threshold').value
        self.depth_bonus = self.get_parameter('depth_validated_bonus').value
        self.alert_range = self.get_parameter('alert_range_m').value

        # --- Subscriber ---
        self.sub = self.create_subscription(
            String, '/detections', self.detection_callback, 10)

        # --- Publisher ---
        self.pub = self.create_publisher(String, '/detections/classified', 10)

        # --- Event counter ---
        self.event_id = 0

        self.get_logger().info('DetectionNode started.')

    def detection_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Malformed detection message.')
            return

        candidates = data.get('candidates', [])
        classified_events = []

        for c in candidates:
            if c['snr'] < self.snr_threshold:
                continue

            classified = self._classify(c)
            if classified:
                classified_events.append(classified)
                level = classified['alert_level']
                name = classified['uav_class']
                self.get_logger().info(
                    f'[{level}] {name} | range={classified["range_m"]}m '
                    f'| bearing={classified["bearing_deg"]}° '
                    f'| confidence={classified["confidence"]:.0%}'
                )

        if classified_events:
            out = String()
            out.data = json.dumps({
                'stamp': time.time(),
                'events': classified_events,
            })
            self.pub.publish(out)

    def _classify(self, candidate: dict) -> dict | None:
        """Assign UAV class and confidence to a detection candidate."""
        x, y, z = candidate['x'], candidate['y'], candidate['z']
        range_m = candidate['range_m']
        snr = candidate['snr']
        depth_ok = candidate['depth_validated']

        # Bearing (degrees from North, clockwise)
        bearing_deg = round((math.degrees(math.atan2(x, y)) + 360) % 360, 1)

        # Altitude estimate (z from radar)
        alt_m = round(z, 1)

        # Confidence: base from SNR + depth validation bonus
        confidence = min(snr + (self.depth_bonus if depth_ok else 0.0), 1.0)

        # Naive group classification by RCS / size proxy (SNR + range heuristic)
        # Group 2 drones tend to appear at higher SNR at same range
        if snr > 0.75 or range_m < 8:
            group = 2
            uav_class = 'DJI Phantom class (Grp 2)'
        else:
            group = 1
            uav_class = 'Small UAV (Grp 1)'

        # Alert level
        if range_m <= self.alert_range and not depth_ok:
            alert_level = 'THREAT'
        elif range_m <= self.alert_range * 1.5:
            alert_level = 'WARNING'
        else:
            alert_level = 'INFO'

        self.event_id += 1
        return {
            'event_id': self.event_id,
            'timestamp': candidate['timestamp'],
            'uav_class': uav_class,
            'group': group,
            'range_m': range_m,
            'altitude_m': alt_m,
            'bearing_deg': bearing_deg,
            'confidence': round(confidence, 3),
            'depth_validated': depth_ok,
            'alert_level': alert_level,
            'x': x, 'y': y, 'z': z,
        }


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
