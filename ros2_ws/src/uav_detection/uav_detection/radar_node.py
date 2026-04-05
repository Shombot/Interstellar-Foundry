#!/usr/bin/env python3
"""
radar_node.py
Interstellar Foundry — Team 7

ROS2 Humble node wrapping the FM24-NP100 24GHz mmWave radar.
Delegates all serial parsing to RadarReader from radar_display.py
(which lives in ~/ on the Jetson, matching its own import convention).

Hardware : FM24-NP100 → Jetson Orin Nano  /dev/ttyTHS1  57600 baud
Platform : ROS2 Humble · Ubuntu 22.04
"""

import sys, os, time, json, struct, math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String, Header

# radar_display.py expects to live in ~/  on the Jetson (its own convention).
# We add the package directory as a fallback so the import works both ways.
sys.path.insert(0, os.path.expanduser("~"))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from radar_display import RadarReader, SPECTRAL_BINS, MAX_SPECTRAL_VAL, DETECTION_RANGE_M


class RadarNode(Node):
    """
    Publishes:
        /radar/raw        sensor_msgs/PointCloud2   – target point at measured range
        /radar/telemetry  std_msgs/String (JSON)    – distance, spectrum, health
    """

    def __init__(self):
        super().__init__('radar_node')

        self.declare_parameter('serial_port',     '/dev/ttyTHS1')
        self.declare_parameter('baud_rate',       57600)   # FM24-NP100 default
        self.declare_parameter('frame_id',        'radar_frame')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('sim_mode',        False)

        self.port     = self.get_parameter('serial_port').value
        self.baud     = self.get_parameter('baud_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.rate_hz  = self.get_parameter('publish_rate_hz').value
        self.sim_mode = self.get_parameter('sim_mode').value

        self.pub_cloud = self.create_publisher(PointCloud2, '/radar/raw',       10)
        self.pub_telem = self.create_publisher(String,      '/radar/telemetry', 10)

        self.reader = None
        if not self.sim_mode:
            self._init_reader()
        else:
            self.get_logger().warn('RadarNode: SIMULATION MODE – synthetic FM24-NP100 data.')

        self.create_timer(1.0 / self.rate_hz, self._cb)
        self.get_logger().info(
            f'RadarNode ready | {self.port} @ {self.baud} | sim={self.sim_mode}')

    def _init_reader(self):
        try:
            self.reader = RadarReader(self.port, self.baud)
            self.reader.connect()
            self.reader.start()
            self.get_logger().info(f'FM24-NP100 connected on {self.port} @ {self.baud} baud')
        except Exception as e:
            self.get_logger().error(f'Radar init failed: {e} – falling back to sim mode.')
            self.reader = None
            self.sim_mode = True

    def _cb(self):
        data = self._sim_data() if (self.sim_mode or self.reader is None) else self.reader.get_data()
        if data['mode'] is None:
            return

        self.pub_cloud.publish(self._to_cloud(data))

        spec = data['spectrum']
        msg = String()
        msg.data = json.dumps({
            'timestamp':   data['last_frame'] or time.time(),
            'mode':        data['mode'],
            'distance_cm': data['distance_cm'],
            'distance_m':  round(data['distance_m'], 3),
            'spectrum':    spec.tolist(),
            'peak_bin':    int(np.argmax(spec)) if np.any(spec > 0) else -1,
            'peak_amp':    float(np.max(spec)),
            'frames':      data['frames'],
            'sim_mode':    self.sim_mode,
            'stale':       (time.time() - data['last_frame']) > 2.0
                           if data['last_frame'] else True,
        })
        self.pub_telem.publish(msg)

    def _to_cloud(self, data):
        """Single-point cloud at the measured range along +X."""
        dist_m   = float(data['distance_m'])
        spec     = data['spectrum']
        intensity = float(np.max(spec) / MAX_SPECTRAL_VAL) if np.any(spec > 0) else 0.0

        hdr = Header()
        hdr.stamp    = self.get_clock().now().to_msg()
        hdr.frame_id = self.frame_id

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg              = PointCloud2()
        msg.header       = hdr
        msg.height       = 1
        msg.width        = 1
        msg.fields       = fields
        msg.is_bigendian = False
        msg.point_step   = 16
        msg.row_step     = 16
        msg.data         = struct.pack('ffff', dist_m, 0.0, 0.0, intensity)
        msg.is_dense     = True
        return msg

    def _sim_data(self):
        t        = time.time()
        dist_m   = 8.0 + 4.0 * math.sin(t * 0.4)
        peak_bin = int((dist_m / DETECTION_RANGE_M) * SPECTRAL_BINS)
        x        = np.arange(SPECTRAL_BINS, dtype=np.float32)
        spectrum = (MAX_SPECTRAL_VAL * 0.85 *
                    np.exp(-0.5 * ((x - peak_bin) / 6.0) ** 2)).astype(np.float32)
        return {
            'mode':        'B',
            'distance_cm': int(dist_m * 100),
            'distance_m':  dist_m,
            'spectrum':    spectrum,
            'history':     [],
            'frames':      int(t * self.rate_hz),
            'last_frame':  t,
        }

    def destroy_node(self):
        if self.reader:
            self.reader.stop()
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
