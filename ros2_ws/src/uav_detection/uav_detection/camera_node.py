#!/usr/bin/env python3
"""
camera_node.py
Interstellar Foundry — Team 7

Reads RGB + depth frames from the OAK-D Pro camera via DepthAI
and publishes them to ROS2 topics.

Hardware: Luxonis OAK-D Pro → Jetson Orin Nano (USB3)
Platform: ROS2 Humble · Ubuntu 22.04
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
import json
import time
import numpy as np


class CameraNode(Node):
    """
    Captures OAK-D Pro RGB and depth frames.
    Publishes:
        /camera/rgb   (sensor_msgs/Image)
        /camera/depth (sensor_msgs/Image)
    """

    def __init__(self):
        super().__init__('camera_node')

        # --- Parameters ---
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('fps', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 400)
        self.declare_parameter('sim_mode', False)

        self.frame_id = self.get_parameter('frame_id').value
        self.fps = self.get_parameter('fps').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.sim_mode = self.get_parameter('sim_mode').value

        # --- Publishers ---
        self.pub_rgb = self.create_publisher(Image, '/camera/rgb', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth', 10)
        self.pub_telem = self.create_publisher(String, '/camera/telemetry', 10)

        # --- DepthAI pipeline ---
        self.pipeline = None
        self.device = None

        if not self.sim_mode:
            self._init_oak_pipeline()
        else:
            self.get_logger().warn('Camera running in SIMULATION MODE.')

        # --- Timer ---
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        self.frame_count = 0

        self.get_logger().info(
            f'CameraNode started | {self.width}x{self.height} @ {self.fps}fps | sim={self.sim_mode}'
        )

    def _init_oak_pipeline(self):
        """Initialize the OAK-D DepthAI pipeline."""
        try:
            import depthai as dai

            pipeline = dai.Pipeline()

            # RGB camera
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setFps(self.fps)

            # Left + Right mono (for depth)
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_right = pipeline.create(dai.node.MonoCamera)
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

            # Stereo depth
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            # XLink outputs
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName('rgb')
            xout_depth.setStreamName('depth')

            cam_rgb.video.link(xout_rgb.input)
            stereo.depth.link(xout_depth.input)

            self.pipeline = pipeline
            self.device = dai.Device(pipeline)
            self.q_rgb = self.device.getOutputQueue('rgb', maxSize=4, blocking=False)
            self.q_depth = self.device.getOutputQueue('depth', maxSize=4, blocking=False)

            self.get_logger().info('OAK-D Pro pipeline initialized.')

        except ImportError:
            self.get_logger().error('depthai not installed. Run: pip install depthai')
            self.sim_mode = True
        except Exception as e:
            self.get_logger().error(f'OAK-D init error: {e}')
            self.sim_mode = True

    def timer_callback(self):
        if self.sim_mode:
            rgb_arr, depth_arr = self._generate_sim_frames()
        else:
            rgb_arr, depth_arr = self._capture_frames()

        if rgb_arr is not None:
            self.pub_rgb.publish(self._np_to_image(rgb_arr, 'rgb8'))
        if depth_arr is not None:
            self.pub_depth.publish(self._np_to_image(depth_arr, '16UC1'))

        telem = {
            'timestamp': time.time(),
            'frame_count': self.frame_count,
            'sim_mode': self.sim_mode,
            'fps': self.fps,
        }
        msg = String()
        msg.data = json.dumps(telem)
        self.pub_telem.publish(msg)
        self.frame_count += 1

    def _capture_frames(self):
        """Pull latest frames from the OAK-D queue."""
        try:
            in_rgb = self.q_rgb.tryGet()
            in_depth = self.q_depth.tryGet()
            rgb = in_rgb.getCvFrame() if in_rgb else None
            depth = in_depth.getFrame() if in_depth else None
            return rgb, depth
        except Exception as e:
            self.get_logger().warn(f'Frame capture error: {e}')
            return None, None

    def _generate_sim_frames(self):
        """Generate placeholder frames for development."""
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.uint16)
        # Add a moving dot to simulate a target
        t = time.time()
        cx = int(self.width * (0.5 + 0.3 * np.sin(t * 0.5)))
        cy = int(self.height * (0.5 + 0.2 * np.cos(t * 0.4)))
        rgb[max(0, cy-10):cy+10, max(0, cx-10):cx+10] = [0, 212, 255]
        depth[max(0, cy-10):cy+10, max(0, cx-10):cx+10] = 8000  # ~8m
        return rgb, depth

    def _np_to_image(self, arr: np.ndarray, encoding: str) -> Image:
        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.encoding = encoding
        if arr.ndim == 3:
            msg.height, msg.width, _ = arr.shape
            msg.step = arr.shape[1] * arr.shape[2]
        else:
            msg.height, msg.width = arr.shape
            msg.step = arr.shape[1] * 2  # uint16
        msg.data = arr.tobytes()
        return msg

    def destroy_node(self):
        if self.device:
            self.device.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
