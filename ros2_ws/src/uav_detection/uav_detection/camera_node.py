#!/usr/bin/env python3
"""
camera_node.py
Interstellar Foundry — Team 7

ROS2 Humble node for the Luxonis OAK-D Pro camera.
Uses the depthai 3.5.0 API matching radar_camera_fusion.py exactly:
  pipeline.create(dai.node.Camera).build(socket)
  cam.requestOutput(size, type, fps)
  output.createOutputQueue()          <-- no XLinkOut nodes

Hardware : OAK-D Pro → Jetson Orin Nano (USB3 blue port)
Platform : ROS2 Humble · Ubuntu 22.04 · depthai 3.5.0
"""

import time, json
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String

# depthai 3.5.0 resolution constants (match radar_camera_fusion.py)
RGB_W,   RGB_H   = 640, 480
DEPTH_W, DEPTH_H = 640, 400


class CameraNode(Node):
    """
    Publishes:
        /camera/rgb       sensor_msgs/Image  (BGR888, 640×480)
        /camera/depth     sensor_msgs/Image  (uint16 mm, 640×480)
        /camera/telemetry std_msgs/String    JSON health info
    """

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('fps',      30)
        self.declare_parameter('sim_mode', False)

        self.frame_id = self.get_parameter('frame_id').value
        self.fps      = self.get_parameter('fps').value
        self.sim_mode = self.get_parameter('sim_mode').value

        self.pub_rgb   = self.create_publisher(Image,  '/camera/rgb',       10)
        self.pub_depth = self.create_publisher(Image,  '/camera/depth',     10)
        self.pub_telem = self.create_publisher(String, '/camera/telemetry', 10)

        self.pipeline = None
        self.rgbQ     = None
        self.dispQ    = None

        if not self.sim_mode:
            self._init_oak()
        else:
            self.get_logger().warn('CameraNode: SIMULATION MODE – no OAK-D hardware required.')

        self.frame_count = 0
        self.create_timer(1.0 / self.fps, self._cb)
        self.get_logger().info(
            f'CameraNode ready | {RGB_W}×{RGB_H} @ {self.fps}fps | sim={self.sim_mode}')

    # ------------------------------------------------------------------
    # depthai 3.5.0 pipeline  (mirrors build_oak_pipeline() exactly)
    # ------------------------------------------------------------------
    def _init_oak(self):
        try:
            import depthai as dai

            pipeline = dai.Pipeline()

            # RGB – CAM_A
            camRgb    = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgbOutput = camRgb.requestOutput(
                (RGB_W, RGB_H), dai.ImgFrame.Type.BGR888p, fps=float(self.fps))

            # Stereo pair
            left       = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            leftOutput = left.requestOutput((DEPTH_W, DEPTH_H), dai.ImgFrame.Type.GRAY8,
                                            fps=float(self.fps))
            right       = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            rightOutput = right.requestOutput((DEPTH_W, DEPTH_H), dai.ImgFrame.Type.GRAY8,
                                              fps=float(self.fps))

            # Stereo depth
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(RGB_W, RGB_H)

            leftOutput.link(stereo.left)
            rightOutput.link(stereo.right)

            # Output queues – depthai 3.5 style (no XLinkOut)
            self.rgbQ  = rgbOutput.createOutputQueue()
            self.dispQ = stereo.disparity.createOutputQueue()

            pipeline.start()
            self.pipeline = pipeline
            self.get_logger().info('OAK-D Pro pipeline (depthai 3.5.0) started.')

        except ImportError:
            self.get_logger().error('depthai not installed. Run: pip3 install depthai==3.5.0')
            self.sim_mode = True
        except Exception as e:
            self.get_logger().error(f'OAK-D init failed: {e} – falling back to sim mode.')
            self.sim_mode = True

    # ------------------------------------------------------------------
    def _cb(self):
        if self.sim_mode or self.pipeline is None:
            rgb_arr, depth_arr = self._sim_frames()
        else:
            rgb_arr, depth_arr = self._capture()

        if rgb_arr is not None:
            self.pub_rgb.publish(self._to_image(rgb_arr, 'bgr8'))
        if depth_arr is not None:
            self.pub_depth.publish(self._to_image(depth_arr, '16UC1'))

        msg = String()
        msg.data = json.dumps({
            'timestamp':   time.time(),
            'frame_count': self.frame_count,
            'fps':         self.fps,
            'sim_mode':    self.sim_mode,
        })
        self.pub_telem.publish(msg)
        self.frame_count += 1

    def _capture(self):
        try:
            rgb_msg  = self.rgbQ.tryGet()
            disp_msg = self.dispQ.tryGet()
            rgb   = rgb_msg.getCvFrame()  if rgb_msg  else None
            depth = disp_msg.getFrame()   if disp_msg else None
            return rgb, depth
        except Exception as e:
            self.get_logger().warn(f'Frame capture error: {e}')
            return None, None

    def _sim_frames(self):
        rgb   = np.zeros((RGB_H, RGB_W, 3), dtype=np.uint8)
        depth = np.zeros((RGB_H, RGB_W),    dtype=np.uint16)
        t  = time.time()
        cx = int(RGB_W * (0.5 + 0.3 * np.sin(t * 0.5)))
        cy = int(RGB_H * (0.5 + 0.2 * np.cos(t * 0.4)))
        rgb  [max(0,cy-10):cy+10, max(0,cx-10):cx+10] = [0, 212, 255]
        depth[max(0,cy-10):cy+10, max(0,cx-10):cx+10] = 8000
        return rgb, depth

    def _to_image(self, arr: np.ndarray, encoding: str) -> Image:
        msg            = Image()
        msg.header     = Header()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.encoding   = encoding
        if arr.ndim == 3:
            msg.height, msg.width, ch = arr.shape
            msg.step = msg.width * ch
        else:
            msg.height, msg.width = arr.shape
            msg.step = msg.width * 2
        msg.data = arr.tobytes()
        return msg

    def destroy_node(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
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
