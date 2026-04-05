#!/usr/bin/env python3
"""
Radar + OAK Camera Fusion Display
----------------------------------
Fuses FM24-NP100 24GHz mmWave radar with OAK stereo camera.
Shows RGB feed, depth colormap, and radar HUD overlay in real-time.

Usage:
    python3 radar_camera_fusion.py
    python3 radar_camera_fusion.py --no-radar       # camera only
    python3 radar_camera_fusion.py --no-camera       # radar only
    python3 radar_camera_fusion.py --port /dev/ttyTHS1 --baud 57600
"""

import argparse
import sys
import os
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.expanduser("~"))
from radar_display import RadarReader, SPECTRAL_BINS, MAX_SPECTRAL_VAL

# Layout
MAIN_W, MAIN_H = 640, 480
DEPTH_W = 240
CANVAS_W = MAIN_W + DEPTH_W

# Colors (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)


def build_oak_pipeline():
    """Build depthai 3.5.0 pipeline: RGB + stereo depth."""
    import depthai as dai

    pipeline = dai.Pipeline()

    # RGB camera (CAM_A)
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgbOutput = camRgb.requestOutput((MAIN_W, MAIN_H), dai.ImgFrame.Type.BGR888p, fps=30.0)

    # Stereo pair (CAM_B = left, CAM_C = right)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    leftOutput = left.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=30.0)

    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rightOutput = right.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=30.0)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(MAIN_W, MAIN_H)

    leftOutput.link(stereo.left)
    rightOutput.link(stereo.right)

    # Create output queues (depthai 3.5 — no XLinkOut nodes needed)
    rgbQ = rgbOutput.createOutputQueue()
    dispQ = stereo.disparity.createOutputQueue()

    return pipeline, rgbQ, dispQ


def draw_radar_hud(frame, data):
    dist_m = data['distance_m']
    mode = data['mode']
    frames = data['frames']
    elapsed = time.time() - data['last_frame'] if data['last_frame'] else float('inf')
    status = "LIVE" if elapsed < 2.0 else "STALE"

    if dist_m < 5.0:
        dist_color = GREEN
    elif dist_m < 15.0:
        dist_color = YELLOW
    else:
        dist_color = RED

    status_color = GREEN if status == "LIVE" else RED

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (290, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 28
    cv2.putText(frame, f"RADAR {status}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    cv2.putText(frame, f"Dist: {dist_m:.2f} m  ({data['distance_cm']} cm)", (10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)

    mode_str = 'B (dist+spec)' if mode == 'B' else 'A (dist)' if mode == 'A' else 'detecting...'
    cv2.putText(frame, f"Mode: {mode_str}", (10, y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    cv2.putText(frame, f"Frames: {frames}", (10, y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    return frame


def draw_spectrum_bar(frame, spectrum):
    h, w = frame.shape[:2]
    bar_h = 80
    bar_y0 = h - bar_h - 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, bar_y0 - 20), (w - 5, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "FMCW Spectrum", (10, bar_y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

    bin_w = max(1, (w - 20) // SPECTRAL_BINS)
    peak_idx = int(np.argmax(spectrum)) if np.any(spectrum > 0) else -1

    for i in range(SPECTRAL_BINS):
        val = float(spectrum[i])
        bh = int((val / MAX_SPECTRAL_VAL) * bar_h)
        if bh < 1:
            continue
        x = 10 + i * bin_w
        ratio = min(val / MAX_SPECTRAL_VAL, 1.0)
        color = (0, int(255 * (1 - ratio)), int(255 * ratio))
        cv2.rectangle(frame, (x, bar_y0 + bar_h - bh),
                      (x + bin_w - 1, bar_y0 + bar_h), color, -1)

    if peak_idx >= 0 and spectrum[peak_idx] > 1:
        px = 10 + peak_idx * bin_w + bin_w // 2
        cv2.putText(frame, f"pk:{peak_idx} amp:{spectrum[peak_idx]:.0f}",
                    (max(px - 30, 10), bar_y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, YELLOW, 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Radar + OAK Camera Fusion Display")
    parser.add_argument('--port', default='/dev/ttyTHS1', help='Radar serial port')
    parser.add_argument('--baud', type=int, default=57600, help='Radar baud rate')
    parser.add_argument('--no-radar', action='store_true', help='Run without radar')
    parser.add_argument('--no-camera', action='store_true', help='Run without camera')
    args = parser.parse_args()

    # --- Initialize radar ---
    radar = None
    if not args.no_radar:
        try:
            radar = RadarReader(args.port, args.baud)
            radar.connect()
            radar.start()
            print(f"Radar connected on {args.port}")
        except Exception as e:
            print(f"Radar unavailable: {e} — continuing camera-only")
            radar = None

    # --- Initialize OAK camera ---
    pipeline = None
    rgbQ = None
    dispQ = None
    if not args.no_camera:
        try:
            pipeline, rgbQ, dispQ = build_oak_pipeline()
            pipeline.start()
            print("OAK camera connected")
        except Exception as e:
            print(f"OAK camera unavailable: {e} — continuing radar-only")
            pipeline = None

    if radar is None and pipeline is None:
        print("ERROR: Neither radar nor camera available. Exiting.")
        sys.exit(1)

    cv2.namedWindow("Radar+Camera Fusion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Radar+Camera Fusion", CANVAS_W, MAIN_H)

    fps_timer = time.time()
    fps_count = 0
    display_fps = 0

    print("Running fusion display — press 'q' to quit")

    try:
        while True:
            # --- Camera frames (non-blocking) ---
            rgb_frame = None
            depth_frame = None
            if pipeline is not None:
                try:
                    rgb_msg = rgbQ.tryGet()
                    disp_msg = dispQ.tryGet()
                    if rgb_msg is not None:
                        rgb_frame = rgb_msg.getCvFrame()
                    if disp_msg is not None:
                        depth_frame = disp_msg.getFrame()
                except Exception:
                    pass

            # --- Radar data ---
            radar_data = None
            if radar is not None:
                radar_data = radar.get_data()

            # --- Main panel (RGB or placeholder) ---
            if rgb_frame is not None:
                main_panel = cv2.resize(rgb_frame, (MAIN_W, MAIN_H))
                if main_panel.ndim == 2:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_GRAY2BGR)
                elif main_panel.shape[2] == 4:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_BGRA2BGR)
            else:
                main_panel = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
                cv2.putText(main_panel, "NO CAMERA", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)

            # --- Radar overlays ---
            if radar_data is not None and radar_data['mode'] is not None:
                main_panel = draw_radar_hud(main_panel, radar_data)
                main_panel = draw_spectrum_bar(main_panel, radar_data['spectrum'])

            # --- Depth side panel ---
            if depth_frame is not None:
                depth_norm = cv2.normalize(depth_frame, None, 0, 255,
                                           cv2.NORM_MINMAX, cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                depth_panel = cv2.resize(depth_color, (DEPTH_W, MAIN_H))
            else:
                depth_panel = np.zeros((MAIN_H, DEPTH_W, 3), dtype=np.uint8)
                cv2.putText(depth_panel, "NO DEPTH", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

            # --- Composite ---
            canvas = np.hstack([main_panel, depth_panel])

            # --- FPS ---
            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                display_fps = fps_count
                fps_count = 0
                fps_timer = now
            cv2.putText(canvas, f"FPS: {display_fps}", (MAIN_W + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

            cv2.imshow("Radar+Camera Fusion", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if radar:
            radar.stop()
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()
