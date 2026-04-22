#!/usr/bin/env python3
"""
Drone Detector — YOLO + Doppler + Stereo Depth + Kalman Fusion
---------------------------------------------------------------
Architecture:
  CAMERA (DETECT + TRACK):
    YOLOv6n on OAK Myriad X VPU detects and classifies airborne objects.
    OAK stereo depth provides 3D position for each detection.
    KCF tracker maintains lock on confirmed drone targets.

  RADAR (RANGE):
    CQRobot 10.525GHz Doppler microwave sensor estimates target range
    from signal-envelope analysis (duty cycle + pulse frequency → depth).
    Also provides motion confirmation and micro-Doppler signatures.

  KALMAN FUSION:
    EKF fuses camera (x, depth) with Doppler radar (range estimate)
    to produce smoothed position, velocity, bearing, and confidence.

Dashboard layout (OpenCV):
  ┌──────────────────┬──────────┐
  │  CAMERA           │  Depth   │
  │  DETECT + TRACK   │  Map     │
  ├──────────────────┴──────────┤
  │  RADAR — RANGING  │  Fused  │
  │  range + waveform │  Scope  │
  └───────────────────┴─────────┘

Usage:
    python3 drone_detector.py                 # full fusion (camera + Doppler)
    python3 drone_detector.py --no-doppler    # camera only
    python3 drone_detector.py --no-camera     # Doppler only
"""

import argparse
import sys
import os
import time
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import cv2

# Hardware imports — only needed when not in video mode
# (guarded in main() so --video works without depthai/doppler installed)


# Simple detection object matching depthai's format for video mode
VideoDet = namedtuple('VideoDet', ['xmin', 'ymin', 'xmax', 'ymax', 'label', 'confidence'])

# Layout
MAIN_W, MAIN_H = 640, 480
NN_W, NN_H = 512, 288
DEPTH_W = 240
BOTTOM_H = 160  # height of bottom sensor panels
CANVAS_W = MAIN_W + DEPTH_W
CANVAS_H = MAIN_H + BOTTOM_H

# Colors (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
MAGENTA = (255, 0, 255)
DARK_GREEN = (0, 100, 0)

DRONE_THRESHOLD = 0.30

# Model paths
SCRIPT_DIR = Path(__file__).parent
BLOB_PATH = SCRIPT_DIR / "yolov6n_model" / "drone_v3.blob"
CONFIG_PATH = SCRIPT_DIR / "yolov6n_model" / "config.json"

# Single-class drone detector (nc=1, class 0 = drone)
NUM_CLASSES = 1
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45


def load_coco_labels():
    return ["drone"]


def parse_yolov8_raw(nn_data):
    """Parse raw YOLOv8 NN output into a list of detection dicts.
    nc=1: output shape (1, 5, N) where 5 = 4 bbox + 1 class score.
    Returns list of dicts with keys: xmin, ymin, xmax, ymax, confidence, label.
    """
    output = np.array(nn_data.getFirstLayerFp16()).reshape(NUM_CLASSES + 4, -1).T
    boxes = output[:, :4]
    scores = output[:, 4:]

    # nc=1: single score column, class always 0
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences > CONF_THRESHOLD
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # xywh to xyxy (normalized by NN input size)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x - w / 2) / NN_W
    y1 = (y - h / 2) / NN_H
    x2 = (x + w / 2) / NN_W
    y2 = (y + h / 2) / NN_H

    # NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=list(zip(x1, y1, x2 - x1, y2 - y1)),
        scores=confidences.tolist(),
        score_threshold=CONF_THRESHOLD,
        nms_threshold=IOU_THRESHOLD,
    )
    if len(indices) == 0:
        return []

    detections = []
    for i in indices.flatten():
        det = argparse.Namespace(
            xmin=float(np.clip(x1[i], 0, 1)),
            ymin=float(np.clip(y1[i], 0, 1)),
            xmax=float(np.clip(x2[i], 0, 1)),
            ymax=float(np.clip(y2[i], 0, 1)),
            confidence=float(confidences[i]),
            label=int(class_ids[i]),
        )
        detections.append(det)
    return detections


# ---------------------------------------------------------------------------
# Camera-only drone scorer
# ---------------------------------------------------------------------------
class DroneScorer:
    def score(self, yolo_label, yolo_conf, depth_m):
        # nc=1: class 0 = drone, the only possible class
        if yolo_label == 0:
            return min(0.95, 0.7 + yolo_conf * 0.25), f"drone {yolo_conf:.0%}"
        return 0.0, "unknown"


# ---------------------------------------------------------------------------
# Drawing — Separate sensor panels
# ---------------------------------------------------------------------------
def draw_drone_boxes(frame, fused_tracks):
    """Draw bounding boxes from fused tracks projected back to camera."""
    for tid, ft in fused_tracks.items():
        if not ft.is_confirmed:
            continue

        # Project fused position back to pixel coords
        # x_m → pixel_x, range → approximate box size
        px = int(MAIN_W / 2 + ft.x * (MAIN_W / 2) / max(ft.y, 1.0))
        py = int(MAIN_H / 2)  # approximate vertical center
        box_size = max(20, int(200 / max(ft.y, 1.0)))  # farther = smaller

        x1 = max(0, px - box_size // 2)
        y1 = max(0, py - box_size // 2)
        x2 = min(MAIN_W, px + box_size // 2)
        y2 = min(MAIN_H, py + box_size // 2)

        is_drone = ft.confidence >= DRONE_THRESHOLD

        # Source indicators
        sources = []
        if ft.has_camera:
            sources.append("CAM")
        if ft.has_doppler:
            sources.append("DOP")
        src_str = "+".join(sources)

        depth_str = f" {ft.range_m:.1f}m" if ft.range_m > 0 else ""

        if is_drone:
            label = f"DRONE {ft.confidence:.0%}{depth_str} [{src_str}]"
            box_color = RED
            cv2.drawMarker(frame, (px, py), CYAN, cv2.MARKER_CROSS, 12, 1)
        else:
            label = f"TGT {ft.confidence:.0%}{depth_str}"
            box_color = YELLOW

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

        # Velocity vector
        if abs(ft.vx) > 0.1 or abs(ft.vy) > 0.1:
            vx_px = int(ft.vx * 10)
            vy_px = int(-ft.vy * 10)  # screen Y is inverted
            cv2.arrowedLine(frame, (px, py), (px + vx_px, py + vy_px),
                            GREEN, 1, tipLength=0.3)
    return frame


def draw_camera_yolo_boxes(frame, yolo_detections, labels, depth_frame,
                           num_tracked):
    """Draw raw YOLO detections with DETECT labels, and camera mode badge."""
    h_frame, w_frame = frame.shape[:2]

    # Camera role badge — top-left
    mode_str = f"CAMERA — DETECT+TRACK" if num_tracked > 0 else "CAMERA — DETECTING"
    mode_col = CYAN if num_tracked > 0 else GREEN
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, MAIN_H - 22), (220, MAIN_H - 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, mode_str, (8, MAIN_H - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, mode_col, 1)

    for det in yolo_detections:
        cx = int((det.xmin + det.xmax) / 2 * MAIN_W)
        cy = int((det.ymin + det.ymax) / 2 * MAIN_H)
        w = int((det.xmax - det.xmin) * MAIN_W)
        h = int((det.ymax - det.ymin) * MAIN_H)
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2

        lbl_idx = det.label
        if isinstance(labels, dict):
            lbl_name = labels.get(lbl_idx, "?")
        else:
            lbl_name = labels[lbl_idx] if lbl_idx < len(labels) else "?"
        conf = det.confidence

        # Single-class drone model: all detections are drones
        color = CYAN
        tag = "DET"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"{tag} {lbl_name} {conf:.0%}" if tag else f"{lbl_name} {conf:.0%}"
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return frame


def draw_hud(frame, num_drones, detection_mode, nn_dets, doppler_ok,
             doppler_range=0.0):
    h, w = frame.shape[:2]
    panel_w, panel_h = 340, 80
    px, py = w - panel_w - 5, 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if num_drones > 0:
        status = f"TRACKING {num_drones} DRONE{'S' if num_drones > 1 else ''}"
        cv2.putText(frame, status, (px + 8, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
    else:
        cv2.putText(frame, "SCANNING...", (px + 8, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

    # Sensor roles
    cv2.putText(frame, f"CAM: DETECT+TRACK  YOLO:{nn_dets}",
                (px + 8, py + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)

    doppler_col = GREEN if doppler_ok else GRAY
    if doppler_ok and doppler_range > 0.1:
        cv2.putText(frame, f"RADAR: RANGING  {doppler_range:.1f}m",
                    (px + 8, py + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, MAGENTA, 1)
    else:
        radar_str = "RANGING" if doppler_ok else "OFF"
        cv2.putText(frame, f"RADAR: {radar_str}",
                    (px + 8, py + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, doppler_col, 1)

    cv2.putText(frame, "KALMAN FUSED", (px + 8, py + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)

    if num_drones > 0 and int(time.time() * 3) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 3)

    return frame


def draw_doppler_panel(panel, doppler_data):
    """
    Draw Doppler radar panel: range estimate, motion state, pulse waveform.
    Role: RADAR RANGING — provides depth estimates from signal envelope.
    """
    h, w = panel.shape[:2]
    panel[:] = (10, 14, 20)

    cv2.putText(panel, "RADAR — 10.525GHz DOPPLER", (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, MAGENTA, 1)
    cv2.putText(panel, "RANGING", (w - 70, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)

    if doppler_data is None or not doppler_data['connected']:
        cv2.putText(panel, "NO SENSOR", (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
        return panel

    motion = doppler_data['motion']
    events = doppler_data['motion_events']
    freq = doppler_data['pulse_freq_hz']
    activity = doppler_data['activity_level']
    age = doppler_data['motion_age_s']
    est_range = doppler_data.get('estimated_range_m', 0.0)
    range_conf = doppler_data.get('range_confidence', 0.0)
    duty = doppler_data.get('duty_cycle', 0.0)

    # --- Range estimate (primary readout) ---
    if est_range > 0.1 and range_conf > 0.05:
        if est_range < 3:
            range_col = RED
        elif est_range < 7:
            range_col = YELLOW
        else:
            range_col = GREEN
        cv2.putText(panel, f"{est_range:.1f}m", (5, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, range_col, 2)
        cv2.putText(panel, f"conf:{range_conf:.0%}", (110, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRAY, 1)
    else:
        cv2.putText(panel, "-- m", (5, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 2)
        cv2.putText(panel, "no return", (110, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRAY, 1)

    # Range bar
    bar_x, bar_y, bar_w, bar_h = 5, 44, w - 10, 8
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (30, 40, 50), -1)
    if est_range > 0.1:
        fill_w = int(min(est_range / 10.0, 1.0) * bar_w)
        range_col2 = RED if est_range < 3 else YELLOW if est_range < 7 else GREEN
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      range_col2, -1)

    # Motion + stats row
    motion_str = "MOTION" if motion else "IDLE"
    motion_col = RED if motion else DARK_GREEN
    cv2.circle(panel, (8, 62), 4, motion_col, -1)
    cv2.putText(panel, motion_str, (18, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, motion_col, 1)
    cv2.putText(panel, f"Freq:{freq:.1f}Hz  Duty:{duty:.0%}  Evt:{events}",
                (100, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.25, GRAY, 1)

    # Activity bar
    bar_y2 = 72
    cv2.rectangle(panel, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + 6),
                  (30, 40, 50), -1)
    fill_w = int(activity * bar_w)
    act_col = RED if activity > 0.7 else YELLOW if activity > 0.3 else GREEN
    cv2.rectangle(panel, (bar_x, bar_y2), (bar_x + fill_w, bar_y2 + 6),
                  act_col, -1)
    cv2.putText(panel, f"Signal: {activity:.0%}", (5, bar_y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, GRAY, 1)

    # Drone signature indicator
    is_drone, d_conf = False, 0.0
    if freq > 5.0 and activity > 0.3:
        is_drone = True
        d_conf = min(1.0, freq / 30.0) * activity
    elif motion and freq > 2.0 and activity > 0.5:
        is_drone = True
        d_conf = 0.3

    if is_drone:
        cv2.putText(panel, f"DRONE SIG {d_conf:.0%}",
                    (w - 110, bar_y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1)

    # Pulse history waveform
    history = doppler_data.get('history', [])
    wave_y0 = 100
    wave_h = h - wave_y0 - 5
    cv2.putText(panel, "Pulse Waveform", (5, wave_y0 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, GRAY, 1)
    cv2.putText(panel, f"Last: {age:.1f}s ago", (w - 90, wave_y0 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, GRAY, 1)
    if history:
        now = time.time()
        recent = [(t, m) for t, m in history if now - t < 3.0]
        if recent:
            t0 = recent[0][0]
            span = max(recent[-1][0] - t0, 0.1)
            for i in range(1, len(recent)):
                x1 = int(5 + (recent[i-1][0] - t0) / span * (w - 10))
                x2 = int(5 + (recent[i][0] - t0) / span * (w - 10))
                y1 = wave_y0 + (0 if recent[i-1][1] else wave_h)
                y2 = wave_y0 + (0 if recent[i][1] else wave_h)
                cv2.line(panel, (x1, y1), (x2, y2), MAGENTA, 1)

    return panel


def draw_fused_scope(panel, fused_tracks, sweep_angle):
    """
    Draw fused radar scope showing Kalman-filtered drone positions.
    Shows bearing and range from the sensor node.
    """
    h, w = panel.shape[:2]
    panel[:] = (10, 14, 20)

    cv2.putText(panel, "KALMAN FUSED SCOPE", (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREEN, 1)

    # Scope geometry
    radius = min(w, h) // 2 - 15
    cx, cy = w // 2, h // 2 + 10

    # Range rings
    for frac in (0.33, 0.66, 1.0):
        r = int(radius * frac)
        cv2.circle(panel, (cx, cy), r, (0, 40, 0), 1)
    cv2.line(panel, (cx - radius, cy), (cx + radius, cy), (0, 40, 0), 1)
    cv2.line(panel, (cx, cy - radius), (cx, cy + radius), (0, 40, 0), 1)

    # Sweep line
    sx = int(cx + radius * np.cos(sweep_angle))
    sy = int(cy - radius * np.sin(sweep_angle))
    cv2.line(panel, (cx, cy), (sx, sy), (0, 150, 0), 1)
    for i in range(1, 4):
        a = sweep_angle - i * 0.15
        tx = int(cx + radius * np.cos(a))
        ty = int(cy - radius * np.sin(a))
        intensity = max(0, 150 - i * 40)
        cv2.line(panel, (cx, cy), (tx, ty), (0, intensity, 0), 1)

    # Range labels
    cv2.putText(panel, "7m", (cx + int(radius * 0.33) - 6, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 60, 0), 1)
    cv2.putText(panel, "14m", (cx + int(radius * 0.66) - 8, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 60, 0), 1)
    cv2.putText(panel, "20m", (cx + radius - 12, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 60, 0), 1)

    # Plot fused tracks
    max_range = 20.0
    for tid, ft in fused_tracks.items():
        if not ft.is_confirmed:
            continue

        # Bearing → angle on scope, range → distance from center
        bearing_rad = np.radians(ft.bearing_deg)
        d_frac = min(ft.range_m / max_range, 1.0)
        dist_px = int(d_frac * radius)

        bx = cx + int(dist_px * np.sin(bearing_rad))
        by = cy - int(dist_px * np.cos(bearing_rad))

        # Color by confidence
        if ft.confidence > 0.6:
            blip_col = RED
        elif ft.confidence > 0.3:
            blip_col = YELLOW
        else:
            blip_col = GREEN

        cv2.circle(panel, (bx, by), 5, blip_col, -1)
        cv2.circle(panel, (bx, by), 8, blip_col, 1)

        # Source indicator dots
        dot_y = by + 12
        if ft.has_camera:
            cv2.circle(panel, (bx - 4, dot_y), 2, CYAN, -1)
        if ft.has_doppler:
            cv2.circle(panel, (bx + 4, dot_y), 2, MAGENTA, -1)

        # Range label
        cv2.putText(panel, f"{ft.range_m:.1f}m",
                    (bx + 10, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1)

        # Velocity vector
        if abs(ft.vx) > 0.1 or abs(ft.vy) > 0.1:
            vx_px = int(ft.vx * 5)
            vy_px = int(-ft.vy * 5)
            cv2.arrowedLine(panel, (bx, by), (bx + vx_px, by + vy_px),
                            GREEN, 1, tipLength=0.3)

    # Origin node
    cv2.circle(panel, (cx, cy), 4, CYAN, -1)
    cv2.circle(panel, (cx, cy), 6, CYAN, 1)

    # Legend
    cv2.circle(panel, (5, h - 8), 2, CYAN, -1)
    cv2.putText(panel, "CAM", (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, GRAY, 1)
    cv2.circle(panel, (45, h - 8), 2, MAGENTA, -1)
    cv2.putText(panel, "DOP", (50, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, GRAY, 1)

    # Outer ring
    cv2.circle(panel, (cx, cy), radius, GREEN, 1)
    return panel


# ---------------------------------------------------------------------------
# OAK pipeline — YOLO on VPU + stereo depth
# ---------------------------------------------------------------------------
def build_oak_pipeline():
    import depthai as dai

    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgbOut = camRgb.requestOutput((MAIN_W, MAIN_H),
                                  dai.ImgFrame.Type.BGR888p, fps=30.0)

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    leftOut = left.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=30.0)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rightOut = right.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=30.0)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(MAIN_W, MAIN_H)
    leftOut.link(stereo.left)
    rightOut.link(stereo.right)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(NN_W, NN_H)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    rgbOut.link(manip.inputImage)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(BLOB_PATH))
    nn.setNumInferenceThreads(2)
    manip.out.link(nn.input)

    rgbQ = rgbOut.createOutputQueue(maxSize=1, blocking=False)
    depthQ = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

    # YOLOv8 raw NN output, parsed on host via parse_yolov8_raw()
    nnQ = nn.out.createOutputQueue(maxSize=1, blocking=False)
    return pipeline, rgbQ, depthQ, nnQ


def build_oak_pipeline_no_nn():
    import depthai as dai

    pipeline = dai.Pipeline()

    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgbOut = camRgb.requestOutput((MAIN_W, MAIN_H),
                                  dai.ImgFrame.Type.BGR888p, fps=30.0)

    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    leftOut = left.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=30.0)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    rightOut = right.requestOutput((640, 400), dai.ImgFrame.Type.GRAY8, fps=30.0)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(MAIN_W, MAIN_H)
    leftOut.link(stereo.left)
    rightOut.link(stereo.right)

    rgbQ = rgbOut.createOutputQueue(maxSize=1, blocking=False)
    depthQ = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

    return pipeline, rgbQ, depthQ, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Drone Detector — YOLO + Doppler Kalman Fusion")
    parser.add_argument('--no-doppler', action='store_true', help='Disable Doppler sensor')
    parser.add_argument('--no-camera', action='store_true', help='Disable OAK camera')
    parser.add_argument('--doppler-pin', type=int, default=105,
                        help='GPIO line for Doppler sensor (default: 105 = Pin 29)')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file — runs YOLO on CPU, no hardware needed')
    args = parser.parse_args()

    video_mode = args.video is not None

    # --- Video mode: load YOLO via ultralytics + open video file ---
    video_cap = None
    yolo_model = None
    if video_mode:
        if not os.path.isfile(args.video):
            print(f"ERROR: Video file not found: {args.video}")
            sys.exit(1)
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: ultralytics not installed. Run: pip3 install ultralytics")
            sys.exit(1)
        video_cap = cv2.VideoCapture(args.video)
        if not video_cap.isOpened():
            print(f"ERROR: Cannot open video: {args.video}")
            sys.exit(1)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {args.video} ({total_frames} frames @ {video_fps:.0f}fps)")
        # Prefer fine-tuned model (drone class), fall back to stock COCO
        FINETUNED_PT = SCRIPT_DIR / "training" / "runs" / "drone_phase2" / "weights" / "best.pt"
        # Check for newest fine-tuned weights first
        DOWNLOAD_CANDIDATES = [
            Path("/Users/rasho/Downloads/best (3).pt"),
            Path("/Users/rasho/Downloads/best (2).pt"),
            Path("/Users/rasho/Downloads/best (1).pt"),
            Path("/Users/rasho/Downloads/best.pt"),
        ]
        loaded = False
        if FINETUNED_PT.exists():
            yolo_model = YOLO(str(FINETUNED_PT))
            print("YOLOv8n (drone-finetuned) loaded for CPU inference")
            loaded = True
        else:
            for pt in DOWNLOAD_CANDIDATES:
                if pt.exists():
                    yolo_model = YOLO(str(pt))
                    print(f"YOLOv8n loaded from {pt.name} for CPU inference")
                    loaded = True
                    break
        if not loaded:
            yolo_model = YOLO("yolov8n.pt")
            print("WARNING: Fine-tuned model not found, using stock YOLOv8n (no drone class)")

    # --- Load labels ---
    if video_mode:
        labels = yolo_model.names  # dict {0: 'drone'}
    else:
        labels = load_coco_labels()

    # --- Initialize hardware sensors (skip in video mode) ---
    doppler = None
    pipeline = rgbQ = depthQ = detQ = None
    has_nn = False

    if not video_mode:
        # Import hardware modules only when needed
        sys.path.insert(0, os.path.expanduser("~"))
        from doppler_reader import DopplerReader

        if not args.no_doppler:
            try:
                doppler = DopplerReader(line=args.doppler_pin)
                doppler.start()
                print(f"CQRobot 10.525GHz Doppler sensor started (GPIO line {args.doppler_pin})")
            except Exception as e:
                print(f"Doppler sensor unavailable: {e} -- continuing without Doppler")

        if not args.no_camera:
            try:
                if BLOB_PATH.exists():
                    pipeline, rgbQ, depthQ, detQ = build_oak_pipeline()
                    has_nn = True
                    print("OAK camera + YOLOv8n drone detector on Myriad X VPU")
                else:
                    pipeline, rgbQ, depthQ, detQ = build_oak_pipeline_no_nn()
                    print("OAK camera (no YOLO blob -- depth-only mode)")
                pipeline.start()
            except Exception as e:
                print(f"OAK camera error: {e}")
                if has_nn:
                    try:
                        pipeline, rgbQ, depthQ, detQ = build_oak_pipeline_no_nn()
                        pipeline.start()
                        has_nn = False
                        print("Fell back to depth-only pipeline")
                    except Exception as e2:
                        print(f"Fallback also failed: {e2}")
                        pipeline = None

        if pipeline is None and doppler is None:
            print("ERROR: No sensors available.")
            sys.exit(1)

    # --- Kalman fusion tracker ---
    from kalman_fusion import MultiDroneTracker
    scorer = DroneScorer()
    dt = 1.0 / (video_fps if video_mode else 30.0)
    fusion_tracker = MultiDroneTracker(dt=dt, confirm_frames=5)

    cv2.namedWindow("Drone Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Detector", CANVAS_W, CANVAS_H)

    fps_timer = time.time()
    fps_count = 0
    display_fps = 0
    last_rgb_frame = None
    last_depth_frame = None
    sweep_angle = 0.0

    mode_str = "VIDEO" if video_mode else "Camera + Doppler Kalman Fusion"
    print("=" * 60)
    print(f"  Drone Detector — {mode_str}")
    if video_mode:
        print(f"  Source:  {args.video}")
        print(f"  YOLO:   YOLOv8n (CPU)")
    else:
        print(f"  Camera:  {'ON' if pipeline else 'OFF'}")
        print(f"  Doppler: {'ON' if doppler else 'OFF'}")
    print("  Press 'q' to quit")
    print("=" * 60)

    try:
        while True:
            # --- Get frames + detections ---
            rgb_frame = depth_frame = None
            yolo_detections = []
            doppler_data = None

            if video_mode:
                ret, frame = video_cap.read()
                if not ret:
                    # Loop video
                    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = video_cap.read()
                    if not ret:
                        break
                rgb_frame = cv2.resize(frame, (MAIN_W, MAIN_H))

                # Run YOLO on CPU
                results = yolo_model(rgb_frame, verbose=False, conf=CONF_THRESHOLD)
                for box in results[0].boxes:
                    x1n = float(box.xyxyn[0][0])
                    y1n = float(box.xyxyn[0][1])
                    x2n = float(box.xyxyn[0][2])
                    y2n = float(box.xyxyn[0][3])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    yolo_detections.append(VideoDet(x1n, y1n, x2n, y2n, cls, conf))

            else:
                # --- OAK-D camera ---
                if pipeline is not None:
                    try:
                        rgb_msg = rgbQ.tryGet()
                        depth_msg = depthQ.tryGet()
                        if rgb_msg is not None:
                            last_rgb_frame = rgb_msg.getCvFrame()
                        if depth_msg is not None:
                            last_depth_frame = depth_msg.getFrame()
                        rgb_frame = last_rgb_frame
                        depth_frame = last_depth_frame

                        if detQ is not None:
                            det_msg = detQ.tryGet()
                            if det_msg is not None:
                                yolo_detections = parse_yolov8_raw(det_msg)
                    except Exception:
                        pass

                # --- Doppler data ---
                if doppler is not None:
                    doppler_data = doppler.get_data()

            # --- Build camera detection candidates for Kalman filter ---
            camera_dets = []  # (x_m, depth_m, confidence, cx_px, cy_px, w_px, h_px, src)
            detection_mode = "FUSION"
            nn_det_count = len(yolo_detections)

            has_depth = depth_frame is not None
            has_rgb = rgb_frame is not None

            def get_depth_at(cx, cy, w, h):
                if not has_depth:
                    return 0.0
                r = max(5, min(w, h) // 4)
                dy = np.clip(cy, r, depth_frame.shape[0] - r - 1)
                dx = np.clip(cx, r, depth_frame.shape[1] - r - 1)
                region = depth_frame[dy-r:dy+r+1, dx-r:dx+r+1]
                valid = region[region > 0]
                return float(np.median(valid)) / 1000.0 if len(valid) > 0 else 0.0

            # --- YOLO detections → camera_dets for Kalman ---
            if has_rgb and yolo_detections:
                gray_frame = cv2.cvtColor(
                    cv2.resize(rgb_frame, (MAIN_W, MAIN_H)),
                    cv2.COLOR_BGR2GRAY)
                for det in yolo_detections:
                    cx = int((det.xmin + det.xmax) / 2 * MAIN_W)
                    cy = int((det.ymin + det.ymax) / 2 * MAIN_H)
                    w = int((det.xmax - det.xmin) * MAIN_W)
                    h = int((det.ymax - det.ymin) * MAIN_H)
                    yolo_label = det.label
                    yolo_conf = det.confidence

                    drone_score, reason = scorer.score(yolo_label, yolo_conf, 0.0)
                    if drone_score == 0.0:
                        continue

                    # Reject bright light sources
                    r = max(3, min(w, h) // 6)
                    ry = np.clip(cy, r, gray_frame.shape[0] - r - 1)
                    rx = np.clip(cx, r, gray_frame.shape[1] - r - 1)
                    patch = gray_frame[ry-r:ry+r+1, rx-r:rx+r+1]
                    if patch.size > 0:
                        mean_val = float(np.mean(patch))
                        bright_ratio = np.sum(patch > 200) / patch.size
                        if mean_val > 200 or bright_ratio > 0.3:
                            continue

                    det_depth_m = get_depth_at(cx, cy, w, h)

                    # Convert pixel x to metres offset from center
                    # Using camera FOV (~70 deg horizontal for OAK-D)
                    hfov_rad = np.radians(70)
                    x_norm = (cx - MAIN_W / 2) / (MAIN_W / 2)  # -1 to +1
                    x_m = x_norm * det_depth_m * np.tan(hfov_rad / 2) if det_depth_m > 0 else x_norm * 5.0

                    if isinstance(labels, dict):
                        lbl_name = labels.get(yolo_label, "?")
                    else:
                        lbl_name = labels[yolo_label] if yolo_label < len(labels) else "?"
                    camera_dets.append((x_m, det_depth_m, drone_score,
                                        cx, cy, w, h, f"Y:{lbl_name}"))

            # --- Run Kalman fusion update ---
            fused_tracks = fusion_tracker.update(
                camera_dets=[(d[0], d[1], d[2]) for d in camera_dets],
                doppler_data=doppler_data,
            )

            # === BUILD DISPLAY ===

            # --- Top row: main camera + depth ---
            if has_rgb:
                main_panel = cv2.resize(rgb_frame, (MAIN_W, MAIN_H)).copy()
                if main_panel.ndim == 2:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_GRAY2BGR)
                elif main_panel.shape[2] == 4:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_BGRA2BGR)
                gray_check = cv2.cvtColor(main_panel, cv2.COLOR_BGR2GRAY)
                if np.mean(gray_check) < 80:
                    lab = cv2.cvtColor(main_panel, cv2.COLOR_BGR2LAB)
                    l_ch, a_ch, b_ch = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l_ch = clahe.apply(l_ch)
                    main_panel = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]),
                                              cv2.COLOR_LAB2BGR)
            else:
                main_panel = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
                cv2.putText(main_panel, "NO CAMERA", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)

            # Count confirmed tracks for display
            num_drones = sum(1 for ft in fused_tracks.values()
                             if ft.is_confirmed and ft.confidence >= DRONE_THRESHOLD)

            # Draw raw YOLO boxes on camera panel
            if has_rgb and yolo_detections:
                main_panel = draw_camera_yolo_boxes(main_panel, yolo_detections,
                                                     labels, depth_frame,
                                                     num_drones)

            # Draw fused drone boxes
            main_panel = draw_drone_boxes(main_panel, fused_tracks)

            # HUD overlay
            doppler_ok = doppler_data is not None and doppler_data.get('connected', False)
            dop_range = (doppler_data.get('estimated_range_m', 0.0)
                         if doppler_data else 0.0)
            main_panel = draw_hud(main_panel, num_drones, detection_mode,
                                  nn_det_count, doppler_ok, dop_range)

            # Depth panel
            if has_depth:
                depth_clipped = np.clip(depth_frame, 0, 15000).astype(np.float32)
                depth_norm = (depth_clipped / 15000.0 * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                depth_color[depth_frame == 0] = 0
                depth_panel = cv2.resize(depth_color, (DEPTH_W, MAIN_H))
                scale_x = DEPTH_W / MAIN_W
                for ft in fused_tracks.values():
                    if ft.is_confirmed and ft.confidence >= DRONE_THRESHOLD:
                        # Project back to pixel for depth panel marker
                        if ft.y > 0:
                            px_x = int((ft.x / (ft.y * np.tan(np.radians(35))) + 1) / 2 * DEPTH_W)
                        else:
                            px_x = DEPTH_W // 2
                        px_x = np.clip(px_x, 0, DEPTH_W - 1)
                        cv2.drawMarker(depth_panel, (px_x, MAIN_H // 2),
                                       MAGENTA, cv2.MARKER_CROSS, 15, 2)
            else:
                depth_panel = np.zeros((MAIN_H, DEPTH_W, 3), dtype=np.uint8)
                cv2.putText(depth_panel, "NO DEPTH", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

            top_row = np.hstack([main_panel, depth_panel])

            # --- Bottom row: Doppler | Fused Scope ---
            doppler_pw = CANVAS_W // 2
            fused_pw = CANVAS_W - doppler_pw

            doppler_panel = np.zeros((BOTTOM_H, doppler_pw, 3), dtype=np.uint8)
            fused_panel = np.zeros((BOTTOM_H, fused_pw, 3), dtype=np.uint8)

            doppler_panel = draw_doppler_panel(doppler_panel, doppler_data)
            sweep_angle += 0.08
            fused_panel = draw_fused_scope(fused_panel, fused_tracks, sweep_angle)

            # Separator line
            cv2.line(doppler_panel, (doppler_pw - 1, 0), (doppler_pw - 1, BOTTOM_H),
                     (40, 60, 80), 1)

            bottom_row = np.hstack([doppler_panel, fused_panel])

            # --- Composite canvas ---
            # Separator line between top and bottom
            sep_line = np.zeros((2, CANVAS_W, 3), dtype=np.uint8)
            sep_line[:] = (40, 60, 80)
            canvas = np.vstack([top_row, sep_line, bottom_row])

            # FPS
            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                display_fps = fps_count
                fps_count = 0
                fps_timer = now
            cv2.putText(canvas, f"FPS: {display_fps}", (MAIN_W + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

            cv2.imshow("Drone Detector", canvas)
            # In video mode, pace playback to ~video fps; press space to pause
            wait_ms = max(1, int(1000 / video_fps)) if video_mode else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if video_cap:
            video_cap.release()
        if doppler:
            doppler.stop()
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()
