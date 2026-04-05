#!/usr/bin/env python3
"""
Drone Detector — YOLO + Radar + Stereo Depth Fusion
----------------------------------------------------
Architecture:
  1. YOLOv6n runs on OAK Myriad X VPU (80 COCO classes, zero Jetson load)
  2. Radar provides range + FMCW micro-Doppler spectral signature
  3. Stereo depth confirms 3D position of detections
  4. Fusion logic:
     - YOLO identifies known objects (bird, person, car → NOT drone)
     - Unidentified objects at radar range + drone micro-Doppler → DRONE
     - Depth-radar range agreement boosts confidence
  5. Fallback: depth-map search when YOLO misses small/far targets

Usage:
    python3 drone_detector.py                 # full fusion
    python3 drone_detector.py --no-radar      # camera-only
    python3 drone_detector.py --no-camera     # radar-only
"""

import argparse
import sys
import os
import time
import json
# from collections import deque
from pathlib import Path

import numpy as np
import cv2

# sys.path.insert(0, os.path.expanduser("~"))
# from radar_display import RadarReader, SPECTRAL_BINS, MAX_SPECTRAL_VAL

# Layout
MAIN_W, MAIN_H = 640, 480
NN_W, NN_H = 512, 288
DEPTH_W = 240
CANVAS_W = MAIN_W + DEPTH_W

# Colors (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
MAGENTA = (255, 0, 255)
ORANGE = (0, 140, 255)

DRONE_THRESHOLD = 0.30

# COCO classes that are airborne / could be a drone — only these show on display
FLYING_CLASSES = {
    4,   # airplane
    14,  # bird
    29,  # frisbee — disc shape in air
    33,  # kite
}

# Everything else is a ground object — never show on dashboard
# (person=0, bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7, boat=8, etc.)

# Model paths
SCRIPT_DIR = Path(__file__).parent
BLOB_PATH = SCRIPT_DIR / "yolov6n_model" / "yolov6n-r2-288x512_openvino_2022.1_6shave.blob"
ARCHIVE_PATH = SCRIPT_DIR / "yolov6n_coco_rvc2.tar.xz"
CONFIG_PATH = SCRIPT_DIR / "yolov6n_model" / "config.json"


def load_coco_labels():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg['model']['heads'][0]['metadata']['classes']


# ---------------------------------------------------------------------------
# Drone tracker — multi-frame confirmation + KCF lock-on
# ---------------------------------------------------------------------------
# Requires N consistent YOLO flying-class detections in the same area
# before classifying as drone. Once confirmed, KCF tracker maintains lock.
CONFIRM_FRAMES = 5   # need 5 hits in the same spot before locking as drone

class DroneTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def _init_kcf(self, frame, t):
        """Start KCF tracker on a confirmed drone."""
        x1 = max(0, t['cx'] - t['w'] // 2)
        y1 = max(0, t['cy'] - t['h'] // 2)
        w = min(t['w'], frame.shape[1] - x1)
        h = min(t['h'], frame.shape[0] - y1)
        if w < 5 or h < 5:
            t['kcf'] = None
            return
        kcf = cv2.TrackerKCF_create()
        kcf.init(frame, (x1, y1, w, h))
        t['kcf'] = kcf

    def update(self, frame, detections):
        """
        detections: list of (cx, cy, w, h, score, depth_m, source)
        Only YOLO flying-class detections should be fed in.
        Returns dict of oid → (cx, cy, w, h, score, depth_m, source)
        """
        # Step 1: advance KCF on confirmed drones
        for oid in list(self.tracks):
            t = self.tracks[oid]
            if t.get('kcf') is not None:
                ok, bbox = t['kcf'].update(frame)
                if ok:
                    bx, by, bw, bh = [int(v) for v in bbox]
                    t['cx'] = bx + bw // 2
                    t['cy'] = by + bh // 2
                    t['w'], t['h'] = bw, bh
                else:
                    t['missed'] += 1
                    t['kcf'] = None
            elif t['confirmed']:
                t['missed'] += 1
            else:
                t['missed'] += 1

        # Step 2: match detections to tracks
        used_det = set()
        for oid in list(self.tracks):
            t = self.tracks[oid]
            best_dist, best_c = 200.0, -1
            for c, det in enumerate(detections):
                if c in used_det:
                    continue
                dist = ((t['cx'] - det[0])**2 + (t['cy'] - det[1])**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_c = c

            if best_c >= 0:
                det = detections[best_c]
                used_det.add(best_c)
                t['cx'], t['cy'] = det[0], det[1]
                t['w'], t['h'] = det[2], det[3]
                t['depth'] = det[5]
                t['source'] = det[6]
                t['missed'] = 0
                t['hits'] += 1

                if t['confirmed']:
                    t['score'] = max(det[4], DRONE_THRESHOLD)
                    # Refresh KCF
                    self._init_kcf(frame, t)
                elif t['hits'] >= CONFIRM_FRAMES:
                    # Enough consistent detections — confirm as drone
                    t['confirmed'] = True
                    t['score'] = max(det[4], DRONE_THRESHOLD)
                    self._init_kcf(frame, t)
                else:
                    t['score'] = det[4]

        # Step 3: new candidate tracks
        for c, det in enumerate(detections):
            if c in used_det:
                continue
            overlaps = any(
                abs(t['cx'] - det[0]) < max(t['w'], det[2]) and
                abs(t['cy'] - det[1]) < max(t['h'], det[3])
                for t in self.tracks.values())
            if overlaps:
                continue

            self.tracks[self.next_id] = {
                'cx': det[0], 'cy': det[1], 'w': det[2], 'h': det[3],
                'score': det[4], 'depth': det[5], 'source': det[6],
                'confirmed': False, 'hits': 1, 'missed': 0, 'kcf': None,
            }
            self.next_id += 1

        # Step 4: prune
        for oid in list(self.tracks):
            t = self.tracks[oid]
            # Confirmed drones persist 3s, unconfirmed candidates expire fast
            limit = 90 if t['confirmed'] else 10
            if t['missed'] > limit:
                del self.tracks[oid]

        # Only return confirmed drones for display
        result = {}
        for oid, t in self.tracks.items():
            if t['confirmed']:
                result[oid] = (t['cx'], t['cy'], t['w'], t['h'],
                               t['score'], t['depth'], t['source'])
        return result


# ---------------------------------------------------------------------------
# Camera-only drone scorer
# ---------------------------------------------------------------------------
class DroneScorer:
    """Scores each detection as drone/not-drone using camera only."""

    def score(self, yolo_label, yolo_conf, depth_m):
        """
        Returns (drone_score, reason_str).
        Only flying/airborne objects pass through. Ground objects → 0.
        """
        # Ground object — not a drone, don't show
        if yolo_label is not None and yolo_label not in FLYING_CLASSES:
            return 0.0, "ground-obj"

        # Bird ��� common false positive but still show it
        if yolo_label == 14:  # bird
            return 0.25, f"bird {yolo_conf:.0%}"

        # Airplane — high drone suspicion, especially at low confidence
        if yolo_label == 4:  # airplane
            score = 0.65 if yolo_conf < 0.5 else 0.50
            return score, "airplane-like"

        # Kite / frisbee — airborne, moderate suspicion
        if yolo_label in (29, 33):
            return 0.40, "airborne-obj"

        # Unknown / unclassified — YOLO missed it, could be a drone
        if yolo_label is None:
            return 0.35, "unidentified"

        return 0.0, "low-evidence"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_drone_boxes(frame, tracker_objects):
    for obj_id, obj_data in tracker_objects.items():
        cx, cy, w, h, score, depth_m, source = obj_data
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        is_drone = score >= DRONE_THRESHOLD
        depth_str = f" {depth_m:.1f}m" if depth_m > 0 else ""

        if is_drone:
            label = f"DRONE {score:.0%}{depth_str} [{source}]"
            box_color = RED
            cv2.drawMarker(frame, (cx, cy), CYAN, cv2.MARKER_CROSS, 12, 1)
        else:
            label = f"{source} {score:.0%}{depth_str}"
            box_color = YELLOW

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    return frame


def draw_hud(frame, num_drones, detection_mode, nn_dets):
    h, w = frame.shape[:2]
    panel_w, panel_h = 310, 70
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

    cv2.putText(frame, f"Mode: {detection_mode}  YOLO:{nn_dets}",
                (px + 8, py + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

    cv2.putText(frame, "Radar: DISABLED", (px + 8, py + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    if num_drones > 0 and int(time.time() * 3) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 3)
    return frame


def draw_radar_scope(frame, tracked, sweep_angle):
    """Draw a radar-style circular scope with drone blips from camera data."""
    h, w = frame.shape[:2]
    radius = 70
    cx, cy = w - radius - 15, 85 + radius  # below HUD panel

    # Dark background circle
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius + 4, (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Range rings
    for r_frac in (0.33, 0.66, 1.0):
        r = int(radius * r_frac)
        cv2.circle(frame, (cx, cy), r, (0, 60, 0), 1)

    # Crosshairs
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (0, 60, 0), 1)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), (0, 60, 0), 1)

    # Sweep line (rotating)
    sweep_x = int(cx + radius * np.cos(sweep_angle))
    sweep_y = int(cy - radius * np.sin(sweep_angle))
    cv2.line(frame, (cx, cy), (sweep_x, sweep_y), (0, 180, 0), 1)

    # Sweep fade trail
    for i in range(1, 4):
        a = sweep_angle - i * 0.15
        sx = int(cx + radius * np.cos(a))
        sy = int(cy - radius * np.sin(a))
        intensity = max(0, 180 - i * 50)
        cv2.line(frame, (cx, cy), (sx, sy), (0, intensity, 0), 1)

    # Label
    cv2.putText(frame, "SCOPE", (cx - 22, cy - radius - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREEN, 1)

    # Range labels
    cv2.putText(frame, "10m", (cx + int(radius * 0.33) - 8, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 80, 0), 1)
    cv2.putText(frame, "20m", (cx + int(radius * 0.66) - 8, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 80, 0), 1)

    # Plot drone blips
    for oid, obj in tracked.items():
        obj_cx, obj_cy, obj_w, obj_h, score, depth_m, source = obj
        if score < DRONE_THRESHOLD:
            continue

        # Map camera X position to radar angle (left=-90°, center=0°, right=+90°)
        # Horizontal position relative to frame center
        norm_x = (obj_cx - MAIN_W / 2) / (MAIN_W / 2)  # -1 to +1

        # Map depth to distance from radar center (0m=center, 20m=edge)
        max_range = 20.0
        d = min(depth_m, max_range) / max_range if depth_m > 0 else 0.8
        dist_px = int(d * radius)

        # Convert to radar XY: X maps to horizontal, depth maps to vertical (up=far)
        blip_x = cx + int(norm_x * dist_px)
        blip_y = cy - dist_px  # up = further away

        # Blip
        cv2.circle(frame, (blip_x, blip_y), 4, (0, 255, 0), -1)
        cv2.circle(frame, (blip_x, blip_y), 7, (0, 200, 0), 1)

        # Distance label
        if depth_m > 0:
            cv2.putText(frame, f"{depth_m:.0f}m",
                        (blip_x + 8, blip_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, GREEN, 1)

    # Outer ring
    cv2.circle(frame, (cx, cy), radius, GREEN, 1)

    return frame


# ---------------------------------------------------------------------------
# OAK pipeline — YOLO on VPU + stereo depth
# ---------------------------------------------------------------------------
def build_oak_pipeline():
    import depthai as dai

    pipeline = dai.Pipeline()

    # RGB camera
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgbOut = camRgb.requestOutput((MAIN_W, MAIN_H),
                                  dai.ImgFrame.Type.BGR888p, fps=30.0)

    # Stereo pair
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

    # Resize for YOLO input
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(NN_W, NN_H)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    rgbOut.link(manip.inputImage)

    # YOLO inference on Myriad X VPU
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(BLOB_PATH))
    nn.setNumInferenceThreads(2)
    manip.out.link(nn.input)

    # Detection parser (host-side, avoids device crash)
    parser = pipeline.create(dai.node.DetectionParser)
    parser.setNNArchive(dai.NNArchive(str(ARCHIVE_PATH)))
    parser.setConfidenceThreshold(0.15)
    parser.setInputImageSize(NN_W, NN_H)
    parser.setRunOnHost(True)
    nn.out.link(parser.input)

    # Output queues
    rgbQ = rgbOut.createOutputQueue(maxSize=1, blocking=False)
    depthQ = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
    detQ = parser.out.createOutputQueue(maxSize=1, blocking=False)

    return pipeline, rgbQ, depthQ, detQ


def build_oak_pipeline_no_nn():
    """Fallback pipeline without YOLO (if blob not found)."""
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
    parser = argparse.ArgumentParser(description="Drone Detector — YOLO+Radar Fusion")
    parser.add_argument('--port', default='/dev/ttyTHS1')
    parser.add_argument('--baud', type=int, default=57600)
    parser.add_argument('--no-radar', action='store_true')
    parser.add_argument('--no-camera', action='store_true')
    args = parser.parse_args()

    # --- Load COCO labels ---
    labels = []
    if CONFIG_PATH.exists():
        labels = load_coco_labels()
        print(f"Loaded {len(labels)} COCO class labels")

    # --- Radar disabled ---
    radar = None
    # if not args.no_radar:
    #     try:
    #         radar = RadarReader(args.port, args.baud)
    #         radar.connect()
    #         radar.start()
    #         print(f"Radar connected on {args.port}")
    #     except Exception as e:
    #         print(f"Radar unavailable: {e} — continuing camera-only")

    # --- Initialize OAK camera + YOLO ---
    pipeline = rgbQ = depthQ = detQ = None
    has_nn = False
    if not args.no_camera:
        try:
            if BLOB_PATH.exists() and ARCHIVE_PATH.exists():
                pipeline, rgbQ, depthQ, detQ = build_oak_pipeline()
                has_nn = True
                print("OAK camera + YOLOv6n on Myriad X VPU")
            else:
                pipeline, rgbQ, depthQ, detQ = build_oak_pipeline_no_nn()
                print("OAK camera (no YOLO blob — depth-only mode)")
            pipeline.start()
        except Exception as e:
            print(f"OAK camera error: {e}")
            # Try fallback without NN
            if has_nn:
                try:
                    pipeline, rgbQ, depthQ, detQ = build_oak_pipeline_no_nn()
                    pipeline.start()
                    has_nn = False
                    print("Fell back to depth-only pipeline")
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    pipeline = None

    if pipeline is None:
        print("ERROR: Camera not available.")
        sys.exit(1)

    # --- Detection components ---
    tracker = DroneTracker()
    scorer = DroneScorer()

    cv2.namedWindow("Drone Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Detector", CANVAS_W, MAIN_H)

    fps_timer = time.time()
    fps_count = 0
    display_fps = 0
    last_rgb_frame = None
    last_depth_frame = None

    sweep_angle = 0.0
    print(f"Drone detector running — press 'q' to quit")

    try:
        while True:
            # --- Camera frames ---
            rgb_frame = depth_frame = None
            yolo_detections = []

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

                    # Get YOLO detections from VPU
                    if detQ is not None:
                        det_msg = detQ.tryGet()
                        if det_msg is not None:
                            yolo_detections = det_msg.detections
                except Exception:
                    pass

            # --- Build detection candidates (camera only) ---
            # Each candidate: (cx, cy, w, h, drone_score, depth_m, source_str)
            detections = []
            detection_mode = "CAMERA"
            nn_det_count = len(yolo_detections)

            has_depth = depth_frame is not None
            has_rgb = rgb_frame is not None

            def get_depth_at(cx, cy, w, h):
                """Get median depth at a detection center."""
                if not has_depth:
                    return 0.0
                r = max(5, min(w, h) // 4)
                dy = np.clip(cy, r, depth_frame.shape[0] - r - 1)
                dx = np.clip(cx, r, depth_frame.shape[1] - r - 1)
                region = depth_frame[dy-r:dy+r+1, dx-r:dx+r+1]
                valid = region[region > 0]
                return float(np.median(valid)) / 1000.0 if len(valid) > 0 else 0.0

            # --- YOLO detections — only flying objects pass ---
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

                    drone_score, reason = scorer.score(
                        yolo_label, yolo_conf, 0.0)
                    if drone_score == 0.0:
                        continue

                    # Reject bright light sources — not drones
                    # Small center patch so glow doesn't dilute the check
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
                    lbl_name = labels[yolo_label] if yolo_label < len(labels) else "?"
                    detections.append((cx, cy, w, h, drone_score,
                                       det_depth_m, f"Y:{lbl_name}"))
                detection_mode = "YOLO"

            # --- Update tracker (multi-frame confirm + KCF lock-on) ---
            display_frame = (cv2.resize(rgb_frame, (MAIN_W, MAIN_H))
                             if has_rgb else
                             np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8))
            tracked = tracker.update(display_frame, detections)

            # --- Display (CLAHE for low-light enhancement) ---
            if has_rgb:
                main_panel = display_frame.copy()
                if main_panel.ndim == 2:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_GRAY2BGR)
                elif main_panel.shape[2] == 4:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_BGRA2BGR)
                # Auto-enhance in low light
                gray_check = cv2.cvtColor(main_panel, cv2.COLOR_BGR2GRAY)
                if np.mean(gray_check) < 80:
                    lab = cv2.cvtColor(main_panel, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    main_panel = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            else:
                main_panel = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
                cv2.putText(main_panel, "NO CAMERA", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)

            main_panel = draw_drone_boxes(main_panel, tracked)

            num_drones = sum(1 for d in tracked.values()
                             if d[4] >= DRONE_THRESHOLD)
            main_panel = draw_hud(main_panel, num_drones,
                                  detection_mode, nn_det_count)

            sweep_angle += 0.08
            main_panel = draw_radar_scope(main_panel, tracked, sweep_angle)

            if has_depth:
                depth_clipped = np.clip(depth_frame, 0, 15000).astype(np.float32)
                depth_norm = (depth_clipped / 15000.0 * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                depth_color[depth_frame == 0] = 0
                depth_panel = cv2.resize(depth_color, (DEPTH_W, MAIN_H))
                scale_x = DEPTH_W / MAIN_W
                for obj_data in tracked.values():
                    if obj_data[4] >= DRONE_THRESHOLD:
                        cv2.drawMarker(
                            depth_panel,
                            (int(obj_data[0] * scale_x), int(obj_data[1])),
                            MAGENTA, cv2.MARKER_CROSS, 15, 2)
            else:
                depth_panel = np.zeros((MAIN_H, DEPTH_W, 3), dtype=np.uint8)
                cv2.putText(depth_panel, "NO DEPTH", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

            canvas = np.hstack([main_panel, depth_panel])

            fps_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                display_fps = fps_count
                fps_count = 0
                fps_timer = now
            cv2.putText(canvas, f"FPS: {display_fps}", (MAIN_W + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

            cv2.imshow("Drone Detector", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # if radar:
        #     radar.stop()
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()