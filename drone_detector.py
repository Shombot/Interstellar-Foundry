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
from collections import deque
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, os.path.expanduser("~"))
from radar_display import RadarReader, SPECTRAL_BINS, MAX_SPECTRAL_VAL

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

DRONE_THRESHOLD = 0.35

# COCO classes that are definitively NOT drones
NON_DRONE_CLASSES = {
    0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13,   # person, vehicles, street
    15, 16, 17, 18, 19, 20, 21, 22, 23,             # animals
    24, 25, 26, 27,                                   # accessories
    39, 40, 41, 42, 43, 44, 45,                      # kitchen
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,         # food
    56, 57, 58, 59, 60, 61,                          # furniture
    62, 63, 64, 65, 66, 67,                          # electronics
    68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, # appliances, misc
}

# COCO classes that could be confused with a drone (ambiguous)
DRONE_AMBIGUOUS_CLASSES = {
    4,   # airplane — small aircraft silhouette similar to drone
    14,  # bird — flies, similar size, key false positive
    29,  # frisbee — disc shape
    33,  # kite — airborne, similar size
    28,  # suitcase (unlikely but rectangular airborne object)
}

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
# Centroid tracker
# ---------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=20):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self.objects[self.next_id] = det
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_cents = np.array([(self.objects[oid][0], self.objects[oid][1])
                              for oid in obj_ids])
        det_cents = np.array([(d[0], d[1]) for d in detections])
        dists = np.linalg.norm(obj_cents[:, None] - det_cents[None, :], axis=2)

        used_obj, used_det, matches = set(), set(), []
        for idx in np.argsort(dists, axis=None):
            r, c = divmod(idx, len(detections))
            if r in used_obj or c in used_det:
                continue
            if dists[r, c] > 150:
                continue
            matches.append((r, c))
            used_obj.add(r)
            used_det.add(c)

        for r, c in matches:
            oid = obj_ids[r]
            self.objects[oid] = detections[c]
            self.disappeared[oid] = 0

        for r in range(len(obj_ids)):
            if r not in used_obj:
                oid = obj_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.objects[oid]
                    del self.disappeared[oid]

        for c in range(len(detections)):
            if c not in used_det:
                self.objects[self.next_id] = detections[c]
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        return self.objects
    

# ---------------------------------------------------------------------------
# Depth range searcher — finds pixel clusters at radar distance
# ---------------------------------------------------------------------------
class DepthRangeSearcher:
    def __init__(self):
        self.min_area = 100
        self.max_area = 25000
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    @staticmethod
    def compute_tolerance_mm(target_mm):
        stereo_error = (target_mm ** 2) / (33000.0 * 8.0) * 3.0
        return max(500.0, min(stereo_error, 4000.0))

    def search(self, depth_frame_mm, target_distance_m):
        target_mm = target_distance_m * 1000.0
        tol_mm = self.compute_tolerance_mm(target_mm)

        depth_f = depth_frame_mm.astype(np.float32)
        mask = ((depth_f > target_mm - tol_mm) &
                (depth_f < target_mm + tol_mm) &
                (depth_f > 0)).astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            roi_depth = depth_frame_mm[y:y+h, x:x+w]
            valid = roi_depth[roi_depth > 0]
            median_depth_m = (float(np.median(valid)) / 1000.0
                              if len(valid) > 0 else target_distance_m)
            candidates.append((x + w // 2, y + h // 2, w, h, median_depth_m))

        candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
        return candidates[:3]

    def search_multi(self, depth_frame_mm, recent_distances):
        if not recent_distances:
            return []
        sorted_d = sorted(set(round(d, 1) for d in recent_distances))
        clusters, current = [], [sorted_d[0]]
        for d in sorted_d[1:]:
            if d - current[-1] < 1.0:
                current.append(d)
            else:
                clusters.append(np.mean(current))
                current = [d]
        clusters.append(np.mean(current))

        all_candidates = []
        for dist_m in clusters[:3]:
            all_candidates.extend(self.search(depth_frame_mm, dist_m))
        all_candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
        return all_candidates[:3]


# ---------------------------------------------------------------------------
# Radar spectral analyzer — micro-Doppler drone signature
# ---------------------------------------------------------------------------
class RadarDroneAnalyzer:
    def __init__(self):
        self.spectral_history = deque(maxlen=12)
        self.drone_detected = False
        self.confidence = 0.0

    def analyze(self, radar_data):
        if radar_data is None or radar_data['mode'] is None:
            return False, 0.0, "no data"

        spectrum = radar_data['spectrum']
        dist_m = radar_data['distance_m']
        self.spectral_history.append(spectrum.copy())

        noise_floor = 1.5
        active_bins = np.sum(spectrum > noise_floor)
        active_ratio = active_bins / SPECTRAL_BINS

        if np.any(spectrum > noise_floor):
            active_indices = np.where(spectrum > noise_floor)[0]
            spectral_spread = (active_indices[-1] - active_indices[0]
                               if len(active_indices) > 1 else 0)
        else:
            spectral_spread = 0

        temporal_var = 0.0
        if len(self.spectral_history) >= 3:
            recent = np.array(list(self.spectral_history)[-5:])
            temporal_var = np.mean(np.var(recent, axis=0))

        peak_amp = float(np.max(spectrum))

        score = 0.0
        # Multi-rotor signature: moderate spectral activity from blade returns
        if 0.08 < active_ratio < 0.65:
            score += 0.25
        elif active_ratio >= 0.65:
            score += 0.10
        # Blade returns spread across frequency bins
        if spectral_spread > 12:
            score += 0.25
        elif spectral_spread > 6:
            score += 0.15
        # Temporal variation from rotating blades
        if temporal_var > 4.0:
            score += 0.25
        elif temporal_var > 1.5:
            score += 0.15
        # Target within plausible drone range with meaningful return
        if 0.5 < dist_m < 20.0 and peak_amp > noise_floor:
            score += 0.25

        self.confidence = min(score, 1.0)
        self.drone_detected = self.confidence > 0.25

        info = (f"bins:{active_bins} spread:{spectral_spread} "
                f"var:{temporal_var:.1f} conf:{self.confidence:.0%}")
        return self.drone_detected, self.confidence, info


# ---------------------------------------------------------------------------
# YOLO + Radar + Depth fusion scorer
# ---------------------------------------------------------------------------
class DroneScorer:
    """Scores each detection as drone/not-drone using all available evidence."""

    def score(self, yolo_label, yolo_conf, depth_m, radar_data,
              radar_conf, radar_detected):
        """
        Returns (drone_score, reason_str).
        drone_score: 0.0 = definitely not drone, 1.0 = definitely drone.
        """
        # --- YOLO class evidence ---
        if yolo_label is not None and yolo_label in NON_DRONE_CLASSES:
            # YOLO confidently says this is a known non-drone object
            if yolo_conf > 0.5:
                return 0.05, f"YOLO:{yolo_conf:.0%} known-obj"
            else:
                return 0.15, f"YOLO:{yolo_conf:.0%} low-conf known"

        # Bird is the primary false positive — needs radar to override
        if yolo_label == 14:  # bird
            if radar_detected and radar_conf > 0.5:
                # Radar says drone micro-Doppler despite YOLO saying bird
                return 0.60, "bird? radar-override"
            else:
                return 0.10, f"bird {yolo_conf:.0%}"

        # Airplane detection — could be a drone at distance
        if yolo_label == 4:  # airplane
            base = 0.50 if yolo_conf < 0.6 else 0.35
            if radar_detected:
                base += 0.25
            return min(base, 1.0), "airplane-like"

        # --- Unknown / unclassified object (YOLO miss or low conf) ---
        # This is where radar + depth do the heavy lifting
        score = 0.0
        reason_parts = []

        # Radar micro-Doppler is the strongest drone indicator
        if radar_detected:
            score += 0.45
            reason_parts.append(f"uDoppler:{radar_conf:.0%}")
        elif radar_conf > 0.2:
            score += radar_conf * 0.30
            reason_parts.append(f"radar:{radar_conf:.0%}")

        # Range match: depth agrees with radar
        if radar_data and radar_data.get('target_present') and depth_m > 0:
            range_diff = abs(depth_m - radar_data['distance_m'])
            if range_diff < 1.0:
                score += 0.30
                reason_parts.append("range-match")
            elif range_diff < 2.5:
                score += 0.15
                reason_parts.append("range-near")

        # Object in air at radar range but YOLO can't classify it → suspicious
        if yolo_label is None:
            if radar_data and radar_data.get('target_present'):
                score += 0.15
                reason_parts.append("unid+radar")
            else:
                score += 0.05
                reason_parts.append("unid")
        elif yolo_label in DRONE_AMBIGUOUS_CLASSES:
            score += 0.10
            reason_parts.append("ambiguous-class")

        return min(score, 1.0), " ".join(reason_parts) if reason_parts else "low-evidence"


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


def draw_hud(frame, num_drones, radar_data, radar_detected, radar_conf,
             radar_info, detection_mode, nn_dets):
    h, w = frame.shape[:2]
    panel_w, panel_h = 310, 120
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

    if radar_data and radar_data.get('target_present'):
        cv2.putText(frame, f"Radar: TARGET {radar_data['distance_m']:.1f}m",
                    (px + 8, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE, 1)
    else:
        cv2.putText(frame, "Radar: no target", (px + 8, py + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)

    cv2.putText(frame, f"Spectral: {radar_info}", (px + 8, py + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, GRAY, 1)

    bar_x, bar_y, bar_w = px + 8, py + 100, panel_w - 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                  (50, 50, 50), -1)
    fill_w = int(bar_w * radar_conf)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 10),
                  RED if radar_detected else GREEN, -1)

    if num_drones > 0 and int(time.time() * 3) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, 3)
    return frame


def draw_radar_hud(frame, data):
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (260, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if data is None or data.get('mode') is None:
        cv2.putText(frame, "RADAR --- [NO DETECTION]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        cv2.putText(frame, "Range: 0.00 m", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        return frame

    elapsed = time.time() - data['last_frame'] if data['last_frame'] else float('inf')
    status = "LIVE" if elapsed < 2.0 else "STALE"
    target = "TGT" if data.get('target_present') else "---"

    cv2.putText(frame, f"RADAR {status} [{target}]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                GREEN if status == "LIVE" else RED, 1)
    cv2.putText(frame, f"Range: {data['distance_m']:.2f} m", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                GREEN if data['distance_m'] < 5 else
                YELLOW if data['distance_m'] < 15 else RED, 1)
    return frame


def draw_spectrum_bar(frame, spectrum):
    h, w = frame.shape[:2]
    bar_h = 50
    bar_y0 = h - bar_h - 5

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, bar_y0 - 15), (w - 5, h - 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "FMCW Spectrum", (10, bar_y0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, CYAN, 1)

    bin_w = max(1, (w - 20) // SPECTRAL_BINS)
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
    return frame


def draw_radar_table(panel, radar_data, radar_detected=False, radar_conf=0.0,
                     radar_info=""):
    """Draw a compact radar values table on the depth side-panel."""
    h, w = panel.shape[:2]

    if radar_data is None or radar_data.get('mode') is None:
        cv2.putText(panel, "RADAR: N/A", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)
        return panel

    spectrum = radar_data['spectrum']
    dist_cm = radar_data['distance_cm']
    dist_m = radar_data['distance_m']
    mode = radar_data['mode']
    frames = radar_data['frames']
    target = radar_data.get('target_present', False)
    elapsed = time.time() - radar_data['last_frame'] if radar_data['last_frame'] else float('inf')

    peak_bin = int(np.argmax(spectrum)) if np.any(spectrum > 0) else 0
    peak_amp = float(spectrum[peak_bin]) if np.any(spectrum > 0) else 0.0
    noise_floor = 1.5
    active_bins = int(np.sum(spectrum > noise_floor))
    active_ratio = active_bins / SPECTRAL_BINS
    mean_amp = float(np.mean(spectrum[spectrum > 0])) if np.any(spectrum > 0) else 0.0

    # Spectral spread
    if np.any(spectrum > noise_floor):
        active_indices = np.where(spectrum > noise_floor)[0]
        spectral_spread = int(active_indices[-1] - active_indices[0]) if len(active_indices) > 1 else 0
    else:
        spectral_spread = 0

    # Table background
    table_h = 225
    table_y = h - table_h - 5
    overlay = panel.copy()
    cv2.rectangle(overlay, (3, table_y), (w - 3, h - 3), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, panel, 0.3, 0, panel)

    # Header
    status_str = "LIVE" if elapsed < 2.0 else "STALE"
    status_clr = GREEN if elapsed < 2.0 else RED
    cv2.putText(panel, "RADAR OUTPUT", (8, table_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
    cv2.putText(panel, status_str, (w - 45, table_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, status_clr, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.30
    lh = 14  # line height
    x_lbl = 8
    x_val = 90
    y = table_y + 30

    rows = [
        ("Mode",      f"{'B (d+spec)' if mode == 'B' else 'A (dist)' if mode == 'A' else '?'}"),
        ("Target",    f"{'YES' if target else 'NO'}"),
        ("Distance",  f"{dist_cm} cm  ({dist_m:.2f} m)"),
        ("Frames",    f"{frames}"),
        ("Peak Bin",  f"{peak_bin}"),
        ("Peak Amp",  f"{peak_amp:.1f} / {MAX_SPECTRAL_VAL}"),
        ("Active",    f"{active_bins} / {SPECTRAL_BINS} bins"),
        ("Active %",  f"{active_ratio:.0%}"),
        ("Spread",    f"{spectral_spread} bins"),
        ("Mean Amp",  f"{mean_amp:.1f}"),
    ]

    for label, value in rows:
        cv2.putText(panel, label, (x_lbl, y), font, fs, GRAY, 1)
        val_clr = WHITE
        if label == "Target":
            val_clr = GREEN if target else GRAY
        elif label == "Distance":
            val_clr = GREEN if dist_m < 5 else YELLOW if dist_m < 15 else RED
        cv2.putText(panel, value, (x_val, y), font, fs, val_clr, 1)
        y += lh

    # --- Drone analysis section ---
    y += 4
    cv2.line(panel, (x_lbl, y), (w - 10, y), (80, 80, 80), 1)
    y += 12
    cv2.putText(panel, "DRONE ANALYSIS", (x_lbl, y), font, 0.35, MAGENTA, 1)
    y += lh

    drone_label = "DETECTED" if radar_detected else "---"
    drone_clr = RED if radar_detected else GRAY
    cv2.putText(panel, "uDoppler", (x_lbl, y), font, fs, GRAY, 1)
    cv2.putText(panel, drone_label, (x_val, y), font, fs, drone_clr, 1)
    y += lh

    conf_pct = f"{radar_conf:.0%}"
    conf_clr = RED if radar_conf >= 0.5 else ORANGE if radar_conf >= 0.25 else GREEN
    cv2.putText(panel, "Drone Conf", (x_lbl, y), font, fs, GRAY, 1)
    cv2.putText(panel, conf_pct, (x_val, y), font, fs, conf_clr, 1)
    y += lh

    # Confidence bar
    bar_w = w - x_lbl - 15
    cv2.rectangle(panel, (x_lbl, y), (x_lbl + bar_w, y + 8), (50, 50, 50), -1)
    fill_w = int(bar_w * radar_conf)
    cv2.rectangle(panel, (x_lbl, y), (x_lbl + fill_w, y + 8), conf_clr, -1)
    # Threshold marker
    thresh_x = x_lbl + int(bar_w * DRONE_THRESHOLD)
    cv2.line(panel, (thresh_x, y - 2), (thresh_x, y + 10), WHITE, 1)

    return panel


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
    parser.setConfidenceThreshold(0.25)
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

    if radar is None and pipeline is None:
        print("ERROR: Neither radar nor camera available.")
        sys.exit(1)

    # --- Detection components ---
    depth_searcher = DepthRangeSearcher()
    radar_analyzer = RadarDroneAnalyzer()
    tracker = CentroidTracker(max_disappeared=20)
    scorer = DroneScorer()

    cv2.namedWindow("Drone Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Detector", CANVAS_W, MAIN_H)

    fps_timer = time.time()
    fps_count = 0
    display_fps = 0
    last_rgb_frame = None
    last_depth_frame = None

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

            # --- Radar ---
            radar_data = radar.get_data() if radar else None
            radar_detected, radar_conf, radar_info = radar_analyzer.analyze(radar_data)

            # --- Build detection candidates ---
            # Each candidate: (cx, cy, w, h, drone_score, depth_m, source_str)
            detections = []
            detection_mode = "IDLE"
            nn_det_count = len(yolo_detections)

            target_present = (radar_data is not None
                              and radar_data.get('target_present', False))
            has_depth = depth_frame is not None
            has_rgb = rgb_frame is not None

            # --- Process YOLO detections ---
            if has_rgb and yolo_detections:
                for det in yolo_detections:
                    # Scale YOLO bbox (normalized 0-1) to frame coordinates
                    cx = int((det.xmin + det.xmax) / 2 * MAIN_W)
                    cy = int((det.ymin + det.ymax) / 2 * MAIN_H)
                    w = int((det.xmax - det.xmin) * MAIN_W)
                    h = int((det.ymax - det.ymin) * MAIN_H)
                    yolo_label = det.label
                    yolo_conf = det.confidence

                    # Get depth at detection center
                    det_depth_m = 0.0
                    if has_depth:
                        r = max(5, min(w, h) // 4)
                        dy = np.clip(cy, r, depth_frame.shape[0] - r - 1)
                        dx = np.clip(cx, r, depth_frame.shape[1] - r - 1)
                        region = depth_frame[dy-r:dy+r+1, dx-r:dx+r+1]
                        valid = region[region > 0]
                        if len(valid) > 0:
                            det_depth_m = float(np.median(valid)) / 1000.0

                    # Score this detection
                    drone_score, reason = scorer.score(
                        yolo_label, yolo_conf, det_depth_m,
                        radar_data, radar_conf, radar_detected)

                    lbl_name = labels[yolo_label] if yolo_label < len(labels) else "?"
                    source = f"Y:{lbl_name}"

                    detections.append((cx, cy, w, h, drone_score,
                                       det_depth_m, source))

                detection_mode = "YOLO+RADAR" if target_present else "YOLO"

            # --- Radar-guided depth search for objects YOLO missed ---
            if target_present and has_depth:
                radar_dist = radar_data['distance_m']
                recent = radar_data.get('recent_distances', [])
                depth_candidates = (
                    depth_searcher.search_multi(depth_frame, recent)
                    if recent else
                    depth_searcher.search(depth_frame, radar_dist))

                # Only add depth candidates that don't overlap with YOLO detections
                for cx, cy, w, h, median_depth_m in depth_candidates:
                    overlaps_yolo = False
                    for yd in detections:
                        dx = abs(cx - yd[0])
                        dy = abs(cy - yd[1])
                        if dx < max(w, yd[2]) and dy < max(h, yd[3]):
                            overlaps_yolo = True
                            break

                    if not overlaps_yolo:
                        # No YOLO match — score based on radar + depth
                        drone_score, reason = scorer.score(
                            None, 0.0, median_depth_m,
                            radar_data, radar_conf, radar_detected)
                        detections.append((cx, cy, w, h, drone_score,
                                           median_depth_m, f"R+D"))
                        if detection_mode == "YOLO":
                            detection_mode = "YOLO+RADAR"
                        elif detection_mode == "IDLE":
                            detection_mode = "RADAR+DEPTH"

            # --- Radar-only (no camera) ---
            elif target_present and not has_rgb:
                detection_mode = "RADAR ONLY"

            # --- Update tracker ---
            tracked = tracker.update(detections)

            # --- Display ---
            if has_rgb:
                main_panel = cv2.resize(rgb_frame, (MAIN_W, MAIN_H))
                if main_panel.ndim == 2:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_GRAY2BGR)
                elif main_panel.shape[2] == 4:
                    main_panel = cv2.cvtColor(main_panel, cv2.COLOR_BGRA2BGR)
            else:
                main_panel = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
                cv2.putText(main_panel, "NO CAMERA", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 2)

            main_panel = draw_drone_boxes(main_panel, tracked)

            if detection_mode == "RADAR ONLY" and radar_data:
                cv2.putText(main_panel,
                            f"RADAR TARGET: {radar_data['distance_m']:.1f}m",
                            (MAIN_W // 2 - 130, MAIN_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)

            num_drones = sum(1 for d in tracked.values()
                             if d[4] >= DRONE_THRESHOLD)
            main_panel = draw_hud(main_panel, num_drones, radar_data,
                                  radar_detected, radar_conf, radar_info,
                                  detection_mode, nn_det_count)

            main_panel = draw_radar_hud(main_panel, radar_data)
            if radar_data is not None and radar_data.get('mode') is not None:
                main_panel = draw_spectrum_bar(main_panel, radar_data['spectrum'])

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

            # Live radar distance readout at top of side panel
            if radar_data is not None and radar_data.get('mode') is not None:
                dist_m = radar_data['distance_m']
                tgt = radar_data.get('target_present', False)
                overlay = depth_panel.copy()
                cv2.rectangle(overlay, (3, 30), (DEPTH_W - 3, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, depth_panel, 0.3, 0, depth_panel)
                cv2.putText(depth_panel, "RADAR DIST", (8, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
                dist_clr = GREEN if dist_m < 5 else YELLOW if dist_m < 15 else RED
                if not tgt:
                    dist_clr = GRAY
                cv2.putText(depth_panel, f"{dist_m:.2f} m",
                            (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, dist_clr, 2)

            # Radar values table on depth side-panel
            depth_panel = draw_radar_table(depth_panel, radar_data,
                                           radar_detected, radar_conf,
                                           radar_info)

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
        if radar:
            radar.stop()
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == '__main__':
    main()