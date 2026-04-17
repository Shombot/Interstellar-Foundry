#!/usr/bin/env python3
"""
Drone Kalman Filter — Camera + Doppler Radar fusion
----------------------------------------------------
Fuses:
  1. OAK stereo camera  → bearing (from pixel x) + depth (stereo range in m)
     Role: DETECT + TRACK (YOLO identification, stereo positioning)
  2. CQRobot 10.525GHz Doppler radar → range estimate + motion gate
     Role: RANGE (signal-envelope depth estimation, motion confirmation)

State vector: [x, y, vx, vy]
  x  = horizontal offset from sensor (metres, + = right)
  y  = range / depth from sensor (metres, + = forward)
  vx = horizontal velocity (m/s)
  vy = range-rate velocity (m/s)

Measurement models:
  Camera:  z_cam = [x_cam, y_cam]  from stereo depth + pixel bearing
  Doppler: z_dop = [range]         from signal-envelope range estimation
           Also modulates process noise via motion gate
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FusedTrack:
    """Result of Kalman-fused drone tracking."""
    track_id: int
    x: float = 0.0           # horizontal offset (m), + = right
    y: float = 0.0           # range from sensor (m)
    vx: float = 0.0          # horizontal velocity (m/s)
    vy: float = 0.0          # range-rate (m/s)
    range_m: float = 0.0     # fused range estimate (m)
    bearing_deg: float = 0.0 # bearing from sensor (deg, 0=ahead)
    confidence: float = 0.0  # overall fusion confidence [0-1]
    camera_conf: float = 0.0
    doppler_conf: float = 0.0
    age_frames: int = 0
    missed_frames: int = 0
    is_confirmed: bool = False
    # Source flags
    has_camera: bool = False
    has_doppler: bool = False


class DroneKalmanFilter:
    """
    Extended Kalman Filter for single-drone tracking.

    Camera provides (x, y) position (DETECT + TRACK).
    Doppler radar provides range estimate + motion gate (RANGE).
    """

    def __init__(self, track_id, dt=1.0/30.0):
        self.track_id = track_id
        self.dt = dt

        # State: [x, y, vx, vy]
        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0  # large initial uncertainty

        # Process noise (tuned for small drone dynamics)
        self.Q_base = np.diag([0.5, 0.5, 1.0, 1.0])

        # Measurement noise covariances
        self.R_camera = np.diag([0.3, 1.0])    # camera: decent x, noisier depth
        self.R_doppler_range = np.array([[4.0]])  # Doppler range: coarse estimate

        self.age = 0
        self.missed = 0
        self.confidence = 0.0
        self.camera_conf = 0.0
        self.doppler_conf = 0.0
        self.confirmed = False

        self._initialized = False

    def initialize(self, x, y):
        """Set initial position from first measurement."""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([1.0, 2.0, 2.0, 2.0])
        self._initialized = True

    @property
    def state(self):
        return self.x.copy()

    @property
    def range_m(self):
        return float(np.sqrt(self.x[0]**2 + self.x[1]**2))

    @property
    def bearing_deg(self):
        return float(np.degrees(np.arctan2(self.x[0], self.x[1])))

    def predict(self, doppler_active=False, doppler_activity=0.0):
        """
        Predict step. Doppler activity modulates process noise:
        - Active motion → lower process noise (trust dynamics model)
        - No motion → higher process noise (target may have stopped/changed)
        """
        if not self._initialized:
            return

        dt = self.dt
        # State transition: constant velocity model
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # Modulate process noise with Doppler confidence
        if doppler_active and doppler_activity > 0.3:
            # Motion confirmed — tighten noise (we trust the model more)
            q_scale = max(0.3, 1.0 - doppler_activity * 0.5)
        else:
            # No motion or no Doppler — inflate noise
            q_scale = 1.5

        Q = self.Q_base * q_scale * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.age += 1

    def update_camera(self, x_m, depth_m, cam_confidence=0.5):
        """
        Update with camera measurement: (x_offset_m, depth_m).
        x_m:     horizontal offset from camera center (metres)
        depth_m: stereo depth (metres)
        """
        if not self._initialized:
            self.initialize(x_m, depth_m)
            self.camera_conf = cam_confidence
            self.missed = 0
            return

        z = np.array([x_m, depth_m])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Scale R by inverse confidence
        r_scale = max(0.5, 2.0 - cam_confidence * 2.0)
        R = self.R_camera * r_scale

        self._kalman_update(z, H, R)
        self.camera_conf = cam_confidence
        self.missed = 0

    def update_doppler_range(self, range_m, range_confidence=0.3):
        """
        Update with Doppler radar range-only measurement.
        The Doppler sensor gives a coarse range estimate from signal envelope
        analysis (duty cycle + pulse frequency → inverse-square-law distance).
        No angle information — only updates range component.
        """
        if not self._initialized:
            return

        # Predicted range from state
        pred_range = np.sqrt(self.x[0]**2 + self.x[1]**2)
        if pred_range < 0.01:
            pred_range = 0.01

        # Jacobian of range = sqrt(x^2 + y^2) w.r.t. state
        H = np.array([[
            self.x[0] / pred_range,
            self.x[1] / pred_range,
            0, 0
        ]])

        z = np.array([range_m])
        innovation = z - np.array([pred_range])

        # Scale noise by inverse confidence (low confidence → high noise)
        r_scale = max(1.0, 3.0 - range_confidence * 4.0)
        R = self.R_doppler_range * r_scale

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ innovation).flatten()
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def mark_missed(self):
        """No measurements this frame."""
        self.missed += 1
        self.camera_conf *= 0.95

    def set_doppler(self, is_drone_sig, doppler_conf):
        """Update Doppler confidence (used in predict and overall confidence)."""
        self.doppler_conf = doppler_conf

    def compute_confidence(self):
        """Overall fusion confidence from camera + Doppler."""
        # Weighted combination
        c = 0.0
        w_total = 0.0

        if self.camera_conf > 0:
            c += self.camera_conf * 0.65
            w_total += 0.65
        if self.doppler_conf > 0:
            c += self.doppler_conf * 0.35
            w_total += 0.35

        if w_total > 0:
            self.confidence = c / w_total
        else:
            self.confidence = 0.0

        # Decay with missed frames
        if self.missed > 0:
            self.confidence *= max(0.1, 1.0 - self.missed * 0.05)

        # Boost when both sensors agree
        if self.camera_conf > 0.1 and self.doppler_conf > 0.1:
            self.confidence = min(1.0, self.confidence * 1.25)

        # Confirm track after sustained confidence
        if self.age > 5 and self.confidence > 0.3:
            self.confirmed = True

        return self.confidence

    def get_fused_track(self) -> FusedTrack:
        """Return a FusedTrack snapshot."""
        self.compute_confidence()
        return FusedTrack(
            track_id=self.track_id,
            x=float(self.x[0]),
            y=float(self.x[1]),
            vx=float(self.x[2]),
            vy=float(self.x[3]),
            range_m=self.range_m,
            bearing_deg=self.bearing_deg,
            confidence=self.confidence,
            camera_conf=self.camera_conf,
            doppler_conf=self.doppler_conf,
            age_frames=self.age,
            missed_frames=self.missed,
            is_confirmed=self.confirmed,
            has_camera=self.camera_conf > 0.05,
            has_doppler=self.doppler_conf > 0.05,
        )

    def _kalman_update(self, z, H, R):
        """Standard Kalman update."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P


class MultiDroneTracker:
    """
    Manages multiple DroneKalmanFilter instances for multi-target tracking.
    Associates new measurements with existing tracks or spawns new ones.
    """

    def __init__(self, dt=1.0/30.0, confirm_frames=5, max_missed=90):
        self.dt = dt
        self.confirm_frames = confirm_frames
        self.max_missed = max_missed
        self.tracks = {}  # track_id → DroneKalmanFilter
        self.next_id = 0

    def predict_all(self, doppler_active=False, doppler_activity=0.0):
        """Run predict on all tracks."""
        for kf in self.tracks.values():
            kf.predict(doppler_active, doppler_activity)

    def update(self, camera_dets, doppler_data=None):
        """
        Update all tracks with new measurements.

        camera_dets: list of (x_m, depth_m, cam_confidence)
            x_m:      horizontal offset in metres from camera center
            depth_m:  stereo depth in metres
        doppler_data: dict from DopplerReader.get_data() or None
        """
        # Doppler info
        doppler_active = False
        doppler_conf = 0.0
        doppler_range = 0.0
        doppler_range_conf = 0.0
        if doppler_data and doppler_data['connected']:
            doppler_active = doppler_data['motion']
            doppler_range = doppler_data.get('estimated_range_m', 0.0)
            doppler_range_conf = doppler_data.get('range_confidence', 0.0)
            # Drone propeller signature: rapid toggling > 5 Hz
            freq = doppler_data['pulse_freq_hz']
            act = doppler_data['activity_level']
            if freq > 5.0 and act > 0.3:
                doppler_conf = min(1.0, freq / 30.0) * act
            elif doppler_active and freq > 2.0 and act > 0.5:
                doppler_conf = 0.3

        has_doppler_range = doppler_range > 0.1 and doppler_range_conf > 0.02

        # Predict all existing tracks
        self.predict_all(doppler_active,
                         doppler_data['activity_level'] if doppler_data else 0.0)

        # Associate camera detections with existing tracks. Gate widens
        # with range because stereo-depth noise grows roughly linearly with
        # distance — a flat 3 m was spawning duplicate tracks on the same
        # drone whenever depth jittered.
        used_det = set()
        for tid, kf in list(self.tracks.items()):
            est_depth = max(0.0, kf.x[1])
            best_dist = 3.0 + 0.15 * est_depth
            best_idx = -1
            for i, det in enumerate(camera_dets):
                if i in used_det:
                    continue
                dx = det[0] - kf.x[0]
                dy = det[1] - kf.x[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                det = camera_dets[best_idx]
                used_det.add(best_idx)
                x_m, depth_m, cam_conf = det[0], det[1], det[2]
                kf.update_camera(x_m, depth_m, cam_conf)
                # Apply Doppler radar range estimate
                if has_doppler_range:
                    kf.update_doppler_range(doppler_range, doppler_range_conf)
                kf.set_doppler(doppler_active, doppler_conf)
            else:
                kf.mark_missed()
                # Even without a camera match, apply Doppler range to missed tracks
                if has_doppler_range and kf.missed <= 3:
                    range_diff = abs(kf.range_m - doppler_range)
                    if range_diff < 5.0:
                        kf.update_doppler_range(doppler_range, doppler_range_conf)
                        kf.set_doppler(doppler_active, doppler_conf)

        # Spawn new tracks for unmatched camera detections
        for i, det in enumerate(camera_dets):
            if i in used_det:
                continue
            x_m, depth_m, cam_conf = det[0], det[1], det[2]
            kf = DroneKalmanFilter(self.next_id, self.dt)
            kf.initialize(x_m, depth_m)
            kf.camera_conf = cam_conf
            if has_doppler_range:
                kf.update_doppler_range(doppler_range, doppler_range_conf)
            kf.set_doppler(doppler_active, doppler_conf)

            self.tracks[self.next_id] = kf
            self.next_id += 1

        # Prune dead tracks
        for tid in list(self.tracks):
            kf = self.tracks[tid]
            limit = self.max_missed if kf.confirmed else 10
            if kf.missed > limit:
                del self.tracks[tid]

        # Build results
        results = {}
        for tid, kf in self.tracks.items():
            ft = kf.get_fused_track()
            if ft.is_confirmed or ft.age_frames < self.confirm_frames:
                results[tid] = ft
        return results
