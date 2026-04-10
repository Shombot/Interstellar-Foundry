#!/usr/bin/env python3
"""
CQRobot 10.525GHz Doppler Microwave Motion Sensor — Threaded Reader
--------------------------------------------------------------------
Reads the GPIO digital output (active LOW) in a background thread and
provides motion state, event count, and pulse frequency analysis.

The sensor outputs a binary signal:
  0 = motion detected (active low)
  1 = no motion

By measuring the rate and duration of motion pulses we can infer
a coarse "activity level" that correlates with drone propeller
micro-Doppler signatures (rapid, periodic toggling).

Wiring:
  Red   -> 3.3V (Pin 1 or 17)
  Black -> GND  (Pin 6, 9, 14, etc.)
  Green -> Pin 29 (GPIO01 / PQ.05 / gpiochip0 line 105)
"""

import subprocess
import threading
import time
from collections import deque

import numpy as np


CHIP = "gpiochip0"
LINE = 105  # GPIO01 = PQ.05 = physical pin 29


class DopplerReader:
    """Background thread that polls the CQRobot Doppler sensor via gpioget."""

    def __init__(self, chip=CHIP, line=LINE, poll_hz=100):
        self.chip = chip
        self.line = line
        self.poll_interval = 1.0 / poll_hz

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # State
        self.motion = False
        self.motion_events = 0          # total rising-edge count
        self.last_motion_time = 0.0     # epoch of last motion=True
        self.last_idle_time = 0.0       # epoch of last motion=False
        self.connected = False

        # Pulse frequency analysis (drone propellers create rapid toggling)
        self._edge_times = deque(maxlen=50)   # timestamps of motion edges
        self.pulse_freq_hz = 0.0              # estimated toggle frequency
        self.activity_level = 0.0             # 0.0–1.0 coarse activity metric

        # Short history for dashboard display
        self.motion_history = deque(maxlen=200)  # (timestamp, motion_bool)

        # Range estimation from signal characteristics
        # The 10.525GHz CW Doppler sensor's detection envelope (duty cycle,
        # pulse rate, sustained activity) correlates with target proximity:
        #   - closer targets → stronger reflected signal → higher duty cycle
        #   - farther targets → weaker signal → sporadic, low duty cycle
        # Max detection range ~7-12m depending on target RCS.
        self.estimated_range_m = 0.0     # coarse range estimate
        self.range_confidence = 0.0      # how much to trust the estimate
        self._duty_cycle = 0.0           # fraction of time in motion state

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _read_gpio(self):
        try:
            r = subprocess.run(
                ["gpioget", "-B", "pull-up", self.chip, str(self.line)],
                capture_output=True, text=True, timeout=1)
            return r.stdout.strip()
        except Exception:
            return None

    def _poll_loop(self):
        prev_motion = None
        # Check sensor presence once
        val = self._read_gpio()
        if val is not None:
            with self.lock:
                self.connected = True

        while self.running:
            val = self._read_gpio()
            if val is None:
                with self.lock:
                    self.connected = False
                time.sleep(0.5)
                continue

            now = time.time()
            motion = (val == "0")  # active low

            with self.lock:
                self.connected = True
                self.motion = motion
                self.motion_history.append((now, motion))

                if motion:
                    self.last_motion_time = now

                # Detect edges (transitions)
                if prev_motion is not None and motion != prev_motion:
                    self._edge_times.append(now)
                    if motion:
                        self.motion_events += 1
                    if not motion:
                        self.last_idle_time = now

                # Compute pulse frequency from recent edges
                edges = list(self._edge_times)
                if len(edges) >= 4:
                    # Only consider edges in last 2 seconds
                    recent = [t for t in edges if now - t < 2.0]
                    if len(recent) >= 4:
                        intervals = [recent[i+1] - recent[i]
                                     for i in range(len(recent)-1)]
                        avg_interval = sum(intervals) / len(intervals)
                        if avg_interval > 0:
                            self.pulse_freq_hz = 1.0 / avg_interval
                        else:
                            self.pulse_freq_hz = 0.0
                    else:
                        self.pulse_freq_hz = max(0, self.pulse_freq_hz - 0.5)
                else:
                    self.pulse_freq_hz = 0.0

                # Activity level: based on recent motion events in last 2s
                recent_events = sum(1 for t, m in self.motion_history
                                    if now - t < 2.0 and m)
                total_recent = sum(1 for t, m in self.motion_history
                                   if now - t < 2.0)
                if total_recent > 0:
                    self.activity_level = min(1.0, recent_events / total_recent)
                else:
                    self.activity_level = 0.0

                # --- Range estimation from signal envelope ---
                # Duty cycle: fraction of recent samples where motion=True
                # Higher duty cycle → closer target (stronger return)
                self._duty_cycle = (recent_events / total_recent
                                    if total_recent > 0 else 0.0)

                if self._duty_cycle > 0.05:
                    # Map duty cycle to range: 100% duty → ~0.5m, ~5% → ~10m
                    # Inverse square law: signal strength ∝ 1/r^2
                    # duty_cycle ∝ signal_strength ∝ 1/r^2
                    # So r ∝ 1/sqrt(duty_cycle)
                    MAX_RANGE = 10.0  # sensor max effective range (m)
                    MIN_RANGE = 0.5
                    raw_range = min(MAX_RANGE,
                                    MIN_RANGE / max(self._duty_cycle, 0.01) ** 0.5)

                    # Refine with pulse frequency — faster toggling = closer
                    if self.pulse_freq_hz > 1.0:
                        freq_factor = max(0.5, 1.0 - self.pulse_freq_hz / 60.0)
                        raw_range *= freq_factor

                    raw_range = np.clip(raw_range, MIN_RANGE, MAX_RANGE)

                    # Smooth the estimate (low-pass filter)
                    alpha = 0.15  # smoothing factor
                    if self.estimated_range_m > 0:
                        self.estimated_range_m = (alpha * raw_range +
                                                  (1 - alpha) * self.estimated_range_m)
                    else:
                        self.estimated_range_m = raw_range

                    # Confidence: higher with more data and sustained motion
                    self.range_confidence = min(1.0,
                        self._duty_cycle * 0.6 +
                        min(self.pulse_freq_hz / 20.0, 0.4))
                else:
                    # No motion — decay range estimate
                    self.estimated_range_m *= 0.98
                    self.range_confidence *= 0.95

            prev_motion = motion
            time.sleep(self.poll_interval)

    def get_data(self):
        """Thread-safe snapshot of current Doppler sensor state."""
        with self.lock:
            now = time.time()
            motion_age = now - self.last_motion_time if self.last_motion_time else float('inf')
            return {
                'connected': self.connected,
                'motion': self.motion,
                'motion_events': self.motion_events,
                'last_motion_time': self.last_motion_time,
                'motion_age_s': motion_age,
                'pulse_freq_hz': self.pulse_freq_hz,
                'activity_level': self.activity_level,
                'duty_cycle': self._duty_cycle,
                'estimated_range_m': self.estimated_range_m,
                'range_confidence': self.range_confidence,
                'history': list(self.motion_history),
            }

    def is_drone_signature(self):
        """
        Heuristic: drone propellers cause rapid, periodic toggling.
        Returns (is_drone_like, confidence).
        """
        with self.lock:
            # Drone propellers typically cause rapid oscillation (>5 Hz toggle)
            if self.pulse_freq_hz > 5.0 and self.activity_level > 0.3:
                conf = min(1.0, self.pulse_freq_hz / 30.0) * self.activity_level
                return True, conf
            # Sustained motion with moderate frequency could be a hovering drone
            if self.motion and self.pulse_freq_hz > 2.0 and self.activity_level > 0.5:
                return True, 0.3
            return False, 0.0
