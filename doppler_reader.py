#!/usr/bin/env python3
"""
CQRobot 10.525GHz Doppler Microwave Motion Sensor — GPIO Reader
----------------------------------------------------------------
Uses python-libgpiod to poll the sensor's digital output at high rate.
Matches the approach of the working C motion_sensor program: simple
input polling, no edge events, no bias flags.

The sensor output (active LOW):
  0 = motion detected ("Somebody is in this area!")
  1 = no motion ("No one!")

Wiring:
  Red   -> 5V (Pin 2 or 4)
  Black -> GND
  Green -> Pin 29 (gpiochip0 line 105)
"""

import threading
import time
from collections import deque

import numpy as np

import gpiod
from gpiod.line import Direction


CHIP_PATH = "/dev/gpiochip0"
LINE = 105  # Pin 29 — confirmed working with C motion_sensor program


class DopplerReader:
    """
    High-rate polling reader matching the working C implementation.
    No edge events, no bias — just rapid get_value() calls.
    """

    def __init__(self, chip_path=CHIP_PATH, line=LINE, poll_hz=100):
        self.chip_path = chip_path
        self.line = line
        self.poll_interval = 1.0 / poll_hz

        self.running = False
        self.poll_thread = None
        self.analysis_thread = None
        self.lock = threading.Lock()

        # State
        self.motion = False
        self.raw_gpio = 1
        self.motion_events = 0
        self.last_motion_time = 0.0
        self.last_idle_time = 0.0
        self.connected = False
        self.edge_count = 0

        # Pulse frequency analysis
        self._edge_times = deque(maxlen=100)
        self.pulse_freq_hz = 0.0
        self.activity_level = 0.0

        # History for dashboard
        self.motion_history = deque(maxlen=400)
        self.motion_history.append((time.time(), False))

        # Range estimation
        self.estimated_range_m = 0.0
        self.range_confidence = 0.0
        self._duty_cycle = 0.0

        self.last_error = ""

    def start(self):
        self.running = True
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()

    def stop(self):
        self.running = False
        for t in (self.poll_thread, self.analysis_thread):
            if t:
                t.join(timeout=2)

    def _poll_loop(self):
        """
        Simple polling loop — same approach as the working C program.
        Request line as INPUT with no bias, poll get_value() rapidly.
        """
        while self.running:
            req = None
            try:
                req = gpiod.request_lines(
                    self.chip_path,
                    consumer="doppler-reader",
                    config={
                        self.line: gpiod.LineSettings(
                            direction=Direction.INPUT,
                        )
                    },
                )

                with self.lock:
                    self.connected = True
                    self.last_error = ""

                prev_motion = None

                while self.running:
                    val = req.get_value(self.line)
                    # gpiod v2: compare against Value enum directly
                    from gpiod.line import Value
                    raw = 0 if val == Value.INACTIVE else 1
                    motion = (raw == 0)  # active LOW
                    now = time.time()

                    with self.lock:
                        self.raw_gpio = raw
                        self.motion = motion
                        self.motion_history.append((now, motion))

                        if motion:
                            self.last_motion_time = now

                        # Detect transitions
                        if prev_motion is not None and motion != prev_motion:
                            self.edge_count += 1
                            self._edge_times.append(now)
                            if motion:
                                self.motion_events += 1
                            else:
                                self.last_idle_time = now

                    prev_motion = motion
                    time.sleep(self.poll_interval)

            except Exception as e:
                with self.lock:
                    self.connected = False
                    self.last_error = f"{type(e).__name__}: {e}"[:200]
            finally:
                if req:
                    try:
                        req.release()
                    except Exception:
                        pass
                if self.running:
                    time.sleep(1)

    def _analysis_loop(self):
        """Compute derived metrics: pulse frequency, duty cycle, range."""
        while self.running:
            now = time.time()
            with self.lock:
                recent_edges = [t for t in self._edge_times if now - t < 2.0]
                if len(recent_edges) >= 2:
                    span = recent_edges[-1] - recent_edges[0]
                    if span > 0:
                        self.pulse_freq_hz = (len(recent_edges) - 1) / span
                    else:
                        self.pulse_freq_hz = 0.0
                else:
                    self.pulse_freq_hz = max(0, self.pulse_freq_hz * 0.8)

                recent_hist = [(t, m) for t, m in self.motion_history
                               if now - t < 2.0]
                if len(recent_hist) >= 2:
                    motion_time = 0.0
                    total_time = 0.0
                    for i in range(1, len(recent_hist)):
                        dt = recent_hist[i][0] - recent_hist[i-1][0]
                        total_time += dt
                        if recent_hist[i-1][1]:
                            motion_time += dt
                    dt_last = now - recent_hist[-1][0]
                    total_time += dt_last
                    if self.motion:
                        motion_time += dt_last
                    if total_time > 0:
                        self._duty_cycle = min(1.0, motion_time / total_time)
                    else:
                        self._duty_cycle = 1.0 if self.motion else 0.0
                else:
                    self._duty_cycle = 1.0 if self.motion else 0.0

                self.activity_level = self._duty_cycle

                recent_edge_count = len(recent_edges)
                has_signal = (self._duty_cycle > 0.01 or
                              recent_edge_count >= 1 or
                              self.motion)

                if has_signal:
                    MAX_RANGE = 10.0
                    MIN_RANGE = 0.5
                    effective_duty = max(self._duty_cycle, 0.02)
                    raw_range = min(MAX_RANGE,
                                    MIN_RANGE / effective_duty ** 0.5)
                    if self.pulse_freq_hz > 0.5:
                        freq_factor = max(0.5, 1.0 - self.pulse_freq_hz / 60.0)
                        raw_range *= freq_factor
                    raw_range = float(np.clip(raw_range, MIN_RANGE, MAX_RANGE))

                    alpha = 0.2
                    if self.estimated_range_m > 0:
                        self.estimated_range_m = (alpha * raw_range +
                                                  (1 - alpha) * self.estimated_range_m)
                    else:
                        self.estimated_range_m = raw_range

                    self.range_confidence = min(1.0,
                        self._duty_cycle * 0.5 +
                        min(self.pulse_freq_hz / 15.0, 0.3) +
                        min(recent_edge_count / 10.0, 0.3))
                else:
                    self.estimated_range_m *= 0.97
                    self.range_confidence *= 0.92

            time.sleep(0.1)

    def get_data(self):
        with self.lock:
            now = time.time()
            motion_age = (now - self.last_motion_time
                          if self.last_motion_time else float('inf'))
            return {
                'connected': self.connected,
                'motion': self.motion,
                'raw_gpio': self.raw_gpio,
                'edge_count': self.edge_count,
                'motion_events': self.motion_events,
                'last_motion_time': self.last_motion_time,
                'motion_age_s': motion_age,
                'pulse_freq_hz': self.pulse_freq_hz,
                'activity_level': self.activity_level,
                'duty_cycle': self._duty_cycle,
                'estimated_range_m': self.estimated_range_m,
                'range_confidence': self.range_confidence,
                'history': list(self.motion_history),
                'gpiomon_error': self.last_error,
            }

    def is_drone_signature(self):
        with self.lock:
            if self.pulse_freq_hz > 5.0 and self.activity_level > 0.3:
                conf = min(1.0, self.pulse_freq_hz / 30.0) * self.activity_level
                return True, conf
            if self.motion and self.pulse_freq_hz > 2.0 and self.activity_level > 0.5:
                return True, 0.3
            return False, 0.0
