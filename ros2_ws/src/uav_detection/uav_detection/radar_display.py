#!/usr/bin/env python3
"""
FM24-NP100 24GHz mmWave Radar Display
--------------------------------------
Displays real-time distance and spectral data from the radar over UART.

Mode A (Pin 6 floating): 8-byte frames, distance only
Mode B (Pin 6 → GND):   134-byte frames, distance + 126 spectral bins

Usage:
    python3 radar_display.py [--port /dev/ttyTHS1] [--baud 57600]
"""

import argparse
import sys
import time
import struct
import threading
from collections import deque

import serial
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Protocol constants ---
HEADER = b'\xff\xff\xff'
TAIL = b'\x00\x00\x00'
FRAME_MODE_A = 8    # 3 header + 2 dist + 3 tail
FRAME_MODE_B = 134  # 3 header + 2 dist + 126 spectral + 3 tail
SPECTRAL_BINS = 126
MAX_SPECTRAL_VAL = 44
DETECTION_RANGE_M = 20.0

# --- Radar reader thread ---
class RadarReader:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.serial = None
        self.running = False
        self.mode = None  # 'A' or 'B', auto-detected

        self.distance_cm = 0
        self.spectrum = np.zeros(SPECTRAL_BINS, dtype=np.float32)
        self.distance_history = deque(maxlen=200)
        self.lock = threading.Lock()
        self.frames_received = 0
        self.last_frame_time = 0

    def connect(self):
        self.serial = serial.Serial(self.port, self.baud, timeout=1)
        self.serial.reset_input_buffer()
        print(f"Connected to {self.port} @ {self.baud} baud")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()

    def _sync_to_header(self, buf):
        """Find the header pattern in the buffer and return the index."""
        idx = buf.find(HEADER)
        return idx

    def _read_loop(self):
        buf = b''
        while self.running:
            try:
                incoming = self.serial.read(self.serial.in_waiting or 1)
                if not incoming:
                    continue
                buf += incoming

                while True:
                    # Find header
                    idx = self._sync_to_header(buf)
                    if idx < 0:
                        # Keep last 2 bytes in case header straddles chunks
                        buf = buf[-2:] if len(buf) > 2 else buf
                        break

                    # Discard bytes before header
                    if idx > 0:
                        buf = buf[idx:]

                    # Try Mode B first (larger frame), then Mode A
                    if len(buf) >= FRAME_MODE_B:
                        if buf[FRAME_MODE_B - 3:FRAME_MODE_B] == TAIL:
                            self._parse_frame_b(buf[:FRAME_MODE_B])
                            buf = buf[FRAME_MODE_B:]
                            continue
                        elif len(buf) >= FRAME_MODE_A and buf[FRAME_MODE_A - 3:FRAME_MODE_A] == TAIL:
                            self._parse_frame_a(buf[:FRAME_MODE_A])
                            buf = buf[FRAME_MODE_A:]
                            continue
                        else:
                            # Bad frame, skip one byte and retry
                            buf = buf[1:]
                            continue
                    elif len(buf) >= FRAME_MODE_A:
                        if buf[FRAME_MODE_A - 3:FRAME_MODE_A] == TAIL:
                            self._parse_frame_a(buf[:FRAME_MODE_A])
                            buf = buf[FRAME_MODE_A:]
                            continue
                        # Not enough for Mode B yet, could still be Mode B partial
                        # Only skip if this clearly isn't a valid Mode A frame
                        if buf[FRAME_MODE_A - 3:FRAME_MODE_A] != TAIL:
                            # Wait for more data (could be Mode B)
                            break
                    else:
                        # Not enough data yet
                        break

            except serial.SerialException as e:
                print(f"Serial error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(0.1)

    def _parse_frame_a(self, frame):
        dist_h, dist_l = frame[3], frame[4]
        distance_cm = (dist_h << 8) | dist_l

        with self.lock:
            self.mode = 'A'
            self.distance_cm = distance_cm
            self.distance_history.append(distance_cm / 100.0)
            self.frames_received += 1
            self.last_frame_time = time.time()

    def _parse_frame_b(self, frame):
        dist_h, dist_l = frame[3], frame[4]
        distance_cm = (dist_h << 8) | dist_l
        spectral_data = np.frombuffer(frame[5:5 + SPECTRAL_BINS], dtype=np.uint8).astype(np.float32)

        with self.lock:
            self.mode = 'B'
            self.distance_cm = distance_cm
            self.spectrum = spectral_data
            self.distance_history.append(distance_cm / 100.0)
            self.frames_received += 1
            self.last_frame_time = time.time()

    def get_data(self):
        with self.lock:
            return {
                'mode': self.mode,
                'distance_cm': self.distance_cm,
                'distance_m': self.distance_cm / 100.0,
                'spectrum': self.spectrum.copy(),
                'history': list(self.distance_history),
                'frames': self.frames_received,
                'last_frame': self.last_frame_time,
            }


# --- Display ---
class RadarDisplay:
    def __init__(self, reader):
        self.reader = reader
        self.fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
        self.fig.suptitle("FM24-NP100 24GHz mmWave Radar", color='white',
                          fontsize=16, fontweight='bold')

        # Create subplots
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                                    left=0.08, right=0.95, top=0.90, bottom=0.08)

        # Top-left: Distance gauge
        self.ax_dist = self.fig.add_subplot(gs[0, 0])
        self._setup_distance_display()

        # Top-right: Spectral plot
        self.ax_spec = self.fig.add_subplot(gs[0, 1])
        self._setup_spectral_plot()

        # Bottom-left: Distance over time
        self.ax_hist = self.fig.add_subplot(gs[1, 0])
        self._setup_history_plot()

        # Bottom-right: Info panel
        self.ax_info = self.fig.add_subplot(gs[1, 1])
        self._setup_info_panel()

    def _setup_distance_display(self):
        ax = self.ax_dist
        ax.set_facecolor('#16213e')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, DETECTION_RANGE_M)
        ax.set_ylabel("Distance (m)", color='white', fontsize=11)
        ax.set_title("Distance", color='#00d4ff', fontsize=13)
        ax.tick_params(colors='white')
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_color('#333')

        # Bar
        self.dist_bar = ax.bar(0.5, 0, width=0.6, color='#00d4ff', alpha=0.8)[0]
        self.dist_text = ax.text(0.5, 1, "-- m", ha='center', va='bottom',
                                  color='white', fontsize=20, fontweight='bold')

    def _setup_spectral_plot(self):
        ax = self.ax_spec
        ax.set_facecolor('#16213e')
        ax.set_xlim(0, SPECTRAL_BINS)
        ax.set_ylim(0, MAX_SPECTRAL_VAL + 5)
        ax.set_xlabel("Spectral Bin", color='white', fontsize=10)
        ax.set_ylabel("Amplitude", color='white', fontsize=10)
        ax.set_title("FMCW Spectrum", color='#00d4ff', fontsize=13)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

        x = np.arange(SPECTRAL_BINS)
        self.spec_line, = ax.plot(x, np.zeros(SPECTRAL_BINS), color='#00ff88',
                                   linewidth=1.2, alpha=0.9)
        self.spec_fill = ax.fill_between(x, 0, np.zeros(SPECTRAL_BINS),
                                          color='#00ff88', alpha=0.15)
        self.spec_peak_marker, = ax.plot([], [], 'rv', markersize=10)
        self.spec_peak_text = ax.text(0, 0, '', color='red', fontsize=9, ha='center')

    def _setup_history_plot(self):
        ax = self.ax_hist
        ax.set_facecolor('#16213e')
        ax.set_ylim(0, DETECTION_RANGE_M)
        ax.set_xlim(0, 200)
        ax.set_xlabel("Samples", color='white', fontsize=10)
        ax.set_ylabel("Distance (m)", color='white', fontsize=10)
        ax.set_title("Distance History", color='#00d4ff', fontsize=13)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

        self.hist_line, = ax.plot([], [], color='#ff6b6b', linewidth=1.5)

    def _setup_info_panel(self):
        ax = self.ax_info
        ax.set_facecolor('#16213e')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Status", color='#00d4ff', fontsize=13)
        for spine in ax.spines.values():
            spine.set_color('#333')

        self.info_text = ax.text(0.05, 0.85, "Waiting for data...",
                                  color='white', fontsize=11, va='top',
                                  family='monospace', transform=ax.transAxes)

    def update(self, frame_num):
        data = self.reader.get_data()
        artists = []

        dist_m = data['distance_m']
        spectrum = data['spectrum']
        history = data['history']
        mode = data['mode']
        frames = data['frames']

        # -- Distance bar --
        self.dist_bar.set_height(dist_m)
        color = '#00d4ff' if dist_m < 5 else '#ffaa00' if dist_m < 15 else '#ff4444'
        self.dist_bar.set_color(color)
        self.dist_text.set_text(f"{dist_m:.2f} m")
        self.dist_text.set_y(min(dist_m + 0.3, DETECTION_RANGE_M - 1))
        artists += [self.dist_bar, self.dist_text]

        # -- Spectral plot --
        self.spec_line.set_ydata(spectrum)
        # Update fill
        self.spec_fill.remove()
        x = np.arange(SPECTRAL_BINS)
        self.spec_fill = self.ax_spec.fill_between(x, 0, spectrum,
                                                     color='#00ff88', alpha=0.15)
        # Mark peak
        if np.any(spectrum > 0):
            peak_idx = np.argmax(spectrum)
            peak_val = spectrum[peak_idx]
            self.spec_peak_marker.set_data([peak_idx], [peak_val])
            self.spec_peak_text.set_position((peak_idx, peak_val + 2))
            self.spec_peak_text.set_text(f"bin {peak_idx}\namp {peak_val:.0f}")
        artists += [self.spec_line, self.spec_fill, self.spec_peak_marker, self.spec_peak_text]

        # -- History plot --
        if history:
            self.hist_line.set_data(range(len(history)), history)
            self.ax_hist.set_xlim(0, max(len(history), 10))
        artists.append(self.hist_line)

        # -- Info panel --
        elapsed = time.time() - data['last_frame'] if data['last_frame'] else float('inf')
        status = "LIVE" if elapsed < 2 else "NO DATA"
        status_color = '#00ff88' if status == "LIVE" else '#ff4444'

        info_str = (
            f"Status:  {status}\n"
            f"Mode:    {'B (dist+spectrum)' if mode == 'B' else 'A (dist only)' if mode == 'A' else 'detecting...'}\n"
            f"Port:    {self.reader.port}\n"
            f"Baud:    {self.reader.baud}\n"
            f"Frames:  {frames}\n"
            f"Dist:    {data['distance_cm']} cm ({dist_m:.2f} m)\n"
        )
        if mode == 'B':
            peak_bin = np.argmax(spectrum) if np.any(spectrum > 0) else 0
            info_str += f"Peak:    bin {peak_bin}, amp {spectrum[peak_bin]:.0f}\n"

        self.info_text.set_text(info_str)
        self.info_text.set_color(status_color)
        artists.append(self.info_text)

        return artists

    def run(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=100, blit=False, cache_frame_data=False)
        plt.show()


# --- Diagnostic mode (no GUI) ---
def run_diagnostic(port, baud):
    """Print raw hex data for debugging wiring / connection issues."""
    print(f"=== Diagnostic Mode ===")
    print(f"Port: {port}  Baud: {baud}")
    print(f"Listening for raw data (Ctrl+C to stop)...\n")

    ser = serial.Serial(port, baud, timeout=1)
    ser.reset_input_buffer()

    try:
        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                hex_str = data.hex(' ')
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
                print(f"[{len(data):3d} bytes] {hex_str}")
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


def main():
    parser = argparse.ArgumentParser(description="FM24-NP100 Radar Display")
    parser.add_argument('--port', default='/dev/ttyTHS1',
                        help='Serial port (default: /dev/ttyTHS1)')
    parser.add_argument('--baud', type=int, default=57600,
                        help='Baud rate (default: 57600)')
    parser.add_argument('--diag', action='store_true',
                        help='Diagnostic mode: print raw hex data')
    args = parser.parse_args()

    if args.diag:
        run_diagnostic(args.port, args.baud)
        return

    reader = RadarReader(args.port, args.baud)
    try:
        reader.connect()
    except serial.SerialException as e:
        print(f"Error: Cannot open {args.port}: {e}")
        print("Check: permissions (chmod 666), correct port, wiring")
        sys.exit(1)

    reader.start()
    print("Starting display... (close window or Ctrl+C to stop)")
    print("If no data appears, run with --diag to debug the connection.")

    display = RadarDisplay(reader)
    try:
        display.run()
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        print("Done.")


if __name__ == '__main__':
    main()
