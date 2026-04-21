"""
Live Drone Detection Dashboard: Oak-D S2 camera + TI IWR6843AOPEVM mmWave radar.

Left half  : Oak-D RGB frame with YOLOv6-nano detections (runs on the Myriad X VPU).
Right half : Top-down PPI plot of the TI radar's detected-points cloud, with the
             nearest target's range / azimuth / Doppler.

The two sensors run independently — the radar reader is a daemon thread that
parses TI's TLV stream, and the camera loop pulls the latest point cloud each
frame. No fusion yet; each panel reports its own coordinates for the operator.
"""
import os
import struct
import threading
import time

import cv2
import numpy as np

try:
    import serial
except ImportError:
    serial = None

import depthai as dai

# ---------- Camera / detector config ----------
MODEL_SLUG = "yolov6-nano"
CONF_THRESHOLD = 0.25
FPS_TARGET = 30
WANT_CLASSES = {
    "airplane", "bird", "surfboard", "cell phone",
    "mouse", "snowboard", "skateboard", "remote",
}
TRACK_TTL_SEC = 0.5             # keep a box alive this long after the last YOLO hit
TRACK_IOU_MATCH = 0.3           # min IoU to treat a new detection as the same track

os.environ.setdefault(
    "DEPTHAI_ZOO_CACHE_PATH",
    os.path.expanduser("~/.depthai_cache"),
)
os.makedirs(os.environ["DEPTHAI_ZOO_CACHE_PATH"], exist_ok=True)

# ---------- TI IWR6843AOPEVM config ----------
# XDS-110 on the EVM enumerates two UARTs; Linux typically assigns
# /dev/ttyUSB0 (CLI) and /dev/ttyUSB1 (Data).
RADAR_CLI_PORT = os.environ.get("RADAR_CLI_PORT", "/dev/ttyUSB0")
RADAR_DATA_PORT = os.environ.get("RADAR_DATA_PORT", "/dev/ttyUSB1")
RADAR_CLI_BAUD = 115200
RADAR_DATA_BAUD = 921600
RADAR_CONFIG_FILE = os.environ.get(
    "RADAR_CONFIG_FILE",
    os.path.expanduser("~/iwr6843aop_drone.cfg"),
)
RADAR_MAX_RANGE_M = 15.0        # horizon of the PPI plot
RADAR_PANEL_W = 480             # pixels — width of the radar side panel
# TI mmWave demo wire format
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_LEN = 40                 # 8-byte magic + 8 × uint32
TLV_DETECTED_POINTS = 1         # payload: N × (x, y, z, velocity) float32
TLV_SIDE_INFO = 7               # payload: N × (snr, noise) uint16 — per-point SNR


class TiMmwRadarReader(threading.Thread):
    """Streams TLV frames from a TI IWR6843AOP demo and exposes the latest cloud."""

    def __init__(self):
        super().__init__(daemon=True)
        self._cli = None
        self._data = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._points = np.empty((0, 5), dtype=np.float32)  # x, y, z, v, snr
        self._frames = 0
        self._last_frame_t = 0.0
        self.status = "init"                                # init / live / stale / error / nodev

    def stop(self):
        self._stop_evt.set()



    def _open(self):
        if serial is None:
            print("pyserial not installed — radar disabled. `pip install pyserial`")
            self.status = "nodev"
            return False
        try:
            self._data = serial.Serial(RADAR_DATA_PORT, RADAR_DATA_BAUD, timeout=0.2)
            self._cli = serial.Serial(RADAR_CLI_PORT, RADAR_CLI_BAUD, timeout=0.2)
        except serial.SerialException as e:
            print(f"Radar serial open failed ({e}); continuing without radar")
            self.status = "nodev"
            return False
        self._send_config()
        return True

    def _send_config(self):
        """Push the TI CLI script line-by-line over the control UART."""
        if not os.path.exists(RADAR_CONFIG_FILE):
            print(
                f"Radar config '{RADAR_CONFIG_FILE}' not found — assuming the "
                "sensor is already running. Set RADAR_CONFIG_FILE to override."
            )
            return
        print(f"Sending radar config from {RADAR_CONFIG_FILE}")
        with open(RADAR_CONFIG_FILE) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("%"):
                    continue
                self._cli.write((line + "\n").encode())
                # The demo CLI echoes back each command; small pause keeps
                # the command FIFO from overflowing on slower firmwares.
                time.sleep(0.02)
                self._cli.reset_input_buffer()

    def _read_exact(self, n, buf):
        """Top up `buf` until it has at least n bytes, or return None on stop."""
        while len(buf) < n:
            if self._stop_evt.is_set():
                return None
            chunk = self._data.read(n - len(buf))
            if not chunk:
                return buf  # caller will re-check length
            buf += chunk
        return buf

    def _read_frame(self, buf):
        """Parse one TLV frame out of the running buffer. Returns (points, new_buf)."""
        # Resync to magic word.
        idx = buf.find(MAGIC_WORD)
        while idx < 0 and not self._stop_evt.is_set():
            buf += self._data.read(256)
            if len(buf) > 8192:
                buf = buf[-1024:]                 # don't grow unbounded on garbage
            idx = buf.find(MAGIC_WORD)
        if self._stop_evt.is_set():
            return None, buf
        buf = buf[idx:]

        buf = self._read_exact(HEADER_LEN, buf)
        if buf is None or len(buf) < HEADER_LEN:
            return None, buf or b""

        # Header (after magic) is 8 uint32: version, totalPacketLen, platform,
        # frameNumber, timeCpuCycles, numDetectedObj, numTLVs, subFrameNumber.
        # Only total_len and numTLVs drive parsing.
        header = struct.unpack_from("<8I", buf, 8)
        total_len = header[1]
        num_tlvs = header[6]

        if total_len < HEADER_LEN or total_len > 65536:
            return None, buf[1:]                  # corrupt; drop one byte and re-sync

        buf = self._read_exact(total_len, buf)
        if buf is None or len(buf) < total_len:
            return None, buf or b""

        xyzv = None                               # TLV 1 payload, shape (N, 4)
        snr = None                                # TLV 7 payload, shape (N,) in 0.1 dB
        off = HEADER_LEN
        for _ in range(num_tlvs):
            if off + 8 > total_len:
                break
            t_type, t_len = struct.unpack_from("<2I", buf, off)
            off += 8
            if off + t_len > total_len:
                break
            if t_type == TLV_DETECTED_POINTS and t_len % 16 == 0:
                n = t_len // 16
                xyzv = np.frombuffer(
                    buf, dtype=np.float32, count=n * 4, offset=off
                ).reshape(-1, 4).copy()
            elif t_type == TLV_SIDE_INFO and t_len % 4 == 0:
                # Each record is (snr_uint16, noise_uint16), SNR in 0.1 dB units.
                m = t_len // 4
                side = np.frombuffer(
                    buf, dtype=np.uint16, count=m * 2, offset=off
                ).reshape(-1, 2)
                snr = side[:, 0].astype(np.float32) * 0.1
            off += t_len

        if xyzv is None:
            points = np.empty((0, 5), dtype=np.float32)
        else:
            if snr is None or len(snr) != len(xyzv):
                snr_col = np.zeros((len(xyzv), 1), dtype=np.float32)
            else:
                snr_col = snr.reshape(-1, 1)
            points = np.hstack([xyzv, snr_col]).astype(np.float32)
        return points, buf[total_len:]

    def run(self):
        if not self._open():
            return
        self.status = "live"
        rolling = b""
        while not self._stop_evt.is_set():
            try:
                pts, rolling = self._read_frame(rolling)
            except Exception as e:
                print(f"Radar read error: {e}")
                self.status = "error"
                time.sleep(0.25)
                continue
            if pts is None:
                continue
            with self._lock:
                self._points = pts
                self._frames += 1
                self._last_frame_t = time.monotonic()

    def snapshot(self):
        with self._lock:
            pts = self._points.copy()
            frames = self._frames
            last_t = self._last_frame_t
        age = time.monotonic() - last_t if last_t else float("inf")
        if self.status in ("nodev", "error"):
            status = self.status
        else:
            status = "live" if age < 1.5 else "stale"
        return pts, frames, status


def _iou(b1, b2):
    """IoU of two (x1, y1, x2, y2) axis-aligned boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def draw_radar_panel(height, points, status, frames, max_range=RADAR_MAX_RANGE_M):
    """Render a top-down PPI view matching the camera frame's height."""
    w = RADAR_PANEL_W
    panel = np.full((height, w, 3), 16, dtype=np.uint8)
    cx = w // 2
    cy = height - 24
    draw_h = cy - 30                              # leave room for title / footer
    ppm = draw_h / max_range                      # pixels per meter

    # Range rings + azimuth graticule
    for frac in (0.25, 0.5, 0.75, 1.0):
        r_px = int(frac * max_range * ppm)
        cv2.circle(panel, (cx, cy), r_px, (40, 80, 40), 1)
        cv2.putText(
            panel, f"{frac * max_range:.0f}m",
            (cx + 4, cy - r_px + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 0), 1,
        )
    for ang_deg in (-60, -30, 0, 30, 60):
        ang = np.radians(ang_deg)
        ex = int(cx + np.sin(ang) * max_range * ppm)
        ey = int(cy - np.cos(ang) * max_range * ppm)
        cv2.line(panel, (cx, cy), (ex, ey), (30, 60, 30), 1)

    # Header
    status_color = {
        "live": (0, 255, 255), "stale": (40, 140, 200),
        "nodev": (120, 120, 120), "error": (0, 0, 255), "init": (120, 120, 120),
    }[status]
    cv2.putText(panel, "mmWave RADAR (top-down)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(panel, f"[{status.upper()}] frames={frames}", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    if status == "nodev":
        cv2.putText(panel, "radar not connected", (10, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        return panel

    # Draw every in-range return in green; remember the strongest one (highest
    # SNR) so we can overlay a red ring + readout for it at the end.
    strongest = None                              # (snr, r, x, y, z, v)
    any_in_range = False
    for x, y, z, v, snr in points:
        if y <= 0 or y > max_range:
            continue
        px = int(cx + x * ppm)
        py = int(cy - y * ppm)
        if not (0 <= px < w and 0 <= py < height):
            continue
        any_in_range = True
        cv2.circle(panel, (px, py), 3, (0, 180, 0), -1)
        if strongest is None or snr > strongest[0]:
            strongest = (float(snr), float(np.hypot(x, y)),
                         float(x), float(y), float(z), float(v))

    if strongest is not None:
        snr_val, r, x, y, z, v = strongest
        px = int(cx + x * ppm)
        py = int(cy - y * ppm)
        cv2.circle(panel, (px, py), 10, (0, 0, 255), 2)
        az_deg = float(np.degrees(np.arctan2(x, y)))
        cv2.putText(
            panel,
            f"TGT  r={r:.2f}m  az={az_deg:+.1f}deg  v={v:+.2f}m/s  "
            f"z={z:+.2f}m  SNR={snr_val:.1f}dB",
            (10, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1,
        )
    elif not any_in_range:
        cv2.putText(panel, "no returns in range", (10, cy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    return panel


def main():
    radar = TiMmwRadarReader()
    radar.start()

    print(f"Opening Oak-D S2 and loading model '{MODEL_SLUG}' from zoo...")
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        model_desc = dai.NNModelDescription(MODEL_SLUG, platform="RVC2")
        nn = pipeline.create(dai.node.DetectionNetwork).build(
            cam, model_desc, fps=FPS_TARGET
        )
        nn.setConfidenceThreshold(CONF_THRESHOLD)
        class_names = nn.getClasses() or []

        det_queue = nn.out.createOutputQueue()
        frame_queue = nn.passthrough.createOutputQueue()

        pipeline.start()
        print("Detection running. Press 'q' to quit.")

        win_name = "Drone Dashboard — Oak-D + TI mmWave"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 480)

        last_t = time.monotonic()
        fps_ema = 0.0
        shape_logged = False
        tracks = []                               # [{bbox, cls, conf, last_seen}]
        next_track_id = 0

        try:
            while pipeline.isRunning():
                img = frame_queue.get()
                dets = det_queue.get()
                if img is None or dets is None:
                    continue

                frame = img.getCvFrame()
                h, w = frame.shape[:2]
                if not shape_logged:
                    print(f"Camera frame shape: {frame.shape} (h={h}, w={w})")
                    shape_logged = True

                now = time.monotonic()

                # Collect this frame's flying-object candidates (no drawing yet).
                frame_dets = []
                for det in dets.detections:
                    cls_id = int(det.label)
                    cls_name = (
                        class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                    ).lower()
                    if cls_name not in WANT_CLASSES:
                        continue
                    bbox = (
                        int(det.xmin * w), int(det.ymin * h),
                        int(det.xmax * w), int(det.ymax * h),
                    )
                    frame_dets.append((bbox, cls_name, float(det.confidence)))

                # Match each detection to the best existing track by IoU. Track
                # that matches → update. Unmatched detection → new track.
                used_track_idxs = set()
                for bbox, cls_name, conf in frame_dets:
                    best_i, best_iou = -1, TRACK_IOU_MATCH
                    for i, tr in enumerate(tracks):
                        if i in used_track_idxs:
                            continue
                        score = _iou(bbox, tr["bbox"])
                        if score > best_iou:
                            best_iou, best_i = score, i
                    if best_i >= 0:
                        tracks[best_i].update(
                            bbox=bbox, cls=cls_name, conf=conf, last_seen=now
                        )
                        used_track_idxs.add(best_i)
                    else:
                        tracks.append({
                            "id": next_track_id, "bbox": bbox, "cls": cls_name,
                            "conf": conf, "last_seen": now,
                        })
                        next_track_id += 1

                # Drop expired tracks, draw the survivors in red.
                tracks = [t for t in tracks if now - t["last_seen"] <= TRACK_TTL_SEC]
                best_det = None                   # (cls, conf, cx_norm, cy_norm)
                for tr in tracks:
                    x1, y1, x2, y2 = tr["bbox"]
                    fresh = (now - tr["last_seen"]) < 0.05
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    tag = "LOCK" if fresh else "HOLD"
                    label = f"{tag} {tr['cls']} {tr['conf']:.2f}"
                    (tw, th), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - th - baseline - 4),
                        (x1 + tw + 4, y1),
                        (0, 0, 255), -1,
                    )
                    cv2.putText(
                        frame, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    )
                    cx_norm = (x1 + x2) / (2 * w)
                    cy_norm = (y1 + y2) / (2 * h)
                    if best_det is None or tr["conf"] > best_det[1]:
                        best_det = (tr["cls"], tr["conf"], cx_norm, cy_norm)

                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst = 1.0 / dt
                    fps_ema = inst if fps_ema == 0 else (0.9 * fps_ema + 0.1 * inst)

                cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if best_det is not None:
                    cls_name, conf, cxn, cyn = best_det
                    cv2.putText(
                        frame,
                        f"CAM tgt {cls_name} conf={conf:.2f}  "
                        f"px=({cxn*w:.0f},{cyn*h:.0f})",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    )

                pts, rframes, rstatus = radar.snapshot()
                radar_panel = draw_radar_panel(h, pts, rstatus, rframes)
                dashboard = np.hstack([frame, radar_panel])

                cv2.imshow(win_name, dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            radar.stop()
            radar.join(timeout=1.0)


if __name__ == "__main__":
    main()
