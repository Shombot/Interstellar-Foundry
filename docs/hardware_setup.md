# Hardware Setup Guide
**Interstellar Foundry — Team 7**

---

## Components

| Part | Connection | Notes |
|------|-----------|-------|
| Jetson Orin Nano | — | Main compute board |
| DFRobot mmWave Radar | UART (`/dev/ttyTHS1`) | Jetson 40-pin header pins 8 (TX) & 10 (RX) |
| OAK-D Pro Camera | USB 3.0 | Use the blue USB3 port |
| Buck Converter | Between battery & Jetson | Step down to 5V/4A for Jetson, 12V for radar |
| Battery Pack | DC barrel / XT60 | Check voltage matches converter input |
| WiFi / Telemetry | USB WiFi dongle or M.2 | For WebSocket dashboard access |

---

## Wiring Diagram (UART — Radar ↔ Jetson)

```
DFRobot mmWave Radar          Jetson Orin Nano (40-pin header)
──────────────────            ─────────────────────────────────
TX  ──────────────────────►  Pin 10  (RX / UART1_RXD / ttyTHS1)
RX  ◄─────────────────────   Pin  8  (TX / UART1_TXD / ttyTHS1)
GND ──────────────────────   Pin  6  (GND)
VCC ──────────────────────   Pin  4  (5V) or powered separately
```

> ⚠️ The DFRobot radar operates at 3.3V logic. Use a level shifter if needed.
> The Jetson GPIO is 3.3V tolerant — direct connection is generally fine.

---

## Power Distribution

```
Battery Pack (12V)
       │
       ├─► Buck Converter → 5V/4A ──► Jetson Orin Nano (barrel jack)
       │
       ├─► Buck Converter → 12V ────► mmWave Radar (if required)
       │
       └─► OAK-D Pro powered via USB from Jetson (up to 900mA on USB3)
```

> The OAK-D Pro draws ~2.5W peak. The Jetson's USB3 port is sufficient
> but monitor current — use a powered USB hub if you see dropouts.

---

## UART Configuration on Jetson

After running `setup_jetson.sh`, the ttyTHS1 serial console is disabled.
Verify with:

```bash
ls -la /dev/ttyTHS1
# Should show: crw-rw---- 1 root dialout ...

# Test UART connection
python3 -c "import serial; s = serial.Serial('/dev/ttyTHS1', 115200, timeout=1); print(s.read(64))"
```

If you see `PermissionError`, log out and back in (group change takes effect on next login).

---

## OAK-D Pro (DepthAI) Setup

```bash
# Install DepthAI
pip3 install depthai

# Udev rules (run once)
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Test camera
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

---

## Physical Mounting

The sensor suite is designed as a cubic frame. Recommended layout:

```
         ┌─────────────────┐
         │   OAK-D Pro     │  ← Face outward, unobstructed
         │   (Front face)  │
         ├─────────────────┤
         │  Jetson Nano    │  ← Center of cube
         │  + Buck Conv.   │
         ├─────────────────┤
         │  mmWave Radar   │  ← Mounted on top or side
         │  Battery Pack   │  ← Bottom, heaviest = lowest CG
         └─────────────────┘
```

- Keep radar clear of metal obstructions
- OAK-D needs line-of-sight forward / upward
- Ensure ventilation around Jetson
