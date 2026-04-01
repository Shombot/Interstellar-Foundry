# Interstellar Foundry — UAV Detection System

**Team 7** · Calkin Garg · Zheng-Yin Lee · Ethan Yee · Parth Patel · Carlos Cordova · Rashod Abdurasulov

A modular, lightweight sensor suite for detecting Group 1 & 2 small UAVs using an FM24-NP100 24GHz mmWave radar and OAK-D Pro stereo camera, running on a Jetson Orin Nano with ROS2 Humble.

---

## Hardware

| Component | Part | Supplier | Cost |
|-----------|------|----------|------|
| Compute | Jetson Orin Nano | Amazon | $245.00 |
| Camera | OAK-D Pro | Neobits | $302.19 |
| Radar | FM24-NP100 24GHz mmWave | DFRobot | $65.90 |
| Battery | Pack | Fat Tire House | $65.00 |
| DC Converter | Buck Converter | Amazon | $12.99 |
| USB Cables | — | Amazon | $5.99 |
| Test Drone | Oddire | Amazon | $39.98 |
| **Total** | | | **~$904** |

**Key specs:**
- Radar: FM24-NP100 · UART `/dev/ttyTHS1` · **57600 baud** · Mode A (8-byte, dist) / Mode B (134-byte, dist + 126 spectral bins)
- Camera: OAK-D Pro · USB3 · **depthai 3.5.0** (`pipeline.create().build()` API, no XLinkOut)
- Platform: Ubuntu 22.04 · ROS2 Humble · Python 3.10

---

## Repository Structure

```
Interstellar-Foundry/
├── ros2_ws/src/uav_detection/
│   ├── uav_detection/
│   │   ├── radar_display.py          ← FM24-NP100 serial driver + matplotlib GUI (standalone)
│   │   ├── radar_camera_fusion.py    ← OpenCV fusion display (standalone, no ROS2)
│   │   ├── radar_node.py             ← ROS2 wrapper for radar_display.RadarReader
│   │   ├── camera_node.py            ← ROS2 wrapper using depthai 3.5.0 API
│   │   ├── fusion_node.py            ← Fuses radar range with camera depth ROI
│   │   ├── detection_node.py         ← Classifies Group 1/2, assigns alert levels
│   │   └── dashboard_bridge.py       ← WebSocket bridge → dashboard (port 9090)
│   ├── launch/uav_detection.launch.py
│   ├── config/params.yaml
│   ├── package.xml
│   └── setup.py
├── dashboard/index.html              ← Web dashboard (live spectrum + telemetry)
├── scripts/
│   ├── setup_jetson.sh               ← One-shot Jetson setup
│   └── setup_mac_dev.sh              ← Mac dev environment
├── docs/
│   ├── hardware_setup.md
│   └── ros2_architecture.md
└── README.md
```

---

## Quick Start

### On the Jetson (Ubuntu 22.04)

```bash
git clone https://github.com/Shombot/Interstellar-Foundry.git
cd Interstellar-Foundry

chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh          # installs ROS2, depthai==3.5.0, copies radar_display.py to ~/

source ~/.bashrc

# Full ROS2 pipeline (hardware required)
ros2 launch uav_detection uav_detection.launch.py

# Simulation mode (no hardware)
ros2 launch uav_detection uav_detection.launch.py sim_mode:=true
```

### Standalone (no ROS2, direct hardware test)

```bash
# Radar only — matplotlib display
python3 ~/radar_display.py --port /dev/ttyTHS1 --baud 57600

# Radar + OAK-D — OpenCV fusion window
python3 ros2_ws/src/uav_detection/uav_detection/radar_camera_fusion.py

# Radar diagnostic (raw hex output)
python3 ~/radar_display.py --diag
```

### On Mac (Development)

```bash
git clone https://github.com/Shombot/Interstellar-Foundry.git
cd Interstellar-Foundry

chmod +x scripts/setup_mac_dev.sh
./scripts/setup_mac_dev.sh         # Python venv + depthai==3.5.0 + copies radar_display.py to ~/

# Open dashboard in browser (mockup mode)
open dashboard/index.html

# Run fusion script camera-less (sim)
source .venv/bin/activate
python3 ros2_ws/src/uav_detection/uav_detection/radar_camera_fusion.py --no-camera
```

---

## FM24-NP100 Radar Protocol

| | Mode A | Mode B |
|-|--------|--------|
| Pin 6 | Floating | → GND |
| Frame size | 8 bytes | 134 bytes |
| Data | Distance only | Distance + 126 spectral bins |
| Header | `0xFF 0xFF 0xFF` | `0xFF 0xFF 0xFF` |
| Tail | `0x00 0x00 0x00` | `0x00 0x00 0x00` |
| Baud | 57600 | 57600 |

---

## Live Dashboard

Open `dashboard/index.html` in any browser.

**Offline (mockup):** open directly — shows demo data including a seeded FMCW spectrum.

**Live (Jetson on same network):**
```
open dashboard/index.html?jetson=<jetson-ip>
```
The dashboard connects to `ws://<jetson-ip>:9090` and renders:
- Live FM24-NP100 FMCW spectrum (126 bins, colour-coded)
- Real-time distance + peak amplitude bars
- Jetson CPU/RAM/temp/battery telemetry
- Detection event feed with alert levels

---

## ROS2 Topics

| Topic | Type | Publisher |
|-------|------|-----------|
| `/radar/raw` | `sensor_msgs/PointCloud2` | radar_node |
| `/radar/telemetry` | `std_msgs/String` JSON | radar_node (incl. spectrum[]) |
| `/camera/rgb` | `sensor_msgs/Image` | camera_node |
| `/camera/depth` | `sensor_msgs/Image` | camera_node |
| `/detections` | `std_msgs/String` JSON | fusion_node |
| `/detections/classified` | `std_msgs/String` JSON | detection_node |

---

## Target UAV Groups

- **Group 1**: < 20 lbs (DJI Mini, Parrot Anafi, Autel EVO Nano…)
- **Group 2**: 21–55 lbs (DJI Phantom, Matrice…)

Reference: [NPS Group 1 & 2 UAS Listing](https://nps.edu/documents/106607930/106914584/Ref+C+MR+Listing+of+non-COTS+Group+1+and+2+Multi-Rotor+UAS+-+4-01-18.pdf)

---

## License

MIT — see [LICENSE](LICENSE)
