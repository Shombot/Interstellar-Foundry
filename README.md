# Interstellar Foundry — UAV Detection System

**Team 7** · Calkin Garg · Zheng-Yin Lee · Ethan Yee · Parth Patel · Carlos Cordova · Rashod Abdurasulov

A modular, lightweight sensor suite for detecting Group 1 & 2 small UAVs using mmWave radar and stereo camera data, running on a Jetson Orin Nano with ROS2 Humble.

---

## System Overview

```
┌─────────────────────────────────────────┐
│           Jetson Orin Nano              │
│         Ubuntu 22.04 + ROS2 Humble      │
│                                         │
│  ┌──────────┐   ┌───────────────────┐   │
│  │ mmWave   │   │  OAK-D Pro Camera │   │
│  │  Radar   │   │  (DepthAI/OAK)    │   │
│  │ (UART)   │   │  (USB3)           │   │
│  └────┬─────┘   └────────┬──────────┘   │
│       │                  │              │
│  ┌────▼──────────────────▼──────────┐   │
│  │        ROS2 Node Graph           │   │
│  │  radar_node ──► fusion_node      │   │
│  │  camera_node──►     │            │   │
│  │                     ▼            │   │
│  │              detection_node      │   │
│  │                     │            │   │
│  │              dashboard_bridge    │   │
│  └─────────────────────┬────────────┘   │
└────────────────────────┼────────────────┘
                         │ WebSocket
                    ┌────▼────┐
                    │Dashboard│  ← browser on any machine
                    │  (HTML) │
                    └─────────┘
```

## Hardware Stack

| Component | Part | Supplier | Cost |
|-----------|------|----------|------|
| Compute | Jetson Orin Nano | Amazon | $245.00 |
| Camera | OAK-D Pro | Neobits | $302.19 |
| Radar | mmWave (DFRobot) | DFRobot | $65.90 |
| Battery | Pack | Fat Tire House | $65.00 |
| DC Converter | Buck Converter | Amazon | $12.99 |
| USB Cables | — | Amazon | $5.99 |
| Test Drone | Oddire | Amazon | $39.98 |
| **Total** | | | **~$904** |

---

## Repository Structure

```
InterstellarFoundry/
├── ros2_ws/                    # ROS2 workspace
│   └── src/
│       └── uav_detection/      # Main ROS2 package
│           ├── uav_detection/  # Python nodes
│           │   ├── radar_node.py
│           │   ├── camera_node.py
│           │   ├── fusion_node.py
│           │   ├── detection_node.py
│           │   └── dashboard_bridge.py
│           ├── launch/
│           │   └── uav_detection.launch.py
│           ├── config/
│           │   └── params.yaml
│           ├── package.xml
│           └── setup.py
├── dashboard/
│   └── index.html              # Web dashboard (open in browser)
├── scripts/
│   ├── setup_jetson.sh         # One-shot Jetson setup script
│   └── setup_mac_dev.sh        # Mac dev environment setup
├── docs/
│   ├── hardware_setup.md
│   └── ros2_architecture.md
├── .gitignore
└── README.md
```

---

## Quick Start

### On the Jetson (Ubuntu 22.04)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/InterstellarFoundry.git
cd InterstellarFoundry

# 2. Run setup (installs ROS2 Humble + dependencies)
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh

# 3. Build ROS2 workspace
cd ros2_ws
colcon build
source install/setup.bash

# 4. Launch the full system
ros2 launch uav_detection uav_detection.launch.py
```

### On Mac (Development)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/InterstellarFoundry.git
cd InterstellarFoundry

# 2. Set up dev environment
chmod +x scripts/setup_mac_dev.sh
./scripts/setup_mac_dev.sh

# 3. Open dashboard in browser
open dashboard/index.html
```

---

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/radar/raw` | `sensor_msgs/PointCloud2` | Raw mmWave radar point cloud |
| `/camera/rgb` | `sensor_msgs/Image` | OAK-D RGB frame |
| `/camera/depth` | `sensor_msgs/Image` | OAK-D depth frame |
| `/detections` | `std_msgs/String` (JSON) | Fused UAV detections |
| `/detections/classified` | `std_msgs/String` (JSON) | Classified UAV events |
| `/system/telemetry` | `std_msgs/String` (JSON) | Jetson hardware stats |
| `/dashboard/events` | `std_msgs/String` (JSON) | Dashboard WebSocket feed |

---

## Dashboard

Open `dashboard/index.html` in any browser. When connected to the same network as the Jetson, it will receive live data via WebSocket on port `9090` (rosbridge).

---

## Target UAV Groups

- **Group 1**: Small UAVs < 20 lbs (DJI Mini, Parrot Anafi, Autel EVO Nano, etc.)
- **Group 2**: UAVs 21–55 lbs (DJI Phantom, Matrice, etc.)

Reference: [NPS Group 1 & 2 UAS Listing](https://nps.edu/documents/106607930/106914584/Ref+C+MR+Listing+of+non-COTS+Group+1+and+2+Multi-Rotor+UAS+-+4-01-18.pdf)

---

## License

MIT License — see [LICENSE](LICENSE)
