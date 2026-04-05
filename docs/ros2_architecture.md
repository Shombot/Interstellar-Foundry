# ROS2 Architecture
**Interstellar Foundry — Team 7**
Platform: ROS2 Humble · Ubuntu 22.04 · Jetson Orin Nano

---

## Node Graph

```
┌─────────────┐     /radar/raw          ┌─────────────┐
│ radar_node  │ ──────────────────────► │             │
│             │     /radar/telemetry    │ fusion_node │
│ (mmWave     │ ──────────────────────► │             │
│  UART)      │                         │             │──► /detections
└─────────────┘                         │             │
                                        │  (radar +   │
┌─────────────┐     /camera/rgb         │   camera    │
│ camera_node │ ──────────────────────► │   fusion)   │
│             │     /camera/depth       │             │
│ (OAK-D Pro  │ ──────────────────────► └─────────────┘
│  DepthAI)   │     /camera/telemetry           │
└─────────────┘                                 │ /detections
                                                ▼
                                    ┌──────────────────────┐
                                    │   detection_node     │
                                    │  (classify Grp1/Grp2 │
                                    │   assign alerts)     │
                                    └──────────┬───────────┘
                                               │ /detections/classified
                                    ┌──────────▼───────────┐
                                    │  dashboard_bridge    │
                                    │  (WebSocket :9090)   │
                                    └──────────┬───────────┘
                                               │ ws://jetson-ip:9090
                                    ┌──────────▼───────────┐
                                    │   dashboard/index    │
                                    │   (any browser)      │
                                    └──────────────────────┘
```

---

## Topic Reference

| Topic | Message Type | Publisher | Subscribers | Rate |
|-------|-------------|-----------|-------------|------|
| `/radar/raw` | `sensor_msgs/PointCloud2` | radar_node | fusion_node | 10 Hz |
| `/radar/telemetry` | `std_msgs/String` (JSON) | radar_node | dashboard_bridge | 10 Hz |
| `/camera/rgb` | `sensor_msgs/Image` | camera_node | — | 30 Hz |
| `/camera/depth` | `sensor_msgs/Image` | camera_node | fusion_node | 30 Hz |
| `/camera/telemetry` | `std_msgs/String` (JSON) | camera_node | dashboard_bridge | 30 Hz |
| `/detections` | `std_msgs/String` (JSON) | fusion_node | detection_node | 5 Hz |
| `/detections/classified` | `std_msgs/String` (JSON) | detection_node | dashboard_bridge | ~5 Hz |
| `/system/telemetry` | `std_msgs/String` (JSON) | dashboard_bridge | — | 0.5 Hz |

---

## Node Parameters

All parameters live in `config/params.yaml` and can be overridden at launch:

```bash
# Override a parameter at runtime
ros2 launch uav_detection uav_detection.launch.py sim_mode:=true

# Or set a parameter on a running node
ros2 param set /detection_node alert_range_m 15.0
```

---

## Simulation Mode

All hardware-facing nodes (`radar_node`, `camera_node`) support `sim_mode: true`.
In sim mode, synthetic data is generated — no Jetson hardware required.

**Useful for:**
- Developing on Mac
- Unit testing pipeline logic
- Demo without hardware present

```bash
ros2 launch uav_detection uav_detection.launch.py sim_mode:=true
```

---

## Useful ROS2 CLI Commands

```bash
# List all running nodes
ros2 node list

# Monitor detections live
ros2 topic echo /detections/classified

# Check topic frequency
ros2 topic hz /radar/raw

# View node parameters
ros2 param list /radar_node
ros2 param get /radar_node sim_mode

# Visualize node graph
rqt_graph
```

---

## Adding a New Node

1. Create `ros2_ws/src/uav_detection/uav_detection/my_node.py`
2. Add an entry_point in `setup.py`:
   ```python
   'my_node = uav_detection.my_node:main',
   ```
3. Add it to `launch/uav_detection.launch.py`
4. Rebuild: `cd ros2_ws && colcon build --symlink-install`
5. Re-source: `source install/setup.bash`
