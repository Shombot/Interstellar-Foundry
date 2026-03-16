#!/usr/bin/env python3
"""
dashboard_bridge.py
Interstellar Foundry — Team 7

Aggregates data from all ROS2 topics and broadcasts it via WebSocket
on port 9090 so the web dashboard (dashboard/index.html) can display
live data from any browser on the same network.

Platform: ROS2 Humble · Ubuntu 22.04
Dependencies: pip install websockets psutil
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import asyncio
import websockets
import json
import time
import threading
import psutil


class DashboardBridge(Node):
    """
    Subscribes to all relevant topics and re-broadcasts as JSON over WebSocket.
    The HTML dashboard connects to ws://<jetson-ip>:9090.
    """

    def __init__(self):
        super().__init__('dashboard_bridge')

        self.declare_parameter('ws_port', 9090)
        self.declare_parameter('broadcast_rate_hz', 5.0)

        self.ws_port = self.get_parameter('ws_port').value
        self.rate_hz = self.get_parameter('broadcast_rate_hz').value

        # Shared state (thread-safe via simple dict + GIL for Python)
        self.state = {
            'detections': [],
            'telemetry': {},
            'radar_telem': {},
            'camera_telem': {},
            'session_start': time.time(),
            'total_detections': 0,
            'active_threats': 0,
        }

        self.connected_clients: set = set()

        # --- ROS2 Subscribers ---
        self.create_subscription(String, '/detections/classified', self._on_classified, 10)
        self.create_subscription(String, '/radar/telemetry', self._on_radar_telem, 10)
        self.create_subscription(String, '/camera/telemetry', self._on_camera_telem, 10)

        # --- Timer for hardware telemetry ---
        self.create_timer(2.0, self._update_hw_telem)

        # --- Start WebSocket server in background thread ---
        self.ws_thread = threading.Thread(target=self._start_ws_server, daemon=True)
        self.ws_thread.start()

        # --- Broadcast timer ---
        self.create_timer(1.0 / self.rate_hz, self._broadcast_state)

        self.get_logger().info(f'DashboardBridge started — ws://0.0.0.0:{self.ws_port}')

    # ------------------------------------------------------------------ #
    # ROS2 Callbacks                                                       #
    # ------------------------------------------------------------------ #

    def _on_classified(self, msg: String):
        try:
            data = json.loads(msg.data)
            events = data.get('events', [])
            # Keep last 50 detections
            self.state['detections'] = (events + self.state['detections'])[:50]
            self.state['total_detections'] += len(events)
            self.state['active_threats'] = sum(
                1 for e in self.state['detections'][:10] if e.get('alert_level') == 'THREAT'
            )
        except Exception as e:
            self.get_logger().warn(f'Bridge parse error (classified): {e}')

    def _on_radar_telem(self, msg: String):
        try:
            self.state['radar_telem'] = json.loads(msg.data)
        except Exception:
            pass

    def _on_camera_telem(self, msg: String):
        try:
            self.state['camera_telem'] = json.loads(msg.data)
        except Exception:
            pass

    def _update_hw_telem(self):
        """Pull Jetson hardware stats via psutil."""
        try:
            mem = psutil.virtual_memory()
            cpu_temp = None
            try:
                temps = psutil.sensors_temperatures()
                for key in ('thermal_fan_est', 'cpu_thermal', 'coretemp'):
                    if key in temps and temps[key]:
                        cpu_temp = temps[key][0].current
                        break
            except Exception:
                pass

            self.state['telemetry'] = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'ram_used_gb': round(mem.used / 1e9, 2),
                'ram_total_gb': round(mem.total / 1e9, 2),
                'cpu_temp_c': cpu_temp,
                'battery_percent': self._get_battery(),
                'uptime_sec': int(time.time() - self.state['session_start']),
                'timestamp': time.time(),
            }
        except Exception as e:
            self.get_logger().warn(f'HW telem error: {e}')

    def _get_battery(self):
        try:
            b = psutil.sensors_battery()
            return round(b.percent, 1) if b else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # WebSocket Server                                                     #
    # ------------------------------------------------------------------ #

    def _start_ws_server(self):
        asyncio.run(self._ws_main())

    async def _ws_main(self):
        async with websockets.serve(self._ws_handler, '0.0.0.0', self.ws_port):
            self.get_logger().info(f'WebSocket server listening on :{self.ws_port}')
            await asyncio.Future()  # run forever

    async def _ws_handler(self, websocket):
        client = websocket.remote_address
        self.connected_clients.add(websocket)
        self.get_logger().info(f'Dashboard client connected: {client}')
        try:
            async for _ in websocket:
                pass  # We don't expect incoming messages from the dashboard
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.discard(websocket)
            self.get_logger().info(f'Dashboard client disconnected: {client}')

    def _broadcast_state(self):
        """Push current state to all connected WebSocket clients."""
        if not self.connected_clients:
            return

        payload = json.dumps({
            'type': 'state_update',
            'timestamp': time.time(),
            'total_detections': self.state['total_detections'],
            'active_threats': self.state['active_threats'],
            'recent_events': self.state['detections'][:10],
            'telemetry': self.state['telemetry'],
            'radar_telem': self.state['radar_telem'],
            'camera_telem': self.state['camera_telem'],
            'uptime_sec': int(time.time() - self.state['session_start']),
        })

        # Schedule async broadcast on the ws event loop
        asyncio.run_coroutine_threadsafe(
            self._async_broadcast(payload),
            asyncio.get_event_loop()
        )

    async def _async_broadcast(self, payload: str):
        dead = set()
        for ws in self.connected_clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        self.connected_clients -= dead


def main(args=None):
    rclpy.init(args=args)
    node = DashboardBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
