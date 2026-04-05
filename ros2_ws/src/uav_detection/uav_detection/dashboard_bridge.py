#!/usr/bin/env python3
"""
dashboard_bridge.py
Interstellar Foundry — Team 7

Aggregates all ROS2 topics and streams live JSON to the web dashboard
(dashboard/index.html) via WebSocket on port 9090.

Spectrum data from the FM24-NP100 (126 bins) is forwarded so the
dashboard can render the FMCW spectrum bar in real-time.

Platform : ROS2 Humble · Ubuntu 22.04
pip deps  : pip3 install websockets psutil
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import asyncio, websockets, json, time, threading, psutil


class DashboardBridge(Node):

    def __init__(self):
        super().__init__('dashboard_bridge')

        self.declare_parameter('ws_port',          9090)
        self.declare_parameter('broadcast_rate_hz', 5.0)

        self.ws_port  = self.get_parameter('ws_port').value
        self.rate_hz  = self.get_parameter('broadcast_rate_hz').value

        self.state = {
            'detections':       [],
            'total_detections': 0,
            'active_threats':   0,
            'radar_telem':      {},   # includes spectrum list from FM24-NP100
            'camera_telem':     {},
            'hw_telem':         {},
            'session_start':    time.time(),
        }
        self.clients: set = set()

        self.create_subscription(String, '/detections/classified', self._on_classified, 10)
        self.create_subscription(String, '/radar/telemetry',       self._on_radar,      10)
        self.create_subscription(String, '/camera/telemetry',      self._on_camera,     10)

        self.create_timer(2.0,              self._update_hw)
        self.create_timer(1.0 / self.rate_hz, self._broadcast_sync)

        # WebSocket server runs in its own thread with its own event loop
        self._loop = asyncio.new_event_loop()
        self._ws_thread = threading.Thread(target=self._start_ws, daemon=True)
        self._ws_thread.start()

        self.get_logger().info(f'DashboardBridge ready – ws://0.0.0.0:{self.ws_port}')

    # ------------------------------------------------------------------ #
    # ROS2 callbacks                                                       #
    # ------------------------------------------------------------------ #

    def _on_classified(self, msg: String):
        try:
            data   = json.loads(msg.data)
            events = data.get('events', [])
            self.state['detections']       = (events + self.state['detections'])[:50]
            self.state['total_detections'] += len(events)
            self.state['active_threats']   = sum(
                1 for e in self.state['detections'][:10]
                if e.get('alert_level') == 'THREAT'
            )
        except Exception:
            pass

    def _on_radar(self, msg: String):
        try:
            self.state['radar_telem'] = json.loads(msg.data)
        except Exception:
            pass

    def _on_camera(self, msg: String):
        try:
            self.state['camera_telem'] = json.loads(msg.data)
        except Exception:
            pass

    def _update_hw(self):
        try:
            mem  = psutil.virtual_memory()
            temp = None
            try:
                for key in ('thermal_fan_est', 'cpu_thermal', 'coretemp'):
                    t = psutil.sensors_temperatures().get(key)
                    if t:
                        temp = t[0].current
                        break
            except Exception:
                pass
            bat = None
            try:
                b = psutil.sensors_battery()
                bat = round(b.percent, 1) if b else None
            except Exception:
                pass

            self.state['hw_telem'] = {
                'cpu_percent':  psutil.cpu_percent(interval=None),
                'ram_used_gb':  round(mem.used  / 1e9, 2),
                'ram_total_gb': round(mem.total / 1e9, 2),
                'cpu_temp_c':   temp,
                'battery_percent': bat,
                'uptime_sec':   int(time.time() - self.state['session_start']),
                'timestamp':    time.time(),
            }
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # WebSocket                                                            #
    # ------------------------------------------------------------------ #

    def _start_ws(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_main())

    async def _ws_main(self):
        async with websockets.serve(self._handler, '0.0.0.0', self.ws_port):
            await asyncio.Future()

    async def _handler(self, ws):
        self.clients.add(ws)
        self.get_logger().info(f'Dashboard connected: {ws.remote_address}')
        try:
            async for _ in ws:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    def _broadcast_sync(self):
        if not self.clients:
            return
        s = self.state
        payload = json.dumps({
            'type':             'state_update',
            'timestamp':        time.time(),
            'total_detections': s['total_detections'],
            'active_threats':   s['active_threats'],
            'recent_events':    s['detections'][:10],
            'telemetry':        s['hw_telem'],
            'radar_telem':      s['radar_telem'],   # contains spectrum[], distance_m, mode, etc.
            'camera_telem':     s['camera_telem'],
            'uptime_sec':       int(time.time() - s['session_start']),
        })
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: str):
        dead = set()
        for ws in list(self.clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        self.clients -= dead


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
