#!/usr/bin/env bash
# =============================================================================
# setup_jetson.sh — Interstellar Foundry · Team 7
#
# One-shot setup for Jetson Orin Nano, Ubuntu 22.04.
# Installs ROS2 Humble, depthai 3.5.0, pyserial, and configures UART.
#
# Usage:
#   chmod +x scripts/setup_jetson.sh && ./scripts/setup_jetson.sh
# =============================================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[[ "$(uname -m)" == "aarch64" ]] || warn "Not aarch64 — script targets Jetson Orin Nano."
[[ "$(lsb_release -rs)" == "22.04" ]] || warn "Expected Ubuntu 22.04."

info "=== Interstellar Foundry — Jetson Setup ==="

# ── System update ──────────────────────────────────────────────────────────
info "Updating system packages..."
sudo apt-get update -q && sudo apt-get upgrade -y -q
success "System updated."

# ── ROS2 Humble ────────────────────────────────────────────────────────────
if ! command -v ros2 &>/dev/null; then
  info "Installing ROS2 Humble..."
  sudo apt-get install -y locales curl software-properties-common
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt-get update -q
  sudo apt-get install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep
  sudo rosdep init 2>/dev/null || true && rosdep update
  success "ROS2 Humble installed."
else
  success "ROS2 Humble already present."
fi

grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc || \
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# ── Python dependencies ────────────────────────────────────────────────────
info "Installing Python dependencies..."
pip3 install --upgrade pip -q
pip3 install -q \
  pyserial \
  numpy \
  opencv-python-headless \
  matplotlib \
  websockets \
  psutil \
  depthai==3.5.0          # must match radar_camera_fusion.py API
success "Python dependencies installed."

# ── Copy radar_display.py to ~/ ────────────────────────────────────────────
# radar_camera_fusion.py imports radar_display from ~/ via sys.path.insert(0, "~")
# This must stay in sync with the copy inside the ROS2 package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RADAR_DISPLAY_SRC="$REPO_ROOT/ros2_ws/src/uav_detection/uav_detection/radar_display.py"

if [ -f "$RADAR_DISPLAY_SRC" ]; then
  cp "$RADAR_DISPLAY_SRC" "$HOME/radar_display.py"
  success "radar_display.py copied to ~/ (required by radar_camera_fusion.py import)."
else
  warn "radar_display.py not found at $RADAR_DISPLAY_SRC — copy manually to ~/."
fi

# ── UART permissions ───────────────────────────────────────────────────────
info "Configuring UART (/dev/ttyTHS1) for FM24-NP100 radar..."
sudo usermod -aG dialout "$USER"
if systemctl is-enabled serial-getty@ttyTHS1.service 2>/dev/null | grep -q enabled; then
  sudo systemctl disable --now serial-getty@ttyTHS1.service
  success "Disabled serial console on ttyTHS1."
fi

# ── OAK-D Pro udev rules ───────────────────────────────────────────────────
info "Setting OAK-D Pro udev rules..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/80-movidius.rules > /dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger
success "OAK-D udev rules set."

# ── Build ROS2 workspace ───────────────────────────────────────────────────
WS="$REPO_ROOT/ros2_ws"
info "Building ROS2 workspace at $WS..."
cd "$WS"
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
success "Workspace built."

SETUP_LINE="source $WS/install/setup.bash"
grep -q "$SETUP_LINE" ~/.bashrc || echo "$SETUP_LINE" >> ~/.bashrc

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!                                 ${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
echo "  Reboot or run: source ~/.bashrc"
echo ""
echo "  Standalone (no ROS2):"
echo "    python3 ~/radar_display.py --port /dev/ttyTHS1 --baud 57600"
echo "    python3 ros2_ws/src/uav_detection/uav_detection/radar_camera_fusion.py"
echo ""
echo "  Full ROS2 pipeline:"
echo "    ros2 launch uav_detection uav_detection.launch.py"
echo ""
echo "  Simulation (no hardware):"
echo "    ros2 launch uav_detection uav_detection.launch.py sim_mode:=true"
echo ""
echo -e "${YELLOW}NOTE:${NC} Log out and back in for UART group permissions to take effect."
