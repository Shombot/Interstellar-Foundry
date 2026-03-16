#!/usr/bin/env bash
# =============================================================================
# setup_jetson.sh
# Interstellar Foundry — Team 7
#
# One-shot setup for Jetson Orin Nano running Ubuntu 22.04.
# Installs ROS2 Humble, Python deps, and configures the UART.
#
# Usage:
#   chmod +x scripts/setup_jetson.sh
#   ./scripts/setup_jetson.sh
# =============================================================================

set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- Sanity checks ----
[[ "$(uname -m)" == "aarch64" ]] || warn "Not running on ARM — this script is for Jetson Orin Nano."
[[ "$(lsb_release -rs)" == "22.04" ]] || warn "Expected Ubuntu 22.04."

info "=== Interstellar Foundry — Jetson Setup ==="

# ---- System update ----
info "Updating system packages..."
sudo apt-get update -q
sudo apt-get upgrade -y -q
success "System updated."

# ---- ROS2 Humble ----
if ! command -v ros2 &>/dev/null; then
  info "Installing ROS2 Humble..."
  sudo apt-get install -y locales
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8

  sudo apt-get install -y software-properties-common curl
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

  sudo apt-get update -q
  sudo apt-get install -y ros-humble-ros-base python3-colcon-common-extensions python3-rosdep

  sudo rosdep init 2>/dev/null || true
  rosdep update
  success "ROS2 Humble installed."
else
  success "ROS2 Humble already installed."
fi

# ---- Add ROS2 to bashrc ----
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  info "Added ROS2 source to ~/.bashrc"
fi

# ---- Python dependencies ----
info "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install \
  pyserial \
  numpy \
  websockets \
  psutil \
  depthai \
  opencv-python-headless

success "Python dependencies installed."

# ---- UART permissions (Jetson ttyTHS1) ----
info "Configuring UART permissions..."
sudo usermod -aG dialout "$USER"
# Disable serial console on ttyTHS1 so we can use it for radar
if systemctl is-enabled serial-getty@ttyTHS1.service 2>/dev/null; then
  sudo systemctl disable serial-getty@ttyTHS1.service
  sudo systemctl stop serial-getty@ttyTHS1.service
  success "Disabled serial console on ttyTHS1."
fi

# ---- Build the ROS2 workspace ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WS="$REPO_ROOT/ros2_ws"

info "Building ROS2 workspace at $WS..."
cd "$WS"
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
success "Workspace built."

# ---- Add workspace to bashrc ----
SETUP_LINE="source $WS/install/setup.bash"
if ! grep -q "$SETUP_LINE" ~/.bashrc; then
  echo "$SETUP_LINE" >> ~/.bashrc
  info "Added workspace source to ~/.bashrc"
fi

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Interstellar Foundry setup complete!               ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Reboot or run: source ~/.bashrc"
echo "  2. Connect radar (UART) and OAK-D (USB3)"
echo "  3. Launch: ros2 launch uav_detection uav_detection.launch.py"
echo "  4. Open dashboard/index.html in a browser on the same network"
echo ""
echo -e "${YELLOW}NOTE:${NC} Log out and back in for UART group permissions to take effect."
