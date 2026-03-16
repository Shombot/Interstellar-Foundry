#!/usr/bin/env bash
# =============================================================================
# setup_mac_dev.sh
# Interstellar Foundry — Team 7
#
# Sets up a Python development environment on macOS for working on
# the UAV detection nodes. ROS2 does not run natively on macOS ARM (M-series),
# so this script sets up the Python deps only for code editing, linting,
# and simulation testing.
#
# Usage:
#   chmod +x scripts/setup_mac_dev.sh
#   ./scripts/setup_mac_dev.sh
# =============================================================================

set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }

info "=== Interstellar Foundry — Mac Dev Setup ==="
info "Platform: $(uname -sm)"

# ---- Check for Homebrew ----
if ! command -v brew &>/dev/null; then
  info "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
success "Homebrew ready."

# ---- Python 3.10+ ----
if ! command -v python3 &>/dev/null; then
  info "Installing Python 3..."
  brew install python@3.11
fi
PY_VER=$(python3 --version)
success "Python: $PY_VER"

# ---- Virtual environment ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv"

if [ ! -d "$VENV" ]; then
  info "Creating virtualenv at $VENV..."
  python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
success "Virtualenv activated."

# ---- Python dependencies ----
info "Installing Python packages..."
pip install --upgrade pip -q
pip install -q \
  pyserial \
  numpy \
  websockets \
  psutil \
  opencv-python \
  ruff \
  pytest

# Install stub packages so imports resolve in the editor
# (rclpy, sensor_msgs, etc. aren't available natively on macOS)
pip install -q \
  rclpy-stubs 2>/dev/null || true

success "Python packages installed."

# ---- VS Code extensions reminder ----
if command -v code &>/dev/null; then
  info "Recommending VS Code extensions..."
  code --install-extension ms-python.python --force 2>/dev/null || true
  code --install-extension ms-python.vscode-pylance --force 2>/dev/null || true
  code --install-extension charliermarsh.ruff --force 2>/dev/null || true
  success "VS Code extensions installed."
fi

# ---- Open dashboard ----
info "Opening dashboard in browser..."
open "$REPO_ROOT/dashboard/index.html" 2>/dev/null || true

echo ""
echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  Mac dev environment ready!                         ${NC}"
echo -e "${GREEN}======================================================${NC}"
echo ""
echo "Activate the venv anytime with:"
echo "  source .venv/bin/activate"
echo ""
echo "Notes:"
echo "  • ROS2 does NOT run natively on macOS Apple Silicon."
echo "  • Edit Python nodes in ros2_ws/src/uav_detection/uav_detection/"
echo "  • Push to GitHub and pull on the Jetson to test with real hardware."
echo "  • The dashboard (dashboard/index.html) works in any browser."
echo "  • When Jetson is running, point the dashboard at ws://<jetson-ip>:9090"
echo ""
warn "For ROS2 development on Mac, consider: docker run --rm -it osrf/ros:humble-desktop"
