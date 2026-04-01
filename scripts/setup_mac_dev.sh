#!/usr/bin/env bash
# =============================================================================
# setup_mac_dev.sh — Interstellar Foundry · Team 7
#
# Sets up a Python dev environment on macOS for editing the UAV detection
# nodes.  ROS2 does not run natively on macOS Apple Silicon, so this script
# gives you the Python deps for linting, unit testing, and running the
# standalone scripts (radar_display.py, radar_camera_fusion.py) in sim mode.
#
# Usage:
#   chmod +x scripts/setup_mac_dev.sh && ./scripts/setup_mac_dev.sh
# =============================================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }

info "=== Interstellar Foundry — Mac Dev Setup ==="
info "Platform: $(uname -sm)"

# ── Homebrew ───────────────────────────────────────────────────────────────
command -v brew &>/dev/null || \
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
success "Homebrew ready."

# ── Python 3.11 ───────────────────────────────────────────────────────────
command -v python3 &>/dev/null || brew install python@3.11
success "Python: $(python3 --version)"

# ── Virtualenv ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv"

[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
success "Virtualenv activated: $VENV"

# ── Python packages ───────────────────────────────────────────────────────
info "Installing Python packages..."
pip install --upgrade pip -q
pip install -q \
  pyserial \
  numpy \
  opencv-python \
  matplotlib \
  websockets \
  psutil \
  ruff \
  pytest

# depthai 3.5.0 — installs on Mac for import/lint checking.
# Hardware features won't work without the OAK-D connected via USB.
pip install -q depthai==3.5.0 || \
  warn "depthai 3.5.0 install failed on this platform — lint stubs unavailable."

success "Python packages installed."

# ── Copy radar_display.py to ~/ ────────────────────────────────────────────
# radar_camera_fusion.py does: sys.path.insert(0, os.path.expanduser("~"))
# On Mac this lets you run: python3 radar_camera_fusion.py --no-camera
RADAR_DISPLAY="$REPO_ROOT/ros2_ws/src/uav_detection/uav_detection/radar_display.py"
cp "$RADAR_DISPLAY" "$HOME/radar_display.py"
success "radar_display.py copied to ~/ (required by radar_camera_fusion.py import)."

# ── VS Code extensions ────────────────────────────────────────────────────
if command -v code &>/dev/null; then
  code --install-extension ms-python.python       --force 2>/dev/null || true
  code --install-extension ms-python.vscode-pylance --force 2>/dev/null || true
  code --install-extension charliermarsh.ruff      --force 2>/dev/null || true
  success "VS Code extensions installed."
fi

# ── Open dashboard ────────────────────────────────────────────────────────
open "$REPO_ROOT/dashboard/index.html" 2>/dev/null || true

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Mac dev environment ready!                      ${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
echo "  Activate venv:  source .venv/bin/activate"
echo ""
echo "  Standalone scripts (Mac — no OAK-D hardware needed):"
echo "    python3 ~/radar_display.py --diag              # hex debug"
echo "    python3 ros2_ws/src/uav_detection/uav_detection/radar_camera_fusion.py --no-camera"
echo ""
echo "  ROS2 on Mac → use Docker:"
echo "    docker run --rm -it osrf/ros:humble-desktop"
echo ""
warn "depthai USB hardware requires the Jetson. Mac = code editing + sim only."
