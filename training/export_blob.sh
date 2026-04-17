#!/usr/bin/env bash
# =============================================================================
# Export fine-tuned YOLOv8n → ONNX → OpenVINO blob for OAK Myriad X
# =============================================================================
#
# Prerequisites:
#   pip install ultralytics openvino-dev
#   (or use tools.luxonis.com for blob conversion)
#
# Usage:
#   bash training/export_blob.sh [path/to/best.pt]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/yolov6n_model"

# Input weights
WEIGHTS="${1:-$SCRIPT_DIR/runs/drone_phase2/weights/best.pt}"

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: Weights not found: $WEIGHTS"
    echo "Usage: bash training/export_blob.sh [path/to/best.pt]"
    exit 1
fi

echo "============================================"
echo "  Export: $WEIGHTS"
echo "  Target: OAK Myriad X (6 shaves)"
echo "============================================"

# --- Step 1: Export to ONNX ---
echo ""
echo "--- Step 1: PyTorch → ONNX ---"
python3 -c "
from ultralytics import YOLO
model = YOLO('$WEIGHTS')
model.export(format='onnx', imgsz=(288, 512), simplify=True, opset=12)
print('ONNX export done.')
"

ONNX_PATH="${WEIGHTS%.pt}.onnx"
echo "ONNX: $ONNX_PATH"

# --- Step 2: ONNX → OpenVINO IR ---
echo ""
echo "--- Step 2: ONNX → OpenVINO IR ---"
mo --input_model "$ONNX_PATH" \
   --input_shape "[1,3,288,512]" \
   --mean_values "[0,0,0]" \
   --scale_values "[255,255,255]" \
   --data_type FP16 \
   --output_dir "${WEIGHTS%.pt}_openvino"

IR_DIR="${WEIGHTS%.pt}_openvino"
IR_XML=$(find "$IR_DIR" -name "*.xml" | head -1)
echo "OpenVINO IR: $IR_XML"

# --- Step 3: OpenVINO IR → MyriadX blob ---
echo ""
echo "--- Step 3: OpenVINO IR → MyriadX blob ---"
echo ""
echo "Option A: Use blobconverter (recommended)"
echo "  pip install blobconverter"
echo ""
python3 -c "
try:
    import blobconverter
    blob_path = blobconverter.from_openvino(
        xml='$IR_XML',
        bin='${IR_XML%.xml}.bin',
        data_type='FP16',
        shaves=6,
        output_dir='$MODEL_DIR',
    )
    print(f'Blob created: {blob_path}')
except ImportError:
    print('blobconverter not installed. Install with: pip install blobconverter')
    print('Or upload the .xml/.bin to https://blobconverter.luxonis.com/')
    print(f'  XML: $IR_XML')
    print(f'  BIN: ${IR_XML%.xml}.bin')
" || true

# --- Step 4: Copy to model directory ---
echo ""
echo "--- Step 4: Deploy ---"
echo "Once you have the .blob file, copy it to:"
echo "  $MODEL_DIR/"
echo ""
echo "Then update config.json to reflect 81 classes (80 COCO + drone)."
echo "The drone_detector.py scorer also needs updating (see below)."
echo ""
echo "============================================"
echo "  Export pipeline complete!"
echo "============================================"
