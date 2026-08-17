#!/usr/bin/env bash
set -euo pipefail

MODEL_ONNX="${1:-./conversion/model.onnx}"
CALIBRATION_DIR="${2:-./conversion/calibration}"
OUTPUT_DIR="${3:-./conversion/output_640}"
MODEL_NAME="${MODEL_NAME:-poker_corner_yolo11n_640}"
ARTIFACT_NAME="${ARTIFACT_NAME:-${MODEL_NAME}_int8_v1}"
INPUT_HEIGHT="${INPUT_HEIGHT:-640}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
OUTPUT_NAMES="${OUTPUT_NAMES:-/model.23/dfl/conv/Conv_output_0,/model.23/Sigmoid_output_0}"
CALIBRATION_COUNT="${CALIBRATION_COUNT:-100}"

test -f "$MODEL_ONNX"
test -d "$CALIBRATION_DIR"
mkdir -p "$OUTPUT_DIR"

TEST_IMAGE="$(find "$CALIBRATION_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) -print -quit)"
test -n "$TEST_IMAGE"

EXTRACTED="$OUTPUT_DIR/${MODEL_NAME}_extract.onnx"
SIMPLIFIED="$OUTPUT_DIR/${MODEL_NAME}.onnx"
python - "$MODEL_ONNX" "$EXTRACTED" "$OUTPUT_NAMES" <<'PY'
import onnx
import sys

source, target, output_names = sys.argv[1:]
model = onnx.load(source)
input_name = model.graph.input[0].name
onnx.utils.extract_model(
    source, target, [input_name], output_names.split(","))
print("ONNX_EXTRACT_OK input=%s outputs=%s" % (input_name, output_names))
PY
python -m onnxsim "$EXTRACTED" "$SIMPLIFIED"

model_transform.py \
  --model_name "$MODEL_NAME" \
  --model_def "$SIMPLIFIED" \
  --input_shapes "[[1,3,${INPUT_HEIGHT},${INPUT_WIDTH}]]" \
  --mean "0,0,0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --channel_format nchw \
  --output_names "$OUTPUT_NAMES" \
  --test_input "$TEST_IMAGE" \
  --test_result "$OUTPUT_DIR/${MODEL_NAME}_top_outputs.npz" \
  --tolerance 0.99,0.99 \
  --mlir "$OUTPUT_DIR/${MODEL_NAME}.mlir"

run_calibration.py "$OUTPUT_DIR/${MODEL_NAME}.mlir" \
  --dataset "$CALIBRATION_DIR" \
  --input_num "$CALIBRATION_COUNT" \
  -o "$OUTPUT_DIR/${MODEL_NAME}_cali_table"

model_deploy.py \
  --mlir "$OUTPUT_DIR/${MODEL_NAME}.mlir" \
  --quantize INT8 \
  --quant_input \
  --calibration_table "$OUTPUT_DIR/${MODEL_NAME}_cali_table" \
  --processor cv181x \
  --test_input "${MODEL_NAME}_in_f32.npz" \
  --test_reference "$OUTPUT_DIR/${MODEL_NAME}_top_outputs.npz" \
  --tolerance 0.9,0.6 \
  --model "$OUTPUT_DIR/${ARTIFACT_NAME}.cvimodel"

python ./scripts/make_poker_corner_mud.py \
  "$OUTPUT_DIR/${ARTIFACT_NAME}.cvimodel" \
  --output "$OUTPUT_DIR/${ARTIFACT_NAME}.mud"

INTERMEDIATE_DIR="$OUTPUT_DIR/intermediate"
mkdir -p "$INTERMEDIATE_DIR"
find . -maxdepth 1 -type f \
  \( -name "${MODEL_NAME}*" -o -name '_weight_map.csv' \) \
  -exec mv -f {} "$INTERMEDIATE_DIR/" \;

echo "CV181X_CONVERSION_OK output=$OUTPUT_DIR"
