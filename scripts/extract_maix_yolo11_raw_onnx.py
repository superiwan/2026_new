#!/usr/bin/env python3
"""Extract the two raw YOLO11 tensors required by MaixCAM CV181x conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


RAW_OUTPUTS = ["/model.23/dfl/conv/Conv_output_0", "/model.23/Sigmoid_output_0"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"missing ONNX source: {source}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing ONNX: {output}")
    model = onnx.load(str(source))
    inputs = {tensor.name: [dim.dim_value for dim in tensor.type.tensor_type.shape.dim] for tensor in model.graph.input}
    if inputs != {"images": [1, 3, 640, 640]}:
        raise SystemExit(f"expected fixed images [1, 3, 640, 640], got {inputs}")
    values = {value for node in model.graph.node for value in node.output}
    missing = [value for value in RAW_OUTPUTS if value not in values]
    if missing:
        raise SystemExit(f"missing raw YOLO11 tensors: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(str(source), str(output), ["images"], RAW_OUTPUTS)
    extracted = onnx.load(str(output))
    onnx.checker.check_model(extracted)
    print("input:", [(item.name, [dim.dim_value for dim in item.type.tensor_type.shape.dim]) for item in extracted.graph.input])
    print("outputs:", [(item.name, [dim.dim_value for dim in item.type.tensor_type.shape.dim]) for item in extracted.graph.output])


if __name__ == "__main__":
    main()
