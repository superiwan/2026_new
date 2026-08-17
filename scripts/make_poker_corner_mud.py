#!/usr/bin/env python3
"""Create and validate a MaixPy MUD file for the poker corner model."""

import argparse
from pathlib import Path


TEMPLATE = """[basic]
type = cvimodel
model = {model}

[extra]
model_type = yolo11
type = detector
input_type = rgb
mean = 0,0,0
scale = 0.00392156862745098,0.00392156862745098,0.00392156862745098
labels = {labels}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cvimodel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", default="corner_mark")
    args = parser.parse_args()

    model = args.cvimodel.resolve()
    if not model.is_file() or model.stat().st_size == 0:
        raise SystemExit("missing or empty cvimodel: %s" % model)
    output = (args.output.resolve() if args.output else
              model.with_suffix(".mud"))
    output.parent.mkdir(parents=True, exist_ok=True)
    content = TEMPLATE.format(model=model.name, labels=args.labels)
    output.write_text(content, encoding="ascii", newline="\n")
    if "model = %s" % model.name not in output.read_text(encoding="ascii"):
        raise SystemExit("MUD model reference validation failed")
    print("MUD_OK path=%s model=%s labels=%s" % (
        output, model.name, args.labels))


if __name__ == "__main__":
    main()
