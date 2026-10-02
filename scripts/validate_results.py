#!/usr/bin/env python3
"""Validate one or more retained benchmark series without recomputing metrics."""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bottleneckshift.validation import validate_series  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("series", type=Path, nargs="+", help="series directories containing manifest.json")
    args = parser.parse_args()
    print(json.dumps([validate_series(path) for path in args.series], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
