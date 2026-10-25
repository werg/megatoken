#!/usr/bin/env python3
"""Create a read-only family-aware analysis of one frozen semantic evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.io import strict_json_loads, write_text_atomic
from megatoken.training.semantic_analysis import analyze_semantic_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = strict_json_loads(args.cohort.read_bytes())
    report = strict_json_loads(args.report.read_bytes())
    analysis = analyze_semantic_evaluation(cohort, report)
    write_text_atomic(
        args.output,
        json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    print(json.dumps(analysis["summary"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
