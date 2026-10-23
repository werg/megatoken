#!/usr/bin/env python3
"""Freeze the source-by-source evidence of availability, preparation and sampling."""

import argparse
import json
from pathlib import Path

from megatoken.data.inventory import source_inventory
from megatoken.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, action="append")
    parser.add_argument("--prepared", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-metadata", type=Path, action="append", default=[])
    parser.add_argument("--access-receipts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalogs = args.catalog or sorted(Path("configs/data").glob("sources.*.json"))
    receipts = [] if args.access_receipts is None else json.loads(args.access_receipts.read_text())
    write_json(
        args.output,
        source_inventory(
            catalogs,
            prepared=args.prepared,
            checkpoints=args.checkpoint_metadata,
            access_receipts=receipts,
        ),
    )


if __name__ == "__main__":
    main()
