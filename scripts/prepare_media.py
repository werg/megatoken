#!/usr/bin/env python3
"""Prepare media events and text anchors with the locked teacher and realizers.

Input JSONL rows, one per event:
  {"modality": "audio"|"image"|"video", "path": "clip.wav", "target": true}
  {"modality": "text", "text": "exact chunk"}          (a byte-E text anchor)

Media rows emit {"path", "record"} (``record`` is the canonical MediaRecord JSON);
text rows emit text-anchor sidecar rows {"text_sha256", "asset"}. All assets are
written under --root. One process keeps the teacher and realizers resident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from megatoken.data.media_prepare import MediaPreparer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Locked pack manifest")
    parser.add_argument("--root", required=True, help="Prepared dataset/asset root")
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True, help="Media record JSONL output")
    parser.add_argument("--anchors", type=Path, help="Text anchor sidecar JSONL output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    preparer = MediaPreparer(args.manifest, args.root, device=args.device)
    counts: dict[str, int] = {}
    with (
        args.records.open("x") as records,
        args.anchors.open("x") if args.anchors else open("/dev/null", "w") as anchors,
    ):
        for line in args.items.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            modality = item["modality"]
            if modality == "text":
                if args.anchors is None:
                    parser.error("Text anchor rows need --anchors")
                payload = item["text"].encode("utf-8")
                asset = preparer.text(payload)
                row = {"text_sha256": hashlib.sha256(payload).hexdigest(), "asset": asdict(asset)}
                anchors.write(json.dumps(row, sort_keys=True) + "\n")
            else:
                record = preparer.media(modality, item["path"], target=item.get("target", True))
                records.write(
                    json.dumps({"path": item["path"], "record": record.to_bytes().decode()}) + "\n"
                )
            counts[modality] = counts.get(modality, 0) + 1
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
