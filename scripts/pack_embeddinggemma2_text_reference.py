"""Package a pinned offline text/document hidden cache as one canonical input event.

Example:
 python scripts/pack_embeddinggemma2_text_reference.py \\
   --descriptor data/teacher/document.json --asset-root data \\
   --output data/teacher/document.ref.json

Embed the resulting reference bytes unchanged in an observed PhysicalRecord
with media_type=TEXT_HIDDEN_RECORD_TYPE and TrackEvent.modality equal to
'document' or 'text_summary'. Do not place these references on policy targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from megatoken.data.text_hidden import TextHiddenReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference = TextHiddenReference.from_cache_descriptor(
        json.loads(args.descriptor.read_text(encoding="utf-8"))
    )
    reference.load(args.asset_root)
    if args.output.exists():
        parser.error("Do not overwrite an existing immutable text reference")
    from megatoken.io import atomic_binary_writer

    with atomic_binary_writer(args.output) as output:
        output.write(reference.to_bytes())
    print(f"{reference.modality}: {reference.tokens} teacher tokens -> one megatoken")


if __name__ == "__main__":
    main()
