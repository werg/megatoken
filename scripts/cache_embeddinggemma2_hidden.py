"""Cache pinned unpooled EmbeddingGemma2 hidden states as native FP32 media assets.

Example:
python scripts/cache_embeddinggemma2_hidden.py --snapshot /models/embeddinggemma-2 \
 --revision <40-hex-sha> --modality image --input img.png --asset-root data/prepared \
 --output data/prepared/teacher/image-1.json

For document or text the input is UTF-8 source text. Split long documents
upstream at auditable source boundaries; never silently truncate 8192 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from megatoken.data.media_assets import TensorAssetStore
from megatoken.omni.embeddinggemma2_hidden import EmbeddingGemma2HiddenExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--modality", choices=("text", "document", "image", "audio", "video"), required=True
    )
    parser.add_argument("--input", required=True, help="Local immutable original source")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--identity-target-sha256",
        default="",
        help="SHA256 of matching raw/codec target tensor asset, required for autoencoder training",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fps-num", type=int, default=0, help="Actual video sampling rate numerator"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=0,
        help="Actual input-audio sampling rate; supply 16000 or 24000",
    )
    parser.add_argument(
        "--fps-den", type=int, default=1, help="Actual video sampling rate denominator"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_symlink():
        parser.error("Input source symlinks are not allowed")
    path = input_path.resolve()
    if not path.is_file():
        parser.error("Input must be a regular local file, not a symlink")
    with path.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.modality == "document":
        if path.suffix.lower() == ".pdf":
            try:
                import pymupdf
            except ImportError as exc:
                raise RuntimeError(
                    "Install megatoken[omni-documents] for PDF source parsing"
                ) from exc
            with pymupdf.open(path) as pdf:
                source = "\\f".join(page.get_text(sort=True) for page in pdf)
            if not source.strip():
                raise ValueError(
                    "Scanned PDF needs visual page extraction, not empty document text"
                )
        elif path.suffix.lower() == ".docx":
            try:
                from docx import Document
            except ImportError as exc:
                raise RuntimeError(
                    "Install megatoken[omni-documents] for DOCX source parsing"
                ) from exc
            doc = Document(path)
            source = "\\n".join(p.text for p in doc.paragraphs if p.text.strip())
        else:
            source = path.read_text(encoding="utf-8")
    elif args.modality == "text":
        source = path.read_text(encoding="utf-8")
    else:
        source = str(path)
    if args.modality in {"text", "document"} and not source.strip():
        parser.error("Empty text/document source has no useful teacher evidence")
    if args.identity_target_sha256:
        if args.modality not in {"audio", "image", "video"} or not re.fullmatch(
            r"[0-9a-f]{64}", args.identity_target_sha256
        ):
            parser.error("Identity target needs a media asset SHA256")
        target_path = Path(args.asset_root) / "media" / f"{args.identity_target_sha256}.f32le"
        if not target_path.is_file() or target_path.is_symlink():
            parser.error("Identity reconstruction requires a staged original target asset")
        with target_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != args.identity_target_sha256:
            parser.error("Staged identity target asset failed its SHA256 verification")
    if args.modality == "audio" and args.sample_rate not in (16000, 24000):
        parser.error("Audio feature assets require an explicit 16000 or 24000 Hz source clock")
    if args.modality != "audio" and args.sample_rate:
        parser.error("Only audio has a source sample clock")
    if args.modality == "video" and (args.fps_num <= 0 or args.fps_den <= 0):
        parser.error("Video feature assets require an explicit positive rational frame clock")
    if args.modality != "video" and args.fps_num:
        parser.error("Only video has an FPS sampling clock")
    extractor = EmbeddingGemma2HiddenExtractor.from_local_snapshot(
        args.snapshot, revision=args.revision, device=args.device
    )
    observation = extractor.encode(args.modality, source, max_tokens=args.max_tokens)
    # Use existing SHA256 content-addressed assets and include original source
    # identity. 'document'/'text' use the standalone text-hidden asset contract.
    features = observation.features[0][observation.mask[0]].numpy().astype("<f4")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.modality in {"audio", "image", "video"}:
        store = TensorAssetStore(args.asset_root)
        asset = store.put(
            features,
            modality=args.modality,
            kind="features",
            extractor=observation.extractor,
            preprocessing_id="chat-template-v1",
            source_family=args.identity_target_sha256 or source_sha,
            fps_num=args.fps_num if args.modality == "video" else 0,
            sample_rate=args.sample_rate if args.modality == "audio" else 0,
            fps_den=args.fps_den,
        )
        descriptor = {}
        from dataclasses import asdict

        descriptor["asset"] = asdict(asset)
    else:
        # The same CAS format as media features, but documents/text are not
        # physical MediaRecords and cannot be parsed as audio/image/video.
        data = features.tobytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("teacher") / f"{digest}.f32le"
        target = Path(args.asset_root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Immutable hidden-state cache collision")
        if not target.exists():
            from megatoken.io import atomic_binary_writer

            with atomic_binary_writer(target) as stream:
                stream.write(data)
        descriptor = {
            "asset": {
                "kind": "features",
                "path": str(relative),
                "sha256": digest,
                "shape": list(features.shape),
                "extractor": observation.extractor,
            }
        }
    descriptor.update(
        {
            "format": "megatoken.embeddinggemma2.hidden.v1",
            "source_sha256": source_sha,
            "teacher_input_sha256": (
                hashlib.sha256(source.encode("utf-8")).hexdigest()
                if args.modality in {"text", "document"}
                else ""
            ),
            "identity_target_sha256": args.identity_target_sha256,
            "modality": args.modality,
            "hidden_tap": "language_model.embedding_projection:input",
            "teacher_revision": args.revision,
            "preprocessing_id": "chat-template-v1",
            "valid_tokens": len(features),
        }
    )
    output.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
