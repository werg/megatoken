#!/usr/bin/env python3
"""Measure how teacher states should become 32 facets: ``halving`` vs ``attention``.

``prepare`` turns small public media sets into prepared feature items with the
locked teacher and realizers (speech: LibriSpeech test-clean parquet, trimmed
to 8 s; images: the Flickr 1k test zip; video: a UCF101 subset tarball,
trimmed to 2 s; long text: HotpotQA paragraphs joined per question, read
compressed). Items are split 90/10 by content hash.

``compare`` trains each readout from the same initialization with only the
encode-and-reconstruct objective (training/autoencode.py) on the same item
stream, and evaluates both on the same held-out items with the same noise:
per-modality realizer reconstruction loss and the raw and centered cosine of
the embedding readout to the teacher's pooled embedding. It also reports
wall time per step and the parameter count of each media path.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

MODALITIES = ("text", "audio", "image", "video")


def _split(family: str) -> str:
    return "validation" if int(family[:8], 16) % 10 == 0 else "train"


def prepare(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    import soundfile

    from megatoken.data.compression import CompressionPolicy, DocumentCompressor
    from megatoken.data.media_prepare import MediaPreparer

    out = args.output
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    preparer = MediaPreparer(args.packs, out / "assets", device=args.device)
    rng = random.Random(args.seed)
    files: list[tuple[str, Path]] = []

    table = pq.read_table(args.librispeech).to_pylist()
    for row in rng.sample(table, min(args.per_modality, len(table))):
        audio, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        path = raw / f"{row['id']}.wav"
        soundfile.write(path, audio[: 8 * rate], rate)
        files.append(("audio", path))

    with zipfile.ZipFile(args.flickr) as archive:
        # The zip also holds a macOS ``__MACOSX/._*.jpg`` metadata stub per image.
        names = sorted(
            n
            for n in archive.namelist()
            if n.lower().endswith(".jpg") and not Path(n).name.startswith("._")
        )
        for name in rng.sample(names, min(args.per_modality, len(names))):
            path = raw / Path(name).name
            path.write_bytes(archive.read(name))
            files.append(("image", path))

    with tarfile.open(args.ucf) as archive:
        members = sorted(
            (m for m in archive.getmembers() if m.name.endswith(".avi")), key=lambda m: m.name
        )
        for member in rng.sample(members, min(args.per_modality, len(members))):
            source = raw / Path(member.name).name
            source.write_bytes(archive.extractfile(member).read())
            clip = source.with_suffix(".mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-t",
                    "2",
                    "-an",
                    str(clip),
                ],
                check=True,
            )
            files.append(("video", clip))

    records = []
    for modality, path in files:
        record = preparer.media(modality, path, target=True)
        records.append(record)
        print(json.dumps({"prepared": modality, "file": path.name}), flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qa_corpora import load_hotpot

    compressor = DocumentCompressor(
        preparer.teacher,
        preparer.store,
        CompressionPolicy(min_bytes=1, roles=frozenset({"document"})),
    )
    questions = load_hotpot(args.hotpot, "train")
    for question in rng.sample(questions, min(args.per_modality, len(questions))):
        document = "\n\n".join(p["payload"].decode() for p in question["paragraphs"]).encode()
        records.extend(descriptor for _, _, descriptor in compressor.descriptors(document))

    with (out / "records.jsonl").open("w") as handle:
        for record in records:
            split = _split(record.source.source_family)
            handle.write(json.dumps({"split": split, "record": record.to_bytes().decode()}) + "\n")
    counts = {m: sum(r.modality == m for r in records) for m in MODALITIES}
    print(json.dumps({"records": counts}), flush=True)


def _trainable(model) -> list:
    modules = [model.core.media_encoder, model.core.frontend, model.embedding_realizer]
    modules += [pack.condition for pack in model.media_realizers.packs.values()]
    seen, params = set(), []
    for module in modules:
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    return params


def _evaluate(model, items, store, center, space, seed) -> dict:
    import torch

    from megatoken.generation.embedding import EmbeddingSpec
    from megatoken.training.autoencode import autoencode_losses

    report = {}
    model.eval()
    with torch.no_grad():
        for modality in MODALITIES:
            group = [item for item in items if item.modality == modality]
            if not group:
                continue
            generator = torch.Generator(device=center.device).manual_seed(seed)
            terms = autoencode_losses(
                model, group, store, space_fingerprint=space, center=center, generator=generator
            )
            value, count = terms["reconstruction"]
            entry = {"reconstruction": float(value) / count} if count else {}
            raws, cents = [], []
            for item in group:
                features = torch.from_numpy(store.load(item.source)).to(center.device)[None]
                readout = model.embedding_realizer(
                    model.core.encode_media(features, extractor=model.config.media_extractor),
                    EmbeddingSpec(space, "document", center.shape[0]),
                )[0]
                target = torch.from_numpy(store.load(item.embedding)).to(center.device)
                raws.append(float(torch.nn.functional.cosine_similarity(readout, target, dim=0)))
                cents.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            readout - center, target - center, dim=0
                        )
                    )
                )
            entry["embedding_cosine"] = sum(raws) / len(raws)
            entry["embedding_centered_cosine"] = sum(cents) / len(cents)
            report[modality] = entry
    model.train()
    return report


def compare(args: argparse.Namespace) -> None:
    import torch

    from megatoken.data.media_assets import MediaRecord, TensorAssetStore
    from megatoken.generation.codec import ByteCodec
    from megatoken.loading import build_model, load_model_config
    from megatoken.omni.loader import install_realizers
    from megatoken.training.autoencode import autoencode_losses

    device = torch.device(args.device)
    store = TensorAssetStore(args.root / "assets")
    splits: dict[str, dict[str, list[MediaRecord]]] = {"train": {}, "validation": {}}
    for line in (args.root / "records.jsonl").read_text().splitlines():
        row = json.loads(line)
        record = MediaRecord.from_bytes(row["record"].encode())
        splits[row["split"]].setdefault(record.modality, []).append(record)
    held_out = [
        item
        for modality in MODALITIES
        for item in splits["validation"].get(modality, [])[: args.eval_items]
    ]
    embeddings = torch.stack(
        [
            torch.from_numpy(store.load(item.embedding))
            for group in splits["train"].values()
            for item in group
        ]
    )
    center = embeddings.mean(0).to(device)
    space = "readout-comparison"
    report = {"items": {k: {m: len(v) for m, v in g.items()} for k, g in splits.items()}}
    for variant in args.variants:
        torch.manual_seed(args.seed)
        # Only the media path trains here: the vocabulary is the byte codec's.
        config = load_model_config(args.model, media_readout=variant, vocab_size=256)
        model = build_model(config, ByteCodec.byte_fallback()).to(device)
        install_realizers(model, str(args.packs), device=device)
        params = _trainable(model)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        media_params = sum(p.numel() for p in model.core.media_encoder.parameters())
        stream = random.Random(args.seed)
        modalities = [m for m in MODALITIES if splits["train"].get(m)]
        curve, seconds = {}, []
        for step in range(args.steps + 1):
            if step % args.eval_every == 0:
                curve[step] = _evaluate(model, held_out, store, center, space, args.seed)
                print(json.dumps({"variant": variant, "step": step, **curve[step]}), flush=True)
            if step == args.steps:
                break
            items = [
                stream.choice(splits["train"][modalities[i % len(modalities)]])
                for i in range(args.batch)
            ]
            started = time.perf_counter()
            terms = autoencode_losses(model, items, store, space_fingerprint=space, center=center)
            loss = sum(value / count for value, count in terms.values() if count)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
        report[variant] = {
            "media_path_parameters": media_params,
            "seconds_per_step": sum(seconds) / len(seconds),
            "curve": curve,
        }
        del model, optimizer
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("prepare")
    item.add_argument("--packs", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--librispeech", type=Path, required=True, help="test.clean parquet")
    item.add_argument("--flickr", type=Path, required=True, help="images_flickr_1k_test.zip")
    item.add_argument("--ucf", type=Path, required=True, help="UCF101_subset.tar.gz")
    item.add_argument("--hotpot", type=Path, required=True, help="HotpotQA distractor dir")
    item.add_argument("--per-modality", type=int, default=400)
    item.add_argument("--seed", type=int, default=20261022)
    item.add_argument("--device", default="cuda")
    item = commands.add_parser("compare")
    item.add_argument("--root", type=Path, required=True, help="prepare --output")
    item.add_argument("--packs", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--model", default="32x512")
    item.add_argument("--variants", nargs="+", default=["halving", "attention"])
    item.add_argument("--steps", type=int, default=1500)
    item.add_argument("--batch", type=int, default=16)
    item.add_argument("--eval-every", type=int, default=250)
    item.add_argument("--eval-items", type=int, default=32)
    item.add_argument("--lr", type=float, default=3e-4)
    item.add_argument("--seed", type=int, default=7)
    item.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else compare(args)


if __name__ == "__main__":
    main()
