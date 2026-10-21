#!/usr/bin/env python3
"""Compile verified aligned bundles into Megatoken's existing native Episode JSONL.

This is an offline *compiler*, not data preparation: it reads only already
supplied immutable bytes/MediaRecords and never downloads, extracts, trains or
applies verifiers. Its output is directly consumable by the native source
adapter with preserve_native_stream=true in the existing preparation engine.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord
from megatoken.data.schema import canonical_json
from megatoken.omni.bundles import (
    AlignedBundle,
    Artifact,
    Edit,
    Relation,
    TaskView,
    compile_edit_trajectory,
    compile_views,
)


def _artifact(value: dict) -> Artifact:
    source = dict(value)
    payload = source.pop("payload_base64", None)
    if payload is None or "payload" in source:
        raise ValueError("Source bytes must be explicitly encoded as payload_base64")
    raw = base64.b64decode(payload, validate=True)
    if source.get("modality") in {"audio", "image", "video"}:
        if source.get("media_type") != MEDIA_RECORD_TYPE:
            raise ValueError("Media artifacts require canonical pinned MediaRecord descriptors")
        ref = MediaRecord.from_bytes(raw)
        if ref.modality != source["modality"]:
            raise ValueError("Aligned modality differs from pinned media descriptor")
    return Artifact(payload=raw, **source)


def compile_bundle(value: dict):
    """Return native Episodes with separate preserved supervision metadata."""
    source = dict(value)
    artifacts = tuple(_artifact(v) for v in source.pop("artifacts"))
    relations = tuple(Relation(**r) for r in source.pop("relations", ()))
    edits = tuple(Edit(**e) for e in source.pop("edits", ()))
    views = source.pop("views", ())
    chains = source.pop("trajectories", ())
    bundle = AlignedBundle(
        artifacts=artifacts,
        relations=relations,
        edits=edits,
        **source,
    )
    compiled_views = []
    for spec in views:
        spec = dict(spec)
        selected_relations = spec.pop("relations", ())
        refs = tuple(
            relations[i]
            if type(i) is int and 0 <= i < len(relations)
            else Relation(**i)
            if isinstance(i, dict)
            else None
            for i in selected_relations
        )
        if any(r is None for r in refs):
            raise ValueError("Task view has an unknown relation")
        task = TaskView(
            inputs=tuple(spec.pop("inputs", ())),
            outputs=tuple(spec.pop("outputs")),
            relations=refs,
            **spec,
        )
        # Do not allow content plans to be relabeled as supplied inputs.
        if task.content_plan and task.content_plan in task.inputs:
            raise ValueError("Generated plans cannot be visible source inputs")
        compiled_views.extend(compile_views(bundle, (task,)))
    for chain in chains:
        steps = []
        for entry in chain["steps"]:
            index = entry["edit_index"]
            if type(index) is not int or index < 0 or index >= len(edits):
                raise ValueError("Edit chain requires certified edit index")
            steps.append(
                (
                    edits[index],
                    entry["output_artifact_id"],
                    tuple(entry.get("related_output_ids", ())),
                )
            )
        compiled_views.append(
            compile_edit_trajectory(
                bundle,
                base_artifact_id=chain["base_artifact_id"],
                steps=tuple(steps),
                request_revision=int(chain.get("request_revision", 0)),
            )
        )
    if not compiled_views:
        raise ValueError("Bundle must declare at least one valid causal task view")
    return compiled_views


def compile_manifest(source: str, output: str) -> dict:
    """Fail closed on mixed-family splits and overwrite, write atomically."""
    source_path = Path(source).resolve()
    destination = Path(output).resolve()
    if source_path == destination or destination.exists():
        raise ValueError("Native task compiler cannot overwrite its input/output")
    prepared = []
    families = {}
    count = 0
    for row in source_path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        views = compile_bundle(json.loads(row))
        for view in views:
            existing = families.setdefault(view.source_family, view.split)
            if existing != view.split:
                raise ValueError(
                    "Counterfactual siblings or source families crossed a frozen split"
                )
            prepared.append(view)
        count += 1
    if not prepared:
        raise ValueError("No valid aligned bundles")
    seen = set()
    lines = []
    for view in prepared:
        if view.episode.episode_id in seen:
            raise ValueError("Duplicate native episode; task view needs unique identity")
        seen.add(view.episode.episode_id)
        lines.append(canonical_json(view.episode.to_dict()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    from megatoken.io import atomic_binary_writer

    with atomic_binary_writer(destination) as handle:
        handle.write(("\n".join(lines) + "\n").encode("utf-8"))
    return {
        "compiler": "aligned-bundles-v1",
        "bundles": count,
        "episodes": len(lines),
        "source_families": len(families),
        "episode_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", required=True, help="JSONL of immutable aligned bundles")
    parser.add_argument("--output", required=True, help="Native Episode JSONL output")
    args = parser.parse_args()
    print(json.dumps(compile_manifest(args.bundles, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
