"""Full-target frozen writer corpus reproduction with original assignment coverage.

All optimized free-memory rows are labels only. This command consumes exactly
the prepared SOURCE E facet views and source-window statistics, verifying the
dataset/checkpoint identities and complete expected stable assignments.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch

from megatoken.io import sha256_file
from megatoken.memory.authoring import (
    FrozenCorpusAuthor,
    FrozenWriterArtifact,
    PairAssignment,
    SourceAssignment,
)
from megatoken.memory.contracts import Footprint, SourceRange
from megatoken.memory.recipe import RetrievalRecipe
from megatoken.training.writer_data import WriterDataset, _shard_tensors


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _footprint(value: list[dict]) -> Footprint:
    return Footprint.from_json(json.dumps(value))


def _source_assignment(item: dict) -> SourceAssignment:
    return SourceAssignment(
        identity=item["identity"],
        content_id=item["content_id"],
        scope=item["scope"],
        footprint=_footprint(item["footprint"]),
        evidence_span=SourceRange(
            **{
                **item["evidence_span"],
                "ancestors": tuple(item["evidence_span"].get("ancestors", ())),
            }
        ),
        asset_ref=item["asset_ref"],
        original_source=item["original_source"],
        original_distance=float(item["original_distance"]),
        batch_index=int(item["batch_index"]),
        output_slot=int(item["output_slot"]),
    )


def _digest_rows(rows: list[dict], *, field: str | None = None) -> str:
    if field == "footprint":
        rows = [row[field] for row in rows]
    return sha256(
        json.dumps(
            rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def reproduce(
    recipe: RetrievalRecipe,
    *,
    source_dataset_path: Path,
    source_metadata_path: Path,
    pair_metadata_path: Path,
    expected_manifest_path: Path,
    source_writer_path: Path,
    pair_writer_path: Path,
    output: Path,
) -> dict:
    if recipe.phase != "corpus_reproduction":
        raise ValueError("Corpus rebuild requires a post-pretraining reproduction recipe")
    targets = json.loads(expected_manifest_path.read_text())
    if targets.get("schema") != "megatoken-reproduction-targets-v1" or (
        targets["space"] != recipe.space.fingerprint
        or targets["source_dataset_sha256"] != sha256_file(source_dataset_path)
        or targets["source_metadata_sha256"] != sha256_file(source_metadata_path)
        or targets["pair_metadata_sha256"] != sha256_file(pair_metadata_path)
    ):
        raise RuntimeError("Full-target dataset/assignment manifest identity differs")
    dataset = WriterDataset.load(source_dataset_path)
    if (
        dataset.kind != "source"
        or dataset.mode != "full"
        or (
            dataset.recipe_fingerprint != recipe.fingerprint
            or dataset.semantic_space_fingerprint != recipe.space.fingerprint
        )
    ):
        raise ValueError("Source writer requires the matching frozen full-context source view")
    source_writer = FrozenWriterArtifact(
        source_writer_path,
        expected_recipe=recipe.fingerprint,
        expected_space=recipe.space.fingerprint,
        expected_model_snapshot=dataset.model_checkpoint_id,
        expected_free_bank=dataset.free_bank_sha256,
        kind="source",
    )
    pair_writer = FrozenWriterArtifact(
        pair_writer_path,
        expected_recipe=recipe.fingerprint,
        expected_space=recipe.space.fingerprint,
        expected_model_snapshot=dataset.model_checkpoint_id,
        expected_free_bank=dataset.free_bank_sha256,
        kind="pair",
    )
    author = FrozenCorpusAuthor(
        source_writer=source_writer,
        pair_writer=pair_writer,
        semantic_space=recipe.space,
        value_dim=recipe.model.semantic_value_dim,
    )
    source_rows = _lines(source_metadata_path)
    by_shard: dict[str, list[dict]] = {}
    for item in source_rows:
        by_shard.setdefault(item["shard"], []).append(item)
    if set(by_shard) != {ref.path for ref in dataset.shards}:
        raise RuntimeError("Source assignment metadata does not cover every frozen E shard")
    root = source_dataset_path.resolve().parent
    for shard in dataset.shards:
        rows = by_shard[shard.path]
        if _digest_rows(rows) != shard.assignment_digest or (
            _digest_rows(rows, field="footprint") != shard.source_footprint_digest
        ):
            raise RuntimeError("Original source assignment/footprint digest differs")
        values = _shard_tensors(root, shard)
        author.add_sources(
            values["encoded"],
            values["valid_blocks"].bool(),
            tuple(_source_assignment(item) for item in rows),
        )
    pair_rows = _lines(pair_metadata_path)
    pairs = []
    for item in pair_rows:
        pairs.append(
            PairAssignment(
                left_assignment=item["left_assignment"],
                right_assignment=item["right_assignment"],
                scope=item["scope"],
                footprint=_footprint(item["footprint"]),
                prior=float(item["prior"]),
                statistics=torch.tensor(item["statistics"], dtype=torch.float32),
            )
        )
    author.add_pairs(pairs)
    expected_point = set(targets["point_assignments"])
    expected_source = set(targets["source_content_ids"])
    expected_pair = {tuple(sorted(pair)) for pair in targets["pair_assignments"]}
    result = author.publish(
        output,
        expected_points=expected_point,
        expected_sources=expected_source,
        expected_pairs=expected_pair,
    )
    receipt = {
        "schema": "megatoken-reproduced-bank-v1",
        "recipe": recipe.fingerprint,
        "space": recipe.space.fingerprint,
        "source_writer_weights": source_writer.manifest["weights_sha256"],
        "pair_writer_weights": pair_writer.manifest["weights_sha256"],
        "original_target_manifest_sha256": sha256_file(expected_manifest_path),
        "snapshot_sha256": sha256_file(result.snapshot),
        "source_count": len(author.sources),
        "point_count": len(author.points),
        "pair_count": len(author.pairs),
        "source_coverage": result.source_coverage,
        "point_coverage": result.point_coverage,
        "pair_coverage": result.pair_coverage,
        "behavioral_uptake_verified": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate full source+KB from frozen writers")
    for arg in (
        "recipe",
        "source-dataset",
        "source-metadata",
        "pair-metadata",
        "expected-manifest",
        "source-writer",
        "pair-writer",
        "output",
    ):
        parser.add_argument(f"--{arg}", required=True, type=Path)
    opts = parser.parse_args(argv)
    receipt = reproduce(
        RetrievalRecipe.load(opts.recipe),
        source_dataset_path=opts.source_dataset,
        source_metadata_path=opts.source_metadata,
        pair_metadata_path=opts.pair_metadata,
        expected_manifest_path=opts.expected_manifest,
        source_writer_path=opts.source_writer,
        pair_writer_path=opts.pair_writer,
        output=opts.output,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
