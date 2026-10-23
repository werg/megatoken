"""Offline, evidence-based source inventory across catalogs and actual runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from megatoken.io import sha256_file

from .media_assets import MEDIA_RECORD_TYPE, MediaRecord
from .recipes import STREAM_PRODUCERS
from .sources import load_source_manifest
from .storage import PreparedDataset


def source_inventory(
    catalogs: list[Path],
    *,
    prepared: list[Path] = (),
    checkpoints: list[Path] = (),
    access_receipts: list[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Report evidence separately; never infer access or exposure from a catalog.

    Preparation counts are per immutable root and split, deliberately not summed
    across copies. Sampling counts are per checkpoint, deliberately not summed
    across successive snapshots of the same run. Access receipts must identify
    an exact revision and retain their observed result and supporting evidence.
    """
    sources: dict[str, dict] = {}
    for catalog in catalogs:
        raw = json.loads(catalog.read_text())
        rows = raw["sources"]
        media = bool(rows and "repository" in rows[0])
        if media:
            from .open_media import load_sources

            specs = load_sources(catalog).values()
        else:
            specs = load_source_manifest(catalog)
        for spec in specs:
            entry = sources.setdefault(
                spec.source_id, {"listed": [], "access": [], "prepared": [], "sampled": []}
            )
            entry["listed"].append(
                {
                    "catalog": str(catalog.resolve()),
                    "catalog_sha256": sha256_file(catalog),
                    "revision": spec.revision,
                    "streams": list(spec.streams) if media else [spec.stream],
                    "location": spec.repository if media else spec.uri,
                    "terms": asdict(spec.terms),
                    "gated_declared": spec.gated if media else None,
                }
            )
    for receipt in access_receipts:
        if (
            not receipt.get("source_id")
            or not receipt.get("revision")
            or receipt.get("result") not in {"accessible", "denied", "downloaded", "failed"}
            or not receipt.get("checked_at")
            or not receipt.get("evidence")
        ):
            raise ValueError(
                "Access receipts need source/revision, observed result, time and evidence"
            )
        if receipt["result"] == "downloaded":
            files = receipt.get("files", ())
            if not files:
                raise ValueError("Downloaded status needs locally checked source files")
            for item in files:
                if sha256_file(item["path"]) != item["sha256"]:
                    raise ValueError("Downloaded source checksum differs from access receipt")
        entry = sources.setdefault(
            receipt["source_id"], {"listed": [], "access": [], "prepared": [], "sampled": []}
        )
        entry["access"].append(receipt)
    for root in prepared:
        with PreparedDataset(root, split=None) as dataset:
            # Read every shard and referenced tensor, not just the small index.
            # An inventory claiming prepared supply must reject incomplete copies.
            for episode in dataset.iter_episodes():
                for record in episode.records:
                    if record.media_type == MEDIA_RECORD_TYPE:
                        media_record = MediaRecord.from_bytes(record.payload)
                        for asset in (
                            media_record.source,
                            media_record.embedding,
                            media_record.target,
                            media_record.rendered,
                        ):
                            if asset is not None:
                                dataset.assets.load(asset)
            # The
            # SQL uses the admitted family identities, not generated task views.
            rows = dataset.connection.execute(
                "SELECT source_id, split, stream, COUNT(*), COUNT(DISTINCT family_id), "
                "SUM(scored_positions) FROM episodes GROUP BY source_id, split, stream"
            )
            for source, split, stream, episodes, families, positions in rows:
                entry = sources.setdefault(
                    source, {"listed": [], "access": [], "prepared": [], "sampled": []}
                )
                entry["prepared"].append(
                    {
                        "root": str(root.resolve()),
                        "manifest_sha256": sha256_file(root / "manifest.json"),
                        "fingerprint": dataset.manifest["fingerprint"],
                        "split": split,
                        "stream": stream,
                        "episodes": episodes,
                        "original_families": families,
                        "scored_positions": positions,
                        "receipts": dataset.manifest["sources"],
                    }
                )
    for checkpoint in checkpoints:
        metadata = json.loads(checkpoint.read_text())
        report = metadata.get("source_sampling")
        if report is None:
            continue  # Historical checkpoints cannot establish source exposure.
        for source, windows in report["source_windows"].items():
            if type(windows) is not int or windows < 0:
                raise ValueError("Sampled window counts must be nonnegative integers")
            entry = sources.setdefault(
                source, {"listed": [], "access": [], "prepared": [], "sampled": []}
            )
            entry["sampled"].append(
                {
                    "checkpoint_metadata": str(checkpoint.resolve()),
                    "metadata_sha256": sha256_file(checkpoint),
                    "global_step": metadata["global_step"],
                    "windows": windows,
                    "complete_run_history": report["complete_history"],
                    "revision_scope": "source_id; bind to this run's admitted preparation receipts",
                }
            )
    return {
        "version": 1,
        "sources": dict(sorted(sources.items())),
        "local_producers": dict(STREAM_PRODUCERS),
        "unknown_access_is_not_denial": True,
        "sampled_windows_are_not_quality_evidence": True,
        "unmeasured_checkpoint_metadata": [
            str(path.resolve())
            for path in checkpoints
            if "source_sampling" not in json.loads(path.read_text())
        ],
    }
