"""Source-catalog import for pinned pretrained semantic teacher vectors.

No teacher call occurs here. The caller's immutable prepared catalog must
identify every original byte span, its full contextual teacher footprint,
source asset checksum and one normalized address in the recipe's named space.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import os

from megatoken.memory.contracts import Footprint, SourceRange
from megatoken.memory.recipe import RetrievalRecipe
from megatoken.memory.sqlite import SQLiteKB
from megatoken.io import sha256_file


def build_source_catalog(
    recipe: RetrievalRecipe, *, catalog: Path, assets_root: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError("Never overwrite the authoritative learned-space catalog")
    if recipe.phase != "coherence_embedding":
        raise ValueError("Literal source catalog construction precedes free-bank growth")
    if (
        recipe.space.teacher_revision.startswith("PIN_")
        or recipe.growth.partition_id.startswith("PIN_")
    ):
        raise ValueError("Pin an actual teacher revision and split partition before import")
    if not catalog.is_file() or not assets_root.is_dir():
        raise FileNotFoundError("Catalog and immutable original source asset root required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-catalog-", dir=output.parent) as tmp:
        working = Path(tmp) / "working.sqlite"
        seen: set[tuple[str, str]] = set()
        bytes_verified: dict[Path, str] = {}
        rows = 0
        with SQLiteKB(working, recipe.space, value_dim=recipe.model.semantic_value_dim) as bank:
            with bank.transaction():
                for number, line in enumerate(catalog.read_text().splitlines(), start=1):
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("semantic_space_fingerprint") != recipe.space.fingerprint:
                        raise ValueError(f"Source {number} mixes incompatible semantic spaces")
                    scope = entry["scope"]
                    identity = (scope, entry["content_id"])
                    if identity in seen:
                        raise ValueError("Duplicate stable source-chunk assignment")
                    seen.add(identity)
                    asset = entry["asset_ref"]
                    if not isinstance(asset, str) or not asset or Path(asset).is_absolute():
                        raise ValueError("Source asset references must be canonical relative paths")
                    source_path = (assets_root / asset).resolve()
                    if not source_path.is_relative_to(assets_root.resolve()) or (
                        not source_path.is_file()
                    ):
                        raise ValueError("Source catalog path escapes immutable asset root")
                    digest = bytes_verified.setdefault(source_path, sha256_file(source_path))
                    if entry["asset_sha256"] != digest:
                        raise ValueError("Original source asset hash differs from catalog")
                    footprint = Footprint.from_json(json.dumps(entry["footprint"]))
                    evidence = SourceRange(**{
                        **entry["evidence_span"],
                        "ancestors": tuple(entry["evidence_span"].get("ancestors", ())),
                    })
                    if evidence not in footprint.ranges or evidence.scope != scope or (
                        evidence.end > source_path.stat().st_size
                    ):
                        raise ValueError("Returned evidence is outside pinned source and context")
                    bank.insert_source(
                        scope=scope, footprint=footprint, key=entry["key"],
                        content_id=entry["content_id"], asset_ref=asset,
                        evidence_span=evidence,
                    )
                    rows += 1
                # Never auto-open the allocator without measured generator
                # coherence and query/document embedding acceptance receipts.
                bank.put_growth_state("readiness", {
                    "coherence": False, "embeddings": False,
                    "space": recipe.space.fingerprint,
                    "recipe": recipe.fingerprint,
                })
            db_id = bank.database_id
            revisions = asdict(bank.revisions())
            bank.backup(Path(tmp) / "catalog.sqlite")
        final = Path(tmp) / "catalog.sqlite"
        report = {
            "schema": "megatoken-source-catalog-v1",
            "recipe": recipe.fingerprint,
            "semantic_space": recipe.space.fingerprint,
            "teacher_revision": recipe.space.teacher_revision,
            "partition_id": recipe.growth.partition_id,
            "source_manifest_sha256": sha256(catalog.read_bytes()).hexdigest(),
            "catalog_sha256": sha256_file(final),
            "database_id": db_id, "sources": rows,
            "verified_original_assets": len(bytes_verified),
            "revisions": revisions,
            "ready_for_free_allocation": False,
        }
        os.replace(final, output)
        # The immutable manifest is independent from the raw source asset tree.
        manifest = output.with_suffix(output.suffix + ".manifest.json")
        manifest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a pinned source ANN/KB catalog")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--assets-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_source_catalog(
        RetrievalRecipe.load(args.recipe), catalog=args.catalog,
        assets_root=args.assets_root, output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
