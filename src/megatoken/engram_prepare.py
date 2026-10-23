"""Prepare both selected native n-gram donors beside verified cloud training data.

Production preparation caches the selected native table shards in the cloud
and reads their selected rows through bounded local file reads. HTTP ranges are
available for small targeted checks. Both modes write self-contained assets.
Run this module inside a preparation sandbox, never on the workstation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from megatoken import cache, publish
from megatoken.donors import DONOR_PROFILES
from megatoken.io import sha256_file, write_json

SELECTED_DONORS = ("deepseek-v4.1-flash-1017", "qwen3.8-flash-next")


def prepare_donors(
    pin_path: str | Path,
    output: str | Path,
    work: str | Path,
    *,
    model: str,
    max_contexts: int,
    seed: int,
    source_mode: str = "local",
) -> dict:
    """Restore one pinned corpus and produce an ordered pair of native assets."""
    if max_contexts < 1:
        raise ValueError("max_contexts must be positive")
    from huggingface_hub import hf_hub_download

    from megatoken import hub
    from megatoken.data import PreparedCorpus
    from megatoken.loading import load_model_config
    from megatoken.native_engram import extract_native_engram

    output, work = Path(output), Path(work)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    pin = publish.load_pin(Path(pin_path))
    restored = work / "prepared"
    publish.fetch(api=hub.api(), keys=cache.load_keys(), pin=pin, dest=restored)
    roots = sorted(path for path in restored.iterdir() if (path / "manifest.json").is_file())
    with PreparedCorpus(roots, split="train") as corpus:
        corpus_identity = corpus.identity
    target = load_model_config(model)
    assets = []
    for name in SELECTED_DONORS:
        profile = DONOR_PROFILES[name]
        metadata = work / "donors" / name
        for filename in (
            "tokenizer.json",
            "config.json",
            "model.safetensors.index.json",
            "LICENSE",
        ):
            hf_hub_download(
                repo_id=profile.repository,
                revision=profile.revision,
                filename=filename,
                local_dir=metadata,
                token=False,
            )
        destination = output / name
        if destination.exists():
            report = json.loads((destination / "engram.json").read_text())
            expected = {
                "algorithm": "native-context-to-byte-hash-v1",
                "donor_profile": name,
                "corpus_fingerprint": corpus_identity,
                "corpus_split": "train",
                "max_contexts": max_contexts,
                "seed": seed,
                "ngram_rows": target.ngram_rows,
                "ngram_orders": list(target.ngram_orders),
                "exact_rows": target.exact_rows,
                "exact_bigrams": target.exact_bigrams,
                "hash_algorithm": "rolling-257-order-seeded-v1",
            }
            if (
                any(report.get(key) != value for key, value in expected.items())
                or report.get("donor_metadata", {}).get("revision") != profile.revision
                or sha256_file(destination / "descriptors.npz") != report.get("descriptor_sha256")
            ):
                raise ValueError(
                    f"Existing native donor asset differs from this preparation: {name}"
                )
        else:
            report = extract_native_engram(
                name,
                metadata,
                roots,
                destination,
                model=model,
                max_contexts=max_contexts,
                seed=seed,
                split="train",
                cache_dir=work / "native-row-cache",
                source_mode=source_mode,
            )
        (destination / "DONOR-LICENSE").write_bytes((metadata / "LICENSE").read_bytes())
        assets.append({"donor": name, "path": str(destination), "extraction": report})
    result = {
        "data_repository": pin["repo"],
        "data_revision": pin["revision"],
        "corpus_fingerprint": corpus_identity,
        "model": model,
        "max_contexts_per_donor": max_contexts,
        "seed": seed,
        "source_mode": source_mode,
        "assets": assets,
        "engram_imports": [item["path"] for item in assets],
    }
    write_json(output / "preparation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", default="/work/engram-preparation")
    parser.add_argument("--model", default="32x512")
    parser.add_argument("--max-contexts", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-mode", choices=("local", "ranges"), default="local")
    args = parser.parse_args()
    if secrets := os.environ.get("MEGATOKEN_CW_SECRETS"):
        token = Path(secrets) / "hf-token"
        if token.is_file():
            os.environ["HF_TOKEN"] = token.read_text().strip()
    print(
        json.dumps(
            prepare_donors(
                args.pin,
                args.output,
                args.work,
                model=args.model,
                max_contexts=args.max_contexts,
                seed=args.seed,
                source_mode=args.source_mode,
            )
        )
    )


if __name__ == "__main__":
    main()
