"""Publish the prepared text data to one Hugging Face dataset repository, and read it back.

Every source's licence decides where its prepared rows go (``terms.redistribute`` in the source
manifests; ``megatoken.data.sources.REDISTRIBUTION_GROUPS``). All of it lives in ONE repository:

    README.md                                   the dataset card
    permissive/<run>/part-0003/<shard>/data-00000.parquet     rows, readable
    permissive/<run>/part-0003/<shard>/root/...               the root's manifest and index
    permissive/<run>/part-0003/sources.json                   attribution of that part
    share_alike/...  non_commercial/...                       the same, one folder per licence group
    encrypted/README.md                         says plainly that the folder is unusable ciphertext
    encrypted/<opaque>/<opaque>.bin             everything that may not be published (megatoken.cache)

A *part* is what one finishing sandbox wrote: several prepared roots (one per shard), each of one
licence group. The sandbox converts the roots of the three public groups to Parquet (the exact
stored episode as one column, a readable transcript beside it, and the index columns) and
encrypts the roots of the fourth. Each part writes only its own folders, so sandboxes upload to
the repository at the same time without ever touching the same path; a commit that races with
another is repeated (``megatoken.hub.retry_conflicts``).

When every part has landed, one final step (``finalize``) rebuilds the dataset card from what the
repository holds and records a *pin*: the repository, one commit of it that contains every part,
and the sha256 of every public file. The pin holds no secret and may be committed.

Reading back (``fetch``) downloads the pinned commit's public files, checks each against the pin,
rebuilds the prepared roots byte for byte (every shard file is checked against the root's own
manifest), and decrypts the encrypted parts with the key. ``decide`` is the rule a training run
follows: a pin that still verifies means fetch, anything else means rebuild (see
``megatoken.cloud_prepare ensure``).

  python -m megatoken.publish fetch --pin FILE --dest DIR [--part N]
  python -m megatoken.publish verify --pin FILE      check the pinned commit without downloading
  python -m megatoken.publish card --repo REPO       rebuild the card from what the repo holds
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sqlite3
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megatoken import cache, hub
from megatoken.io import sha256_file
from megatoken.redistribution import REDISTRIBUTION_GROUPS

PIP = ["pyarrow", *cache.PIP]  # what a sandbox needs beside the project's own requirements
PUBLIC_GROUPS = REDISTRIBUTION_GROUPS[:-1]
PIN_SCHEMA = "megatoken.data-pin/1"
RECEIPT_SCHEMA = "megatoken.data-receipt/1"
SOURCES_SCHEMA = "megatoken.data-sources/1"
PARQUET_BYTES = 1 << 30  # a Parquet file is closed once it passes this size
ROW_GROUP_BYTES = 256 << 20  # text held in memory per Parquet row group
ROW_GROUP_ROWS = 2000
ROOT_FOLDER = "root"  # a root's own files (manifest, index, reports), beside its Parquet files
SOURCES_FILE = "sources.json"
CARD = "README.md"
ENCRYPTED_CARD = f"{cache.ENCRYPTED_DIR}/README.md"
DEFAULT_CONTACT = "open a discussion in this repository's Community tab"
SUBSET_LICENCES = {
    "permissive": "each source's own permissive licence (see the attribution table)",
    "share_alike": "CC-BY-SA-4.0",
    "non_commercial": "CC-BY-NC-4.0",
}
SUBSET_BLURBS = {
    "permissive": "Sources under permissive licences (MIT, Apache-2.0, BSD, CC0, CC-BY, ODC-By, "
    "ODbL, PDDL, Unlicense or close equivalents).",
    "share_alike": "Sources under share-alike licences (CC-BY-SA), published under CC-BY-SA-4.0.",
    "non_commercial": "Sources under non-commercial licences (CC-BY-NC, CC-BY-NC-SA), published "
    "under the same non-commercial licence.",
}


class PublishError(RuntimeError):
    pass


# ---- reading a prepared root --------------------------------------------------------------


def readable(episode: Mapping[str, Any]) -> list[dict[str, str]]:
    """The transcript of a stored episode as ``{role, text}`` messages: the parts of one logical
    record joined, media shown as a placeholder naming its type and size."""
    messages: list[dict[str, str]] = []
    last = None
    for record in episode["records"]:
        payload = base64.b64decode(record["payload"])
        media = record.get("media_type") or ""
        text = (
            payload.decode("utf-8", "replace")
            if media.startswith("text/")
            else (f"[{media or 'binary'}: {len(payload)} bytes]")
        )
        key = (record["role"], record.get("logical_id") or record["record_id"])
        if messages and key == last:
            messages[-1]["text"] += text
        else:
            messages.append({"role": record["role"], "text": text})
            last = key
    return messages


def root_rows(root: Path) -> Iterator[dict[str, Any]]:
    """Every episode of a prepared root in order: the index columns, the stored line verbatim
    (checked against the index's hash) and the readable transcript."""
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise PublishError(f"{root} is not a finished prepared root")
    handle = None
    opened = ""
    with closing(sqlite3.connect(f"file:{root / 'index.sqlite'}?mode=ro", uri=True)) as index:
        for row in index.execute(
            "SELECT entry_id,episode_id,shard,offset,length,sha256,split,family_id,stream,"
            "source_id,tokens,scored_positions,outcome,verified_success FROM episodes "
            "ORDER BY entry_id"
        ):
            entry, episode_id, shard, offset, length, digest = row[:6]
            if shard != opened:
                if handle is not None:
                    handle.close()
                handle, opened = (root / shard).open("rb"), shard
            assert handle is not None
            handle.seek(offset)
            line = handle.read(length)
            if hashlib.sha256(line).hexdigest() != digest:
                raise PublishError(
                    f"{root.name}/{shard}: episode {episode_id} differs from its index"
                )
            text = line.decode("utf-8")
            yield {
                "entry_id": entry,
                "shard": shard,
                "episode_id": episode_id,
                "source_id": row[9],
                "stream": row[8],
                "split": row[6],
                "family_id": row[7],
                "tokens": row[10],
                "scored_positions": row[11],
                "outcome": row[12],
                "verified_success": bool(row[13]),
                "messages": readable(json.loads(text)),
                "tools": json.loads(text).get("tools_json", "[]"),
                "episode_json": text.rstrip("\n"),
            }
    if handle is not None:
        handle.close()


def schema() -> Any:
    import pyarrow as pa

    text = pa.large_string()
    return pa.schema(
        [
            ("entry_id", pa.int64()),
            ("shard", text),
            ("episode_id", text),
            ("source_id", text),
            ("stream", text),
            ("split", text),
            ("family_id", text),
            ("tokens", pa.int64()),
            ("scored_positions", pa.int64()),
            ("outcome", text),
            ("verified_success", pa.bool_()),
            ("messages", pa.list_(pa.struct([("role", text), ("text", text)]))),
            ("tools", text),
            ("episode_json", text),
        ]
    )


class _Files:
    """Writes rows as Parquet files of about ``target`` bytes each, in row-group-sized steps."""

    def __init__(self, directory: Path, target: int) -> None:
        self.directory, self.target = directory, target
        self.writer: Any = None
        self.path: Path | None = None
        self.count = 0
        self.written: list[Path] = []

    def add(self, rows: list[dict[str, Any]]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.writer is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory / f"data-{self.count:05d}.parquet"
            self.writer = pq.ParquetWriter(self.path, schema(), compression="zstd")
            self.count += 1
        self.writer.write_table(pa.Table.from_pylist(rows, schema=schema()))
        assert self.path is not None
        if self.path.stat().st_size >= self.target:
            self.close()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            assert self.path is not None
            self.written.append(self.path)
            self.writer = None


def convert_root(root: Path, directory: Path, target: int = PARQUET_BYTES) -> dict[str, Any]:
    """Write a prepared root as Parquet files plus its own small files (``root/…``) under
    ``directory``; returns the episodes and tokens per source."""
    if directory.exists():
        shutil.rmtree(directory)
    files = _Files(directory, target)
    pending: list[dict[str, Any]] = []
    size = 0
    counts: dict[str, dict[str, int]] = {}
    for row in root_rows(root):
        pending.append(row)
        size += len(row["episode_json"]) + sum(len(m["text"]) for m in row["messages"])
        seen = counts.setdefault(row["source_id"], {"episodes": 0, "tokens": 0})
        seen["episodes"] += 1
        seen["tokens"] += row["tokens"]
        if size >= ROW_GROUP_BYTES or len(pending) >= ROW_GROUP_ROWS:
            files.add(pending)
            pending, size = [], 0
    if pending:
        files.add(pending)
    files.close()
    shards = {item["path"] for item in json.loads((root / "manifest.json").read_text())["shards"]}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            path.is_file()
            and relative.as_posix() not in shards
            and path.name not in cache.SKIP_NAMES
        ):
            target_path = directory / ROOT_FOLDER / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target_path)
    return counts


def restore_root(files: Mapping[str, Path], dest: Path) -> int:
    """Rebuild a prepared root from the files of its published folder (``files`` maps each path
    relative to the folder to a local file). The shard files come from the Parquet rows and are
    each checked against the root's own manifest, the index against its recorded hash; the
    manifest is written last, as it is the root's commit marker. Returns the episode count."""
    import pyarrow.parquet as pq

    own = {
        name.removeprefix(f"{ROOT_FOLDER}/"): path
        for name, path in files.items()
        if name.startswith(f"{ROOT_FOLDER}/")
    }
    if "manifest.json" not in own:
        raise PublishError(f"{dest.name}: the published root has no manifest")
    manifest = json.loads(own["manifest.json"].read_text())
    dest.mkdir(parents=True, exist_ok=True)
    # A failed repeat must not leave an old completion marker over partially replaced data.
    (dest / "manifest.json").unlink(missing_ok=True)
    for name, path in own.items():
        if name != "manifest.json":
            target = dest / cache._safe_relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    if (
        sha256_file(dest / cache._safe_relative(manifest["index"]["path"]))
        != manifest["index"]["sha256"]
    ):
        raise PublishError(f"{dest.name}: the published index does not match the manifest")
    handles: dict[str, Any] = {}
    digests: dict[str, Any] = {}
    episodes = 0
    try:
        for name in sorted(n for n in files if n.endswith(".parquet")):
            for batch in pq.ParquetFile(files[name]).iter_batches(
                batch_size=1000, columns=["shard", "episode_json"]
            ):
                for shard, text in zip(
                    batch.column("shard").to_pylist(),
                    batch.column("episode_json").to_pylist(),
                    strict=True,
                ):
                    if shard not in handles:
                        target = dest / cache._safe_relative(shard)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        handles[shard] = target.with_name(target.name + ".partial").open("wb")
                        digests[shard] = hashlib.sha256()
                    line = text.encode("utf-8") + b"\n"
                    handles[shard].write(line)
                    digests[shard].update(line)
                    episodes += 1
    finally:
        for handle in handles.values():
            handle.close()
    expected = {item["path"]: item for item in manifest["shards"]}
    if set(handles) != set(expected):
        raise PublishError(f"{dest.name}: the published rows do not cover the root's shards")
    for shard, item in expected.items():
        target = dest / cache._safe_relative(shard)
        partial = target.with_name(target.name + ".partial")
        if digests[shard].hexdigest() != item["sha256"]:
            partial.unlink(missing_ok=True)
            raise PublishError(f"{dest.name}/{shard}: rebuilt shard does not match the manifest")
        partial.replace(target)
    shutil.copyfile(own["manifest.json"], dest / "manifest.json")
    return episodes


# ---- naming, attribution ------------------------------------------------------------------


def part_dir(group: str, run: str, part: int) -> str:
    return f"{group}/{run}/part-{part:04d}"


def upstream_url(uri: str) -> str:
    """Where a source lives on the Hub, for the attribution table; empty for a local file."""
    prefix = "hf://datasets/"
    return f"https://huggingface.co/datasets/{uri[len(prefix) :]}" if uri.startswith(prefix) else ""


def attribution(sources: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """What the card says about each source, from its manifest entry."""
    return {
        source["source_id"]: {
            "source_id": source["source_id"],
            "upstream": upstream_url(source["uri"]),
            "revision": source["revision"],
            "license": source["terms"].get("license", ""),
            "license_url": source["terms"].get("license_url", ""),
        }
        for source in sources
    }


def _listing(files: Sequence[Path], base: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(base).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]


# ---- storing a part -----------------------------------------------------------------------


def store_part(
    *,
    keys: cache.Keys,
    api: Any,
    repo: str,
    run: str,
    part: int,
    prepared: Path,
    work: Path,
    groups: Mapping[str, str],
    sources: Mapping[str, Mapping[str, str]],
    parquet_bytes: int = PARQUET_BYTES,
    piece_bytes: int = cache.PIECE_BYTES,
    segment: int = cache.SEGMENT_BYTES,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Publish the roots under ``prepared`` and return the part's receipt.

    ``groups`` maps each root's folder name to its licence group, ``sources`` each source id to
    its attribution (see ``attribution``). Roots of the public groups become Parquet in the
    part's own folders; the roots of group ``no`` are sealed and uploaded by ``cache.store``.
    """
    roots = sorted(p.name for p in prepared.iterdir() if (p / "manifest.json").is_file())
    unknown = [name for name in roots if name not in groups]
    if unknown:
        raise PublishError(f"No licence group is known for the roots {unknown}")
    if invalid := sorted({groups[name] for name in roots} - set(REDISTRIBUTION_GROUPS)):
        raise PublishError(f"Unknown licence groups: {invalid}")
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    staging = work / "public"
    shutil.rmtree(staging, ignore_errors=True)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for name in roots:
        group = groups[name]
        if group in PUBLIC_GROUPS:
            found = convert_root(
                prepared / name, staging / part_dir(group, run, part) / name, parquet_bytes
            )
            for source_id, seen in found.items():
                total = counts.setdefault(group, {}).setdefault(
                    source_id, {"episodes": 0, "tokens": 0}
                )
                total["episodes"] += seen["episodes"]
                total["tokens"] += seen["tokens"]
            log(f"converted {name} ({group})")
    for group, found in counts.items():
        if unlisted := sorted(set(found) - set(sources)):
            raise PublishError(f"The rows of {unlisted} have no source to attribute them to")
        listing = {
            "schema": SOURCES_SCHEMA,
            "run": run,
            "part": part,
            "group": group,
            "sources": [
                {**sources[source_id], **seen} for source_id, seen in sorted(found.items())
            ],
        }
        target = staging / part_dir(group, run, part) / SOURCES_FILE
        target.write_text(json.dumps(listing, indent=1))
    public = (
        _listing([p for p in staging.rglob("*") if p.is_file()], staging)
        if staging.exists()
        else []
    )
    if public:
        remote = cache.Remote(api)
        hub.retry_conflicts(
            lambda: api.upload_large_folder(
                repo_id=repo, folder_path=str(staging), repo_type="dataset"
            )
        )
        head = remote.head(repo)
        verify_files(remote, repo, head, public, work / "verify")
        shutil.rmtree(work / "verify", ignore_errors=True)
        log(f"uploaded and verified {len(public)} public files")
    encrypted = [name for name in roots if groups[name] == "no"]
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "run": run,
        "part": part,
        "repo": repo,
        "public": public,
        "encrypted": None,
        "episodes": {
            group: sum(seen["episodes"] for seen in found.values())
            for group, found in counts.items()
        },
        "roots": {name: groups[name] for name in roots},
    }
    if encrypted:
        sealed = cache.store(
            keys=keys,
            api=api,
            repo=repo,
            run=run,
            part=part,
            source=prepared,
            work=work / "cache",
            only=encrypted,
            piece_bytes=piece_bytes,
            segment=segment,
            log=log,
        )
        receipt["encrypted"] = sealed["folder"]
    return receipt


def verify_files(
    remote: cache.Remote,
    repo: str,
    revision: str,
    files: Sequence[Mapping[str, Any]],
    scratch: Path,
) -> None:
    """Every listed file is in ``repo`` at ``revision`` with its size and sha256 (the Hub records
    the hash of large files; a small file is downloaded and hashed). Raises ``PublishError``."""
    folders: dict[str, dict[str, Any]] = {}
    for item in files:
        top = "/".join(item["path"].split("/")[:3])  # <group>/<run>/part-N: one part's folder
        if top not in folders:
            try:
                folders[top] = remote.tree(repo, revision, top)
            except cache.CacheError as error:
                raise PublishError(f"{repo}: {top} is missing at {revision}") from error
        found = folders[top].get(item["path"])
        if found is None or found.size != item["bytes"]:
            raise PublishError(f"{repo}: {item['path']} is missing or the wrong size")
        if remote.remote_sha256(repo, item["path"], revision, found, scratch) != item["sha256"]:
            raise PublishError(f"{repo}: {item['path']} differs from what was uploaded")


# ---- the pin ------------------------------------------------------------------------------


def make_pin(
    run: str, repo: str, revision: str, receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The pin of a whole run from its parts' receipts: the repository, one commit that holds
    every part, and what each part put there. No secret."""
    parts = sorted(receipts, key=lambda receipt: receipt["part"])
    if [receipt["part"] for receipt in parts] != list(range(len(parts))):
        raise PublishError("The receipts do not cover parts 0..n-1 exactly once")
    if any(receipt["repo"] != repo for receipt in parts):
        raise PublishError("The receipts name different repositories")
    if any(receipt["run"] != run for receipt in parts):
        raise PublishError("The receipts belong to a different run")
    return {
        "schema": PIN_SCHEMA,
        "run": run,
        "repo": repo,
        "revision": revision,
        "parts": [
            {
                "part": receipt["part"],
                "public": receipt["public"],
                "encrypted": receipt["encrypted"],
            }
            for receipt in parts
        ],
    }


def load_pin(path: Path) -> dict[str, Any]:
    pin = json.loads(Path(path).read_text())
    if (
        pin.get("schema") != PIN_SCHEMA
        or not pin.get("run")
        or not pin.get("repo")
        or not pin.get("revision")
        or not pin.get("parts")
    ):
        raise PublishError(f"{path} is not a data pin")
    return pin


def verify_pin(api: Any, keys: cache.Keys | None, pin: Mapping[str, Any], scratch: Path) -> None:
    """Check the pinned commit without downloading the data: every public file is there with
    its recorded size and sha256, and every encrypted part's manifest decrypts and every piece
    is stored at its recorded size and hash. Raises ``PublishError`` or ``CacheError``."""
    remote = cache.Remote(api)
    try:
        for part in pin["parts"]:
            verify_files(remote, pin["repo"], pin["revision"], part["public"], scratch)
            if part["encrypted"]:
                if keys is None:
                    raise PublishError("The pin has encrypted parts and no key was given")
                cache.verify_part(
                    api,
                    keys,
                    repo=pin["repo"],
                    revision=pin["revision"],
                    run=pin["run"],
                    part=part["part"],
                    folder=part["encrypted"],
                    scratch=scratch,
                )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@dataclass(frozen=True)
class Decision:
    mode: str  # "stored": fetch the pin; "rebuild": prepare the data again
    reason: str


def decide(pin_path: Path | None, keys: cache.Keys, api: Any, scratch: Path) -> Decision:
    """Fetch if the pin names a commit that still verifies, otherwise rebuild.

    A missing pin, repo or commit, a file or piece that is gone or altered, or a key that no
    longer opens the manifests all mean rebuild. A network failure is not an answer and
    propagates.
    """
    if pin_path is None or not Path(pin_path).is_file():
        return Decision("rebuild", "no pin has been recorded for this run")
    try:
        verify_pin(api, keys, load_pin(Path(pin_path)), scratch)
    except (PublishError, cache.CacheError, KeyError, ValueError) as error:
        return Decision("rebuild", f"the stored copy does not verify: {error}")
    return Decision("stored", "the pinned commit verifies: every public file and encrypted piece")


# ---- the card -----------------------------------------------------------------------------


def _count(value: int) -> str:
    return f"{value:,}"


def _size(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError


def subset_licence(group: str, entries: Sequence[Mapping[str, Any]]) -> str:
    if group == "non_commercial" and any("nc-sa" in e["license"].lower() for e in entries):
        return "CC-BY-NC-SA-4.0"
    return SUBSET_LICENCES[group]


def build_card(
    *,
    repo: str,
    subsets: Mapping[str, Sequence[Mapping[str, Any]]],
    encrypted: Mapping[str, int],
    contact: str = DEFAULT_CONTACT,
) -> str:
    """The dataset card: the encrypted notice first, then what each subset is, its licence, and
    the per-source attribution grouped by subset. ``subsets`` maps a public group to its sources
    (``source_id``, ``upstream``, ``revision``, ``license``, ``license_url``, ``episodes``,
    ``tokens``), ``encrypted`` is ``{parts, bytes}``."""
    present = [group for group in PUBLIC_GROUPS if subsets.get(group)]
    head = [
        "---",
        "license: other",
        "license_name: per-subset-see-card",
        "pretty_name: Megatoken training data",
    ]
    head += ["task_categories:", "- text-generation"]
    if present:
        head.append("configs:")
    for group in present:
        head += [
            f"- config_name: {group}",
            "  data_files:",
            "  - split: train",
            f"    path: {group}/**/*.parquet",
        ]
    head.append("---")
    lines = head + [
        "",
        "# Megatoken training data",
        "",
        "> **The `encrypted/` folder of this repository is an ENCRYPTED cache of preprocessed "
        "machine-learning training data for the megatoken project. Its contents cannot be used "
        "without the project's private key, which is not published. It holds no usable data for "
        "anyone else: please do not try to read it.** The other folders are ordinary, readable "
        f"Parquet datasets. Contact: {contact}.",
        "",
        "Prepared conversations, tool-use and agent trajectories, reasoning traces and code "
        "sessions used to train megatoken models. Each row is one *episode*: a transcript made "
        "of role-tagged records, with the tool definitions it used. Episodes were parsed from the "
        "public sources listed below and checked. Within each preparation run, episodes are "
        "deduplicated across licence groups (the copy from the most permissive source is kept) "
        "and split into train, validation and test by family, so related episodes stay on one "
        "side. Different runs may contain overlapping episodes and families; consumers combining "
        "runs should deduplicate and establish their splits again.",
        "",
        "## Subsets and licences",
        "",
        "The repository's top-level licence is `other` because the subsets differ. "
        "Each subset has one licence group; rows of different groups are never mixed in a file.",
        "",
        "| Subset | What it holds | Licence of the subset | Sources | Episodes |",
        "|---|---|---|---:|---:|",
    ]
    for group in present:
        entries = subsets[group]
        lines.append(
            f"| `{group}` | {SUBSET_BLURBS[group]} | {subset_licence(group, entries)} | "
            f"{_count(len(entries))} | {_count(sum(e['episodes'] for e in entries))} |"
        )
    lines.append(
        f"| `encrypted` | Everything whose licence does not clearly allow redistribution. "
        f"Ciphertext only, unusable without the private key. | not published | not listed | "
        f"{_count(encrypted.get('parts', 0))} encrypted parts, {_size(encrypted.get('bytes', 0))} |"
    )
    lines += [
        "",
        "Licences are the ones the publishers state on each source's Hub page at the revision "
        "listed in the tables; we have not verified them independently. Following a source's "
        "licence (attribution, share-alike, non-commercial use) is your responsibility when you "
        "use its rows. The `permissive` subset adds no terms of its own; the `share_alike` "
        "subset must be shared under CC-BY-SA-4.0; the `non_commercial` subset is for "
        "non-commercial use only.",
        "",
        "## Loading",
        "",
        "```python",
        "from datasets import load_dataset",
        f'rows = load_dataset("{repo}", "permissive", split="train", streaming=True)',
        "```",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `source_id`, `stream`, `split`, `family_id` | the source, the kind of data, the train, "
        "validation or test split, and the family that keeps related episodes together |",
        "| `messages` | readable transcript: a list of `{role, text}` |",
        "| `tools` | the tool definitions of the episode (JSON text) |",
        "| `tokens`, `scored_positions`, `outcome`, `verified_success` | measured when the episode was prepared |",
        "| `episode_json` | the exact stored episode (payloads in base64) from which megatoken rebuilds its training files |",
        "",
        "## Attribution",
    ]
    for group in present:
        entries = sorted(subsets[group], key=lambda e: e["source_id"])
        lines += [
            "",
            f"### `{group}`",
            "",
            "| Source | Upstream | Revision | Licence | Licence URL | Episodes |",
            "|---|---|---|---|---|---:|",
        ]
        for entry in entries:
            upstream = (
                f"[{entry['upstream'].rsplit('datasets/', 1)[-1]}]({entry['upstream']})"
                if entry["upstream"]
                else "local file"
            )
            licence_url = f"[link]({entry['license_url']})" if entry["license_url"] else ""
            licence = entry["license"].replace("|", "/")
            lines.append(
                f"| {entry['source_id']} | {upstream} | `{entry['revision'][:12]}` | {licence} | "
                f"{licence_url} | {_count(entry['episodes'])} |"
            )
    lines += [
        "",
        "## Contact and takedown",
        "",
        f"If you hold rights to material in this repository and want it removed, or if a licence "
        f"above is wrong: {contact}. Rows are removed on request.",
        "",
    ]
    return "\n".join(lines)


def encrypted_card(contact: str) -> str:
    return (
        "# ENCRYPTED cache: nothing here is usable without a private key\n\n"
        "This folder is an ENCRYPTED cache of preprocessed machine-learning training data for the "
        "megatoken project. Every file is ciphertext (AES-256-GCM) under a private key that is not "
        "published, with opaque names, so the contents cannot be read or used without that key. "
        "It is stored here only so the project can fetch it back; it is not a dataset for anyone "
        f"else. Contact: {contact}.\n"
    )


def repo_state(
    api: Any, repo: str, revision: str, scratch: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """What the card describes, read from the repository at ``revision``: each public group's
    sources (summed over every part of every run) and the encrypted parts' count and size."""
    remote = cache.Remote(api)
    tree = remote.tree(repo, revision)
    listings = sorted(
        path
        for path in tree
        if path.rsplit("/", 1)[-1] == SOURCES_FILE and path.split("/")[0] in PUBLIC_GROUPS
    )

    def read(path: str) -> dict[str, Any]:
        local = remote.download(repo, path, revision, scratch)
        try:
            return json.loads(local.read_text())
        finally:
            local.unlink(missing_ok=True)

    with ThreadPoolExecutor(8) as pool:
        parts = list(pool.map(read, listings))
    summed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for part in parts:
        for entry in part["sources"]:
            seen = summed.setdefault(part["group"], {}).setdefault(
                (entry["source_id"], entry["revision"]), {**entry, "episodes": 0, "tokens": 0}
            )
            seen["episodes"] += entry["episodes"]
            seen["tokens"] += entry["tokens"]
    folders = {
        path.split("/")[1]
        for path in tree
        if path.startswith(f"{cache.ENCRYPTED_DIR}/") and path.count("/") == 2
    }
    size = sum(
        item.size
        for path, item in tree.items()
        if path.startswith(f"{cache.ENCRYPTED_DIR}/")
        and path.count("/") == 2
        and getattr(item, "size", None)
    )
    return (
        {group: list(found.values()) for group, found in summed.items()},
        {"parts": len(folders), "bytes": size},
    )


def refresh_card(
    api: Any,
    repo: str,
    *,
    contact: str = DEFAULT_CONTACT,
    scratch: Path,
    log: Callable[[str], None] = print,
) -> str:
    """Rebuild the card from what ``repo`` holds and commit it (and the encrypted folder's
    notice) if they changed; returns the commit that holds everything, the new one or the head."""
    return hub.retry_conflicts(
        lambda: _refresh_card(api, repo, contact=contact, scratch=scratch, log=log)
    )


def _refresh_card(
    api: Any,
    repo: str,
    *,
    contact: str,
    scratch: Path,
    log: Callable[[str], None],
) -> str:
    """One optimistic update; if another run uploads, rebuild from the new head on retry."""
    from huggingface_hub import CommitOperationAdd

    remote = cache.Remote(api)
    head = remote.head(repo)
    try:
        subsets, encrypted = repo_state(api, repo, head, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    wanted = {
        CARD: build_card(repo=repo, subsets=subsets, encrypted=encrypted, contact=contact),
        ENCRYPTED_CARD: encrypted_card(contact),
    }
    tree = remote.tree(repo, head)
    changed = []
    for path, text in wanted.items():
        found = tree.get(path)
        current = None
        if found is not None:
            local = remote.download(repo, path, head, scratch)
            current = local.read_text()
            local.unlink(missing_ok=True)
        if current != text:
            changed.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=text.encode()))
    shutil.rmtree(scratch, ignore_errors=True)
    if not changed:
        return head
    info = api.create_commit(
        repo_id=repo,
        operations=changed,
        commit_message="Rebuild the dataset card",
        repo_type="dataset",
        parent_commit=head,
    )
    log(f"dataset card rebuilt in {info.oid}")
    return str(info.oid)


def finalize(
    *,
    api: Any,
    keys: cache.Keys | None,
    repo: str,
    run: str,
    receipts: Sequence[Mapping[str, Any]],
    contact: str = DEFAULT_CONTACT,
    scratch: Path,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The last step of a run, once every part has landed: rebuild the card, pin the commit
    that holds everything and check that the pin verifies. Returns the pin."""
    revision = refresh_card(api, repo, contact=contact, scratch=scratch / "card", log=log)
    pin = make_pin(run, repo, revision, receipts)
    verify_pin(api, keys, pin, scratch / "verify")
    return pin


# ---- reading back -------------------------------------------------------------------------


def _roots_of(part: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """A part's public files grouped by root: ``{root: {name inside its folder: file entry}}``."""
    roots: dict[str, dict[str, dict[str, Any]]] = {}
    for item in part["public"]:
        pieces = item["path"].split("/")
        if len(pieces) < 5:  # <group>/<run>/part-N/sources.json
            continue
        roots.setdefault(pieces[3], {})["/".join(pieces[4:])] = item
    return roots


def fetch(
    *,
    api: Any,
    keys: cache.Keys | None,
    pin: Mapping[str, Any],
    dest: Path,
    parts: Sequence[int] | None = None,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Restore the pinned run into ``dest`` as prepared roots, one folder per shard: public
    files are downloaded from the pinned commit and checked against the pin before use, the
    rest is fetched, verified and decrypted by ``cache.fetch_part``. A rerun skips encrypted
    files already there. Raises ``PublishError`` or ``cache.CacheError`` on any mismatch."""
    remote = cache.Remote(api)
    dest.mkdir(parents=True, exist_ok=True)
    scratch = dest.with_name(dest.name + ".fetching")
    chosen = [part for part in pin["parts"] if parts is None or part["part"] in parts]
    if not chosen:
        raise PublishError("The pin has none of the requested parts")
    roots = episodes = 0
    try:
        for part in chosen:
            for root, files in sorted(_roots_of(part).items()):
                local: dict[str, Path] = {}

                def download(name: str, files: Mapping[str, Mapping[str, Any]] = files) -> Path:
                    item = files[name]
                    path = remote.download(pin["repo"], item["path"], pin["revision"], scratch)
                    if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                        raise PublishError(f"{item['path']}: the download does not match the pin")
                    return path

                with ThreadPoolExecutor(max(1, workers)) as pool:
                    for name, path in zip(files, pool.map(download, files), strict=True):
                        local[name] = path
                episodes += restore_root(local, dest / root)
                for path in local.values():
                    path.unlink(missing_ok=True)
                roots += 1
                log(f"part {part['part']}: restored {root}")
            if part["encrypted"]:
                if keys is None:
                    raise PublishError("The pin has encrypted parts and no key was given")
                cache.fetch_part(
                    keys=keys,
                    api=api,
                    repo=pin["repo"],
                    revision=pin["revision"],
                    run=pin["run"],
                    part=part["part"],
                    folder=part["encrypted"],
                    dest=dest,
                    scratch=scratch,
                    workers=workers,
                    log=log,
                )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "run": pin["run"],
        "parts": [p["part"] for p in chosen],
        "public_roots": roots,
        "public_episodes": episodes,
    }


# ---- command line -------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(prog="python -m megatoken.publish", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("fetch", help="restore a pinned run as prepared roots")
    item.add_argument("--pin", type=Path, required=True)
    item.add_argument("--dest", type=Path, required=True)
    item.add_argument("--part", type=int, action="append", help="only this part (repeatable)")
    item.add_argument("--workers", type=int, default=4)
    item = commands.add_parser("verify", help="check the pinned commit without downloading data")
    item.add_argument("--pin", type=Path, required=True)
    item = commands.add_parser("card", help="rebuild the dataset card from what the repo holds")
    item.add_argument("--repo", required=True)
    item.add_argument("--contact", default=DEFAULT_CONTACT)
    args = parser.parse_args(argv)
    try:
        api = hub.api()
        scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "megatoken-publish"
        if args.command == "card":
            result: Any = {
                "revision": refresh_card(api, args.repo, contact=args.contact, scratch=scratch)
            }
        else:
            pin = load_pin(args.pin)
            keys = cache.load_keys() if any(p["encrypted"] for p in pin["parts"]) else None
            if args.command == "fetch":
                result = fetch(
                    api=api,
                    keys=keys,
                    pin=pin,
                    dest=args.dest,
                    parts=args.part,
                    workers=args.workers,
                )
            else:
                verify_pin(api, keys, pin, scratch)
                result = {"verified_parts": [p["part"] for p in pin["parts"]]}
    except (PublishError, cache.CacheError) as error:
        print(f"publish: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
