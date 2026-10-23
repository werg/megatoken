"""Pinned, streaming source readers. Importing this module performs no network I/O."""

from __future__ import annotations

import codecs
import gzip
import itertools
import json
import random
import re
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from megatoken.io import content_hash, sha256_file, write_json
from megatoken.redistribution import REDISTRIBUTION_GROUPS

PARQUET_BATCH_BYTES = 64 << 20


@dataclass(frozen=True, slots=True)
class TermsReview:
    approved: bool = False
    license: str = "unreviewed"
    license_url: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""
    notes: str = ""
    # ``permissive``, ``share_alike`` or ``non_commercial``: the prepared rows may be published
    # publicly under that group's licence; ``no`` (and anything unrecorded): only as an
    # encrypted copy. ``redistribute_basis`` says why, in a sentence.
    redistribute: str = ""
    redistribute_basis: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("Terms approval must be an explicit JSON boolean")
        if self.redistribute and self.redistribute not in REDISTRIBUTION_GROUPS:
            raise ValueError(f"redistribute is one of {REDISTRIBUTION_GROUPS}")

    @property
    def group(self) -> str:
        """The licence group the source is published in; unrecorded means ``no``."""
        return self.redistribute or "no"

    def require_approval(self) -> None:
        if not self.approved or not self.reviewed_by or not self.reviewed_at:
            raise ValueError(
                "Source terms are unapproved; record approved/reviewed_by/reviewed_at in the source manifest"
            )


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    uri: str
    format: str
    adapter: str
    stream: str
    revision: str
    terms: TermsReview = TermsReview()
    files: tuple[SourceFile, ...] = ()
    subset: str | None = None
    split: str = "train"
    max_rows: int | None = None
    field_map: Mapping[str, str | list[str]] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    # ``(index, count)``: this shard reads every ``count``-th row, starting at ``index``, of
    # the rows its files hold. Row numbers stay those of the whole file list.
    stripe: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not all((self.source_id, self.uri, self.adapter, self.stream, self.revision)):
            raise ValueError(
                "Sources require identity, URI, adapter, stream, and a pinned revision"
            )
        if self.format not in {"jsonl", "parquet", "hf"}:
            raise ValueError("Source format must be jsonl, parquet, or hf")
        if self.format == "hf" and not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(
                "Hugging Face sources require an immutable 40-character commit, not main/a tag"
            )
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive, or null for an explicitly unbounded run")
        if self.stripe is not None:
            index, count = self.stripe
            if not (type(index) is int and type(count) is int and 0 <= index < count):
                raise ValueError("stripe must be (index, count) with 0 <= index < count")
            object.__setattr__(self, "stripe", (index, count))

    @property
    def fingerprint(self) -> str:
        """What is read, not who approved it or where the files sit on this machine."""
        value = asdict(self)
        if value["stripe"] is None:  # Sources written before striping keep their identity.
            del value["stripe"]
        value["terms"] = {"license": self.terms.license, "license_url": self.terms.license_url}
        value["files"] = [{**item, "path": Path(item["path"]).name} for item in value["files"]]
        if self.format != "hf":
            value["uri"] = Path(self.uri).name
        if value["options"].get("lineage_join"):
            join = value["options"]["lineage_join"]
            value["options"] = {
                **value["options"],
                "lineage_join": {**join, "path": Path(join["path"]).name},
            }
        return content_hash(value)

    @property
    def repository_id(self) -> str:
        return (
            self.uri.removeprefix("hf://datasets/")
            .removeprefix("hf://")
            .removeprefix("https://huggingface.co/datasets/")
        )

    def validate_local_files(self) -> None:
        self.terms.require_approval()
        if self.format == "hf":
            return
        if not self.files:
            raise ValueError("Local sources require an explicit checksummed files list")
        for source_file in self.files:
            path = Path(source_file.path)
            if not source_file.sha256:
                raise ValueError(f"Missing source checksum: {path}")
            if source_file.size_bytes is not None and path.stat().st_size != source_file.size_bytes:
                raise ValueError(f"Source size changed: {path}")
            if sha256_file(path) != source_file.sha256:
                raise ValueError(f"Source checksum changed: {path}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, base: Path | None = None) -> SourceSpec:
        obj = dict(value)
        # Human-readable acquisition notes do not silently enter reader options.
        for key in ("notes", "readiness", "publisher_card", "metadata_checked_at"):
            obj.pop(key, None)
        obj["terms"] = TermsReview(**obj.get("terms", {}))
        files = []
        for entry in obj.get("files", ()):
            item = SourceFile(**entry)
            if base is not None and obj.get("format") != "hf" and not Path(item.path).is_absolute():
                item = replace(item, path=str((base / item.path).resolve()))
            files.append(item)
        obj["files"] = tuple(files)
        options = dict(obj.get("options", {}))
        if base is not None and options.get("lineage_join"):
            join = dict(options["lineage_join"])
            location = Path(join["path"])
            if not location.is_absolute():
                join["path"] = str((base / location).resolve())
            options["lineage_join"] = join
        obj["options"] = options
        if obj.get("stripe") is not None:
            obj["stripe"] = tuple(obj["stripe"])
        return cls(**obj)


def parquet_group_rows(reader: Any, group: int, max_batch_rows: int = 128) -> Iterator[dict]:
    """One row group's rows, decoded in batches of about ``PARQUET_BATCH_BYTES``.

    Arrow cannot hand back a nested column over 2 GiB in one piece, and long trajectories
    reach that within a few rows, so a batch holds as many rows as fit in the budget.
    """
    meta = reader.metadata.row_group(group)
    row_bytes = meta.total_byte_size / max(meta.num_rows, 1)
    batch_size = max(1, min(max_batch_rows, int(PARQUET_BATCH_BYTES / max(row_bytes, 1))))
    for batch in reader.iter_batches(batch_size=batch_size, row_groups=[group]):
        yield from batch.to_pylist()


def load_source_manifest(path: str | Path) -> tuple[SourceSpec, ...]:
    location = Path(path)
    obj = json.loads(location.read_text())
    if obj.get("schema_version") != 1:
        raise ValueError("Unsupported source manifest schema")
    sources = tuple(
        SourceSpec.from_dict(value, base=location.resolve().parent) for value in obj["sources"]
    )
    if len({source.source_id for source in sources}) != len(sources):
        raise ValueError("Source IDs must be unique within one manifest")
    return sources


def iter_source_rows(
    source: SourceSpec, *, start_row: int = 0
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream deterministic logical rows, only this stripe's if the source has one.

    A stripe is a shard's share of a source too big to split by file: every shard reads
    the whole file list and keeps its own rows, which keep their global numbers.
    """
    rows = _read_rows(source, start_row=start_row)
    if source.stripe is None:
        return rows
    index, count = source.stripe
    return ((number, row) for number, row in rows if number % count == index)


def _hub_rows(source: SourceSpec) -> Iterator[Mapping[str, Any]]:
    """The rows of a Hub source; with ``explode``, one per element of a list column."""
    rows = _hub_file_rows(source)
    if source.options.get("explode"):
        rows = _exploded(rows, source.options["explode"])
    yield from rows


def _exploded(rows: Iterator[Mapping[str, Any]], field: str) -> Iterator[Mapping[str, Any]]:
    """One row per element of a list-valued column (several runs stored in one task row).

    The element's own fields are added to the parent's other columns, an element's overriding
    a parent's of the same name, and ``explode_index`` says which element it was.
    """
    for row in rows:
        items = row.get(field)
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except json.JSONDecodeError:
                continue
        if not isinstance(items, list):
            continue
        parent = {key: value for key, value in row.items() if key != field}
        for index, item in enumerate(items):
            yield {
                **parent,
                **(item if isinstance(item, Mapping) else {"value": item}),
                "explode_index": index,
            }


_FILE_BUILDERS = {
    ".parquet": "parquet",
    ".jsonl": "json",
    ".jsonl.gz": "json",
    ".jsonl.zst": "json",
    ".json": "json",
}


def _file_suffix(path: str) -> str:
    name = path.rsplit("/", 1)[-1].lower()
    for suffix in (".jsonl.gz", ".jsonl.zst"):
        if name.endswith(suffix):
            return suffix
    return name[name.rfind(".") :] if "." in name else ""


def _hub_file_rows(source: SourceSpec) -> Iterator[Mapping[str, Any]]:
    """The rows of a Hub source, one listed file at a time.

    ``datasets`` reads a list of Parquet files as one table type fixed by the first file and
    fails on a later file whose messages carry another field (a ``type`` beside ``role`` and
    ``content``). Each file is its own dataset here, with its own columns.
    """
    if source.options.get("whole_files"):
        yield from _whole_file_rows(source)
        return
    kwargs: dict[str, Any] = {
        "path": source.repository_id,
        "name": source.subset,
        "split": source.split,
        "revision": source.revision,
        "streaming": True,
    }
    # Parquet files are batched by bytes; this further caps the rows in one batch.
    max_batch_rows = int(source.options.get("read_batch_rows", 128))
    for item in source.files or (None,):
        if item is not None:
            builder = _FILE_BUILDERS.get(_file_suffix(item.path))
            if builder is not None:
                # A listed file is read by its format's own builder: the repository's card may
                # declare columns that this file's rows do not have, and ``datasets`` would
                # then refuse to cast them.
                address = item.path
                if not address.startswith("hf://"):
                    address = f"hf://datasets/{source.repository_id}@{source.revision}/{address}"
                if builder == "parquet":
                    yield from _retry_hub_file(
                        address,
                        lambda address=address: _hub_parquet_rows(address, max_batch_rows),
                    )
                    continue
                if _file_suffix(item.path).startswith(".jsonl"):
                    yield from _retry_hub_file(
                        address, lambda address=address: _hub_jsonl_rows(address)
                    )
                    continue
                if _file_suffix(item.path) == ".json":
                    yield from _retry_hub_file(
                        address, lambda address=address: _hub_json_file_rows(address)
                    )
                    continue
                yield from _load_dataset()(
                    builder,
                    data_files={source.split: [address]},
                    split=source.split,
                    streaming=True,
                )
                continue
            kwargs["data_files"] = {source.split: [item.path]}
        yield from _load_dataset()(**kwargs)


def _retry_hub_file(address: str, reader: Any) -> Iterator[dict[str, Any]]:
    """Retry transient Hub errors while retaining exact row position.

    ``prepare_sources`` commits its cursor after each ingested row. This local
    retry also recovers when a ranged file read is rate limited after yielding
    rows: it reopens the pinned file and skips only rows already emitted.
    """
    emitted = 0
    deadline: float | None = None
    last_error: Exception | None = None
    attempt = 0
    while deadline is None or time.monotonic() < deadline:
        attempt += 1
        row_index = 0
        try:
            for row in reader():
                if row_index >= emitted:
                    emitted += 1
                    yield row
                row_index += 1
            return
        except Exception as error:
            delay = _hub_retry_delay(error, attempt)
            if delay is None:
                raise
            if deadline is None:
                deadline = time.monotonic() + 60 * 60
            remaining = deadline - time.monotonic()
            if delay >= remaining:
                raise
            last_error = error
            print(
                f"Hub file read was rate limited or unavailable; retrying in {delay:.0f}s "
                f"(attempt {attempt}, {emitted} rows already read)",
                flush=True,
            )
            time.sleep(delay)
    if last_error is not None:
        raise last_error


def _hub_retry_delay(error: Exception, attempt: int) -> float | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    name = type(error).__name__.lower()
    if status not in {429, 500, 502, 503, 504} and not any(
        token in name
        for token in (
            "timeout",
            "connectionerror",
            "connecterror",
            "readerror",
            "networkerror",
            "remoteprotocol",
        )
    ):
        return None
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") if headers is not None else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                return max(
                    0.0, (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return min(300.0, 5.0 * 2 ** min(attempt - 1, 6)) + random.uniform(0, 1.5)


def _load_dataset() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face streaming requires the 'data' extra (datasets)") from exc
    return load_dataset


def _hub_jsonl_rows(address: str) -> Iterator[dict[str, Any]]:
    """Read independent JSON objects without imposing one nested schema on later rows.

    Execution reports and tool messages legitimately add fields or change a field's type
    between episodes, including within one file. Arrow schema inference cannot preserve them.
    """
    from contextlib import ExitStack

    from huggingface_hub import HfFileSystem

    with ExitStack() as stack:
        handle = stack.enter_context(HfFileSystem().open(address, "rb", block_size=8 << 20))
        if address.endswith(".gz"):
            handle = stack.enter_context(gzip.GzipFile(fileobj=handle))
        elif address.endswith(".zst"):
            import io

            import zstandard

            reader = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(handle))
            handle = stack.enter_context(io.BufferedReader(reader))
        for number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at line {number + 1} in {address}")
            yield row


def _hub_json_file_rows(address: str) -> Iterator[dict[str, Any]]:
    """Stream a Hub ``.json`` file as JSONL or a top-level array of objects.

    Some published data files use a ``.json`` suffix for JSON Lines. The Hub
    JSON builder infers one nested Arrow schema for all rows and fails when a
    later row adds, removes or changes a nested field. Decode independent rows
    directly, while retaining support for conventional JSON arrays.
    """
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(address, "rb", block_size=8 << 20) as handle:
        prefix = handle.read(64 << 10)
        first = prefix.lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
        chunks = itertools.chain((prefix,), iter(lambda: handle.read(1 << 20), b""))
        chunks = _strip_json_bom(chunks)
        if first == b"[":
            yield from _json_array_objects(chunks, address)
        else:
            yield from _json_object_stream(chunks, address)


def _strip_json_bom(chunks: Iterator[bytes]) -> Iterator[bytes]:
    first = next(chunks, b"")
    yield first.removeprefix(codecs.BOM_UTF8)
    yield from chunks


def _json_object_stream(chunks: Iterator[bytes], address: str) -> Iterator[dict[str, Any]]:
    """Read whitespace-separated JSON objects, including a pretty-printed single object."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    json_decoder = json.JSONDecoder()
    text = ""
    position = 0
    ended = False
    row_number = 0
    chunk_iter = iter(chunks)

    def refill() -> bool:
        nonlocal text, ended
        try:
            block = next(chunk_iter)
        except StopIteration:
            text += decoder.decode(b"", final=True)
            ended = True
            return False
        text += decoder.decode(block)
        return True

    while not ended or position < len(text):
        if position > 1 << 20:
            text = text[position:]
            position = 0
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            if not refill():
                return
            continue
        try:
            row, end = json_decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            if ended:
                raise ValueError(f"Invalid JSON object at row {row_number} in {address}") from error
            refill()
            continue
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at row {row_number} in {address}")
        position = end
        if position == len(text) and not ended:
            refill()
        if position < len(text) and not text[position].isspace():
            raise ValueError(f"Expected whitespace between JSON objects in {address}")
        row_number += 1
        yield row


def _json_array_objects(chunks: Iterator[bytes], address: str) -> Iterator[dict[str, Any]]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    json_decoder = json.JSONDecoder()
    text = ""
    position = 0
    started = False
    ended = False
    expect_value = True
    after_comma = False
    row_number = 0
    chunk_iter = iter(chunks)

    def refill() -> bool:
        nonlocal text, ended
        try:
            block = next(chunk_iter)
        except StopIteration:
            text += decoder.decode(b"", final=True)
            ended = True
            return False
        text += decoder.decode(block)
        return True

    while not ended or position < len(text):
        if position > 1 << 20:
            text = text[position:]
            position = 0
        while position < len(text) and text[position].isspace():
            position += 1
        if not started:
            if position == len(text):
                if not refill():
                    raise ValueError(f"Expected a JSON array in {address}")
                continue
            if text[position] != "[":
                raise ValueError(f"Expected a JSON array in {address}")
            position += 1
            started = True
            continue
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            if not refill():
                raise ValueError(f"Unterminated JSON array in {address}")
            continue
        if not expect_value:
            if text[position] == "]":
                position += 1
                while True:
                    if position < len(text) and text[position:].strip():
                        raise ValueError(f"Trailing content after JSON array in {address}")
                    if not refill():
                        return
            if text[position] != ",":
                raise ValueError(f"Expected a comma between JSON array rows in {address}")
            position += 1
            expect_value = True
            after_comma = True
            continue
        if text[position] == "]":
            if after_comma:
                raise ValueError(f"Trailing comma in JSON array in {address}")
            position += 1
            while True:
                if position < len(text) and text[position:].strip():
                    raise ValueError(f"Trailing content after JSON array in {address}")
                if not refill():
                    return
        try:
            row, end = json_decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            if ended:
                raise ValueError(f"Invalid JSON array row in {address}") from error
            refill()
            continue
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON objects in array in {address}")
        yield row
        row_number += 1
        position = end
        expect_value = False
        after_comma = False


def _hub_parquet_rows(address: str, max_batch_rows: int) -> Iterator[dict]:
    """A Hub Parquet file's rows by HTTP range reads, in byte-budgeted batches.

    ``datasets`` decodes a whole row group's batch at once and fails on row groups whose
    nested columns pass 2 GiB (long terminal sessions); this reads them in small batches.
    """
    import pyarrow.parquet as parquet
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(address, "rb", block_size=32 << 20) as handle:
        # Stream each column chunk in pages instead of reading it whole (a chunk can be GBs).
        reader = parquet.ParquetFile(handle, buffer_size=8 << 20, pre_buffer=False)
        for group in range(reader.num_row_groups):
            yield from parquet_group_rows(reader, group, max_batch_rows)


def _whole_file_rows(source: SourceSpec) -> Iterator[Mapping[str, Any]]:
    """One row per listed file, for repositories that keep one session or document per file.

    ``whole_files`` is ``"jsonl"`` (a row ``{"source_file", "events": [one object per line]}``)
    or ``"json"`` (``{"source_file", "document": the parsed file}``). A file name may contain
    ``*`` wildcards, which are expanded against the pinned revision in sorted order. A line
    that is not a JSON object is skipped; an unreadable file is an error, not a missing row.
    """
    import gzip

    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise RuntimeError("Hub file reading requires huggingface_hub") from exc
    mode = source.options["whole_files"]
    if mode not in {"jsonl", "json"}:
        raise ValueError("whole_files must be 'jsonl' or 'json'")
    filesystem = HfFileSystem()
    prefix = f"hf://datasets/{source.repository_id}@{source.revision}/"
    for item in source.files:
        location = item.path.removeprefix("hf://")
        names = sorted(filesystem.glob(location)) if "*" in location else [location]
        for name in names:
            with filesystem.open(name, "rb") as raw:
                data = raw.read()
            if name.endswith(".gz"):
                data = gzip.decompress(data)
            relative = name.removeprefix(prefix.removeprefix("hf://"))
            if mode == "json":
                yield {"source_file": relative, "document": json.loads(data)}
                continue
            events = []
            for line in data.splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
            yield {"source_file": relative, "events": events}


def _read_rows(source: SourceSpec, *, start_row: int) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream deterministic logical rows; Parquet skips completed row groups.

    A Hugging Face streaming resume counts logical rows sequentially. This avoids
    format-dependent Arrow ``skip`` behavior but reads preceding rows again; it
    never stores a whole dataset in Python memory. Local JSONL/GZip resumes scan
    old lines without parsing them.
    """
    source.terms.require_approval()
    if start_row < 0:
        raise ValueError("start_row cannot be negative")
    if source.max_rows is not None and start_row >= source.max_rows:
        return
    if source.format == "hf":
        # datasets 4.8.5 Arrow skip can repeat fully skipped Parquet row groups.
        # Count actual emitted rows so persisted logical cursors stay exact.
        for row_index, row in enumerate(
            itertools.islice(_hub_rows(source), start_row, source.max_rows), start=start_row
        ):
            yield row_index, dict(row)
        return
    row_index = 0
    for source_file in source.files:
        path = Path(source_file.path)
        if source.format == "jsonl":
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    if source.max_rows is not None and row_index >= source.max_rows:
                        return
                    current = row_index
                    row_index += 1
                    if current < start_row:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"Expected a JSON object at source row {current} in {path}"
                        )
                    yield current, obj
        else:
            try:
                import pyarrow.parquet as parquet
            except ImportError as exc:
                raise RuntimeError("Parquet streaming requires the 'data' extra (pyarrow)") from exc
            reader = parquet.ParquetFile(path)
            for group in range(reader.num_row_groups):
                meta = reader.metadata.row_group(group)
                count = meta.num_rows
                if row_index + count <= start_row:
                    row_index += count
                    continue
                for row in parquet_group_rows(reader, group):
                    if source.max_rows is not None and row_index >= source.max_rows:
                        return
                    current = row_index
                    row_index += 1
                    if current >= start_row:
                        yield current, row


def pin_local_source(
    path: str | Path,
    *,
    source_id: str,
    adapter: str,
    stream: str,
    terms: TermsReview,
    max_rows: int | None = None,
) -> SourceSpec:
    file = Path(path).resolve()
    sha = sha256_file(file)
    format = "parquet" if file.suffix == ".parquet" else "jsonl"
    return SourceSpec(
        source_id=source_id,
        uri=str(file),
        format=format,
        adapter=adapter,
        stream=stream,
        revision=f"sha256:{sha}",
        files=(SourceFile(str(file), sha, file.stat().st_size),),
        terms=terms,
        max_rows=max_rows,
    )


def pin_hf_source(
    repository_id: str,
    *,
    source_id: str,
    adapter: str,
    stream: str,
    revision: str = "main",
    subset: str | None = None,
    split: str = "train",
    terms: TermsReview | None = None,
    max_rows: int | None = 1000,
) -> SourceSpec:
    """Resolve metadata only; no corpus download and no terms acceptance."""
    terms = terms or TermsReview()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Revision resolution requires huggingface_hub") from exc
    info = HfApi().dataset_info(repository_id, revision=revision)
    return SourceSpec(
        source_id=source_id,
        uri=f"hf://datasets/{repository_id}",
        format="hf",
        adapter=adapter,
        stream=stream,
        revision=info.sha,
        subset=subset,
        split=split,
        terms=terms,
        max_rows=max_rows,
    )


def write_source_manifest(path: str | Path, sources: tuple[SourceSpec, ...]) -> None:
    write_json(path, {"schema_version": 1, "sources": [asdict(source) for source in sources]})
