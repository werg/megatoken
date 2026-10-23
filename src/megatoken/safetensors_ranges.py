"""Small, pinned HTTP-range reads from safetensors without downloading a shard."""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

RangeFetch = Callable[[int, int], bytes]


@dataclass(frozen=True)
class TensorLayout:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


class SafetensorsRangeReader:
    """Read selected complete first-axis rows from an immutable shard URL.

    The caller binds the URL to a pinned revision and verifies the LFS digest via
    metadata. Every request also checks the exact HTTP Content-Range response. Rows
    are coalesced into bounded contiguous requests; unrelated shard bytes are never
    downloaded.
    """

    _DTYPES = {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
        "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
        "I8": torch.int8,
        "U8": torch.uint8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "BOOL": torch.bool,
    }

    def __init__(
        self,
        url: str,
        tensor_name: str,
        *,
        file_size: int,
        expected_sha256: str,
        headers: dict[str, str] | None = None,
        fetch: RangeFetch | None = None,
        max_header_bytes: int = 16 << 20,
        max_range_bytes: int = 8 << 20,
        max_gap_rows: int = 0,
        range_workers: int = 16,
        row_cache: str | None = None,
    ) -> None:
        if (
            file_size <= 8
            or max_header_bytes < 1
            or max_range_bytes < 1
            or max_gap_rows < 0
            or range_workers < 1
        ):
            raise ValueError("Invalid shard size or range limits")
        if len(expected_sha256) != 64:
            raise ValueError("Expected a pinned SHA256 for the donor shard")
        self.url = url
        self.expected_sha256 = expected_sha256
        self.file_size = file_size
        self.max_range_bytes = max_range_bytes
        self.max_gap_rows = max_gap_rows
        self.range_workers = range_workers
        self._cache_path = row_cache
        self._cache_id = f"{expected_sha256}:{tensor_name}"
        self._stats_lock = threading.Lock()
        self.requests = 0
        self.bytes_received = 0
        self._client = None
        if row_cache:
            with sqlite3.connect(row_cache) as database:
                database.execute(
                    "CREATE TABLE IF NOT EXISTS rows "
                    "(cache_id TEXT NOT NULL, row_id INTEGER NOT NULL, payload BLOB NOT NULL, "
                    "PRIMARY KEY(cache_id, row_id))"
                )
        self._headers = dict(headers or {})
        self._fetch = fetch or self._http_range
        prefix = self._fetch(0, 7)
        if len(prefix) != 8:
            raise ValueError("Safetensors shard has a truncated header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 1 <= header_size <= max_header_bytes or 8 + header_size > file_size:
            raise ValueError("Safetensors header length is invalid or exceeds the configured bound")
        raw_header = self._fetch(8, 8 + header_size - 1)
        if len(raw_header) != header_size:
            raise ValueError("Safetensors shard has a truncated header")
        try:
            document = json.loads(raw_header)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Safetensors shard header is invalid JSON") from exc
        item = document.get(tensor_name)
        if not isinstance(item, dict):
            raise ValueError(f"Pinned shard does not contain tensor {tensor_name!r}")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        dtype = item.get("dtype")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension < 1 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or 8 + header_size + offsets[1] > file_size
            or dtype not in self._DTYPES
            or self._DTYPES[dtype] is None
        ):
            raise ValueError(f"Tensor {tensor_name!r} has invalid safetensors metadata")
        self.dtype = dtype
        self.torch_dtype = self._DTYPES[dtype]
        self.layout = TensorLayout(
            tensor_name,
            dtype,
            tuple(shape),
            8 + header_size + offsets[0],
            8 + header_size + offsets[1],
        )
        elements = 1
        for dimension in shape[1:]:
            elements *= dimension
        element_size = torch.empty((), dtype=self.torch_dtype).element_size()
        self.row_bytes = elements * element_size
        if self.row_bytes > self.max_range_bytes:
            raise ValueError("One donor tensor row exceeds the configured maximum range size")
        if self.row_bytes * shape[0] != self.layout.data_end - self.layout.data_start:
            raise ValueError("Tensor byte length does not match its declared shape and dtype")

    def _http_range(self, start: int, end: int) -> bytes:
        import httpx

        if self._client is None:
            self._client = httpx.Client(
                follow_redirects=True,
                timeout=90,
                limits=httpx.Limits(
                    max_connections=self.range_workers, max_keepalive_connections=16
                ),
            )
        for attempt in range(8):
            try:
                with self._client.stream(
                    "GET",
                    self.url,
                    headers={**self._headers, "Range": f"bytes={start}-{end}"},
                ) as response:
                    with self._stats_lock:
                        self.requests += 1
                    content_range = response.headers.get("Content-Range", "")
                    expected = f"bytes {start}-{end}/{self.file_size}"
                    if response.status_code in {429, 500, 502, 503, 504}:
                        if attempt < 7:
                            time.sleep(min(30, 0.5 * (2**attempt)))
                            continue
                    if response.status_code != 206 or content_range != expected:
                        raise ValueError(
                            "Donor server did not honor the bounded pinned byte range "
                            f"({response.status_code}, {content_range!r})"
                        )
                    expected_bytes = end - start + 1
                    body_parts = []
                    received = 0
                    for part in response.iter_bytes():
                        received += len(part)
                        if received > expected_bytes:
                            raise ValueError("Donor server exceeded the bounded range response")
                        body_parts.append(part)
                    if received != expected_bytes:
                        raise ValueError("Donor server returned a truncated tensor range")
                    body = b"".join(body_parts)
                    with self._stats_lock:
                        self.bytes_received += len(body)
                    return body
            except httpx.HTTPError as exc:
                if attempt == 7:
                    raise ValueError("Donor shard range request failed after retries") from exc
                time.sleep(min(30, 0.5 * (2**attempt)))
        raise ValueError("Donor shard range request failed after retries")

    def close(self) -> None:
        """Release the pooled HTTP connections after an extraction finishes."""
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
            self._client = None

    def __del__(self):
        self.close()

    def _cached_rows(self, rows: list[int]) -> dict[int, bytes]:
        if not self._cache_path or not rows:
            return {}
        found: dict[int, bytes] = {}
        with sqlite3.connect(self._cache_path) as database:
            for start in range(0, len(rows), 500):
                part = rows[start : start + 500]
                placeholders = ",".join("?" for _ in part)
                query = (
                    "SELECT row_id,payload FROM rows WHERE cache_id=? AND row_id IN ("
                    + placeholders
                    + ")"
                )
                found.update(
                    (int(row), bytes(payload))
                    for row, payload in database.execute(query, (self._cache_id, *part))
                )
        return found

    def _save_cached_rows(self, rows: dict[int, torch.Tensor]) -> None:
        if not self._cache_path or not rows:
            return
        payloads = [
            (
                self._cache_id,
                row,
                bytes(value.contiguous().view(torch.uint8).numpy()),
            )
            for row, value in rows.items()
        ]
        with sqlite3.connect(self._cache_path) as database:
            database.executemany(
                "INSERT OR IGNORE INTO rows(cache_id,row_id,payload) VALUES(?,?,?)", payloads
            )

    def read_rows(self, rows: list[int] | tuple[int, ...]) -> dict[int, torch.Tensor]:
        """Return requested rows in the tensor's original dtype, keyed by row ID."""
        requested = sorted(set(rows))
        if any(type(row) is not int or not 0 <= row < self.layout.shape[0] for row in requested):
            raise ValueError("Requested donor row is outside the tensor")
        if not requested:
            return {}
        cached = self._cached_rows(requested)
        missing = [row for row in requested if row not in cached]
        groups: list[list[int]] = []
        for row in missing:
            if (
                groups
                and row - groups[-1][-1] <= self.max_gap_rows + 1
                and (row - groups[-1][0] + 1) * self.row_bytes <= self.max_range_bytes
            ):
                groups[-1].append(row)
            else:
                groups.append([row])
        result: dict[int, torch.Tensor] = {}

        def read_group(group: list[int]) -> tuple[list[int], bytes]:
            first, last = group[0], group[-1]
            start = self.layout.data_start + first * self.row_bytes
            end = start + (last - first + 1) * self.row_bytes - 1
            raw = self._fetch(start, end)
            if len(raw) != end - start + 1:
                raise ValueError("Donor server returned a truncated tensor range")
            return group, raw

        with ThreadPoolExecutor(max_workers=self.range_workers) as pool:
            blocks = list(pool.map(read_group, groups))
        result: dict[int, torch.Tensor] = {
            row: torch.frombuffer(bytearray(raw), dtype=self.torch_dtype)
            .reshape(self.layout.shape[1:])
            .clone()
            for row, raw in cached.items()
        }
        for group, raw in blocks:
            first = group[0]
            block = torch.frombuffer(bytearray(raw), dtype=self.torch_dtype).reshape(
                len(raw) // self.row_bytes, *self.layout.shape[1:]
            )
            for row in group:
                index = row - first
                result[row] = block[index].clone()
        self._save_cached_rows({row: result[row] for row in missing})
        return result
