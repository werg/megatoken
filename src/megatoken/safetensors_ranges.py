"""Small, pinned HTTP-range reads from safetensors without downloading a shard."""

from __future__ import annotations

import email.utils
import json
import logging
import math
import random
import sqlite3
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC
from urllib.parse import urlsplit

import torch

RangeFetch = Callable[[int, int], bytes]
MAX_RETRY_WAIT_SECONDS = 3600.0
_LOG = logging.getLogger(__name__)


def retry_wait_seconds(headers, attempt: int, remaining: float, *, now=None) -> float:
    """Return a jittered backoff that honors numeric or HTTP-date Retry-After."""
    from datetime import datetime

    named = None
    raw = headers.get("Retry-After", "")
    if raw:
        try:
            named = max(0.0, float(raw))
        except (TypeError, ValueError):
            try:
                retry_at = email.utils.parsedate_to_datetime(raw)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                current = now or datetime.now(UTC)
                named = max(0.0, (retry_at - current).total_seconds())
            except (TypeError, ValueError, OverflowError):
                named = None
    base = named if named is not None else min(2.0**attempt, 300.0)
    jitter_cap = max(0.0, min(1.0 + base / 4.0, remaining - base))
    return min(base + random.uniform(0.0, jitter_cap), remaining)


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
        max_wait_seconds: float = MAX_RETRY_WAIT_SECONDS,
    ) -> None:
        if (
            file_size <= 8
            or max_header_bytes < 1
            or max_range_bytes < 1
            or max_gap_rows < 0
            or range_workers < 1
            or max_wait_seconds <= 0
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
        self.max_wait_seconds = max_wait_seconds
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
        self._cdn_url: str | None = None
        self._cdn_headers = {
            name: value
            for name, value in self._headers.items()
            if name.casefold() != "authorization"
        }
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
        attempt = 0
        wait_started = time.monotonic()
        deadline = wait_started + self.max_wait_seconds
        while True:
            try:
                request_url = self._cdn_url or self.url
                request_headers = self._headers if self._cdn_url is None else self._cdn_headers
                retry_delay = None
                retry_status = None
                refresh_url = False
                with self._client.stream(
                    "GET",
                    request_url,
                    headers={**request_headers, "Range": f"bytes={start}-{end}"},
                ) as response:
                    with self._stats_lock:
                        self.requests += 1 + len(getattr(response, "history", ()))
                    if (
                        self._cdn_url is not None
                        and response.status_code in {401, 403}
                        and attempt < 7
                    ):
                        # Signed Hub CDN URLs expire. Return to the pinned resolve
                        # URL, which can mint a fresh signed URL; auth is sent only
                        # to that original Hub host, never to the CDN host.
                        self._cdn_url = None
                        refresh_url = True
                    content_range = response.headers.get("Content-Range", "")
                    expected = f"bytes {start}-{end}/{self.file_size}"
                    if refresh_url:
                        pass
                    elif response.status_code in {429, 500, 502, 503, 504}:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            waited = self.max_wait_seconds
                            raise ValueError(
                                f"Donor shard request remained HTTP {response.status_code} "
                                f"after waiting {waited:.0f}s for rate limits"
                            )
                        retry_status = response.status_code
                        retry_delay = retry_wait_seconds(response.headers, attempt, remaining)
                    elif response.status_code != 206 or content_range != expected:
                        raise ValueError(
                            "Donor server did not honor the bounded pinned byte range "
                            f"({response.status_code}, {content_range!r})"
                        )
                    if not refresh_url and retry_delay is None and retry_status is None:
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
                        resolved_url = str(response.url)
                        if urlsplit(resolved_url).netloc != urlsplit(self.url).netloc:
                            self._cdn_url = resolved_url
                        return body
                if refresh_url:
                    attempt += 1
                    continue
                if retry_delay is not None:
                    _LOG.warning(
                        "Donor shard request received HTTP %s; retrying in %.1fs "
                        "(waited %.1fs of %.0fs)",
                        retry_status,
                        retry_delay,
                        time.monotonic() - wait_started,
                        self.max_wait_seconds,
                    )
                    time.sleep(retry_delay)
                    attempt += 1
                    continue
            except httpx.HTTPError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError(
                        "Donor shard request failed after waiting "
                        f"{self.max_wait_seconds:.0f}s for network recovery"
                    ) from exc
                retry_delay = retry_wait_seconds({}, attempt, remaining)
                _LOG.warning(
                    "Donor shard request hit a network error; retrying in %.1fs "
                    "(waited %.1fs of %.0fs)",
                    retry_delay,
                    time.monotonic() - wait_started,
                    self.max_wait_seconds,
                )
                time.sleep(retry_delay)
                attempt += 1

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


class LocalSafetensorsRowReader:
    """Read sparse tensor rows from a previously pinned local safetensors shard."""

    _DTYPES = SafetensorsRangeReader._DTYPES

    def __init__(self, path: str, tensor_name: str) -> None:
        from safetensors import safe_open

        self.path = path
        self.tensor_name = tensor_name
        self._handle = safe_open(path, framework="pt", device="cpu")
        self._handle.__enter__()
        try:
            self._tensor = self._handle.get_slice(tensor_name)
            self.dtype = self._tensor.get_dtype()
            self.torch_dtype = self._DTYPES.get(self.dtype)
            shape = tuple(self._tensor.get_shape())
            if (
                not shape
                or any(type(dimension) is not int or dimension < 1 for dimension in shape)
                or self.torch_dtype is None
            ):
                raise ValueError(f"Local tensor {tensor_name!r} has unsupported metadata")
            self.shape = shape
            row_elements = math.prod(shape[1:])
            self.row_bytes = row_elements * torch.empty((), dtype=self.torch_dtype).element_size()
        except BaseException:
            self.close()
            raise
        self.requests = 0
        self.bytes_received = 0

    def read_rows(self, rows: list[int] | tuple[int, ...]) -> dict[int, torch.Tensor]:
        requested = sorted(set(rows))
        if any(type(row) is not int or not 0 <= row < self.shape[0] for row in requested):
            raise ValueError("Requested donor row is outside the tensor")
        if not requested:
            return {}
        values = self._tensor[requested]
        self.requests += 1
        self.bytes_received += len(requested) * self.row_bytes
        return {row: values[index].clone() for index, row in enumerate(requested)}

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            self._handle = None
            handle.__exit__(None, None, None)

    def __del__(self):
        self.close()
