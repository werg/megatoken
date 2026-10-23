"""Fetching media from published URLs inside a preparation job.

Some public datasets publish only a table whose rows name their images by URL (the pictures sit
on a publisher's server, not in the repository). ``Fetcher`` downloads those pictures from the
sandbox that prepares them, never from a workstation:

- every body is stored under the SHA-256 of its content (``blobs/``), so a picture that many rows
  name, or that two URLs both serve, is kept and decoded once;
- every URL's outcome is remembered (``urls/``): the content hash, or the reason it is dead, so
  a retried shard or a second row naming the same URL asks the server nothing;
- a dead link (an HTTP error that will not go away: 404, 403, 410, ...) is skipped and counted;
  a busy server (429, 5xx, timeouts) is retried with growing waits (``Retry-After`` honoured) and
  counted as ``failed`` if it never answers;
- the counts are in ``Fetcher.stats`` and, per source, ``FETCH_STATS`` (printed and written to the
  prepared root by ``scripts/prepare_open_media.py``).

``fetched_records`` applies a fetcher to a stream of table rows, a few rows ahead so that the
network waits overlap.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

USER_AGENT = "megatoken-data-preparation/1.0 (research dataset preparation)"
# HTTP statuses that mean "try again later"; every other error status is a dead link.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
FETCH_STATS: dict[str, Counter[str]] = {}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class _Host:
    """One web host's politeness: a limit on simultaneous requests, and a shared pause after it
    says "slow down" (429 or 503), growing while it keeps saying so."""

    def __init__(self, workers: int):
        self.slots = threading.BoundedSemaphore(workers)
        self.resume_at = 0.0
        self.strikes = 0


class Fetcher:
    """Download URLs with a content-addressed cache, polite retries and a count of every outcome."""

    def __init__(
        self,
        cache: Path,
        *,
        timeout: float = 30.0,
        retries: int = 6,
        backoff: float = 4.0,
        host_workers: int = 4,
        host_limits: Mapping[str, int] | None = None,
        max_bytes: int = 64 << 20,
        cache_bytes: int = 20 << 30,
        stats: Counter[str] | None = None,
        open_url: Callable[[urllib.request.Request, float], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cache = Path(cache)
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self._clock = clock
        self._host_workers, self._host_limits = host_workers, dict(host_limits or {})
        self._hosts: dict[str, _Host] = {}
        self.max_bytes, self.cache_bytes = max_bytes, cache_bytes
        self.stats = stats if stats is not None else Counter()
        self._open = open_url or (
            lambda request, timeout: urllib.request.urlopen(request, timeout=timeout)
        )
        self._sleep = sleep
        self._lock = threading.Lock()
        self._gates: dict[str, threading.Lock] = {}
        (self.cache / "blobs").mkdir(parents=True, exist_ok=True)
        (self.cache / "urls").mkdir(parents=True, exist_ok=True)
        self._size = sum(path.stat().st_size for path in (self.cache / "blobs").iterdir())

    # -- cache ---------------------------------------------------------------------------------

    def _note(self, url: str) -> Path:
        return self.cache / "urls" / _digest(url)

    def _blob(self, digest: str) -> Path:
        return self.cache / "blobs" / digest

    def _remembered(self, url: str) -> tuple[bytes | None, bool]:
        """(body, known): the cached body, or None for a remembered dead link."""
        try:
            note = self._note(url).read_text()
        except FileNotFoundError:
            return None, False
        if note.startswith("dead:"):
            return None, True
        try:
            body = self._blob(note).read_bytes()
        except FileNotFoundError:
            return None, False
        os.utime(self._blob(note))
        return body, True

    def _store(self, url: str, body: bytes) -> None:
        digest = hashlib.sha256(body).hexdigest()
        path = self._blob(digest)
        with self._lock:
            if not path.exists():
                # Write beside the blob and rename, so a crash never leaves a truncated body
                # that a later run would trust as complete.
                partial = path.with_name(path.name + ".partial")
                partial.write_bytes(body)
                os.replace(partial, path)
                self._size += len(body)
                self._evict(keep=digest)
            self._note(url).write_text(digest)

    def _evict(self, keep: str) -> None:
        """Delete the least recently used bodies once the cache is over its size."""
        if self._size <= self.cache_bytes:
            return
        blobs = sorted((self.cache / "blobs").iterdir(), key=lambda path: path.stat().st_mtime)
        for path in blobs:
            if self._size <= self.cache_bytes * 0.9:
                break
            if path.name != keep:
                self._size -= path.stat().st_size
                path.unlink(missing_ok=True)

    # -- network -------------------------------------------------------------------------------

    def get(self, url: str) -> bytes | None:
        """The body of ``url``, or None for a dead or unreachable link (counted either way)."""
        with self._lock:
            gate = self._gates.setdefault(url, threading.Lock())
        try:
            with gate:  # two rows naming one URL at once download it once
                return self._get(url)
        finally:
            with self._lock:
                self._gates.pop(url, None)

    def _host(self, url: str) -> _Host:
        name = urllib.parse.urlsplit(url).hostname or ""
        limit = next(
            (n for suffix, n in self._host_limits.items() if name.endswith(suffix)),
            self._host_workers,
        )
        with self._lock:
            return self._hosts.setdefault(name, _Host(limit))

    def _get(self, url: str) -> bytes | None:
        body, known = self._remembered(url)
        if known:
            self.stats["cached" if body is not None else "dead_cached"] += 1
            return body
        reason = "failed"
        host = self._host(url)
        for attempt in range(self.retries + 1):
            pause = host.resume_at - self._clock()
            if pause > 0:  # every request to this host waits while it asks for peace
                self._sleep(pause)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with host.slots, self._open(request, self.timeout) as response:
                    body = response.read(self.max_bytes + 1)
                host.strikes = 0
                if len(body) > self.max_bytes:
                    reason = "too_big"
                    break
                if not body:
                    reason = "empty"
                    break
                self._store(url, body)
                self.stats["fetched"] += 1
                self.stats["bytes"] += len(body)
                return body
            except urllib.error.HTTPError as error:
                self.stats[f"http_{error.code}"] += 1
                if error.code not in RETRY_STATUSES:
                    reason = "dead"
                    break
                host.strikes += 1
                wait = min(self.backoff * 2 ** (host.strikes - 1), 60.0)
                retry_after = error.headers.get("Retry-After") if error.headers else None
                if retry_after and retry_after.isdigit():
                    wait = min(float(retry_after), 60.0)
                host.resume_at = max(host.resume_at, self._clock() + wait)
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                ConnectionError,
                OSError,
                ValueError,
            ):
                self.stats["network_errors"] += 1
                if attempt < self.retries:
                    self._sleep(min(self.backoff * 2**attempt, 60.0))
        self.stats[reason] += 1
        if reason != "failed":  # a server that never answered may answer on a retried shard
            self._note(url).write_text(f"dead:{reason}")
        return None


def _opener(open_url):
    return open_url or (lambda request, timeout: urllib.request.urlopen(request, timeout=timeout))


def content_length(
    url: str,
    *,
    timeout: float = 60.0,
    open_url: Callable[[urllib.request.Request, float], Any] | None = None,
) -> int:
    """The size of a published file (HEAD)."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with _opener(open_url)(request, timeout) as response:
        return int(response.headers["Content-Length"])


def download(
    url: str,
    destination: Path,
    *,
    sha256: str | None = None,
    timeout: float = 60.0,
    attempts: int = 12,
    open_url: Callable[[urllib.request.Request, float], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """A whole published file saved at ``destination``, byte-exact.

    Servers (a CDN in front of a bucket) sometimes end a long response early without any error,
    so the file is built by range requests that resume where the last one stopped, until its
    size equals the server's ``Content-Length``; with ``sha256`` the finished file must also match
    that pinned digest. An existing file that passes the same checks is reused."""
    destination = Path(destination)
    size = content_length(url, timeout=timeout, open_url=open_url)

    def intact(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size != size:
            return False
        return sha256 is None or hashlib.sha256(path.read_bytes()).hexdigest() == sha256

    if intact(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    failures = 0
    while True:
        have = partial.stat().st_size if partial.exists() else 0
        if have >= size:
            break
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={have}-"}
        )
        try:
            with _opener(open_url)(request, timeout) as response:
                if getattr(response, "status", 206) == 200 and have:
                    partial.unlink()  # the server ignored the range: start over
                    have = 0
                with partial.open("ab") as out:
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as error:
            failures += 1
            if failures >= attempts:
                raise OSError(f"Could not download {url}: {error}") from error
            sleep(min(2.0**failures, 30.0))
            continue
        if (partial.stat().st_size if partial.exists() else 0) == have:
            failures += 1  # an empty answer: no progress
            if failures >= attempts:
                raise OSError(f"Download of {url} made no progress at byte {have}")
            sleep(min(2.0**failures, 30.0))
    partial.replace(destination)
    if not intact(destination):
        destination.unlink()
        raise OSError(f"{url} does not match its pinned size or SHA-256")
    return destination


# -- applying a fetcher to table rows -------------------------------------------------------


def _paths(value: Any, parts: Sequence[str]) -> Iterator[tuple[Any, str | int]]:
    """The (container, key) places a dotted path with ``*`` for every list item points at."""
    head, rest = parts[0], parts[1:]
    if head == "*":
        if isinstance(value, list):
            for index in range(len(value)):
                if rest:
                    yield from _paths(value[index], rest)
                else:
                    yield value, index
        return
    key: str | int = int(head) if isinstance(value, list) and head.isdigit() else head
    try:
        child = value[key]
    except (KeyError, IndexError, TypeError):
        return
    if rest:
        yield from _paths(child, rest)
    else:
        yield value, key


def url_places(record: Mapping[str, Any], fields: Iterable[str]) -> list[tuple[Any, str | int]]:
    return [place for spec in fields for place in _paths(record, spec.split("."))]


def fetched_records(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    fetcher: Fetcher,
    *,
    workers: int = 8,
    ahead: int = 16,
) -> Iterator[dict[str, Any]]:
    """Rows with the URLs in ``fields`` (dotted paths; ``*`` takes every item of a list) replaced
    by the picture's bytes (``{"bytes": ...}``), or by None where the link is dead.

    The rows are copied, never changed. A few rows are in flight at once; order is kept.
    """
    import copy

    def resolve(row: dict[str, Any]) -> dict[str, Any]:
        places = [
            (container, key)
            for container, key in url_places(row, fields)
            if isinstance(container[key], str)
        ]
        dead = 0
        for container, key in places:
            body = fetcher.get(container[key])
            if body is None:
                dead += 1
            container[key] = {"bytes": body, "path": None} if body is not None else None
        fetcher.stats["rows"] += 1
        fetcher.stats["rows_with_dead_links"] += bool(dead)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        window: deque[Future[dict[str, Any]]] = deque()
        for record in records:
            window.append(pool.submit(resolve, copy.deepcopy(dict(record))))
            if len(window) >= ahead:
                yield window.popleft().result()
        while window:
            yield window.popleft().result()
