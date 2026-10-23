"""Fetching pictures from URLs: caching by content, dead links, busy servers."""

from __future__ import annotations

import http.server
import io
import threading
import urllib.error
from collections import Counter

import pytest
from PIL import Image

from megatoken.data.fetch import Fetcher, fetched_records


def _png(seed: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (seed * 40 % 255, 90, 150)).save(buffer, format="PNG")
    return buffer.getvalue()


class _Server:
    """A local web server: ``/ok/N.png`` serves a picture, ``/same/N.png`` the same picture as
    ``/ok/N.png``, ``/gone`` is 404, ``/busy`` is 503 twice and then a picture, ``/limited`` is
    always 429."""

    def __init__(self):
        outer = self
        self.hits: Counter[str] = Counter()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.hits[self.path] += 1
                name = self.path.rsplit("/", 1)[-1]
                if self.path.startswith(("/ok/", "/same/")):
                    body = _png(int(name.split(".")[0]))
                elif self.path == "/busy" and outer.hits[self.path] > 2:
                    body = _png(9)
                elif self.path == "/busy":
                    self.send_response(503)
                    self.send_header("Retry-After", "1")
                    self.end_headers()
                    return
                elif self.path == "/limited":
                    self.send_response(429)
                    self.end_headers()
                    return
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def web():
    server = _Server()
    yield server
    server.close()


def _fetcher(tmp_path, **extra):
    waits: list[float] = []
    fetcher = Fetcher(tmp_path / "cache", backoff=0.01, sleep=waits.append, **extra)
    fetcher.waits = waits
    return fetcher


def test_a_picture_is_downloaded_once_and_a_second_url_with_the_same_content_is_stored_once(
    web, tmp_path
):
    fetcher = _fetcher(tmp_path)
    assert fetcher.get(f"{web.url}/ok/1.png") == _png(1)
    assert fetcher.get(f"{web.url}/ok/1.png") == _png(1)
    assert fetcher.get(f"{web.url}/same/1.png") == _png(1)
    assert web.hits["/ok/1.png"] == 1
    assert len(list((tmp_path / "cache/blobs").iterdir())) == 1
    assert fetcher.stats["fetched"] == 2 and fetcher.stats["cached"] == 1


def test_a_second_fetcher_over_the_same_cache_asks_the_server_nothing(web, tmp_path):
    first = _fetcher(tmp_path)
    assert first.get(f"{web.url}/ok/2.png") and first.get(f"{web.url}/gone") is None
    second = _fetcher(tmp_path)
    assert second.get(f"{web.url}/ok/2.png") == _png(2) and second.get(f"{web.url}/gone") is None
    assert web.hits["/ok/2.png"] == 1 and web.hits["/gone"] == 1
    assert second.stats["cached"] == 1 and second.stats["dead_cached"] == 1


def test_dead_links_are_skipped_and_counted_but_a_busy_server_is_retried(web, tmp_path):
    fetcher = _fetcher(tmp_path)
    assert fetcher.get(f"{web.url}/gone") is None
    assert fetcher.stats["dead"] == 1 and fetcher.stats["http_404"] == 1
    assert web.hits["/gone"] == 1  # a dead link is not retried
    assert fetcher.get(f"{web.url}/busy") == _png(9)
    assert web.hits["/busy"] == 3 and fetcher.waits == [1.0, 1.0]
    assert fetcher.get(f"{web.url}/limited") is None
    assert fetcher.stats["failed"] == 1 and web.hits["/limited"] == 5
    # A server that never answered is asked again by the next run; a dead link is not.
    assert _fetcher(tmp_path).get(f"{web.url}/limited") is None
    assert web.hits["/limited"] == 10


def test_a_body_over_the_limit_and_an_unreachable_host_are_counted(web, tmp_path):
    fetcher = _fetcher(tmp_path, max_bytes=100, retries=1)
    assert fetcher.get(f"{web.url}/ok/3.png") is None and fetcher.stats["too_big"] == 1
    assert fetcher.get("http://127.0.0.1:9/never") is None
    assert fetcher.stats["failed"] == 1 and fetcher.stats["network_errors"] == 2


def test_the_cache_evicts_the_least_recently_used_pictures_over_its_size(web, tmp_path):
    fetcher = _fetcher(tmp_path, cache_bytes=2 * len(_png(1)) + 10)
    for number in (1, 2, 3, 4):
        fetcher.get(f"{web.url}/ok/{number}.png")
    assert len(list((tmp_path / "cache/blobs").iterdir())) <= 3
    assert fetcher.get(f"{web.url}/ok/4.png") == _png(4)  # the newest stays


def test_rows_get_their_urls_replaced_in_order_with_dead_links_as_none(web, tmp_path):
    rows = [
        {
            "id": number,
            "first": f"{web.url}/ok/{number}.png",
            "turns": [{"out": f"{web.url}/ok/{number + 10}.png"}, {"out": f"{web.url}/gone"}],
            "text": "not a url field",
        }
        for number in range(1, 6)
    ]
    original = [dict(row) for row in rows]
    fetcher = _fetcher(tmp_path)
    result = list(fetched_records(rows, ["first", "turns.*.out"], fetcher, workers=3, ahead=2))
    assert [row["id"] for row in result] == [1, 2, 3, 4, 5]
    assert result[0]["first"]["bytes"] == _png(1) and result[0]["turns"][0]["out"]["bytes"] == _png(
        11
    )
    assert result[0]["turns"][1]["out"] is None and result[0]["text"] == "not a url field"
    assert rows == original  # the input rows are not changed
    assert fetcher.stats["rows"] == 5 and fetcher.stats["rows_with_dead_links"] == 5
    assert fetcher.stats["dead"] == 1 and fetcher.stats["dead_cached"] == 4


def test_http_errors_other_than_retryable_ones_are_dead_links(tmp_path):
    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    fetcher = Fetcher(tmp_path / "cache", open_url=refuse, sleep=lambda seconds: None)
    assert fetcher.get("https://example.invalid/a.jpg") is None
    assert fetcher.stats["dead"] == 1 and fetcher.stats["http_403"] == 1
