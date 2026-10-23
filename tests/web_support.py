"""A small local web site for tests of sources that are published as URLs (no network)."""

from __future__ import annotations

import http.server
import re
import threading
from collections import Counter


class Site:
    """Serves ``files`` (path -> bytes) with HEAD and byte ranges.

    ``cut`` lists paths whose first full-length answer is ended early after half the bytes, as a
    CDN sometimes does; ``hits`` counts the requests per path."""

    def __init__(self, files: dict[str, bytes], cut: tuple[str, ...] = ()):
        self.files, self.hits, self._cut = files, Counter(), set(cut)
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _serve(self, send_body: bool) -> None:
                outer.hits[self.path] += 1
                body = outer.files.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                start = 0
                match = re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
                if match:
                    start = int(match.group(1))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                chunk = body[start:]
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                if not send_body:
                    return
                if self.path in outer._cut:
                    outer._cut.discard(self.path)
                    self.wfile.write(chunk[: len(chunk) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(chunk)

            def do_GET(self):
                self._serve(True)

            def do_HEAD(self):
                self._serve(False)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
