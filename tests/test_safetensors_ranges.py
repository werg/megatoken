import json
import struct

import pytest
import torch

from megatoken.safetensors_ranges import SafetensorsRangeReader


def make_shard(dtype="U8", data=bytes(range(10))):
    columns = 2
    bytes_per_element = 1 if dtype == "U8" else 2
    shape = [len(data) // (columns * bytes_per_element), columns]
    header = json.dumps(
        {"table": {"dtype": dtype, "shape": shape, "data_offsets": [0, len(data)]}},
        separators=(",", ":"),
    ).encode()
    padding = (-len(header)) % 8
    header += b" " * padding
    return struct.pack("<Q", len(header)) + header + data


def test_safetensors_range_reader_only_fetches_header_and_requested_rows(tmp_path):
    shard = make_shard()
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return shard[start : end + 1]

    reader = SafetensorsRangeReader(
        "https://example.invalid/pinned.safetensors",
        "table",
        file_size=len(shard),
        expected_sha256="a" * 64,
        fetch=fetch,
        max_range_bytes=2,
        row_cache=str(tmp_path / "rows.sqlite"),
    )
    rows = reader.read_rows([4, 1, 1])
    assert set(rows) == {1, 4}
    assert rows[1].tolist() == [2, 3]
    assert rows[4].tolist() == [8, 9]
    assert len(calls) == 4
    assert all((end - start + 1) < len(shard) for start, end in calls)
    assert reader.read_rows([1, 4])[1].tolist() == [2, 3]
    assert len(calls) == 4


def test_safetensors_range_reader_decodes_bfloat16_rows():
    values = torch.tensor([[1.25, -2.0], [3.5, 4.0]], dtype=torch.bfloat16)
    raw = bytes(values.view(torch.uint8).flatten().tolist())
    shard = make_shard("BF16", raw)
    reader = SafetensorsRangeReader(
        "https://example.invalid/pinned.safetensors",
        "table",
        file_size=len(shard),
        expected_sha256="b" * 64,
        fetch=lambda start, end: shard[start : end + 1],
    )
    loaded = reader.read_rows([0, 1])
    assert torch.equal(loaded[0], values[0])
    assert torch.equal(loaded[1], values[1])


def test_safetensors_range_reader_rejects_invalid_shape_metadata():
    shard = make_shard()
    with pytest.raises(ValueError, match="invalid safetensors metadata"):
        SafetensorsRangeReader(
            "https://example.invalid/pinned.safetensors",
            "table",
            file_size=len(shard) - 1,
            expected_sha256="c" * 64,
            fetch=lambda start, end: shard[start : end + 1],
        )


def test_http_reader_rejects_servers_that_ignore_range_before_reading_body(monkeypatch):
    import httpx

    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_bytes(self):
            raise AssertionError("The whole shard body must not be consumed")

    class Client:
        def __init__(self, **kwargs):
            pass

        def stream(self, *args, **kwargs):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(httpx, "Client", Client)
    with pytest.raises(ValueError, match="did not honor"):
        SafetensorsRangeReader(
            "https://example.invalid/pinned.safetensors",
            "table",
            file_size=1024,
            expected_sha256="d" * 64,
        )


def test_http_reader_caches_signed_cdn_url_and_refreshes_without_auth_leak():
    shard = make_shard()
    origin = "https://huggingface.co/model/resolve/revision/table.safetensors"
    cdn1 = "https://cdn.example/first?signature=one"
    cdn2 = "https://cdn.example/second?signature=two"
    reader = SafetensorsRangeReader(
        origin,
        "table",
        file_size=len(shard),
        expected_sha256="e" * 64,
        headers={"Authorization": "Bearer private-token"},
        fetch=lambda start, end: shard[start : end + 1],
    )

    class Response:
        def __init__(self, status, url, expected, history=()):
            self.status_code = status
            self.url = url
            self.history = history
            self.headers = {"Content-Range": expected}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_bytes(self):
            yield b"xy"

    class Client:
        def __init__(self):
            self.calls = []
            self.responses = [
                Response(206, cdn1, f"bytes 10-11/{len(shard)}", history=(object(),)),
                Response(206, cdn1, f"bytes 10-11/{len(shard)}"),
                Response(403, cdn1, ""),
                Response(206, cdn2, f"bytes 10-11/{len(shard)}", history=(object(),)),
                Response(206, cdn2, f"bytes 10-11/{len(shard)}"),
            ]

        def stream(self, method, url, *, headers):
            self.calls.append((method, url, dict(headers)))
            return self.responses.pop(0)

        def close(self):
            pass

    client = Client()
    reader._client = client
    assert reader._http_range(10, 11) == b"xy"
    assert reader._http_range(10, 11) == b"xy"
    assert reader._http_range(10, 11) == b"xy"
    assert reader._http_range(10, 11) == b"xy"
    assert [url for _, url, _ in client.calls] == [origin, cdn1, cdn1, origin, cdn2]
    assert client.calls[0][2]["Authorization"] == "Bearer private-token"
    assert client.calls[1][2].get("Authorization") is None
    assert client.calls[2][2].get("Authorization") is None
    assert client.calls[3][2]["Authorization"] == "Bearer private-token"
    assert client.calls[4][2].get("Authorization") is None
    reader.close()
