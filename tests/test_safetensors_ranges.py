import json
import struct
from datetime import UTC

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


def test_local_safetensors_reader_selects_rows_without_loading_entire_table(tmp_path):
    from safetensors.torch import save_file

    from megatoken.safetensors_ranges import LocalSafetensorsRowReader

    path = tmp_path / "table.safetensors"
    values = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    save_file({"table": values}, str(path))
    reader = LocalSafetensorsRowReader(str(path), "table")
    selected = reader.read_rows([9, 2, 2])
    assert sorted(selected) == [2, 9]
    assert torch.equal(selected[2], values[2])
    assert torch.equal(selected[9], values[9])
    assert reader.bytes_received == 2 * 4 * 4
    reader.close()


def test_local_native_fp8_weights_and_e8m0_scales_keep_their_numeric_values(tmp_path):
    from safetensors.torch import save_file

    from megatoken.safetensors_ranges import LocalSafetensorsRowReader

    path = tmp_path / "native-fp8.safetensors"
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127, 128], [129, 130]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    save_file({"weight": weights, "scale": scales}, str(path))
    weight_reader = LocalSafetensorsRowReader(str(path), "weight")
    scale_reader = LocalSafetensorsRowReader(str(path), "scale")
    assert weight_reader.read_rows([1])[1].float().tolist() == [3.0, 4.0]
    assert scale_reader.read_rows([0])[0].float().tolist() == [1.0, 2.0]
    assert scale_reader.read_rows([1])[1].float().tolist() == [4.0, 8.0]
    weight_reader.close()
    scale_reader.close()


def test_local_reader_handles_a_hundred_gb_shard_without_whole_file_mapping(tmp_path, monkeypatch):
    import safetensors

    from megatoken.safetensors_ranges import LocalSafetensorsRowReader

    # Sparse file: the declared tensor is large, while only two pages contain data.
    rows = 100_000_000_000
    header = json.dumps(
        {"table": {"dtype": "U8", "shape": [rows, 1], "data_offsets": [0, rows]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    path = tmp_path / "large.safetensors"
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)) + header)
        handle.write(b"\x07")
        handle.seek(8 + len(header) + rows - 1)
        handle.write(b"\x09")

    def forbid_mapping(*args, **kwargs):
        raise AssertionError("A local row reader must not map the whole shard")

    monkeypatch.setattr(safetensors, "safe_open", forbid_mapping)
    reader = LocalSafetensorsRowReader(str(path), "table")
    try:
        selected = reader.read_rows([0, rows - 1])
        assert selected[0].item() == 7 and selected[rows - 1].item() == 9
        assert reader.bytes_received == 2
    finally:
        reader.close()


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


def test_retry_after_accepts_seconds_and_http_dates(monkeypatch):
    from datetime import datetime

    from megatoken import safetensors_ranges

    monkeypatch.setattr(safetensors_ranges.random, "uniform", lambda low, high: high / 2)
    current = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)
    assert safetensors_ranges.retry_wait_seconds({"Retry-After": "12"}, 0, 60, now=current) == 14
    assert (
        safetensors_ranges.retry_wait_seconds(
            {"Retry-After": "Fri, 23 Oct 2026 12:00:10 GMT"}, 0, 60, now=current
        )
        == 11.75
    )
    assert safetensors_ranges.retry_wait_seconds({"Retry-After": "100"}, 0, 15, now=current) == 15


def test_rate_limit_waits_until_one_hour_budget_instead_of_quick_attempt_cap(monkeypatch, caplog):
    from megatoken import safetensors_ranges

    shard = make_shard()
    reader = SafetensorsRangeReader(
        "https://huggingface.co/model/resolve/revision/table.safetensors",
        "table",
        file_size=len(shard),
        expected_sha256="f" * 64,
        fetch=lambda start, end: shard[start : end + 1],
        max_wait_seconds=10,
    )

    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(safetensors_ranges.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(safetensors_ranges.time, "sleep", sleep)
    monkeypatch.setattr(safetensors_ranges.random, "uniform", lambda low, high: 0.0)

    class Response:
        def __init__(self):
            self.status_code = 429
            self.headers = {"Retry-After": "4"}
            self.url = reader.url
            self.history = ()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_bytes(self):
            raise AssertionError("429 response body is not consumed")

    class Client:
        def stream(self, *args, **kwargs):
            return Response()

        def close(self):
            pass

    reader._client = Client()
    with pytest.raises(ValueError, match="HTTP 429 after waiting 10s"):
        reader._http_range(10, 11)
    assert sleeps == [4, 4, 2]
    assert "retrying in" in caplog.text and "waited 8.0s of 10s" in caplog.text
    reader.close()
