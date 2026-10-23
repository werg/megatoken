"""Real optional reader integration; all input data is generated locally."""

import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import pytest
from prepare_support import interrupt_after_rows

from megatoken.data import DedupConfig, TermsReview, pin_local_source, prepare_sources
from megatoken.data.sources import SourceFile, SourceSpec, iter_source_rows
from megatoken.generation.codec import ByteCodec


@pytest.fixture
def parquet_source(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    bodies = [
        "First: decomposed e\u0301 must stay decomposed.",
        "Second: CRLF\r\nand a final newline\n",
        "Third: Ελληνικά and 日本語.",
        None,
        'Fifth: literal "quotes", a backslash \\, and NUL \0.',
        "Sixth: supplementary Unicode 🛰 remains exact.",
        "Seventh: outside the declared source row cap.",
    ]
    rows = [
        {"uuid": f"row-{index}", "body": body, "metadata": {"original_id": f"family-{index}"}}
        for index, body in enumerate(bodies)
    ]
    schema = arrow.schema(
        [
            ("uuid", arrow.string()),
            ("body", arrow.string()),
            ("metadata", arrow.struct([("original_id", arrow.string())])),
        ]
    )
    path = tmp_path / "documents.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows, schema=schema), path, row_group_size=2)
    with parquet.ParquetFile(path) as reader:
        assert reader.num_row_groups == 4
    source = pin_local_source(
        path,
        source_id="local-parquet-fixture",
        adapter="document",
        stream="natural_documents",
        terms=TermsReview(True, "local test fixture", reviewed_by="test", reviewed_at="2026-10-18"),
        max_rows=6,
    )
    return rows, replace(source, field_map={"text": "body", "family": "metadata.original_id"})


def test_local_parquet_row_groups_resume_rejections_and_byte_exact_preparation(
    tmp_path, parquet_source, monkeypatch
):
    rows, source = parquet_source
    source.validate_local_files()
    expected_rows = list(enumerate(rows[:6]))
    assert list(iter_source_rows(source)) == expected_rows
    # Both row-group boundaries and offsets inside a group preserve logical indices.
    for start_row in (1, 2, 3, 5, 6):
        actual = list(iter_source_rows(source, start_row=start_row))
        assert actual == expected_rows[start_row:]
        assert all(tuple(row) == ("uuid", "body", "metadata") for _, row in actual)

    codec = ByteCodec.byte_fallback()
    options = {
        "dedup": DedupConfig(train_fraction=1, validation_fraction=0),
        "max_record_bytes": 9,
        "max_record_tokens": 7,
    }
    with pytest.raises(ValueError, match="row 3: Missing 'text'"):
        prepare_sources([source], tmp_path / "strict", codec, **options)

    output = tmp_path / "resumed"
    with monkeypatch.context() as crash:
        interrupt_after_rows(crash, 3)
        with pytest.raises(InterruptedError):
            prepare_sources([source], output, codec, reject_invalid=True, **options)
    with prepare_sources([source], output, codec, reject_invalid=True, **options) as prepared:
        episodes = list(prepared.iter_episodes())
        actual_payloads = {
            episode.lineage.source_row: b"".join(record.payload for record in episode.records)
            for episode in episodes
        }
        assert actual_payloads == {
            index: row["body"].encode("utf-8")
            for index, row in expected_rows
            if row["body"] is not None
        }
        assert all(
            episode.lineage.family_id == f"family-{episode.lineage.source_row}"
            for episode in episodes
        )
        assert all(len(episode.records) > 1 for episode in episodes)
        assert prepared.manifest["census"]["accepted_episodes"] == 5
        assert prepared.manifest["census"]["rejected_rows"] == 1
        assert prepared.manifest["census"]["U"] == sum(map(len, actual_payloads.values()))
        resumed_episodes = [episode.to_dict() for episode in episodes]
        resumed_census = prepared.manifest["census"]

    with prepare_sources(
        [source], tmp_path / "uninterrupted", codec, reject_invalid=True, **options
    ) as uninterrupted:
        assert [episode.to_dict() for episode in uninterrupted] == resumed_episodes
        assert uninterrupted.manifest["census"] == resumed_census
    rejections = json.loads((output / "rejections.json").read_text())
    assert rejections["total_rejected"] == 1
    assert rejections["rows"][0]["row_index"] == 3


def test_hf_reader_passes_pinned_identity_and_uses_real_local_streaming(
    parquet_source, monkeypatch
):
    hub = pytest.importorskip("huggingface_hub")
    rows, source = parquet_source
    calls = []

    def local_open(_filesystem, path, mode="rb", **kwargs):
        # Preserve the real pinned Hub address while replacing only remote I/O;
        # `_hub_parquet_rows` still exercises the actual PyArrow row-group reader.
        calls.append((path, mode, kwargs))
        assert path == f"hf://datasets/local/fixture@{'a' * 40}/documents.parquet"
        return open(source.files[0].path, mode)

    monkeypatch.setattr(hub.HfFileSystem, "open", local_open)
    hf_source = replace(
        source,
        format="hf",
        uri="hf://datasets/local/fixture",
        revision="a" * 40,
        subset="fixture-subset",
        files=(replace(source.files[0], path="documents.parquet"),),
    )
    assert list(iter_source_rows(hf_source, start_row=3)) == list(enumerate(rows[3:6], 3))
    assert calls == [
        (
            "hf://datasets/local/fixture@" + "a" * 40 + "/documents.parquet",
            "rb",
            {"block_size": 32 << 20},
        )
    ]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"messages":[{"role":"user","content":"a"}]}\n'
        b'{"messages":[{"role":"assistant","content":"b","is_train":false}]}\n',
        b"[\n"
        b' {"messages":[{"role":"user","content":"a"}]},\n'
        b' {"messages":[{"role":"assistant","content":"b","is_train":false}]}\n'
        b"]\n",
        b'\xef\xbb\xbf{\n  "text": "pretty object"\n}\n',
    ],
    ids=("jsonl-with-json-suffix", "json-array", "bom-pretty-object"),
)
def test_hf_json_reader_handles_heterogeneous_rows_without_arrow_schema(monkeypatch, payload):
    hub = pytest.importorskip("huggingface_hub")
    address = f"hf://datasets/local/fixture@{'a' * 40}/mixed.json"
    calls = []

    def local_open(_filesystem, path, mode="rb", **kwargs):
        calls.append((path, mode, kwargs))
        assert path == address
        return io.BytesIO(payload)

    monkeypatch.setattr(hub.HfFileSystem, "open", local_open)
    source = SourceSpec(
        source_id="mixed-json-fixture",
        uri="hf://datasets/local/fixture",
        format="hf",
        adapter="chat",
        stream="chat",
        revision="a" * 40,
        terms=TermsReview(True, "local test", reviewed_by="test", reviewed_at="2026-10-23"),
        files=(SourceFile("mixed.json"),),
    )
    rows = list(iter_source_rows(source))
    if payload.startswith(b"\xef\xbb\xbf"):
        assert rows == [(0, {"text": "pretty object"})]
    else:
        assert rows == [
            (0, {"messages": [{"role": "user", "content": "a"}]}),
            (1, {"messages": [{"role": "assistant", "content": "b", "is_train": False}]}),
        ]
    assert list(iter_source_rows(source, start_row=1)) == rows[1:]
    assert calls == [(address, "rb", {"block_size": 8 << 20})] * 2


def test_hf_jsonl_reader_handles_a_row_spanning_multiple_remote_blocks(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    text = "x" * (70 << 10) + "🛰"
    payload = json.dumps({"text": text}, ensure_ascii=False).encode() + b"\n"
    monkeypatch.setattr(
        hub.HfFileSystem,
        "open",
        lambda *_args, **_kwargs: io.BytesIO(payload),
    )
    source = SourceSpec(
        source_id="long-json-fixture",
        uri="hf://datasets/local/fixture",
        format="hf",
        adapter="document",
        stream="natural_documents",
        revision="a" * 40,
        terms=TermsReview(True, "local test", reviewed_by="test", reviewed_at="2026-10-23"),
        files=(SourceFile("mixed.json"),),
    )
    assert list(iter_source_rows(source)) == [(0, {"text": text})]


def test_hf_json_reader_retries_after_rate_limit_without_repeating_rows(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    import megatoken.data.sources as sources

    first = {"text": "first"}
    second = {"text": "x" * (70 << 10) + "🛰"}
    payload = (json.dumps(first) + "\n" + json.dumps(second, ensure_ascii=False) + "\n").encode()
    calls = []
    sleeps = []

    class RateLimited(Exception):
        def __init__(self):
            self.response = SimpleNamespace(status_code=429, headers={"Retry-After": "17"})

    class FailsAfterPrefix(io.BytesIO):
        def read(self, size=-1):
            if self.tell() >= 64 << 10:
                raise RateLimited()
            return super().read(size)

    def local_open(_filesystem, path, mode="rb", **kwargs):
        calls.append(path)
        return FailsAfterPrefix(payload) if len(calls) == 1 else io.BytesIO(payload)

    monkeypatch.setattr(hub.HfFileSystem, "open", local_open)
    monkeypatch.setattr(sources.time, "sleep", sleeps.append)
    monkeypatch.setattr(sources.random, "uniform", lambda *_args: 0.0)
    source = SourceSpec(
        source_id="rate-limited-json-fixture",
        uri="hf://datasets/local/fixture",
        format="hf",
        adapter="document",
        stream="natural_documents",
        revision="a" * 40,
        terms=TermsReview(True, "local test", reviewed_by="test", reviewed_at="2026-10-23"),
        files=(SourceFile("mixed.json"),),
    )
    assert list(iter_source_rows(source)) == [(0, first), (1, second)]
    assert sleeps == [17.0]
    assert len(calls) == 2


def test_hf_retry_delay_accepts_http_date_retry_after():
    import megatoken.data.sources as sources

    retry_at = datetime.now(UTC) + timedelta(seconds=25)
    error = RuntimeError("rate limit")
    error.response = SimpleNamespace(
        status_code=429, headers={"Retry-After": format_datetime(retry_at, usegmt=True)}
    )
    delay = sources._hub_retry_delay(error, 1)
    assert delay is not None and 0 <= delay <= 25


@pytest.mark.parametrize("name", ("ReadError", "ConnectError", "NetworkError"))
def test_hf_retry_delay_recognizes_httpx_network_errors(name):
    import megatoken.data.sources as sources

    error_type = type(name, (Exception,), {})
    delay = sources._hub_retry_delay(error_type(), 1)
    assert delay is not None and delay >= 5


@pytest.mark.parametrize(
    "payload",
    (b"[{},,{}]", b"[{ },]", b"[{}] trailing"),
    ids=("duplicate-comma", "trailing-comma", "trailing-content"),
)
def test_hf_json_array_rejects_malformed_separators_and_trailing_content(payload):
    import megatoken.data.sources as sources

    with pytest.raises(ValueError):
        list(sources._json_array_objects(iter((payload,)), "fixture.json"))


def test_hf_json_object_stream_requires_a_record_separator():
    import megatoken.data.sources as sources

    with pytest.raises(ValueError, match="whitespace between"):
        list(sources._json_object_stream(iter((b'{}{"next":true}',)), "fixture.json"))


def test_parquet_rows_too_large_to_batch_together_come_back_one_at_a_time(
    parquet_source, monkeypatch
):
    import megatoken.data.sources as sources

    rows, source = parquet_source
    monkeypatch.setattr(sources, "PARQUET_BATCH_BYTES", 1)
    expected_rows = list(enumerate(rows[:6]))
    assert list(iter_source_rows(source)) == expected_rows
    assert list(iter_source_rows(source, start_row=3)) == expected_rows[3:]
