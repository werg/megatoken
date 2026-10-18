"""Real optional reader integration; all input data is generated locally."""

import json
from dataclasses import replace

import pytest

from megatoken.data import DedupConfig, TermsReview, pin_local_source, prepare_sources
from megatoken.data.sources import iter_source_rows
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
    tmp_path, parquet_source
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
    with pytest.raises(InterruptedError):
        prepare_sources([source], output, codec, reject_invalid=True, _interrupt_after=3, **options)
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
    tmp_path, parquet_source, monkeypatch
):
    datasets = pytest.importorskip("datasets")
    rows, source = parquet_source
    load_dataset = datasets.load_dataset
    calls = []

    def local_loader(**kwargs):
        # Replace only Hub resolution; exercise datasets' actual Parquet iterator
        # and our logical resume offset. No remote source or terms are consulted.
        calls.append(kwargs)
        return load_dataset(
            "parquet",
            data_files=kwargs["data_files"],
            split=kwargs["split"],
            streaming=kwargs["streaming"],
            cache_dir=str(tmp_path / "datasets-cache"),
        )

    monkeypatch.setattr(datasets, "load_dataset", local_loader)
    hf_source = replace(
        source,
        format="hf",
        uri="hf://datasets/local/fixture",
        revision="a" * 40,
        subset="fixture-subset",
    )
    assert list(iter_source_rows(hf_source, start_row=3)) == list(enumerate(rows[3:6], 3))
    assert calls == [
        {
            "path": "local/fixture",
            "name": "fixture-subset",
            "split": "train",
            "revision": "a" * 40,
            "streaming": True,
            "data_files": {"train": [source.files[0].path]},
        }
    ]
