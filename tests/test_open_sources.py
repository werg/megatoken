"""The open text-source manifests: pinned, licence-recorded, and mapped to the stage plan's streams."""

from __future__ import annotations

from pathlib import Path

import pytest

from megatoken.data.adapters import AdapterContext, adapt_row
from megatoken.data.open_media import STREAMS
from megatoken.data.quality import QualityReject, validate_row_filters
from megatoken.data.recipes import (
    STREAM_PRODUCERS,
    resolve_stream_producer,
    stage_recipe,
)
from megatoken.data.sources import load_source_manifest
from megatoken.generation.codec import ByteCodec

CONFIGS = Path(__file__).parents[1] / "configs/data"
SOURCES = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.open.json")}

ROWS = {
    "fineweb-edu-10bt": {
        "text": "Photosynthesis lets plants make food from light.",
        "id": "<urn:uuid:1>",
        "url": "http://example.org/photosynthesis",
    },
    "stackv2-edu-0000": {
        "text": "def f():\n    return 1\n",
        "metadata": {"repo_name": "owner/repo", "path": "/pkg/mod.py", "language": "Python"},
    },
    "cosmopedia-stanford": {"prompt": "Write a textbook unit", "text": "Unit 1. Cells ..."},
    "openmathinstruct2-1m": {
        "problem": "What is 2+2?",
        "generated_solution": "Adding gives 4. \\boxed{4}",
        "expected_answer": "4",
    },
    "openthoughts-114k": {
        "system": "Think carefully.",
        "conversations": [
            {"from": "user", "value": "Hi"},
            {"from": "assistant", "value": "<|begin_of_thought|>x<|end_of_thought|> Hello"},
        ],
    },
    "oasst-top1": {
        "text": "<|im_start|>user\nHola<|im_end|>\n<|im_start|>assistant\nHola!<|im_end|>\n"
    },
    "smoltalk-new": {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        "source": "smol-rewrite",
    },
    "opencodeinstruct": {"input": "Write f", "output": "```python\ndef f(): pass\n```"},
}


def test_every_source_is_pinned_licensed_and_approved_by_the_maintainer():
    for source in (*SOURCES.values(), *load_source_manifest(CONFIGS / "sources.open-mb.json")):
        assert len(source.revision) == 40 and source.format == "hf"
        assert source.terms.license not in ("", "unreviewed"), source.source_id
        assert source.terms.license_url.startswith("https://"), source.source_id
        # Approval is the maintainer's decision, recorded with who and when.
        assert source.terms.approved and source.terms.reviewed_by and source.terms.reviewed_at
        source.terms.require_approval()


def test_streams_cover_the_stage_plan():
    text = {s.stream for s in SOURCES.values()}
    for manifest in (
        "sources.agentic.json",
        "sources.agentic-coding.json",
        "sources.later-stages.json",
    ):
        text |= {s.stream for s in load_source_manifest(CONFIGS / manifest)}
    covered = (
        text
        | STREAMS
        | set(STREAM_PRODUCERS)
        | {
            "long_document_reading",
            "base_reading",
            "collection_reading",
        }
    )
    assert set(stage_recipe("P1").mixture) - covered == set()
    mb_text = {s.stream for s in load_source_manifest(CONFIGS / "sources.open-mb.json")}
    assert set(stage_recipe("MB").mixture) <= mb_text | STREAMS | set(STREAM_PRODUCERS)


def test_procedural_streams_resolve_to_real_local_producers():
    for stream, path in STREAM_PRODUCERS.items():
        producer = resolve_stream_producer(stream)
        assert callable(producer), (stream, path)
        assert ":" in path and not path.startswith(("http:", "https:"))
    downloaded_streams = {source.stream for source in SOURCES.values()}
    for manifest in (
        "sources.agentic.json",
        "sources.agentic-coding.json",
        "sources.later-stages.json",
        "sources.open-mb.json",
    ):
        downloaded_streams.update(
            source.stream for source in load_source_manifest(CONFIGS / manifest)
        )
    assert not {"dynamic_execution", "duplex_execution"} & downloaded_streams
    assert STREAM_PRODUCERS["duplex_execution"].endswith(
        "megatoken.data.execution_traces:prepare_traces"
    )


@pytest.mark.parametrize("source_id", sorted(ROWS))
def test_fixture_row_becomes_an_episode_in_the_declared_stream(source_id):
    source = SOURCES[source_id]
    context = AdapterContext(
        source_id=source.source_id,
        revision=source.revision,
        uri=source.uri,
        stream=source.stream,
        field_map=source.field_map,
        options=source.options,
        codec=ByteCodec.byte_fallback(),
        max_record_tokens=1024,
    )
    validate_row_filters(ROWS[source_id], source.options.get("row_filters", []))
    episode = adapt_row(ROWS[source_id], source.adapter, context)
    assert episode.stream == source.stream
    assert any(record.policy_mask for record in episode.records)
    if source_id.startswith("stackv2"):
        assert episode.lineage.family_id == "owner/repo"
    if source_id == "fineweb-edu-10bt":
        assert episode.lineage.family_id == "http://example.org/photosynthesis"


def test_smoltalk_keeps_only_the_apache_subsets():
    rules = SOURCES["smoltalk-new"].options["row_filters"]
    validate_row_filters({"source": "smol-summarize"}, rules)
    with pytest.raises(QualityReject):
        validate_row_filters({"source": "numina-cot-100k"}, rules)
