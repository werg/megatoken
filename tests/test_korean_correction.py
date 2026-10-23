import json

import pytest

from megatoken.data.adapters import AdapterContext, adapt_row
from megatoken.data.korean_correction import (
    CORRECTED_SOURCE_ID,
    KOREAN_SOURCE_ID,
    KOREAN_SOURCE_REVISION,
    build_correction_corpus,
    correct_korean_row,
    korean_correction_source,
)
from megatoken.data.schema import Episode, canonical_json, digest_bytes
from megatoken.data.sources import SourceFile, SourceSpec, TermsReview, load_source_manifest


def _source() -> SourceSpec:
    return SourceSpec(
        source_id=KOREAN_SOURCE_ID,
        uri="hf://datasets/heegyu/glaive-function-calling-v2-ko",
        format="hf",
        adapter="simulated_tool_roles",
        stream="tool_use",
        revision=KOREAN_SOURCE_REVISION,
        terms=TermsReview(
            approved=True,
            license="apache-2.0",
            license_url="https://huggingface.co/datasets/heegyu/glaive-function-calling-v2-ko",
            reviewed_by="reviewer",
            reviewed_at="2026-10-23",
            redistribute="permissive",
            redistribute_basis="publisher states Apache-2.0",
        ),
        files=(SourceFile("train.json"),),
        field_map={},
        options={"quality": {"min_scored_positions": 1}, "allow_pending_final_action": True},
    )


def _row(*, function_description=None):
    if function_description is None:
        function_description = json.dumps(
            {
                "name": "lookup",
                "description": "Find an item.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        )
    return {
        "system_message": "Use the available functions when they help.",
        "function_description": function_description,
        "conversations": [
            {"role": "user", "content": "Find the item named blue."},
            {
                "role": "function-call",
                "content": json.dumps({"name": "lookup", "arguments": {"query": "blue"}}),
            },
            {"role": "function-response", "content": "The item is available."},
            {"role": "assistant", "content": "I found the item."},
        ],
        "conversations_ko": [
            {"role": "user", "content": "파란색이라고 이름 붙은 항목을 찾아 주세요."},
            {
                "role": "function-call",
                "content": json.dumps({"name": "lookup", "arguments": {"query": "파란색"}}),
            },
            {"role": "function-response", "content": "항목을 찾았습니다."},
            {"role": "assistant", "content": "항목을 찾았습니다."},
        ],
    }


def test_correction_uses_korean_field_and_merges_exact_english_family_keys():
    source = _source()
    corrected = korean_correction_source(source)
    assert corrected.source_id == CORRECTED_SOURCE_ID
    assert corrected.field_map["messages"] == "conversations_ko"
    assert source.field_map == {}

    row = _row()
    episode = correct_korean_row(row, 27, original_source=source, corrected_source=corrected)

    assert episode.lineage.source_id == CORRECTED_SOURCE_ID
    assert episode.lineage.revision == source.revision
    assert episode.lineage.source_row == 27
    assert episode.lineage.raw_sha256 == digest_bytes(canonical_json(row).encode("utf-8"))
    assert "Find the item named blue." not in b"".join(r.payload for r in episode.records).decode()
    assert (
        "파란색이라고 이름 붙은 항목을 찾아 주세요."
        in b"".join(r.payload for r in episode.records).decode()
    )
    assert episode.records[0].role == "system"
    assert episode.records[0].payload == row["system_message"].encode()
    assert episode.records[0].policy_mask is False
    assert (
        sum(
            record.role == "assistant" and record.action_id is not None
            for record in episode.records
        )
        == 1
    )
    assert sum(record.result_of is not None for record in episode.records) == 1
    assert json.loads(episode.tools_json)[0]["name"] == "lookup"
    assert len(episode.meridians) == len(episode.records)

    english = adapt_row(
        row,
        source.adapter,
        AdapterContext(
            source_id=source.source_id,
            revision=source.revision,
            uri=source.uri,
            stream=source.stream,
            row_index=27,
            field_map=source.field_map,
            options=source.options,
        ),
    )
    korean = adapt_row(
        row,
        corrected.adapter,
        AdapterContext(
            source_id=corrected.source_id,
            revision=corrected.revision,
            uri=corrected.uri,
            stream=corrected.stream,
            row_index=27,
            field_map=corrected.field_map,
            options=corrected.options,
        ),
    )
    assert set(english.lineage.family_keys).issubset(episode.lineage.family_keys)
    assert set(korean.lineage.family_keys).issubset(episode.lineage.family_keys)


@pytest.mark.parametrize("description", ["not-json", json.dumps({"description": "missing name"})])
def test_unusable_function_description_is_retained_as_unscored_document(description):
    episode = correct_korean_row(
        _row(function_description=description), 0, original_source=_source()
    )

    context = [record for record in episode.records if record.role == "document"]
    assert len(context) == 1
    assert context[0].payload == description.encode("utf-8")
    assert context[0].policy_mask is False
    assert json.loads(episode.tools_json) == []


def test_build_writes_distinct_checksummed_native_addendum(tmp_path):
    source = _source()
    output = tmp_path / "correction"
    english_invalid = _row()
    english_invalid["conversations"] = []
    receipt = build_correction_corpus(
        [(0, _row()), (1, _row(function_description="not-json")), (2, english_invalid)],
        original_source=source,
        output_dir=output,
    )

    assert receipt.input_rows == 3
    assert receipt.emitted_rows == 2
    assert receipt.rejected_rows == 1
    assert receipt.rejected_row_indices == (2,)
    assert receipt.rows_linked_to_english_family == 2
    assert receipt.rows_with_system_context == 2
    assert receipt.rows_with_normalized_tools == 1
    assert receipt.rows_with_verbatim_tool_context == 1
    sources = load_source_manifest(output / "source-manifest.json")
    assert len(sources) == 1
    sources[0].validate_local_files()
    episodes = [
        Episode.from_dict(json.loads(line))
        for line in (output / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(episodes) == 2
    assert episodes[0].lineage.source_id == CORRECTED_SOURCE_ID
    assert (
        json.loads((output / "receipt.json").read_text())["output_sha256"] == receipt.output_sha256
    )
    with pytest.raises(FileExistsError):
        build_correction_corpus([], original_source=source, output_dir=output)
