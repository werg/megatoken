"""Operational parser commands exercise the same preparation and replay contracts."""

import json

from test_data_cli import invoke
from test_parser_review import ByteCodec, capture_fixture, make_source

from megatoken.cli import main
from megatoken.data.review import ParserReviewQueue


def test_inspection_preserves_nested_source_and_reports_the_plan(tmp_path):
    payload = (
        b"# A chapter\r\n\r\nA paragraph introduces the code.\r\n\r\n"
        b"```python\r\ndef example():\r\n    return '<p>Hello world.</p>'\r\n```\r\n"
    )
    path = tmp_path / "chapter.md"
    path.write_bytes(payload)
    output = tmp_path / "inspection.json"
    result = invoke(
        [
            "chunks",
            "inspect",
            "--input",
            path,
            "--include-payload",
            "--chat-template",
            "none",
            "--output",
            output,
        ]
    )
    report = json.loads(output.read_text())
    assert result["bytes"] == len(payload)
    assert b"".join(bytes.fromhex(c["payload_hex"]) for c in report["chunks"]) == payload
    assert set(report["languages"]) >= {"markdown", "python", "html"}
    assert report["identity"]["config"]["target_tokens"] == 512
    assert report["identity"]["config"]["soft_max_tokens"] == 768
    assert report["planner"]["evaluated_edges"] > 0


def test_coverage_command_reports_real_installed_grammars(tmp_path):
    output = tmp_path / "coverage.json"
    result = invoke(["chunks", "coverage", "--output", output])
    report = json.loads(output.read_text())
    assert result["validated"] and report["validated"]
    assert result["grammar_count"] >= 173
    assert result["failures"] == {}
    assert {"python", "rust", "cpp", "julia", "typescript", "markdown"} <= set(report["grammars"])


def test_prepare_loads_policy_then_applies_stage_capacity(tmp_path):
    source = tmp_path / "source.jsonl"
    invoke(["demo", "--output", source, "--count", 2])
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "target_tokens: 192\nsoft_max_tokens: 288\nhard_max_tokens: 512\n"
        "tree_distance_budget: 6\nlimits:\n  max_depth: 8\n"
    )
    result = invoke(
        [
            "prepare",
            "--demo",
            source,
            "--output",
            tmp_path / "prepared",
            "--chunking-config",
            policy,
            "--max-record-tokens",
            128,
        ]
    )
    config = result["chunking"]["config"]
    assert (config["target_tokens"], config["soft_max_tokens"], config["hard_max_tokens"]) == (
        128,
        128,
        128,
    )
    assert config["limits"]["max_depth"] == 8
    assert config["tree_distance_budget"] == 6
    assert result["parser_review"]["total"] == 0


def test_review_commands_preserve_a_failed_repair_receipt_and_return_failure(
    tmp_path, monkeypatch, capsys
):
    row = {"text": "Small captured text."}
    source = make_source(tmp_path, [row], options={"quality": {"min_tokens": 100}})
    path = tmp_path / "review.sqlite"
    with ParserReviewQueue(path) as queue:
        sample_id, _ = capture_fixture(queue, source, row)
    task = invoke(["chunks", "review", "claim", "--queue", path, "--owner", "agent-a"])
    assert task["task"]["sample_id"] == sample_id
    invoke(
        [
            "chunks",
            "review",
            "renew",
            "--queue",
            path,
            "--owner",
            "agent-a",
            "--sample",
            sample_id,
            "--lease-seconds",
            3600,
        ]
    )
    exported = invoke(
        [
            "chunks",
            "review",
            "export",
            "--queue",
            path,
            "--output",
            tmp_path / "tasks",
        ]
    )
    assert len(exported["tasks"]) == 1
    monkeypatch.setattr("megatoken.data.cli._codec", lambda _: ByteCodec())
    receipt_path = tmp_path / "replay.json"
    status = main(
        list(
            map(
                str,
                [
                    "--json-output",
                    receipt_path,
                    "chunks",
                    "review",
                    "replay",
                    "--queue",
                    path,
                    "--owner",
                    "agent-a",
                    "--sample",
                    sample_id,
                    "--codec",
                    "fixture",
                    "--fix-ref",
                    "commit:parser-amendment",
                ],
            )
        )
    )
    capsys.readouterr()
    receipt = json.loads(receipt_path.read_text())
    assert status == 1 and not receipt["success"]
    assert "quality:length" in receipt["error"]
    with ParserReviewQueue(path) as queue:
        assert queue.get(sample_id)["status"] == "claimed"
    invoke(
        [
            "chunks",
            "review",
            "release",
            "--queue",
            path,
            "--owner",
            "agent-a",
            "--sample",
            sample_id,
        ]
    )
    listed = invoke(["chunks", "review", "list", "--queue", path, "--status", "pending"])
    assert listed["summary"]["pending"] == 1


def test_tabular_inspection_exposes_row_parser_through_public_cli(tmp_path):
    payload = b'name,note\r\nAlice,"Hello,\r\nworld"\r\nBob,"Other"\r\n'
    source = tmp_path / "records.csv"
    source.write_bytes(payload)
    output = tmp_path / "records.json"
    invoke(
        [
            "chunks",
            "inspect",
            "--input",
            source,
            "--content-kind",
            "csv",
            "--parser-backend",
            "fallback",
            "--chat-template",
            "none",
            "--include-payload",
            "--output",
            output,
        ]
    )
    report = json.loads(output.read_text())
    assert "delimited-records-v1:csv" in report["parsers"]
    assert b"".join(bytes.fromhex(chunk["payload_hex"]) for chunk in report["chunks"]) == payload
