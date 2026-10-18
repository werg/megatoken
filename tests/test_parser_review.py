"""Parser failures become reproducible repair tasks, never silent training rows."""

from __future__ import annotations

import ast
import json
import sqlite3
from contextlib import ExitStack, closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatoken.data.chunking import ChunkingConfig
from megatoken.data.dedup import DedupConfig
from megatoken.data.prepare import prepare_sources
from megatoken.data.review import ParserReviewPolicy, ParserReviewQueue
from megatoken.data.schema import Episode, Lineage, canonical_json, make_records
from megatoken.data.sources import TermsReview, pin_local_source


class ByteCodec:
    fingerprint = "review-byte-codec-v1"

    def encode(self, payload):
        return list(payload)

    def decode(self, ids):
        return bytes(ids)


def make_source(tmp_path, rows, *, name="source", adapter="document", options=None):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    source = pin_local_source(
        path,
        source_id=name,
        adapter=adapter,
        stream="natural_documents",
        terms=TermsReview(True, "fixture", reviewed_by="test", reviewed_at="2026-10-18"),
    )
    return replace(source, options=options or {})


def capture_fixture(queue, source, row, *, parser_identity=None):
    codec = ByteCodec()
    config = ChunkingConfig()
    identity = parser_identity or config.identity(codec) | {"prior_parser_fixture": "before-fix"}
    episode = Episode(
        "fixture",
        make_records(row["text"].encode(), logical_id="body", role="document", policy_mask=True),
        Lineage("fixture", "revision", "family"),
    )
    sample_id = queue.capture(
        source=source,
        row_index=0,
        raw_row=row,
        adapted_row=row,
        episode=episode,
        parser_identity=identity,
        chunking=config,
        policy=ParserReviewPolicy(),
        reasons=("native_parser_error",),
    )
    return sample_id, episode


def test_review_classification_preserves_lenient_coverage_and_immutable_native_data():
    policy = ParserReviewPolicy()
    assert (
        policy.classify(
            [
                "grammar_unavailable",
                "recovered_syntax",
                "syntax_error",
                "unclosed_fence",
                "chat_no_template",
                "external_reference_unread",
                "invalid_utf8",
                "forced_grapheme_cut",
                "forced_token_cut",
            ]
        )
        == ()
    )
    assert policy.classify(
        ["native_parser_error", "node_budget_exhausted", "chat_ambiguous_template"]
    ) == (
        "chat_ambiguous_template",
        "native_parser_error",
        "node_budget_exhausted",
    )
    strict = ParserReviewPolicy(
        review_incomplete_syntax=True, review_recovered_syntax=True, review_unknown_languages=True
    )
    assert strict.classify(["grammar_unavailable", "unclosed_fence", "syntax_error"]) == (
        "grammar_unavailable",
        "syntax_error",
        "unclosed_fence",
    )
    assert not ParserReviewPolicy(enabled=False).classify(["native_parser_error"])
    # Native tapes/records do not enter parser policy at all.
    assert policy.reasons(None, adapter="native") == ()


def test_every_emitted_resource_budget_enters_review_by_default():
    """Audit parser-owned diagnostic emitters, including worker failure wrappers."""
    import megatoken.data.parsing.types as parser_types

    emitted = set()
    for path in Path(parser_types.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            position = 3 if name == "_failure_scan" else 0
            if name not in {"Diagnostic", "diagnostic", "_diagnostic", "_failure_scan"}:
                continue
            if len(node.args) <= position:
                continue
            emitted.update(
                item.value
                for item in ast.walk(node.args[position])
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and item.value.endswith(("_budget", "_budget_exhausted"))
            )
    assert {
        "prose_parse_byte_budget",
        "prose_evidence_budget",
        "prose_boundary_budget",
        "prose_diagnostic_budget",
        "prose_protected_region_budget",
        "native_parser_message_budget",
        "depth_budget_exhausted",
        "node_budget_exhausted",
    } <= emitted
    assert set(ParserReviewPolicy().classify(tuple(emitted))) == emitted


@pytest.mark.parametrize(
    ("payload", "limits", "expected"),
    [
        (b"A longer sentence.", {"max_parse_bytes": 4}, "prose_parse_byte_budget"),
        (b"One.\n\nTwo.\n\nThree.", {"max_regions": 1}, "prose_evidence_budget"),
        (
            b"One sentence. Two sentences. Three sentences.",
            {"max_nodes": 2},
            "prose_boundary_budget",
        ),
        (b"\xff x \xfe y \xfd", {"max_regions": 1}, "prose_diagnostic_budget"),
        (
            b"https://one.test https://two.test https://three.test",
            {"max_regions": 1},
            "prose_protected_region_budget",
        ),
    ],
)
def test_actual_prose_resource_failures_are_reviewable(payload, limits, expected):
    from megatoken.data.parsing.prose import analyze_prose
    from megatoken.data.parsing.types import ParseLimits

    parsed = analyze_prose(payload, refine=True, limits=ParseLimits(**limits))
    codes = tuple(item.code for item in parsed.diagnostics)
    assert expected in codes
    assert expected in ParserReviewPolicy().classify(codes)


def test_severe_syntax_coverage_and_invalid_evidence_are_hard_but_local_recovery_is_not():
    codes = (
        "low_syntax_coverage",
        "invalid_parser_evidence",
        "icu_refinement_failed",
        "recovered_syntax",
        "syntax_error",
        "missing_syntax",
        "grammar_unavailable",
    )
    assert ParserReviewPolicy().classify(codes) == (
        "icu_refinement_failed",
        "invalid_parser_evidence",
        "low_syntax_coverage",
    )


def test_capture_is_idempotent_exact_and_exported_as_an_agent_task(tmp_path):
    row = {
        "text": "你好.\r\n<script>const x = '</tag>';</script>",
        "metadata": {"label": "audit only"},
    }
    source = make_source(tmp_path, [row])
    with ParserReviewQueue(tmp_path / "review.sqlite") as queue:
        sample_id, episode = capture_fixture(queue, source, row)
        repeated, _ = capture_fixture(queue, source, row)
        assert sample_id == repeated
        assert queue.summary()["pending"] == 1
        capture = queue.get(sample_id)["capture"]
        assert json.loads(capture["raw_row_json"]) == row
        assert Episode.from_dict(capture["episode"]) == episode
        bundle = queue.export_tasks(tmp_path / "tasks")
        assert len(bundle["tasks"]) == 1
        assert (tmp_path / "tasks" / sample_id / "raw-row.json").read_bytes() == canonical_json(
            row
        ).encode()
        assert (tmp_path / "tasks" / sample_id / "payload-00000.bin").read_bytes() == row[
            "text"
        ].encode()
        instructions = (tmp_path / "tasks" / sample_id / "AGENT_TASK.md").read_text()
        assert "untrusted data" in instructions
        assert row["text"] not in instructions


def test_claims_are_exclusive_expire_and_require_the_owner(tmp_path):
    row = {"text": "A normal sentence."}
    source = make_source(tmp_path, [row])
    path = tmp_path / "review.sqlite"
    with ParserReviewQueue(path) as first, ParserReviewQueue(path) as second:
        sample_id, _ = capture_fixture(first, source, row)
        assert first.claim("agent-a")["sample_id"] == sample_id
        first.renew(sample_id, "agent-a", lease_seconds=3600)
        assert second.claim("agent-b") is None
        with pytest.raises(ValueError, match="claiming owner"):
            second.release(sample_id, "agent-b")
        with pytest.raises(ValueError, match="claiming owner"):
            second.renew(sample_id, "agent-b")
        with pytest.raises(ValueError, match="claiming owner"):
            second.replay(sample_id, ByteCodec(), owner="agent-b", fix_ref="PR #123")
        first.connection.execute(
            "UPDATE parser_review SET lease_until=0 WHERE sample_id=?", (sample_id,)
        )
        first.connection.commit()
        assert second.claim("agent-b")["owner"] == "agent-b"
        second.release(sample_id, "agent-b")
        assert first.summary()["pending"] == 1


def test_replay_requires_new_parser_and_passes_the_normal_adapter(tmp_path):
    row = {"text": "A normal sentence.", "family_id": "original-parent"}
    source = make_source(tmp_path, [row])
    codec = ByteCodec()
    with ParserReviewQueue(tmp_path / "review.sqlite") as queue:
        same_id, _ = capture_fixture(
            queue, source, row, parser_identity=ChunkingConfig().identity(codec)
        )
        queue.claim("agent", sample_id=same_id)
        with pytest.raises(ValueError, match="changed parser identity"):
            queue.replay(same_id, codec, owner="agent", fix_ref="new-label-only")
        sample_id, _ = capture_fixture(queue, source, row)
        queue.claim("agent", sample_id=sample_id)
        with pytest.raises(ValueError, match="cannot change"):
            queue.replay(
                sample_id,
                codec,
                owner="agent",
                fix_ref="fix",
                chunking=ChunkingConfig(target_tokens=1),
            )
        receipt = queue.replay(sample_id, codec, owner="agent", fix_ref="commit:parser-fix")
        assert receipt["success"]
        assert receipt["rebuild_required"]
        assert queue.get(sample_id)["status"] == "verified"
        assert queue.summary()["verified"] == 1
        with pytest.raises(ValueError, match="claiming owner"):
            queue.replay(sample_id, codec, owner="agent", fix_ref="again")


@pytest.mark.parametrize("shared_connection", [False, True])
def test_replay_lease_check_and_finalization_cannot_be_reclaimed_between_writes(
    tmp_path, monkeypatch, shared_connection
):
    import megatoken.data.review as review_module

    row = {"text": "The exact captured source."}
    source = make_source(tmp_path, [row])
    path = tmp_path / "race.sqlite"
    clock = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(review_module, "time", SimpleNamespace(time=lambda: clock.value))
    with ExitStack() as stack:
        database = (
            stack.enter_context(closing(sqlite3.connect(path))) if shared_connection else path
        )
        first = stack.enter_context(ParserReviewQueue(database))
        second = stack.enter_context(ParserReviewQueue(path))
        second.connection.execute("PRAGMA busy_timeout=0")
        sample_id, _ = capture_fixture(first, source, row)
        first.claim("agent-a", sample_id=sample_id, lease_seconds=1)
        reclaim_results = []

        def reclaim_before_attempt_insert(action, table, *_):
            if action == sqlite3.SQLITE_INSERT and table == "parser_review_attempts":
                # SQLite prepares this INSERT after the final lease SELECT.
                # A deferred transaction used to permit a reclaim in this gap.
                clock.value = 1002.0
                try:
                    reclaim_results.append(second.claim("agent-b", sample_id=sample_id))
                except sqlite3.OperationalError as error:
                    reclaim_results.append(str(error))
            return sqlite3.SQLITE_OK

        first.connection.set_authorizer(reclaim_before_attempt_insert)
        try:
            receipt = first.replay(sample_id, ByteCodec(), owner="agent-a", fix_ref="parser-fix")
        finally:
            first.connection.set_authorizer(None)
        assert reclaim_results == ["database is locked"]
        assert receipt["success"]
        assert not first.connection.in_transaction
        evidence = second.get(sample_id)
        assert (evidence["status"], evidence["owner"]) == ("verified", "agent-a")
        assert (
            second.connection.execute("SELECT count(*) FROM parser_review_attempts").fetchone()[0]
            == 1
        )
        assert second.claim("agent-b", sample_id=sample_id) is None


def test_replay_rejects_a_lease_reclaimed_while_the_parser_runs(tmp_path, monkeypatch):
    import megatoken.data.review as review_module

    row = {"text": "The exact captured source."}
    source = make_source(tmp_path, [row])
    clock = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(review_module, "time", SimpleNamespace(time=lambda: clock.value))
    path = tmp_path / "reclaimed.sqlite"
    with ParserReviewQueue(path) as first, ParserReviewQueue(path) as second:
        sample_id, _ = capture_fixture(first, source, row)
        first.claim("agent-a", sample_id=sample_id, lease_seconds=1)
        original_adapter = review_module.adapt_row

        def reclaim_during_parse(*args, **kwargs):
            episode = original_adapter(*args, **kwargs)
            clock.value = 1002.0
            assert second.claim("agent-b", sample_id=sample_id)["owner"] == "agent-b"
            return episode

        monkeypatch.setattr(review_module, "adapt_row", reclaim_during_parse)
        with pytest.raises(ValueError, match="ownership expired or changed"):
            first.replay(sample_id, ByteCodec(), owner="agent-a", fix_ref="parser-fix")
        assert not first.connection.in_transaction
        evidence = second.get(sample_id)
        assert (evidence["status"], evidence["owner"]) == ("claimed", "agent-b")
        assert (
            second.connection.execute("SELECT count(*) FROM parser_review_attempts").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("caller_transaction", [False, True])
def test_failed_replay_finalization_rolls_back_the_attempt_and_preserves_caller_work(
    tmp_path, caller_transaction
):
    row = {"text": "The exact captured source."}
    source = make_source(tmp_path, [row])
    path = tmp_path / "rollback.sqlite"
    with ExitStack() as stack:
        database = (
            stack.enter_context(closing(sqlite3.connect(path))) if caller_transaction else path
        )
        queue = stack.enter_context(ParserReviewQueue(database))
        sample_id, _ = capture_fixture(queue, source, row)
        queue.claim("agent", sample_id=sample_id)
        queue.connection.executescript("""
            CREATE TABLE caller_work(value TEXT);
            CREATE TRIGGER reject_verification BEFORE UPDATE OF status ON parser_review
            WHEN NEW.status='verified'
            BEGIN SELECT RAISE(ABORT,'fixture verification failure'); END;
        """)
        if caller_transaction:
            queue.connection.execute("INSERT INTO caller_work VALUES('preserve me')")
        with pytest.raises(sqlite3.IntegrityError, match="fixture verification failure"):
            queue.replay(sample_id, ByteCodec(), owner="agent", fix_ref="parser-fix")
        assert queue.get(sample_id)["status"] == "claimed"
        assert (
            queue.connection.execute("SELECT count(*) FROM parser_review_attempts").fetchone()[0]
            == 0
        )
        assert queue.connection.in_transaction is caller_transaction
        if caller_transaction:
            assert (
                queue.connection.execute("SELECT value FROM caller_work").fetchone()[0]
                == "preserve me"
            )
            queue.connection.rollback()
            assert queue.connection.execute("SELECT count(*) FROM caller_work").fetchone()[0] == 0


def test_successful_shared_replay_does_not_commit_the_callers_deferred_transaction(tmp_path):
    row = {"text": "The exact captured source."}
    source = make_source(tmp_path, [row])
    path = tmp_path / "shared-replay.sqlite"
    with closing(sqlite3.connect(path)) as connection, ParserReviewQueue(path) as observer:
        queue = ParserReviewQueue(connection)
        sample_id, _ = capture_fixture(queue, source, row)
        queue.claim("agent", sample_id=sample_id)
        connection.execute("BEGIN")
        receipt = queue.replay(sample_id, ByteCodec(), owner="agent", fix_ref="parser-fix")
        assert receipt["success"]
        assert connection.in_transaction
        assert queue.get(sample_id)["status"] == "verified"
        assert observer.get(sample_id)["status"] == "claimed"
        assert (
            observer.connection.execute("SELECT count(*) FROM parser_review_attempts").fetchone()[0]
            == 0
        )
        connection.rollback()
        assert queue.get(sample_id)["status"] == "claimed"
        assert connection.execute("SELECT count(*) FROM parser_review_attempts").fetchone()[0] == 0


def test_replay_does_not_waive_quality_or_source_filters(tmp_path):
    row = {"text": "small"}
    source = make_source(tmp_path, [row], options={"quality": {"min_tokens": 30}})
    with ParserReviewQueue(tmp_path / "review.sqlite") as queue:
        sample_id, _ = capture_fixture(queue, source, row)
        queue.claim("agent", sample_id=sample_id)
        receipt = queue.replay(sample_id, ByteCodec(), owner="agent", fix_ref="parser-fix")
        assert not receipt["success"]
        assert "quality:length" in receipt["error"]
        assert queue.get(sample_id)["status"] == "claimed"
        assert queue.summary()["verified"] == 0
    source = make_source(
        tmp_path,
        [row],
        name="filtered",
        options={"row_filters": [{"path": "allowed", "op": "eq", "value": True}]},
    )
    with ParserReviewQueue(tmp_path / "filtered.sqlite") as queue:
        sample_id, _ = capture_fixture(queue, source, row)
        queue.claim("agent", sample_id=sample_id)
        receipt = queue.replay(sample_id, ByteCodec(), owner="agent", fix_ref="parser-fix")
        assert not receipt["success"]
        assert "source_filter:" in receipt["error"]


def test_capture_checksums_and_bound_identity_detect_sample_rewriting(tmp_path):
    row = {"text": "The original sample."}
    source = make_source(tmp_path, [row])
    with ParserReviewQueue(tmp_path / "review.sqlite") as queue:
        sample_id, _ = capture_fixture(queue, source, row)
        captured = queue.connection.execute(
            "SELECT capture_json FROM parser_review WHERE sample_id=?", (sample_id,)
        ).fetchone()[0]
        changed = captured.replace("The original sample.", "Rewritten source!!!")
        queue.connection.execute(
            "UPDATE parser_review SET capture_json=? WHERE sample_id=?", (changed, sample_id)
        )
        queue.connection.commit()
        with pytest.raises(ValueError, match="checksum"):
            queue.get(sample_id)


def test_prepare_queues_hard_framing_without_accepting_or_losing_ancestry(tmp_path):
    source = make_source(
        tmp_path,
        [
            {"text": "This is unframed text.", "family_id": "family-a"},
            {
                "text": "<|im_start|>user\nQuestion?<|im_end|><|im_start|>assistant\nAnswer.<|im_end|>",
                "family_id": "family-b",
            },
        ],
        adapter="serialized_chat",
        options={"chat_template": "chatml"},
    )
    root = tmp_path / "prepared"
    with prepare_sources(
        [source], root, ByteCodec(), dedup=DedupConfig(train_fraction=1, validation_fraction=0)
    ) as dataset:
        assert len(dataset) == 1
        assert dataset[0].lineage.family_id == "family-b"
        assert dataset.manifest["parser_review"]["pending"] == 1
        assert dataset.manifest["census"]["parser_review_rows"] == 1
    with ParserReviewQueue(root / "preparation.sqlite") as queue:
        assert queue.summary()["pending"] == 1
        assert queue.connection.execute("SELECT 1 FROM families WHERE key='family-a'").fetchone()
        item = queue.list_items()[0]
        assert (
            json.loads(queue.get(item["sample_id"])["capture"]["raw_row_json"])["text"]
            == "This is unframed text."
        )
    with pytest.raises(ValueError, match="different pipeline/source identity"):
        prepare_sources([source], root, ByteCodec())


@pytest.mark.parametrize("adapter", ["serialized_chat", "document"])
def test_masked_chat_over_frame_budget_enters_review_before_mask_validation(tmp_path, adapter):
    from megatoken.data.parsing.types import ParseLimits

    row = {
        "text": "".join(
            f"<|im_start|>{role}\n{body}<|im_end|>"
            for role, body in (
                ("user", "First question?"),
                ("assistant", "First answer."),
                ("user", "Second question?"),
                ("assistant", "Second answer."),
            )
        ),
        "train_turns_mask": [False, True, False, True],
        "family_id": "original-prompt-family",
    }
    source = make_source(tmp_path, [row], adapter=adapter)
    root = tmp_path / "prepared"
    config = ChunkingConfig(limits=ParseLimits(max_regions=2))
    with prepare_sources([source], root, ByteCodec(), chunking=config) as dataset:
        assert len(dataset) == 0
        assert dataset.manifest["parser_review"]["pending"] == 1
    with ParserReviewQueue(root / "preparation.sqlite") as queue:
        capture = queue.get(queue.list_items()[0]["sample_id"])["capture"]
        assert {"chat_frame_limit", "chat_unframed_tail"} <= set(capture["reasons"])
        assert json.loads(capture["raw_row_json"]) == row
        assert capture["episode"] is None
        assert queue.connection.execute("SELECT next_row FROM source_progress").fetchone()[0] == 1
        assert queue.connection.execute(
            "SELECT 1 FROM families WHERE key='original-prompt-family'"
        ).fetchone()


@pytest.mark.parametrize("adapter", ["serialized_chat", "document"])
@pytest.mark.parametrize(
    "tail",
    [
        "Unfinished answer",
        "```python\nprint(1)\n<|im_end|>"
        "<|im_start|>user\nSecond question?<|im_end|>"
        "<|im_start|>assistant\nSecond answer.<|im_end|>",
    ],
    ids=["truncated-body", "unfinished-fence"],
)
def test_masked_chat_with_unfinished_body_enters_preparation_review(tmp_path, adapter, tail):
    from megatoken.data.parsing.chat import parse_chat

    row = {
        "text": ("<|im_start|>user\nFirst question?<|im_end|><|im_start|>assistant\n" + tail),
        "train_turns_mask": [False, True, False, True],
        "family_id": "original-prompt-family",
    }
    # Transport truncation or a literal shield can leave fewer recognized
    # frames than the original four-turn mask describes.
    parsed = parse_chat(row["text"].encode())
    assert parsed.detected and len(parsed.frames) < len(row["train_turns_mask"])
    assert any(item.code == "chat_incomplete_body" for item in parsed.diagnostics)
    source = make_source(tmp_path, [row], adapter=adapter)
    root = tmp_path / "prepared"
    with prepare_sources([source], root, ByteCodec()) as dataset:
        assert len(dataset) == 0
        assert dataset.manifest["parser_review"]["pending"] == 1
    with ParserReviewQueue(root / "preparation.sqlite") as queue:
        capture = queue.get(queue.list_items()[0]["sample_id"])["capture"]
        assert capture["reasons"] == ["chat_incomplete_body"]
        assert json.loads(capture["raw_row_json"]) == row
        assert capture["episode"] is None
        assert queue.connection.execute("SELECT next_row FROM source_progress").fetchone()[0] == 1
        assert queue.connection.execute(
            "SELECT 1 FROM families WHERE key='original-prompt-family'"
        ).fetchone()


@pytest.mark.parametrize(
    ("tail", "allow_incomplete"),
    [
        ("<|im_start|>assistant", False),
        ("<|im_start|>assistant\nunfinished answer", True),
    ],
)
def test_incomplete_chat_opt_in_and_generation_headers_remain_unscored_context(
    tmp_path, tail, allow_incomplete
):
    row = {
        "text": "<|im_start|>user\nQuestion?<|im_end|>" + tail,
        "train_turns_mask": [False, True],
    }
    source = make_source(
        tmp_path,
        [row],
        adapter="serialized_chat",
        options={"allow_incomplete_serialized": allow_incomplete},
    )
    with prepare_sources([source], tmp_path / "prepared", ByteCodec()) as dataset:
        assert len(dataset) == 1
        assert dataset.manifest["parser_review"]["pending"] == 0
        assert b"".join(record.payload for record in dataset[0].records) == row["text"].encode()
        assert not any(record.policy_mask for record in dataset[0].records)


def test_review_capture_and_source_cursor_commit_together_on_resume(tmp_path):
    rows = [{"text": "unframed", "family_id": f"family-{index}"} for index in range(3)]
    source = make_source(
        tmp_path, rows, adapter="serialized_chat", options={"chat_template": "chatml"}
    )
    root = tmp_path / "interrupted"
    with pytest.raises(InterruptedError):
        prepare_sources([source], root, ByteCodec(), _interrupt_after=1)
    with ParserReviewQueue(root / "preparation.sqlite") as queue:
        assert queue.summary()["total"] == 1
        assert queue.connection.execute("SELECT next_row FROM source_progress").fetchone()[0] == 1
    with prepare_sources([source], root, ByteCodec()) as dataset:
        assert len(dataset) == 0
        assert dataset.manifest["parser_review"]["total"] == 3
    with prepare_sources([source], root, ByteCodec()) as dataset:
        assert dataset.manifest["parser_review"]["total"] == 3


@pytest.mark.parametrize("adapter", ["serialized_chat", "document"])
def test_unparsed_benchmark_variant_retains_the_same_prompt_family(tmp_path, adapter):
    prefix = "<|im_start|>user\nShared question?<|im_end|><|im_start|>assistant\n"
    source = make_source(
        tmp_path,
        [
            {"text": prefix + "Incomplete benchmark", "benchmark_tags": ["fixture-benchmark"]},
            {"text": prefix + "A complete answer.<|im_end|>"},
        ],
        adapter=adapter,
    )
    with prepare_sources([source], tmp_path / "prepared", ByteCodec()) as dataset:
        assert len(dataset) == 0
        assert dataset.manifest["parser_review"]["pending"] == 1
        assert dataset.manifest["census"]["quarantined"] == 1


@pytest.mark.parametrize("adapter", ["serialized_chat", "document"])
@pytest.mark.parametrize(
    "payload",
    [
        "<|im_start|>tool\nResult<|im_end|>",
        "<|start|>assistant to=python<|message|>1+1<|call|>",
        "<|im_start|>assistant\n<tool_call>{}</tool_call><|im_end|>",
    ],
)
def test_unsupported_serialized_dependencies_are_queued_without_inventing_actions(
    tmp_path, adapter, payload
):
    row = {"text": payload}
    source = make_source(tmp_path, [row], adapter=adapter)
    root = tmp_path / "prepared"
    with prepare_sources([source], root, ByteCodec()) as dataset:
        assert len(dataset) == 0
        assert dataset.manifest["parser_review"]["pending"] == 1
    with ParserReviewQueue(root / "preparation.sqlite") as queue:
        capture = queue.get(queue.list_items()[0]["sample_id"])["capture"]
        assert capture["reasons"] == ["chat_missing_dependency_ids"]
        assert json.loads(capture["raw_row_json"]) == row
        assert capture["episode"] is None


def test_shared_queue_capture_rolls_back_with_the_caller_transaction(tmp_path):
    row = {"text": "Original."}
    source = make_source(tmp_path, [row])
    connection = sqlite3.connect(tmp_path / "shared.sqlite")
    connection.row_factory = sqlite3.Row
    queue = ParserReviewQueue(connection)
    with pytest.raises(RuntimeError), connection:
        capture_fixture(queue, source, row)
        raise RuntimeError("crash before source cursor commit")
    assert queue.summary()["total"] == 0
    connection.close()


def test_fault_diagnostics_quarantine_then_require_a_clean_full_replay(tmp_path, monkeypatch):
    import megatoken.data.prepare as preparation
    import megatoken.data.review as review_module

    row = {"text": "Ordinary source text.", "family_id": "parent"}
    source = make_source(tmp_path, [row])
    original_adapter = preparation.adapt_row
    original_identity = ChunkingConfig.identity
    revision = {"value": 0}

    def identity(config, codec):
        return original_identity(config, codec) | {"fixture_parser_revision": revision["value"]}

    def fault_adapter(*args, **kwargs):
        episode = original_adapter(*args, **kwargs)
        return replace(
            episode,
            records=tuple(
                replace(record, chunk_diagnostics=("native_parser_error",))
                for record in episode.records
            ),
        )

    monkeypatch.setattr(ChunkingConfig, "identity", identity)
    monkeypatch.setattr(preparation, "adapt_row", fault_adapter)
    root = tmp_path / "prepared"
    with prepare_sources([source], root, ByteCodec()) as dataset:
        assert len(dataset) == 0
        assert dataset.manifest["parser_review"]["pending"] == 1
    with ParserReviewQueue(root / "preparation.sqlite") as queue:
        item = queue.claim("repair-agent")
        capture = item["capture"]
        assert capture["episode"]["records"][0]["chunk_diagnostics"] == ["native_parser_error"]
        monkeypatch.setattr(review_module, "adapt_row", fault_adapter)
        revision["value"] = 1
        failed = queue.replay(
            item["sample_id"], ByteCodec(), owner="repair-agent", fix_ref="candidate-fix"
        )
        assert not failed["success"]
        assert "native_parser_error" in failed["error"]
        monkeypatch.setattr(review_module, "adapt_row", original_adapter)
        revision["value"] = 2
        success = queue.replay(
            item["sample_id"], ByteCodec(), owner="repair-agent", fix_ref="correct-fix"
        )
        assert success["success"]
        assert queue.summary()["verified"] == 1
    # A queue resolution cannot alter the old manifest or admit bytes into old shards.
    from megatoken.data.storage import PreparedDataset

    with PreparedDataset(root, split=None) as frozen:
        assert len(frozen) == 0
        assert frozen.manifest["parser_review"]["pending"] == 1
    with pytest.raises(ValueError, match="different pipeline/source identity"):
        prepare_sources([source], root, ByteCodec())


def test_native_records_are_not_reparsed_or_queued_for_old_parser_diagnostics(tmp_path):
    records = make_records(
        b"A preserved native record.", logical_id="native", role="document", policy_mask=True
    )
    records = tuple(
        replace(record, chunk_diagnostics=("native_parser_error",)) for record in records
    )
    episode = Episode("native-episode", records, Lineage("native-source", "r", "native-parent"))
    source = make_source(tmp_path, [episode.to_dict()], adapter="native")
    with prepare_sources([source], tmp_path / "native-prepared", ByteCodec()) as dataset:
        assert len(dataset) == 1
        assert dataset[0].records == records
        assert dataset.manifest["parser_review"]["total"] == 0
