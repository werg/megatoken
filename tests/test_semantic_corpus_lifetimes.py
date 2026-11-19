"""Production preparation retains complete private audits without retaining every candidate."""

import copy
import gc
import hashlib
import json
import os
import weakref
from contextlib import contextmanager
from pathlib import Path

import pytest

from megatoken import io as durable_io
from megatoken.data import semantic_corpus
from megatoken.data.host_operations import CONTRACT_DIGEST
from megatoken.data.semantic_admission import native_admission
from megatoken.data.semantic_contract import CONTRACT
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec

PRODUCER = "a" * 40
CONSUMER = "b" * 40


class PrivateProbe(dict):
    pass


@pytest.fixture
def production_codec():
    path = Path(os.environ.get("MEGATOKEN_TOKENIZER_JSON", ".validation/tokenizer.json"))
    if not path.is_file():
        pytest.skip("Dedicated semantic CI downloads the exact production tokenizer")
    return ByteCodec.from_tokenizer_json(path, expected_sha256=LING_TOKENIZER_SHA256)


def admitted_document(codec, index):
    document = copy.deepcopy(CONTRACT["fixtures"]["envelope"]["valid"][0])
    for row in document["public"]["timeline"]:
        row["session"] = f"session{index}"
    # These are distinct retained-source cases, not renamed copies. Lifecycle
    # acknowledgements add enough words to enable near-duplicate detection;
    # unique source comments keep this lifetime test independent of that gate.
    program = next(row for row in document["public"]["timeline"] if row["kind"] == "program")
    program["source"] += "\n// " + " ".join(f"case{index}_word{word}" for word in range(24))
    program["digest"] = hashlib.sha256(program["source"].encode()).hexdigest()
    for row in document["public"]["timeline"]:
        if row["kind"] == "invocation":
            row["sourceDigest"] = program["digest"]
    document["sidecar"].update(
        admission="admitted",
        representation="compact",
        selectionOrigin="certified_teacher",
        referenceStatus="agreed",
        catalogDigest=CONTRACT_DIGEST,
        sourceLineage=f"streaming-fixture-{index}",
        sourceProvenance={"license": "test-fixture", "trustTier": "repository-fixture"},
    )
    document["sidecar"]["nativeAdmission"] = native_admission(
        document, codec=codec, consumer_revision=CONSUMER
    )
    return document


def unit_source_receipt(document):
    # This unit fixture isolates preparation and lifetime behavior. Actual
    # source checks are exercised by the separately pinned native corpus CI.
    return {
        "schema": semantic_corpus.SOURCE_SCHEMA,
        "episodeDigest": semantic_corpus.public_digest(document),
        "producerRevision": PRODUCER,
        "catalogDigest": CONTRACT_DIGEST,
        "referenceStatus": "agreed",
        "optimizerSteps": 0,
        "journalValidated": True,
        "replayValidated": True,
        "sourceValidated": True,
    }


def prepare(candidates, output, codec):
    return semantic_corpus.prepare_semantic_corpus(
        candidates,
        output,
        codec=codec,
        producer_revision=PRODUCER,
        consumer_revision=CONSUMER,
        source_verifier=unit_source_receipt,
    )


def test_preparation_releases_old_private_evidence_and_publishes_complete_audit_first(
    tmp_path, monkeypatch, production_codec
):
    codec = production_codec
    probes = []
    original_rows = semantic_corpus.DedupStore.accepted_rows

    def bounded_rows(store):
        previous = []
        for row in original_rows(store):
            if len(previous) >= 2:
                gc.collect()
                assert previous[-2]() is None
            current = PrivateProbe(dict(row))
            previous.append(weakref.ref(current))
            yield current

    monkeypatch.setattr(semantic_corpus.DedupStore, "accepted_rows", bounded_rows)

    def candidates():
        for index in range(6):
            gc.collect()
            # One prior candidate can be held by the caller's loop variables;
            # earlier private evidence must already have been released.
            if index >= 2:
                assert probes[index - 2]() is None
            document = admitted_document(codec, index)
            probe = PrivateProbe(index=index, payload="x" * 65536)
            probes.append(weakref.ref(probe))
            document["sidecar"]["privateLifetimeProbe"] = probe
            yield document

    original_export = semantic_corpus.export_prepared

    def export_after_audit(store, root, **kwargs):
        audits = json.loads((root / "semantic-audit.json").read_text())
        assert [row["supervision"]["privateLifetimeProbe"]["index"] for row in audits] == list(
            range(6)
        )
        assert "semantic-audit.json" not in kwargs["sidecars"]
        assert not (root / "manifest.json").exists()
        return original_export(store, root, **kwargs)

    monkeypatch.setattr(semantic_corpus, "export_prepared", export_after_audit)
    manifest, report = prepare(candidates(), tmp_path, codec)
    assert report["counts"]["admitted"] == 6
    assert report["accepted"] == 6
    assert manifest["complete"] is True
    assert json.loads((tmp_path / "semantic-report.json").read_text()) == report


def test_audit_write_failure_does_not_publish_partial_audit_or_manifest(
    tmp_path, monkeypatch, production_codec
):
    codec = production_codec
    previous = b'["previous complete evidence"]\n'
    (tmp_path / "semantic-audit.json").write_bytes(previous)
    original_writer = durable_io.atomic_binary_writer

    @contextmanager
    def failing_writer(path):
        with original_writer(path) as handle:

            class FailingOutput:
                writes = 0

                def write(self, value):
                    self.writes += 1
                    if self.writes == 10:
                        raise OSError("simulated audit storage failure")
                    return handle.write(value)

            yield FailingOutput()

    monkeypatch.setattr(durable_io, "atomic_binary_writer", failing_writer)
    with pytest.raises(OSError, match="audit storage failure"):
        prepare([admitted_document(codec, 0)], tmp_path, codec)
    assert (tmp_path / "semantic-audit.json").read_bytes() == previous
    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob(".semantic-audit.json.*"))


def test_all_refusals_retain_report_and_empty_complete_audit(tmp_path):
    codec = ByteCodec.byte_fallback()
    with pytest.raises(ValueError, match="No admitted native episodes"):
        prepare([{"status": "unsupported", "reason": "outside declared profile"}], tmp_path, codec)
    assert json.loads((tmp_path / "semantic-audit.json").read_text()) == []
    report = json.loads((tmp_path / "semantic-report.json").read_text())
    assert report["counts"] == {"attempted": 1, "unsupported": 1}
    assert report["refusals"][0]["reason"] == "outside declared profile"
    assert not (tmp_path / "manifest.json").exists()
