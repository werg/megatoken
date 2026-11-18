"""Bound private evidence retention without changing its JSON contents.

Storage integration tests replace external admission gates; native/source
semantics are exercised separately by the harness and producer integration.
"""

import gc
import json
import weakref

import pytest

from megatoken.data import semantic_corpus
from megatoken.data.semantic_harness_v5 import adapt_selected_harness_v5
from megatoken.generation.codec import ByteCodec
from megatoken.io import atomic_json_array_writer, sha256_file, write_json
from test_semantic_harness_v5 import fixture


@pytest.mark.parametrize(
    "items",
    [[], [{"unicode": "café\n雪", "nested": [[], {"b": 2, "a": [True, None]}]}],
     [None, 1, "quoted\"value", [1, 2], {"z": -0.0, "a": False}]],
)
def test_streamed_array_matches_existing_json_bytes_and_hashes(tmp_path, items):
    streamed, expected = tmp_path / "streamed.json", tmp_path / "expected.json"
    with atomic_json_array_writer(streamed) as append:
        for item in items:
            append(item)
        assert not streamed.exists()
    write_json(expected, items)
    assert streamed.read_bytes() == expected.read_bytes()
    assert sha256_file(streamed) == sha256_file(expected)


def test_interrupted_or_invalid_append_preserves_complete_evidence(tmp_path):
    path = tmp_path / "audit.json"
    write_json(path, [{"previous": "complete"}])
    previous = path.read_bytes()
    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_json_array_writer(path) as append:
            append({"first": "valid"})
            raise RuntimeError("interrupted")
    assert path.read_bytes() == previous
    with pytest.raises(ValueError, match="incomplete JSON"):
        with atomic_json_array_writer(path) as append:
            # Catching a failed item cannot accidentally publish partial JSON.
            with pytest.raises(ValueError):
                append({"bad": float("nan")})
    assert path.read_bytes() == previous
    assert sorted(item.name for item in tmp_path.iterdir()) == ["audit.json"]


def controlled_storage_pipeline(monkeypatch):
    codec = ByteCodec.byte_fallback()
    episode, _ = adapt_selected_harness_v5(fixture(), allow_local=True, codec=codec)
    monkeypatch.setattr(semantic_corpus, "native_admission",
                        lambda document, **_: {"native": document["sidecar"]["number"]})
    monkeypatch.setattr(semantic_corpus, "verify_admission", lambda *args, **kwargs: True)
    monkeypatch.setattr(semantic_corpus, "adapt_selected_harness_v5", lambda *args, **kwargs: (episode, None))
    return codec, episode


def test_preparation_releases_previous_private_sidecars_and_keeps_full_audit(tmp_path, monkeypatch):
    codec, episode = controlled_storage_pipeline(monkeypatch)
    references = []

    class TrackedSidecar(dict):
        pass

    def candidates():
        for number in range(8):
            gc.collect()
            # The caller's current loop variable may retain its preceding row;
            # a growing audit list would keep every previous row alive here.
            assert sum(reference() is not None for reference in references) <= 1
            side = TrackedSidecar(admission="admitted", representation="compact", number=number,
                                  privateEvidence=("evidence-" + str(number)) * 8192)
            references.append(weakref.ref(side))
            yield {"public": {}, "sidecar": side}
            del side

    manifest, report = semantic_corpus.prepare_semantic_corpus(
        candidates(), tmp_path, codec=codec, producer_revision="a" * 40,
        consumer_revision="b" * 40,
        source_verifier=lambda document: {"source": document["sidecar"]["number"]},
    )
    assert manifest["complete"] is True
    assert report["counts"]["admitted"] == 8
    assert report["accepted"] == 1  # Repeated controlled episode, eight audits.
    gc.collect()
    assert all(reference() is None for reference in references)
    expected = [dict(episodeId=episode.episode_id, source={"source": number}, native={"native": number},
                     supervision=dict(admission="admitted", representation="compact", number=number,
                                      privateEvidence=("evidence-" + str(number)) * 8192))
                for number in range(8)]
    assert json.loads((tmp_path / "semantic-audit.json").read_text()) == expected
    write_json(tmp_path / "expected-audit.json", expected)
    assert sha256_file(tmp_path / "semantic-audit.json") == sha256_file(tmp_path / "expected-audit.json")


def test_preparation_failure_does_not_publish_partial_audit_or_manifest(tmp_path, monkeypatch):
    codec, _ = controlled_storage_pipeline(monkeypatch)

    def interrupted():
        yield {"sidecar": {"admission": "admitted", "representation": "compact", "number": 0}}
        raise RuntimeError("candidate source interrupted")

    with pytest.raises(RuntimeError, match="source interrupted"):
        semantic_corpus.prepare_semantic_corpus(interrupted(), tmp_path, codec=codec,
            producer_revision="a" * 40, consumer_revision="b" * 40, source_verifier=lambda _: {})
    assert not (tmp_path / "semantic-audit.json").exists()
    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob(".semantic-audit.json.*"))
