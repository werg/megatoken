"""Immutable-input admission reuse never reuses model state or changed evidence."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_semantic_v8_training_policy import prepared_v8_fixture

from megatoken.data import PreparedCorpus
from megatoken.generation.codec import ByteCodec
from megatoken.training.semantic_v8 import SemanticV8EpisodeAdmissionCache


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    document, codec, policy, output = prepared_v8_fixture(tmp_path, direct_source=True)
    with PreparedCorpus([str(output)], split="train") as corpus:
        episode, _ = corpus.fetch(0)
    import megatoken.data.native_v8 as native

    calls = []
    original = native.prepare_native_v8_training

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "prepare_native_v8_training", counted)
    options = {
        "codec": codec,
        "policy": policy,
        "registry": document["sidecar"]["splitIdentity"]["registry"],
    }
    return episode, options, calls


def test_adjacent_windows_reuse_complete_admission_but_return_independent_receipts(admitted):
    episode, options, calls = admitted
    cache = SemanticV8EpisodeAdmissionCache()
    first = cache.validate(episode, **options)
    first["sourceId"] = "caller-mutated-result"
    second = cache.validate(episode, **options)
    assert second["sourceId"] == "source-1"
    assert len(calls) == 1
    # A fresh model or process has no reusable admission result.
    assert SemanticV8EpisodeAdmissionCache().validate(episode, **options) == second
    assert len(calls) == 2


@pytest.mark.parametrize(
    "changed", ["source-proof", "physical-record", "registry", "policy", "geometry", "split"]
)
def test_changed_admission_inputs_never_hit_an_existing_positive_entry(admitted, changed):
    episode, options, _ = admitted
    cache = SemanticV8EpisodeAdmissionCache()
    cache.validate(episode, **options)
    replacement = episode
    changed_options = dict(options)
    if changed == "source-proof":
        annotations = json.loads(episode.supervision.annotations_json)
        annotations["semantic_execution"]["sourceAdmission"]["sourceIdentityDigest"] = "0" * 64
        replacement = replace(
            episode,
            supervision=replace(episode.supervision, annotations_json=json.dumps(annotations)),
        )
    elif changed == "physical-record":
        records = list(episode.records)
        records[0] = replace(records[0], payload=b"x" + records[0].payload[1:])
        replacement = replace(episode, records=tuple(records))
    elif changed == "registry":
        changed_options["registry"] = deepcopy(options["registry"])
        changed_options["registry"]["sources"][0]["source"] += " "
    elif changed == "policy":
        changed_options["policy"] = {**options["policy"], "consumerRevision": "3" * 40}
    elif changed == "geometry":
        changed_options["max_positions"] = 1
    else:
        changed_options["expected_split"] = "validation"
    with pytest.raises(ValueError):
        cache.validate(replacement, **changed_options)


def test_equal_codec_labels_do_not_reuse_admission_across_codec_instances(admitted):
    episode, options, calls = admitted
    cache = SemanticV8EpisodeAdmissionCache()
    cache.validate(episode, **options)
    replacement = ByteCodec.byte_fallback()
    assert replacement.fingerprint == options["codec"].fingerprint
    cache.validate(episode, **{**options, "codec": replacement})
    assert len(calls) == 2
