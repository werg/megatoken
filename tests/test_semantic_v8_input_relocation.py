"""Content-bound V8 resume across storage moves, with explicit fixture admission."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig


def fixture_codec(path, *, vocab_size):
    """Test-only byte vocabulary; this is not production tokenizer admission."""
    assert Path(path).read_bytes() == b"explicit byte-fallback relocation fixture\n"
    return ByteCodec.byte_fallback()


def assert_same_state(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            assert_same_state(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_same_state(a, b, f"{path}[{index}]")
    else:
        assert type(left) is type(right) and left == right, path


def test_real_v8_next_update_survives_input_relocation_in_a_fresh_process(tmp_path, monkeypatch):
    """Three actual updates: one initial and two independent continuations."""
    from test_semantic_v8_training_policy import prepared_v8_fixture

    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    _, _codec, policy, prepared = prepared_v8_fixture(tmp_path)
    tokenizer = tmp_path / "tokenizer.fixture"
    tokenizer.write_bytes(b"explicit byte-fallback relocation fixture\n")
    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", fixture_codec)
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec=str(tokenizer),
        datasets=[str(prepared)],
        output=str(tmp_path / "initial"),
        device="cpu",
        precision="fp32",
        max_steps=1,
        checkpoint_every=1,
        keep_checkpoints=1,
        microbatch_episodes=1,
        accumulate_scored=1,
        accumulate_ingested=1,
        max_accumulation_windows=1,
        allow_unverified_data=True,
        semantic_v8_policy=policy,
    )
    first = train(config)
    first_payload, first_metadata = load_checkpoint(first["checkpoint"])
    assert first_payload["global_step"] == 1
    baseline = train(replace(config, output=str(tmp_path / "baseline")), resume=first["checkpoint"])
    baseline_payload, baseline_metadata = load_checkpoint(baseline["checkpoint"])
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    relocated_data = relocated / "prepared"
    relocated_codec = relocated / "tokenizer.fixture"
    shutil.copytree(prepared, relocated_data)
    shutil.copyfile(tokenizer, relocated_codec)
    moved = replace(
        config,
        codec=str(relocated_codec),
        datasets=[str(relocated_data)],
        output=str(relocated / "continuation"),
    )
    config_path = relocated / "resume.json"
    config_path.write_text(json.dumps(asdict(moved)) + "\n")
    # The resumed process cannot accidentally fall back to the original paths.
    prepared.rename(tmp_path / "original-data-unavailable")
    tokenizer.rename(tmp_path / "original-codec-unavailable")
    assert not prepared.exists() and not tokenizer.exists()
    code = """
import json, sys, torch
from tests.test_semantic_v8_input_relocation import fixture_codec
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train
import megatoken.training.semantic_v8 as semantic_v8
torch.set_num_threads(1)
semantic_v8.load_semantic_v8_codec=fixture_codec
config=TrainConfig.from_dict(json.loads(open(sys.argv[1]).read()))
result=train(config,resume=sys.argv[2])
print(json.dumps({'checkpoint':result['checkpoint'],'global_step':result['global_step']}))
"""
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    process = subprocess.run(
        [sys.executable, "-c", code, str(config_path), first["checkpoint"]],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    (tmp_path / "relocated-process.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout.strip().splitlines()[-1])
    actual, metadata = load_checkpoint(result["checkpoint"])
    assert actual["global_step"] == baseline_payload["global_step"] == 2
    keys = set(baseline_payload) - {"train_config"}
    assert set(actual) - {"train_config"} == keys
    for key in sorted(keys):
        assert_same_state(baseline_payload[key], actual[key], key)
    baseline_bank = torch.load(
        Path(baseline["checkpoint"]) / baseline_metadata["transducer_checkpoint"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    relocated_bank = torch.load(
        Path(result["checkpoint"]) / metadata["transducer_checkpoint"]["file"],
        map_location="cpu",
        weights_only=True,
    )
    assert_same_state(baseline_bank, relocated_bank, "transducer")
    assert actual["train_config"] == asdict(moved)
    assert metadata["config_fingerprint"] != baseline_metadata["config_fingerprint"]
    for key in (
        "dataset_fingerprint",
        "codec_fingerprint",
        "semantic_v8_lineage_digest",
        "semantic_v8_position_digest",
    ):
        assert metadata[key] == baseline_metadata[key], key
    assert (
        actual["semantic_v8_position_exposure"]["previousDigest"]
        == first_payload["semantic_v8_position_exposure"]["digest"]
    )
    report = {
        "schema": "semantic-v8-input-relocation-evidence/1",
        "executedRevision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "device": "cpu",
            "precision": "fp32",
        },
        "scope": "One controlled source with injected source and byte-tokenizer admission; no production-tokenizer, source, hosted, release or learned-quality qualification",
        "optimizerUpdatesExecuted": 3,
        "trajectories": "one initial update; two separate next updates from the same checkpoint",
        "originalInputPathsAbsent": True,
        "freshProcessContinuation": True,
        "comparedPayloadFields": sorted(keys),
        "transducerStateEqual": True,
        "firstCheckpointMetadata": first_metadata,
        "baselineCheckpointMetadata": baseline_metadata,
        "relocatedCheckpointMetadata": metadata,
        "positionExposure": actual["semantic_v8_position_exposure"],
        "learnedSuccess": None,
    }
    (tmp_path / "relocation-proof.json").write_text(json.dumps(report, indent=2) + "\n")


def empty_prepared_root(root, codec, chunking, *, identity):
    from megatoken.data import DedupConfig, DedupStore
    from megatoken.data.storage import export_prepared

    with DedupStore(root.with_suffix(".sqlite"), DedupConfig()) as store:
        store.freeze()
        export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=[],
            pipeline_fingerprint=identity,
            chunking=chunking,
        )
    return root


@pytest.fixture(scope="module")
def input_bound_checkpoint(tmp_path_factory):
    """One actual update shared by the refusal probes; never alter that checkpoint."""
    from test_semantic_v8_training_policy import prepared_v8_fixture

    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.data.chunking import ChunkingConfig
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("input-bound-checkpoint")
    _, codec, policy, prepared = prepared_v8_fixture(root)
    manifest = json.loads((prepared / "manifest.json").read_text())
    chunking = ChunkingConfig.from_identity(manifest["chunking"], codec)
    empty = empty_prepared_root(root / "empty", codec, chunking, identity="empty-v8-input-fixture")
    tokenizer = root / "tokenizer.fixture"
    tokenizer.write_bytes(b"explicit byte-fallback relocation fixture\n")
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec=str(tokenizer),
        datasets=[str(prepared), str(empty)],
        output=str(root / "initial"),
        device="cpu",
        precision="fp32",
        max_steps=1,
        checkpoint_every=1,
        keep_checkpoints=1,
        microbatch_episodes=1,
        accumulate_scored=1,
        accumulate_ingested=1,
        max_accumulation_windows=1,
        allow_unverified_data=True,
        semantic_v8_policy=policy,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(semantic_v8, "load_semantic_v8_codec", fixture_codec)
        first = train(config)
    payload, metadata = load_checkpoint(first["checkpoint"])
    assert payload["global_step"] == 1
    assert len(payload["semantic_training_v8"]["preparedManifestFingerprints"]) == 2
    assert payload["semantic_v8_position_exposure"]["optimizerUpdates"] == 1
    return config, first["checkpoint"], prepared, empty, codec, chunking


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("resealed_manifest", "admitted input contents"),
        ("changed_index", "index checksum"),
        ("changed_shard", "[Ss]hard checksum"),
        ("changed_registry", "registry"),
        ("missing_empty_root", "admitted input contents"),
        ("extra_empty_root", "admitted input contents"),
        ("duplicate_root", "listed twice"),
        ("changed_policy", "selected policy"),
        ("changed_codec", "different codec"),
        ("changed_window", "configuration changed max_window_records"),
        ("changed_loss", "configuration changed loss_weights"),
    ],
)
def test_relocated_input_substitution_refuses_before_model_construction(
    tmp_path, monkeypatch, input_bound_checkpoint, mutation, message
):
    import megatoken.training.engine as engine
    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.io import content_hash

    config, checkpoint, original, original_empty, codec, chunking = input_bound_checkpoint
    prepared, empty, tokenizer = (
        tmp_path / "prepared",
        tmp_path / "empty",
        tmp_path / "tokenizer.fixture",
    )
    shutil.copytree(original, prepared)
    shutil.copytree(original_empty, empty)
    shutil.copyfile(config.codec, tokenizer)
    changed = replace(
        config,
        codec=str(tokenizer),
        datasets=[str(prepared), str(empty)],
        output=str(tmp_path / "refused"),
    )
    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", fixture_codec)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "resealed_manifest":
        manifest["pipeline_fingerprint"] += "-altered"
        manifest["fingerprint"] = content_hash(
            {key: value for key, value in manifest.items() if key != "fingerprint"}
        )
        manifest_path.write_text(json.dumps(manifest) + "\n")
    elif mutation in {"changed_index", "changed_shard"}:
        relative = (
            manifest["index"]["path"]
            if mutation == "changed_index"
            else manifest["shards"][0]["path"]
        )
        with (prepared / relative).open("ab") as output:
            output.write(b"changed bytes without a matching manifest\n")
    elif mutation == "changed_registry":
        registry_path = prepared / "semantic-v8-families.json"
        registry = json.loads(registry_path.read_text())
        registry["seed"] += 1
        registry_path.write_text(json.dumps(registry) + "\n")
    elif mutation == "missing_empty_root":
        changed = replace(changed, datasets=[str(prepared)])
    elif mutation == "extra_empty_root":
        extra = empty_prepared_root(
            tmp_path / "extra", codec, chunking, identity="different-empty-input"
        )
        changed = replace(changed, datasets=[str(prepared), str(empty), str(extra)])
    elif mutation == "duplicate_root":
        changed = replace(changed, datasets=[str(prepared), str(empty), str(empty)])
    elif mutation == "changed_policy":
        changed = replace(
            changed, semantic_v8_policy={**config.semantic_v8_policy, "consumerRevision": "c" * 40}
        )
    elif mutation == "changed_codec":
        altered = ByteCodec(tuple(bytes([value]) for value in reversed(range(256))))
        assert altered.fingerprint != codec.fingerprint
        monkeypatch.setattr(
            semantic_v8, "load_semantic_v8_codec", lambda _path, *, vocab_size: altered
        )
    elif mutation == "changed_window":
        changed = replace(changed, max_window_records=config.max_window_records - 1)
    elif mutation == "changed_loss":
        changed = replace(
            changed,
            loss_weights={**config.loss_weights, "lexical": config.loss_weights["lexical"] + 0.25},
        )
    constructions = []

    def forbidden_model(*args, **kwargs):
        constructions.append(True)
        raise AssertionError("Changed input reached model construction")

    monkeypatch.setattr(engine, "build_model", forbidden_model)
    with pytest.raises(ValueError, match=message):
        engine.train(changed, resume=checkpoint)
    assert constructions == []
    assert not (Path(changed.output) / "checkpoints").exists()


def test_relocated_production_tokenizer_does_not_accept_unpinned_bytes(tmp_path):
    from megatoken.training.semantic_v8 import load_semantic_v8_codec

    source = tmp_path / "tokenizer.json"
    source.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum|SHA|sha256|tokenizer"):
        load_semantic_v8_codec(source, vocab_size=256)


def test_relocation_cannot_remove_the_explicit_v8_training_policy(input_bound_checkpoint):
    config, *_ = input_bound_checkpoint
    with pytest.raises(ValueError, match="Semantic training requires a frozen experiment"):
        replace(config, semantic_v8_policy=None)
