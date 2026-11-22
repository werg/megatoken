"""Release trust boundaries, with one explicitly controlled real checkpoint."""

import copy
import json
import shutil
import sqlite3
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.io import content_hash, sha256_file
from megatoken.training.config import TrainConfig
from scripts import verify_semantic_v8_release as release
from tests.test_semantic_launch_v8 import two_split_supply as two_split_supply
from tests.test_semantic_v8_training_policy import POLICY, prepared_v8_fixture


def descriptor_fixture(root):
    for name in (
        "source-qualification",
        "tokenizer",
        "prepared/000",
        "training/checkpoint",
        "serving",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    for name in (
        "source-qualification/source-1.admitted.json",
        "tokenizer/tokenizer.json",
        "training/recipe.json",
    ):
        (root / name).write_text("{}\n")
    return {
        "schema": release.DESCRIPTOR_SCHEMA,
        "producerRevision": POLICY["producerRevision"],
        "consumerRevision": POLICY["consumerRevision"],
        "contractDigest": release.CONTRACT_DIGEST,
        "fixtureDigest": release.FIXTURE_DIGEST,
        "interoperabilityDigest": release.INTEROPERABILITY_DIGEST,
        "roles": {
            "qualification": "source-qualification",
            "tokenizer": "tokenizer/tokenizer.json",
            "prepared": [
                {
                    "path": "prepared/000",
                    "sourceArtifacts": [
                        {"caseId": "case-1", "path": "source-qualification/source-1.admitted.json"}
                    ],
                }
            ],
            "checkpoint": "training/checkpoint",
            "recipe": "training/recipe.json",
            "served": "serving",
        },
    }


def test_release_descriptor_and_hidden_member_inventory(tmp_path):
    descriptor = descriptor_fixture(tmp_path)
    lock = tmp_path / "source-qualification/.campaign.lock"
    lock.touch()
    paths = release.validate_descriptor(descriptor, tmp_path)
    assert paths["prepared"] == tmp_path / "prepared/000"
    rows = release.bundle_inventory(tmp_path)
    assert any(row["path"] == "source-qualification/.campaign.lock" for row in rows)
    assert release.bundle_inventory(tmp_path) == rows


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../outside",
        "/absolute",
        "x/../y",
        "x/./y",
        "x//y",
        "x/",
        "x\\y",
        "C:x",
        "x\x00y",
    ],
)
def test_release_paths_refuse_escapes_and_aliases(name):
    with pytest.raises(ValueError, match="canonical relative"):
        release.canonical_relative(name)


def test_release_symlinks_refused_in_role_tree_and_ancestors(tmp_path):
    real = tmp_path / "real"
    descriptor = descriptor_fixture(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink ancestor"):
        release.checked_root(alias / "prepared")
    (real / "source-qualification/link").symlink_to(real / "training/recipe.json")
    with pytest.raises(ValueError, match="symlink"):
        release.bundle_inventory(real)
    descriptor["roles"]["prepared"][0]["sourceArtifacts"][0]["path"] = "source-qualification/link"
    with pytest.raises(ValueError, match="symlink"):
        release.validate_descriptor(descriptor, real)


@pytest.mark.parametrize(
    "mutation",
    ["field", "contract", "two-roots", "duplicate-case", "outside-source", "nested-role"],
)
def test_release_descriptor_refuses_ambiguous_membership(tmp_path, mutation):
    value = descriptor_fixture(tmp_path)
    if mutation == "field":
        value["unknown"] = True
    elif mutation == "contract":
        value["contractDigest"] = "0" * 64
    elif mutation == "two-roots":
        value["roles"]["prepared"] *= 2
    elif mutation == "duplicate-case":
        value["roles"]["prepared"][0]["sourceArtifacts"] *= 2
    elif mutation == "outside-source":
        value["roles"]["prepared"][0]["sourceArtifacts"][0]["path"] = "training/recipe.json"
    else:
        value["roles"]["checkpoint"] = "prepared/000"
    with pytest.raises(ValueError):
        release.validate_descriptor(value, tmp_path)


def recipe_fixture():
    return {
        "purpose": "diagnostic",
        "phase": "SC",
        "model": "tiny",
        "codec": "tokenizer/tokenizer.json",
        "datasets": ["prepared/000"],
        "max_steps": 1,
        "semantic_v8_policy": POLICY,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("codec", "../tokenizer.json"),
        ("datasets", ["prepared/*"]),
        ("model", "external-model.json"),
        ("packs", "external-packs.json"),
        ("text_anchors", "external-anchors.jsonl"),
        ("hindsight_targets", "external-hindsight.jsonl"),
        ("engram_import", "external-donor"),
        ("engram_imports", ["external-donor"]),
        ("use_stage_mixture", True),
        ("use_effort_mixture", True),
    ],
)
def test_release_recipe_refuses_unbound_assets_and_routes(tmp_path, key, value):
    descriptor = descriptor_fixture(tmp_path)
    raw = {**recipe_fixture(), key: value}
    with pytest.raises(ValueError):
        release.release_recipe(raw, descriptor, tmp_path)


def test_release_recipe_preserves_physical_provenance_and_binds_only_declared_inputs(tmp_path):
    descriptor = descriptor_fixture(tmp_path)
    raw = recipe_fixture()
    before = copy.deepcopy(raw)
    parsed = release.release_recipe(raw, descriptor, tmp_path)
    assert parsed.codec == str(tmp_path / "tokenizer/tokenizer.json")
    assert parsed.datasets == [str(tmp_path / "prepared/000")]
    assert raw == before
    wrong_policy = {**POLICY, "consumerRevision": "c" * 40}
    with pytest.raises(ValueError, match="policy differs"):
        release.release_recipe({**raw, "semantic_v8_policy": wrong_policy}, descriptor, tmp_path)


def test_release_source_membership_rejects_two_cases_for_one_physical_source(tmp_path):
    descriptor = descriptor_fixture(tmp_path)
    artifact = {
        "public": {"schema": "semantic-execution/8"},
        "sidecar": {
            "datasetBinding": {
                "sourceId": "source-1",
                "split": "train",
                "registryDigest": POLICY["registryDigest"],
            }
        },
    }
    path = tmp_path / descriptor["roles"]["prepared"][0]["sourceArtifacts"][0]["path"]
    path.write_text(json.dumps(artifact))
    second = path.with_name("different-case.admitted.json")
    second.write_bytes(path.read_bytes())
    descriptor["roles"]["prepared"][0]["sourceArtifacts"].append(
        {"caseId": "different-case", "path": second.relative_to(tmp_path).as_posix()}
    )
    with pytest.raises(ValueError, match="unique membership"):
        release.source_membership(descriptor, tmp_path, POLICY["registryDigest"])


def test_release_strict_json_refuses_duplicate_keys_and_nonfinite_numbers(tmp_path):
    path = tmp_path / "ambiguous.json"
    for raw in ('{"schema":1,"schema":2}', '{"value":NaN}'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            release.read_json(path)


@pytest.mark.parametrize(("left", "right"), [(False, 0), (True, 1), (1, 1.0), ([False], [0])])
def test_native_json_comparison_preserves_scalar_types(left, right):
    assert release.same_json({"value": left}, {"value": copy.deepcopy(left)})
    assert not release.same_json({"value": left}, {"value": right})


def checkpoint_metadata_fixture(directory):
    directory.mkdir()
    (directory / "state.pt").write_bytes(b"not loaded by metadata-only controls\n")
    (directory / "bank.pt").write_bytes(b"not loaded by metadata-only controls\n")
    metadata = {"transducer_checkpoint": {"file": "bank.pt"}}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize(
    "mutation", ["duplicate", "nonfinite", "latest-root", "latest-marker", "symlink-attachment"]
)
def test_checkpoint_companion_metadata_is_strict_and_never_resolves_pointers(tmp_path, mutation):
    directory = tmp_path / ("latest.json" if mutation == "latest-root" else "checkpoint")
    checkpoint_metadata_fixture(directory)
    if mutation == "duplicate":
        (directory / "metadata.json").write_text(
            '{"transducer_checkpoint":{"file":"bank.pt"},"transducer_checkpoint":{"file":"bank.pt"}}'
        )
    elif mutation == "nonfinite":
        (directory / "metadata.json").write_text(
            '{"transducer_checkpoint":{"file":"bank.pt"},"value":NaN}'
        )
    elif mutation == "latest-marker":
        (directory / "latest.json").write_text("{}")
    elif mutation == "symlink-attachment":
        (directory / "bank.pt").unlink()
        (directory / "bank.pt").symlink_to(directory / "state.pt")
    before = {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    with pytest.raises(ValueError):
        release.immutable_checkpoint_metadata(directory)
    assert {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    } == before


@pytest.mark.parametrize("name", ["manifest.json", "model_config.json", "semantic-v8-lineage.json"])
def test_served_companion_json_refuses_duplicate_fields_before_loading(tmp_path, monkeypatch, name):
    import megatoken.loading
    import megatoken.served  # Import before patching its captured load_trained dependency.

    checkpoint = tmp_path / "checkpoint"
    metadata = checkpoint_metadata_fixture(checkpoint)
    served = tmp_path / "served"
    served.mkdir()
    for file in ("manifest.json", "model_config.json", "semantic-v8-lineage.json"):
        (served / file).write_text('{"key":1,"key":1}' if file == name else "{}")

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Ambiguous JSON must fail before model or checkpoint loading")

    monkeypatch.setattr(megatoken.loading, "load_trained", forbidden_load)
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        release.served_consistency(
            {"checkpoint": checkpoint, "served": served}, {"model_config": {}}, metadata
        )


def test_reprepared_semantics_compare_every_physical_field_and_index_scope(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    document, codec, _policy, prepared = prepared_v8_fixture(first)
    _, _, _, regenerated = prepared_v8_fixture(
        second, document=copy.deepcopy(document), codec=codec
    )
    original = release.prepared_semantics(prepared)
    fresh = release.prepared_semantics(regenerated)
    sources = [{"sourceId": "source-1", "split": "train"}]
    proof = release.compare_prepared_semantics(original, fresh, sources)
    assert proof["sourceCount"] == proof["episodeCount"] == 1
    assert proof["recordCount"] > proof["policyRecordCount"] > 0
    for name in ("episodes", "scopePositions", "indexRows", "membership"):
        altered = copy.deepcopy(fresh)
        altered[name] = altered[name][:-1]
        with pytest.raises(ValueError, match="differ from fresh sources"):
            release.compare_prepared_semantics(original, altered, sources)
    altered = copy.deepcopy(fresh)
    altered["manifestSemantics"]["complete"] = 1
    with pytest.raises(ValueError, match="differ from fresh sources"):
        release.compare_prepared_semantics(original, altered, sources)


def test_reprepare_uses_physical_prepared_order_without_rewriting_inputs(
    tmp_path, monkeypatch, two_split_supply
):
    """Real two-source prepared order differs from the controlled journal order."""
    _, _, config, _codec, model, prepared = two_split_supply
    bundled = release.prepared_semantics(prepared)
    assert [row["sourceId"] for row in bundled["membership"]] == ["source-0", "source-1"]
    descriptor = descriptor_fixture(tmp_path)
    source_rows = []
    for row in reversed(bundled["membership"]):
        relative = f"source-qualification/{row['sourceId']}.original.json"
        (tmp_path / relative).write_text(
            json.dumps(
                {
                    "public": {"schema": "semantic-execution/8"},
                    "sidecar": {
                        "datasetBinding": {
                            "sourceId": row["sourceId"],
                            "split": row["split"],
                            "registryDigest": config.semantic_v8_policy["registryDigest"],
                        }
                    },
                }
            )
        )
        source_rows.append({"caseId": row["sourceId"], "path": relative})
    descriptor["roles"]["prepared"][0]["sourceArtifacts"] = source_rows
    sources = release.source_membership(
        descriptor, tmp_path, config.semantic_v8_policy["registryDigest"]
    )
    assert [row["sourceId"] for row in sources] == ["source-1", "source-0"]
    before_bundle = release.bundle_inventory(tmp_path)
    before_prepared = release.bundle_inventory(prepared)
    expected = [
        str(tmp_path / f"source-qualification/{source}.original.json")
        for source in ("source-0", "source-1")
    ]
    observed = []

    def prepare(command, **_kwargs):
        inputs = [command[i + 1] for i, value in enumerate(command) if value == "--input"]
        observed.append(inputs)
        assert inputs == expected
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "report": {
                        "counts": {"attempted": 2, "admitted": 2, "refused": 0},
                        "refusals": [],
                    }
                }
            ),
        )

    monkeypatch.setattr(release.subprocess, "run", prepare)
    monkeypatch.setattr(release, "prepared_semantics", lambda _root: copy.deepcopy(bundled))
    regenerated, _ = release.regenerate_prepared(
        tmp_path,
        descriptor,
        {"tokenizer": tmp_path / "tokenizer/tokenizer.json"},
        producer_root=tmp_path,
        consumer_root=tmp_path,
        model_config=model,
        timeout_seconds=1,
    )
    assert observed == [expected]
    assert release.compare_prepared_semantics(bundled, regenerated, sources)["sourceCount"] == 2
    assert release.bundle_inventory(tmp_path) == before_bundle
    assert release.bundle_inventory(prepared) == before_prepared


@pytest.mark.parametrize("mutation", ["cached-label", "free-mask"])
def test_resealed_prepared_label_change_is_not_hidden_by_valid_file_hashes(tmp_path, mutation):
    _, _, _, prepared = prepared_v8_fixture(tmp_path)
    original = release.prepared_semantics(prepared)
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard = prepared / manifest["shards"][0]["path"]
    value = json.loads(shard.read_text())
    record = next(row for row in value["records"] if row["policy_mask"])
    if mutation == "cached-label":
        record["cached_ids"][0] = (record["cached_ids"][0] + 1) % 256
    else:
        record["free_spans"] = []
    line = json.dumps(value, separators=(",", ":")).encode() + b"\n"
    shard.write_bytes(line)
    index = prepared / manifest["index"]["path"]
    with sqlite3.connect(index) as connection:
        connection.execute(
            "UPDATE episodes SET length=?,sha256=?",
            (len(line), sha256_file(shard)),
        )
    manifest["shards"][0].update(sha256=sha256_file(shard), bytes=len(line))
    manifest["index"]["sha256"] = sha256_file(index)
    manifest["fingerprint"] = content_hash(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    manifest_path.write_text(json.dumps(manifest))
    altered = release.prepared_semantics(prepared)
    with pytest.raises(ValueError, match="differ from fresh sources"):
        release.compare_prepared_semantics(
            altered, original, [{"sourceId": "source-1", "split": "train"}]
        )


@pytest.mark.parametrize("changed", ["value", "dtype", "shape", "key", "scalar-type"])
def test_actual_loaded_state_comparison_rejects_plausible_forgery(changed):
    original = {"weight": torch.tensor([1.0, 2.0]), "bank": {"count": 1}}
    altered = copy.deepcopy(original)
    if changed == "value":
        altered["weight"][0] = 9.0
    elif changed == "dtype":
        altered["weight"] = altered["weight"].double()
    elif changed == "shape":
        altered["weight"] = altered["weight"].reshape(1, 2)
    elif changed == "key":
        altered["extra"] = 0
    else:
        altered["bank"]["count"] = True
    with pytest.raises(ValueError, match="Served state"):
        release.assert_same_state(original, altered, "test")


def test_checkpoint_refusal_never_repairs_or_moves_input_files(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"not a checkpoint\n")
    (checkpoint / "metadata.json").write_text("{}\n")
    aside = checkpoint / ".superseded-side-1"
    aside.mkdir()
    (aside / "preserved.bin").write_bytes(b"original evidence must stay in place\n")
    before = release.bundle_inventory(checkpoint)
    with pytest.raises(ValueError):
        release.checkpoint_consistency(None, None, None, {"checkpoint": checkpoint}, [])
    assert release.bundle_inventory(checkpoint) == before


def test_real_controlled_checkpoint_export_and_resealed_weight_refusal(tmp_path, monkeypatch):
    """One real optimizer update; fixture source/byte codec are explicitly injected.

    This exercises real checkpoint, cursor, position history, tensor export and
    ServedBundle loading. It does not qualify a production tokenizer, source
    replay, the complete release descriptor or the active paired repositories.
    """
    from safetensors.torch import load_file, save_file

    import megatoken.training.semantic_v8 as semantic_v8
    from megatoken.generation.codec import ByteCodec
    from megatoken.loading import load_model_config
    from megatoken.served import export_served
    from megatoken.training.engine import train

    torch.set_num_threads(1)
    _, codec, policy, prepared = prepared_v8_fixture(tmp_path)
    tokenizer = tmp_path / "tokenizer.fixture"
    tokenizer.write_bytes(b"explicit controlled release byte codec\n")

    def fixture_codec(path, *, vocab_size):
        assert type(vocab_size) is int and vocab_size == codec.vocab_size
        assert Path(path).read_bytes() == tokenizer.read_bytes()
        return ByteCodec.byte_fallback()

    monkeypatch.setattr(semantic_v8, "load_semantic_v8_codec", fixture_codec)
    config = TrainConfig(
        purpose="diagnostic",
        phase="SC",
        model="tiny",
        codec=str(tokenizer),
        datasets=[str(prepared)],
        output=str(tmp_path / "fit"),
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
    result = train(config)
    served = tmp_path / "served"
    export_served(result["checkpoint"], served, codec_path=str(tokenizer))
    relocated = tmp_path / "copied-data"
    shutil.copytree(prepared, relocated)
    moved = TrainConfig.from_dict({**asdict(config), "datasets": [str(relocated)]})
    model_config = load_model_config(
        config.model,
        expert_backend=config.expert_backend,
        checkpoint_stages=config.gradient_checkpointing,
    )
    paths = {
        "prepared": relocated,
        "checkpoint": Path(result["checkpoint"]),
        "tokenizer": tokenizer,
        "served": served,
    }
    payload, metadata, checkpoint = release.checkpoint_consistency(
        moved,
        model_config,
        codec,
        paths,
        [{"sourceId": "source-1", "split": "train"}],
    )
    assert checkpoint["globalStep"] == 1
    assert checkpoint["positionComparison"]["unexpectedPhysicalTargets"] == []
    proof = release.served_consistency(paths, payload, metadata)
    assert proof["actualModelTensorsEqual"] and proof["actualTransducerBankEqual"]
    assert proof["additionalLoadInvalidationVersionIncrement"] == 1

    def reseal_served(directory, field, file, lineage_field):
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[field] = sha256_file(directory / file)
        lineage_path = directory / "semantic-v8-lineage.json"
        lineage = json.loads(lineage_path.read_text())
        lineage[lineage_field] = manifest[field]
        lineage["digest"] = stable_algorithm_hash(
            {k: v for k, v in lineage.items() if k != "digest"}
        )
        lineage_path.write_text(json.dumps(lineage))
        manifest["semantic_v8_lineage_sha256"] = sha256_file(lineage_path)
        manifest["semantic_v8_lineage_digest"] = lineage["digest"]
        manifest_path.write_text(json.dumps(manifest))

    config_forgery = tmp_path / "served-config-forgery"
    shutil.copytree(served, config_forgery)
    config_path = config_forgery / "model_config.json"
    exported_config = json.loads(config_path.read_text())
    assert exported_config["checkpoint_stages"] is False
    exported_config["checkpoint_stages"] = 0
    config_path.write_text(json.dumps(exported_config))
    reseal_served(config_forgery, "config_sha256", "model_config.json", "configSha256")
    with pytest.raises(ValueError, match="Served configuration"):
        release.served_consistency({**paths, "served": config_forgery}, payload, metadata)

    duplicate_config = tmp_path / "served-duplicate-config"
    shutil.copytree(served, duplicate_config)
    config_path = duplicate_config / "model_config.json"
    raw = config_path.read_text().rstrip()
    config_path.write_text(raw[:-1] + ',"checkpoint_stages":false}')
    reseal_served(duplicate_config, "config_sha256", "model_config.json", "configSha256")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        release.served_consistency({**paths, "served": duplicate_config}, payload, metadata)

    version_forgery = tmp_path / "served-version-forgery"
    shutil.copytree(served, version_forgery)
    manifest_path = version_forgery / "manifest.json"
    forged_manifest = json.loads(manifest_path.read_text())
    assert forged_manifest["model_version"] == 1
    forged_manifest["model_version"] = True
    manifest_path.write_text(json.dumps(forged_manifest))
    with pytest.raises(ValueError, match="Served configuration"):
        release.served_consistency({**paths, "served": version_forgery}, payload, metadata)

    bank_forgery = tmp_path / "served-bank-forgery"
    shutil.copytree(served, bank_forgery)
    bank = torch.load(bank_forgery / "transducer.pt", weights_only=True)
    bank["bank"]["version"] += 1
    torch.save(bank, bank_forgery / "transducer.pt")
    reseal_served(bank_forgery, "transducer_sha256", "transducer.pt", "transducerSha256")
    with pytest.raises(ValueError, match="exported.transducer.bank.version"):
        release.served_consistency({**paths, "served": bank_forgery}, payload, metadata)

    weights = load_file(str(served / "model.safetensors"))
    key = next(name for name, value in weights.items() if value.numel() > 0)
    weights[key] = weights[key].clone()
    weights[key].reshape(-1)[0] += 1
    save_file(weights, str(served / "model.safetensors"))
    reseal_served(served, "model_sha256", "model.safetensors", "modelIdentity")
    with pytest.raises(ValueError, match="Served state differs from the checkpoint: model"):
        release.served_consistency(paths, payload, metadata)
