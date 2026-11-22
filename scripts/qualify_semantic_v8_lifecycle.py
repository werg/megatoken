#!/usr/bin/env python3
"""Verify ordinary V8 optimizer/resume/export and real JSONL serving evidence.

Run preparation, two bounded normal trainer invocations and export first. This
collector requires their actual artifacts, a clean pinned consumer checkout and
the ordinary pinned tokenizer. It never injects a decoder or changes admission.
Generated output is retained without making a learned-success claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Service:
    def __init__(self, root, served, output, name):
        self.output = output
        self.name = name
        self.calls = []
        self.stderr = (output / f"{name}-stderr.log").open("w")
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(root / "scripts/serve_semantic_v8.py"),
                "--served",
                str(served),
                "--device",
                "cpu",
                "--threads",
                "1",
                "--max-tokens",
                "4",
                "--max-fragments",
                "1",
                "--profile-output",
                str(output / f"{name}-profile.json"),
            ],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
        )

    def __enter__(self):
        return self

    def __exit__(self, error_type, *_):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        finally:
            self.process.stdout.close()
            self.stderr.close()
            write_json(self.output / f"{self.name}-rpc.json", self.calls)
        if error_type is None:
            require(
                self.process.returncode == 0, f"Native service {self.name} did not close cleanly"
            )

    def call(self, operation, **fields):
        request = {"id": f"{self.name}-{len(self.calls)}", "op": operation, **fields}
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as ready:
            ready.register(self.process.stdout, selectors.EVENT_READ)
            if not ready.select(timeout=180):
                raise TimeoutError(f"Native V8 service timed out on {operation}")
        line = self.process.stdout.readline()
        require(line, f"Native V8 service exited on {operation}; inspect {self.name}-stderr.log")
        response = json.loads(line)
        self.calls.append({"request": request, "response": response})
        require(response.get("id") == request["id"], "Native service changed the request ID")
        return response


def success(response):
    require(response.get("ok") is True, f"Native RPC refused: {response.get('error')}")
    return response["result"]


def original_attempt(response):
    attempt = (
        response["result"].get("selection")
        if response.get("ok")
        else response["error"].get("evidence")
    )
    require(isinstance(attempt, dict), "Selection failed before producing model evidence")
    require(
        attempt.get("origin") == "actual-model-selected"
        and attempt.get("modelSelected") is True
        and attempt.get("generatedTokens", 0) > 0
        and attempt.get("fragments"),
        "Qualification requires original tokens from the loaded model",
    )
    return attempt


def qualify_serving(root, served, source, output, manifest):
    example = read_json(source)
    orchestration = "orchestrationTranscript" in example.get("sidecar", {})
    context = copy.deepcopy(example["contexts"][0])
    public = copy.deepcopy(example["public"])
    public["timeline"] = public["timeline"][: context["visibleThrough"]]
    selection = {
        "session": public["timeline"][0]["fields"]["session"],
        "public": public,
        "contexts": [],
        "context": context,
    }
    identity = {
        "seed": 17,
        "checkpointIdentity": manifest["model_sha256"],
        "tokenizerIdentity": manifest["codec_fingerprint"],
        "trainingIdentity": manifest["semantic_v8_lineage_digest"],
    }
    controls = []
    with Service(root, served, output, "original") as service:
        for label, invalid in (
            (
                "omitted",
                {key: value for key, value in identity.items() if key != "trainingIdentity"},
            ),
            ("null", {**identity, "trainingIdentity": None}),
            ("substituted", {**identity, "trainingIdentity": "0" * 64}),
        ):
            response = service.call("initialize", **invalid)
            require(response.get("ok") is False, f"Serving accepted {label} training identity")
            controls.append({"kind": label, "error": response["error"]})
        initialized = success(service.call("initialize", **identity))
        require(initialized["trainingIdentity"] == identity["trainingIdentity"], "Lineage changed")
        select_operation = "select"
        if orchestration:
            opened = success(
                service.call(
                    "open-orchestration",
                    session=selection["session"],
                    public={**example["public"], "timeline": example["public"]["timeline"][:2]},
                    origin="model",
                    childBudget=100000,
                )
            )
            selection = {"session": selection["session"], "track": opened["track"]}
            select_operation = "select-orchestration"
        first_response = service.call(select_operation, **selection)
        first = original_attempt(first_response)
        packet = success(service.call("checkpoint"))
        write_json(output / "live-checkpoint.json", packet)
        if first_response["ok"]:
            success(
                service.call(
                    "reject",
                    session=selection["session"],
                    selectionId=first["selectionId"],
                    reason="Lifecycle qualification does not acknowledge generated semantics",
                )
            )
        expected = original_attempt(service.call(select_operation, **selection))
    with Service(root, served, output, "restored") as service:
        restored = success(service.call("restore", checkpoint=packet))
        actual = original_attempt(service.call(select_operation, **selection))
    compared = (
        "selectionId",
        "fragments",
        "payloadHex",
        "generatedTokens",
        "checkpointIdentity",
        "tokenizerIdentity",
        "trainingIdentity",
    )
    for key in compared:
        require(actual[key] == expected[key], f"Fresh-process next selection differs: {key}")
    require(restored["restored"] is True, "Live state did not restore")
    require(restored["reexecutedSelections"] == 0, "Unexpected acknowledged model selections")
    return {
        "initialization": initialized,
        "identityRefusals": controls,
        "firstSelection": first,
        "firstSelectionError": first_response.get("error"),
        "restoration": restored,
        "nextSelectionComparedFields": list(compared),
        "nextSelectionExact": True,
        "acknowledgedModelSelections": 0,
        "selectionFamily": "orchestration" if orchestration else "semantic",
        "learnedSuccess": None,
        "scope": "Actual trained CPU weights and pinned tokenizer; original token selection and fresh-process initial-state/RNG continuation. No semantic record was acknowledged.",
    }


def qualify_orchestration_preparation(source, prepared_path, report):
    """Inspect real saved rows without mistaking materialization for exposure."""
    from megatoken.data import PreparedCorpus
    from megatoken.data.semantic_orchestration_training_v8 import (
        validate_orchestration_records_v8,
        validate_orchestration_transcript_v8,
    )

    example = read_json(source)
    if "orchestrationTranscript" not in example.get("sidecar", {}):
        require(not report["counts"].get("orchestrationRecords"), "Detached orchestration count")
        return None
    validation = validate_orchestration_transcript_v8(example)
    require(
        report["counts"].get("orchestrationRecords") == 4,
        "Ordinary preparation omitted lifecycle targets",
    )
    with PreparedCorpus([prepared_path], split="train") as corpus:
        require(len(corpus) == 1, "Orchestration qualification needs its single admitted episode")
        episode = next(corpus.iter_episodes())
        require(
            validate_orchestration_records_v8(episode, validation),
            "Saved physical lifecycle records differ from the source-qualified transcript",
        )
        targets = [
            record
            for record in episode.records
            if record.logical_id.startswith("semantic:v8:orchestration:target:")
        ]
        require(
            len({record.logical_id for record in targets}) == 4
            and all(record.policy_mask for record in targets),
            "Saved lifecycle targets lost loss-bearing physical parts",
        )
        semantic = json.loads(episode.supervision.annotations_json)["semantic_execution"]
        return {
            "transcriptDigest": validation["transcriptDigest"],
            "logicalLifecycleTargets": 4,
            "physicalLifecycleTargetParts": len(targets),
            "trackOwners": sorted(track.track_id for track in episode.tracks),
            "nativeTrainingDigest": semantic["nativeAdmission"]["nativeTrainingDigest"],
            "workAccounting": example["sidecar"]["orchestrationTranscript"]["policy"][
                "workAccounting"
            ],
            "optimizerExposurePerTarget": None,
            "actualModelFragmentBudgetEquivalence": None,
        }


def qualify_checkpoints(
    first_path,
    resumed_path,
    prepared,
    served,
    manifest,
    registry_digest,
    allowed_source_ids=None,
):
    import torch

    from megatoken.semantic_lineage_v8 import checkpoint_lineage_v8, validate_served_lineage_v8
    from megatoken.training.checkpoint import load_checkpoint

    first, first_metadata = load_checkpoint(first_path, map_location="cpu")
    resumed, resumed_metadata = load_checkpoint(resumed_path, map_location="cpu")
    first_lineage = checkpoint_lineage_v8(first, first_metadata)
    resumed_lineage = checkpoint_lineage_v8(resumed, resumed_metadata)
    require(
        first_lineage is not None and first_lineage == resumed_lineage, "Training lineage changed"
    )
    require(
        first["global_step"] == 1 and resumed["global_step"] == 2, "Expected one step then resume"
    )
    require(
        first_lineage["preparedManifestFingerprints"] == [prepared["fingerprint"]],
        "Checkpoint does not bind this single prepared corpus",
    )
    require(manifest["model_version"] == 2, "Serving bundle did not export the resumed step")
    require(
        manifest["checkpoint_sha256"] == resumed_metadata["state_sha256"],
        "Serving bundle names another training checkpoint",
    )
    moments1 = first["optimizer"]["moments"]
    moments2 = resumed["optimizer"]["moments"]
    require(moments1 and set(moments1) <= set(moments2), "Optimizer lost existing moment owners")
    advanced = [
        name
        for name in moments1
        if float(moments1[name].get("step", 0)) == 1 and float(moments2[name].get("step", 0)) == 2
    ]
    require(advanced, "No retained optimizer moment advanced from step one to two")
    changed = [
        name
        for name in first["optimizer"]["master_weights"]
        if not torch.equal(
            first["optimizer"]["master_weights"][name],
            resumed["optimizer"]["master_weights"][name],
        )
    ]
    require(changed, "Resumed training did not update model master weights")
    target_exposure = qualify_target_exposure(
        first,
        resumed,
        registry_digest=registry_digest,
        allowed_source_ids=allowed_source_ids,
    )
    served_lineage = read_json(served / "semantic-v8-lineage.json")
    validate_served_lineage_v8(served_lineage, manifest)
    require(served_lineage["training"] == resumed_lineage, "Serving lineage changed")
    return {
        "firstGlobalStep": first["global_step"],
        "resumedGlobalStep": resumed["global_step"],
        "firstCheckpointSha256": first_metadata["state_sha256"],
        "resumedCheckpointSha256": resumed_metadata["state_sha256"],
        "retainedMomentOwners": len(moments1),
        "advancedMomentOwners": len(advanced),
        "changedMasterWeights": len(changed),
        "changedMasterWeightNames": changed,
        "trainingLineage": resumed_lineage,
        "servingLineageDigest": served_lineage["digest"],
        "optimizerTargetExposure": target_exposure,
    }



def qualify_target_exposure(first, resumed, *, registry_digest, allowed_source_ids=None):
    from megatoken.training.semantic_v8 import (
        semantic_v8_target_exposure_report,
        validate_semantic_v8_target_exposure,
    )

    first_state = first.get("semantic_v8_target_exposure")
    resumed_state = resumed.get("semantic_v8_target_exposure")
    require(
        first_state is not None and resumed_state is not None,
        "V8 lifecycle checkpoint lost its target-exposure census",
    )
    first_state = validate_semantic_v8_target_exposure(
        first_state, registry_digest=registry_digest
    )
    resumed_state = validate_semantic_v8_target_exposure(
        resumed_state, registry_digest=registry_digest
    )
    require(
        first_state["startingGlobalStep"] == 0
        and first_state["lastGlobalStep"] == first["global_step"] == 1
        and first_state["optimizerUpdates"] == 1,
        "First checkpoint target-exposure history is not exactly optimizer step one",
    )
    require(
        resumed_state["startingGlobalStep"] == first_state["startingGlobalStep"]
        and resumed_state["lastGlobalStep"] == resumed["global_step"] == 2
        and resumed_state["optimizerUpdates"] == 2,
        "Resumed target-exposure history is not the exact step-one to step-two continuation",
    )
    allowed = None if allowed_source_ids is None else set(allowed_source_ids)
    for source_id, source in first_state["sources"].items():
        require(source_id in resumed_state["sources"], "Resume dropped a scored source")
        if allowed is not None:
            require(
                source_id in allowed,
                "Target exposure includes a source outside the support inventory",
            )
        old_source = source
        new_source = resumed_state["sources"][source_id]
        require(
            old_source["split"] == new_source["split"] == "train",
            "Target exposure changed split",
        )
        for logical_id, old_target in old_source["logicalTargets"].items():
            require(
                logical_id in new_source["logicalTargets"],
                "Resume dropped a scored logical target",
            )
            new_target = new_source["logicalTargets"][logical_id]
            for record_id, old_record in old_target["records"].items():
                require(
                    record_id in new_target["records"],
                    "Resume dropped a scored physical record",
                )
                new_record = new_target["records"][record_id]
                require(
                    set(old_record["windowIndices"]) <= set(new_record["windowIndices"])
                    and old_record["scoredOccurrences"] <= new_record["scoredOccurrences"]
                    and old_record["optimizerUpdates"] <= new_record["optimizerUpdates"]
                    and old_record["firstGlobalStep"] == new_record["firstGlobalStep"]
                    and old_record["lastGlobalStep"] <= new_record["lastGlobalStep"]
                    and all(
                        old_record["lossDenominators"][name]
                        <= new_record["lossDenominators"].get(name, 0)
                        for name in old_record["lossDenominators"]
                    ),
                    "Resume target-exposure history is not cumulative from step one",
                )
    if allowed is not None:
        require(
            set(resumed_state["sources"]) <= allowed,
            "Target exposure includes a source outside the support inventory",
        )
    first_report = semantic_v8_target_exposure_report(first_state)
    resumed_report = semantic_v8_target_exposure_report(resumed_state)
    return {
        "schema": resumed_report["schema"],
        "registryDigest": registry_digest,
        "historyContinued": True,
        "firstCheckpointDigest": first_state["digest"],
        "resumedCheckpointDigest": resumed_state["digest"],
        "firstCheckpointCounts": {
            key: first_report[key]
            for key in (
                "sourceCount",
                "logicalTargetCount",
                "physicalTargetRecordCount",
                "scoredOccurrences",
                "lossDenominators",
            )
        },
        "resumedCounts": {
            key: resumed_report[key]
            for key in (
                "sourceCount",
                "logicalTargetCount",
                "physicalTargetRecordCount",
                "scoredOccurrences",
                "lossDenominators",
            )
        },
        "positiveWeightLossDenominators": resumed_report["lossDenominators"],
        "sources": resumed_report["sources"],
        "allTargetsExposed": False,
    }

def read_support_source_inventory(path, registry_digest):
    value = read_json(path)
    require(
        isinstance(value, dict)
        and set(value) == {"schema", "registryDigest", "sourceIds"}
        and value["schema"] == "semantic-v8-source-population/1"
        and value["registryDigest"] == registry_digest,
        "Support inventory is malformed or bound to another registry",
    )
    ids = value["sourceIds"]
    require(
        isinstance(ids, list)
        and len(ids) == 64
        and all(isinstance(item, str) and item for item in ids)
        and len(set(ids)) == 64,
        "Support inventory must contain exactly 64 unique source IDs",
    )
    return tuple(sorted(ids))


def validate_support_population(prepared_root, source_path, inventory_path):
    from megatoken.data import PreparedCorpus
    from megatoken.data.semantic_corpus_v8 import _verify_binding
    from megatoken.data.semantic_json import stable_algorithm_hash

    prepared_root = Path(prepared_root)
    registry = read_json(prepared_root / "semantic-v8-families.json")
    registry_digest = registry.get("digest") if isinstance(registry, dict) else None
    expected = read_support_source_inventory(inventory_path, registry_digest)
    require(
        isinstance(registry, dict)
        and registry.get("schema") == "semantic-region-family-registry/1"
        and stable_algorithm_hash(
            {key: value for key, value in registry.items() if key != "digest"}
        )
        == registry_digest,
        "Prepared support registry digest is invalid",
    )
    source_rows = registry.get("sources")
    require(
        isinstance(source_rows, list)
        and len(source_rows) == 64
        and {row.get("id") for row in source_rows if isinstance(row, dict)} == set(expected),
        "Prepared support registry differs from the immutable 64-source inventory",
    )
    split_by_id = {}
    for component in registry.get("components", []):
        require(
            isinstance(component, dict)
            and component.get("split") in {"train", "validation", "test"}
            and isinstance(component.get("sourceIds"), list),
            "Prepared support registry has a malformed split component",
        )
        for source_id in component["sourceIds"]:
            require(
                source_id in expected and source_id not in split_by_id,
                "Prepared support registry duplicates or substitutes a source",
            )
            split_by_id[source_id] = component["split"]
    require(set(split_by_id) == set(expected), "Prepared support registry omits source assignments")
    sample_document = read_json(source_path)
    sample_identity, sample_member, sample_component, sample_binding = _verify_binding(
        sample_document, registry
    )
    sample_source_id = sample_identity["sourceId"]
    require(
        sample_source_id in expected and sample_component["split"] == "train",
        "Lifecycle sample must be a source-bound member of the frozen training inventory",
    )
    counts = {split: 0 for split in ("train", "validation", "test")}
    observed = set()
    sample_semantic = None
    with PreparedCorpus([str(prepared_root)], split=None) as corpus:
        for episode in corpus.iter_episodes():
            try:
                semantic = json.loads(episode.supervision.annotations_json)["semantic_execution"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError("Prepared support episode has no source identity") from error
            source_id, split = semantic.get("sourceId"), episode.lineage.frozen_split
            require(
                source_id in split_by_id
                and source_id not in observed
                and split_by_id[source_id] == split,
                "Prepared support episode differs from its frozen inventory or split",
            )
            observed.add(source_id)
            counts[split] += 1
            if source_id == sample_source_id:
                sample_semantic = semantic
    require(observed == set(expected), "Prepared support corpus is missing an inventory source")
    require(isinstance(sample_semantic, dict), "Prepared support corpus lacks the selected sample")
    public_artifact = {"public": sample_document["public"], "contexts": sample_document["contexts"]}
    if "orchestrationTranscript" in sample_document["sidecar"]:
        public_artifact["sidecar"] = sample_document["sidecar"]
    require(
        sample_semantic.get("datasetBinding") == sample_binding
        and sample_semantic.get("publicArtifact") == public_artifact,
        "Lifecycle sample public/context/source bytes differ from the prepared training episode",
    )
    require(
        counts == {"train": 48, "validation": 12, "test": 4},
        "Prepared support corpus differs from its frozen 48/12/4 source split",
    )
    return {
        "registryDigest": registry_digest,
        "sourceCount": len(expected),
        "sourceInventory": list(expected),
        "splitCounts": counts,
        "sampleSourceId": sample_source_id,
        "sampleDatasetBinding": sample_binding,
        "sampleMember": sample_member,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-ids", type=Path)
    for name in (
        "consumer-root",
        "prepared",
        "source",
        "first-checkpoint",
        "resumed-checkpoint",
        "served",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    options = parser.parse_args()
    driver = file_identity(__file__)
    root = options.consumer_root.resolve()
    options.output.mkdir(parents=True, exist_ok=False)
    output = options.output.resolve()
    sys.path.insert(0, str(root / "src"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    )
    require(not dirty, "Qualification consumer checkout must be clean")
    prepared = read_json(options.prepared / "manifest.json")
    report = read_json(options.prepared / "semantic-v8-report.json")
    require(report["consumerRevision"] == head, "Prepared data names another consumer revision")
    if options.expected_source_ids:
        support = validate_support_population(
            options.prepared, options.source, options.expected_source_ids
        )
        require(
            report["counts"]["admitted"] == support["sourceCount"] and not report["refusals"],
            "Prepared support report differs from the immutable source inventory",
        )
        require(
            report.get("registryDigest") == support["registryDigest"],
            "Prepared support report names another registry",
        )
        orchestration = None
    else:
        support = None
        require(
            report["counts"]["admitted"] == 1 and not report["refusals"],
            "Expected one admitted source",
        )
        orchestration = qualify_orchestration_preparation(options.source, options.prepared, report)
    manifest = read_json(options.served / "manifest.json")
    registry = read_json(options.prepared / "semantic-v8-families.json")
    checkpoint = qualify_checkpoints(
        options.first_checkpoint,
        options.resumed_checkpoint,
        prepared,
        options.served,
        manifest,
        registry_digest=registry["digest"],
        allowed_source_ids=None if support is None else support["sourceInventory"],
    )
    serving = qualify_serving(root, options.served.resolve(), options.source, output, manifest)
    artifacts = [
        options.source,
        options.prepared / "manifest.json",
        options.prepared / "semantic-v8-report.json",
        *([options.expected_source_ids] if options.expected_source_ids else []),
    ]
    for directory in (options.first_checkpoint, options.resumed_checkpoint, options.served):
        artifacts.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    summary = {
        "schema": "semantic-ordinary-lifecycle-qualification/8",
        "consumerRevision": head,
        "driver": driver,
        "preparation": report,
        "preparedManifestFingerprint": prepared["fingerprint"],
        **(
            {
                "supportPopulation": support,
                "fullLaunchQualified": False,
                "supportSourceInventory": file_identity(options.expected_source_ids),
            }
            if support is not None
            else {}
        ),
        "optimizerResumeExport": checkpoint,
        "serving": serving,
        **({"orchestrationPreparation": orchestration} if orchestration is not None else {}),
        "artifacts": [file_identity(path) for path in artifacts],
        "learnedSuccess": None,
        "limits": [
            "Two optimizer steps do not qualify learned execution quality or prove exposure of every source target.",
            *(
                []
                if support is not None
                else [
                    "One source family and two optimizer steps do not qualify learned execution quality."
                ]
            ),
            "Fresh-process recovery retained original failed/unacknowledged selections; accepted-record recovery was not exercised.",
            "Artifact hashes identify local complete bundles. Durable external bundle publication is a separate recorded step.",
        ],
    }
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "resumedStep": 2,
                "nextSelectionExact": True,
                **(
                    {
                        "optimizerTargetExposure": {
                            key: value
                            for key, value in checkpoint["optimizerTargetExposure"][
                                "resumedCounts"
                            ].items()
                            if key != "lossDenominators"
                        }
                        | {
                            "registryDigest": checkpoint["optimizerTargetExposure"][
                                "registryDigest"
                            ],
                            "positiveWeightLossDenominators": checkpoint[
                                "optimizerTargetExposure"
                            ]["positiveWeightLossDenominators"],
                            "historyContinued": checkpoint["optimizerTargetExposure"][
                                "historyContinued"
                            ],
                            "allTargetsExposed": False,
                        }
                    }
                    if "optimizerTargetExposure" in checkpoint
                    else {}
                ),
                "learnedSuccess": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
