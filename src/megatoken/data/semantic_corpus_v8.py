"""Fresh source/native admission and ordinary prepared shards for V8 traces."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from megatoken.data.native_v8 import prepare_native_v8_training, validate_v8_artifact
from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.source_identity_v8 import source_evidence_binding_v8
from megatoken.generation.codec import LING_TOKENIZER_SHA256
from megatoken.io import write_json

from .chunking import ChunkingConfig
from .dedup import DedupConfig, DedupStore
from .storage import export_prepared


def _verify_binding(document, registry):
    side = document.get("sidecar", {})
    identity = side.get("splitIdentity") if isinstance(side, dict) else None
    if (
        not isinstance(identity, dict)
        or identity.get("schema") != "semantic-region-split-identity/1"
    ):
        raise ValueError("V8 artifact has no immutable source-family split identity")
    if identity.get("registry") != registry:
        raise ValueError("V8 artifact split identity differs from the complete frozen registry")
    digest = stable_algorithm_hash(
        {key: value for key, value in registry.items() if key != "digest"}
    )
    if digest != registry.get("digest") or identity.get("registryDigest") != digest:
        raise ValueError("V8 family registry digest is invalid")
    member = next(
        (row for row in registry["sources"] if row["id"] == identity.get("sourceId")), None
    )
    component = next(
        (row for row in registry["components"] if identity.get("sourceId") in row["sourceIds"]),
        None,
    )
    if member is None or component is None:
        raise ValueError("V8 source is absent from the frozen registry")
    if (
        identity.get("sourceDigest") != member.get("sourceDigest")
        or identity.get("coreDigest") != member.get("coreDigest")
        or identity.get("componentId") != component.get("id")
        or identity.get("members") != component.get("members")
        or identity.get("split") != component.get("split")
        or side.get("split") != component.get("split")
        or side.get("sourceComponent") != component.get("id")
        or side.get("familyRegistryDigest") != digest
    ):
        raise ValueError("V8 frozen split assignment or exact source membership changed")
    binding = side.get("datasetBinding")
    expected = {
        "schema": "semantic-region-dataset-binding/1",
        "traceFormat": "semantic-trace/8",
        "episodeDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        **source_evidence_binding_v8(side),
        "registryDigest": digest,
        "componentId": component["id"],
        "split": component["split"],
        "sourceId": member["id"],
        "family": member["family"],
        "sourceDigest": member["sourceDigest"],
        "coreDigest": member["coreDigest"],
    }
    if binding != expected:
        raise ValueError("V8 binding does not cover original source, public records and contexts")
    return identity, member, component, binding


def _native_admission(
    document,
    *,
    codec,
    tokenizer_evidence,
    consumer_revision,
    source_receipt,
    binding,
    max_positions,
    max_history,
):
    result = validate_v8_artifact(document)
    episode, training = prepare_native_v8_training(
        document, codec=codec, max_positions=max_positions, max_history=max_history
    )
    checks = {
        **result["checks"],
        "masks": "passed" if training["maskValidation"] else "failed",
        "sourceBinding": "passed",
        "tokenizer": "passed",
    }
    if any(
        checks.get(key) != "passed"
        for key in ("replay", "contexts", "continuation", "masks", "sourceBinding", "tokenizer")
    ):
        raise ValueError("Fresh V8 native checks did not all pass")
    if (
        not isinstance(tokenizer_evidence, dict)
        or tokenizer_evidence.get("schema") != "semantic-tokenizer-admission/1"
        or tokenizer_evidence.get("status") != "passed"
        or tokenizer_evidence.get("identity") != codec.fingerprint
        or tokenizer_evidence.get("sha256") != LING_TOKENIZER_SHA256
        or tokenizer_evidence.get("verification") != "pinned-tokenizer-bytes"
    ):
        raise ValueError("V8 admission requires a verified pinned tokenizer and codec identity")
    receipt = {
        "schema": "semantic-native-admission/8",
        "traceFormat": "semantic-trace/8",
        "contextFormat": "semantic-context/8",
        "representation": "mixed",
        "componentGate": "native-consumer-only",
        "publicDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        "sourceArtifactDigest": binding["sourceArtifactDigest"],
        **(
            {"sourceIdentityDigest": binding["sourceIdentityDigest"]}
            if "sourceIdentityDigest" in binding
            else {}
        ),
        "sourceVerification": source_receipt,
        "sourceProfile": document.get("sidecar", {}).get("profile"),
        "protocolProfile": document.get("sidecar", {}).get("protocolProfile"),
        "protocolExtensions": document.get("sidecar", {}).get("protocolExtensions", []),
        "datasetBindingDigest": stable_algorithm_hash(binding),
        "consumerRevision": consumer_revision,
        "contractDigest": CONTRACT_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "interoperabilityDigest": INTEROPERABILITY_DIGEST,
        "tokenizer": {
            "status": "passed",
            "identity": codec.fingerprint,
            "sha256": LING_TOKENIZER_SHA256,
        },
        "checks": checks,
        "counts": {
            "records": len(document["public"]["timeline"]),
            "contexts": training["contexts"],
            "generatedRecords": training["generatedRecords"],
            "algorithmRecords": result["counts"].get("algorithmRecords", 0),
            "regionRecords": result["counts"].get("regionRecords", 0),
            "detailedRecords": result["counts"].get("detailedRecords", 0),
            "protocolRecords": result["counts"].get("protocolRecords", 0),
            "generatedTokens": training["generatedTokens"],
            "observedTokens": training["observedTokens"],
        },
        "framing": {
            "kind": training["framing"],
            "eosTokenId": training["eosTokenId"],
            "terminalTokens": training["terminalTokens"],
            "physicalRecords": training["physicalRecords"],
        },
        "nativeTrainingDigest": training["nativeTrainingDigest"],
        "optimizerSteps": 0,
        "learningClaim": None,
        "admitted": True,
    }
    return episode, training, receipt


def semantic_behaviors_v8(episode):
    """Derive the small stage coverage vocabulary from V8 public records."""
    annotations = json.loads(episode.supervision.annotations_json)
    semantic = annotations.get("semantic_execution", {})
    if semantic.get("schema") != "semantic-execution/8":
        raise ValueError("V8 behavior census requires a V8 prepared episode")
    artifact = semantic.get("publicArtifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("public"), dict):
        raise ValueError("V8 behavior census requires retained public records")
    rows = artifact["public"].get("timeline", [])
    kinds = [row["kind"] for row in rows]
    fields = [row["fields"] for row in rows]
    behaviors = {"semantic_v8"}
    if "program" in kinds and "invocation" in kinds:
        behaviors.add("semantic_code_invocation")
    if "request" in kinds:
        behaviors.add("semantic_tool_feedback")
    if "await" in kinds or "resume" in kinds:
        behaviors.add("semantic_async")
    if any(row.get("activation") == "callback" for row in fields):
        behaviors.add("semantic_callback")
    if any(row.get("activation") == "module-init" for row in fields):
        behaviors.add("semantic_initialization")
    if "cancel" in kinds:
        behaviors.add("semantic_cancel")
    if "reconcile" in kinds or "callback-reattach" in kinds:
        behaviors.add("semantic_recovery")
    return behaviors


def prepare_semantic_corpus_v8(
    candidates,
    output,
    *,
    codec,
    producer_revision,
    consumer_revision,
    source_verifier,
    tokenizer_evidence,
    max_positions=8192,
    max_history=1024,
):
    """Prepare only candidates with fresh source correspondence and native receipts.

    Each candidate is a bound V8 artifact. ``source_verifier`` must freshly replay
    source correspondence for that artifact and return its producer receipt.
    """
    if not callable(source_verifier) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in (producer_revision, consumer_revision)
    ):
        raise ValueError(
            "V8 preparation requires fresh source replay and exact repository revisions"
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() or (root / "semantic-v8-dedup.sqlite").exists():
        raise ValueError("V8 preparation requires a new output directory")
    counts, refusals = Counter(), []
    registry = None
    with DedupStore(
        root / "semantic-v8-dedup.sqlite",
        DedupConfig(
            split_seed="semantic-region-family-components/1", max_descendants_per_parent=20000
        ),
    ) as store:
        for index, candidate in enumerate(candidates):
            counts["attempted"] += 1
            document = (
                candidate.get("exported", candidate) if isinstance(candidate, dict) else candidate
            )
            try:
                if (
                    not isinstance(document, dict)
                    or document.get("public", {}).get("schema") != "semantic-execution/8"
                ):
                    raise ValueError("Only exact semantic-execution/8 artifacts can be prepared")
                identity = document.get("sidecar", {}).get("splitIdentity", {})
                candidate_registry = identity.get("registry")
                if registry is None:
                    registry = candidate_registry
                elif registry != candidate_registry:
                    raise ValueError("All V8 candidates must use one complete family registry")
                identity, member, component, binding = _verify_binding(document, registry)
                source = source_verifier(document)
                expected_source = {
                    "schema": "semantic-native-source-recheck/1",
                    "producerRevision": producer_revision,
                    "producerTreeClean": True,
                    "sourceArtifactDigest": binding["sourceArtifactDigest"],
                    "publicDigest": binding["episodeDigest"],
                    "contextDigest": binding["contextsDigest"],
                    "sourceValidated": True,
                    "projectionValidated": True,
                    "replayValidated": True,
                    "familyBindingValidated": True,
                    "registryDigest": registry["digest"],
                    "datasetBindingDigest": stable_algorithm_hash(binding),
                }
                if "sourceIdentityDigest" in binding:
                    expected_source.update(
                        sourceIdentityDigest=binding["sourceIdentityDigest"], identityValidated=True
                    )
                if not isinstance(source, dict) or any(
                    source.get(key) != value for key, value in expected_source.items()
                ):
                    raise ValueError(
                        "Fresh V8 producer receipt does not bind the exact source/public/context bytes"
                    )
                episode, training, native = _native_admission(
                    document,
                    codec=codec,
                    consumer_revision=consumer_revision,
                    tokenizer_evidence=tokenizer_evidence,
                    source_receipt=source,
                    binding=binding,
                    max_positions=max_positions,
                    max_history=max_history,
                )
                semantic = json.loads(episode.supervision.annotations_json)
                semantic["semantic_geometry"] = {
                    "schema": "semantic-native-geometry/8",
                    "maxHistory": max_history,
                    "maxPositions": max_positions,
                }
                semantic["semantic_execution"].update(
                    {
                        "continuation": "teacher_conditioned",
                        "admission": "admitted",
                        "targetEligibility": "source_admitted",
                        "modelExecutionOutcome": "unmeasured",
                        "sourceId": member["id"],
                        "componentId": component["id"],
                        "members": component["members"],
                        "split": component["split"],
                        "registryDigest": registry["digest"],
                        "datasetBinding": binding,
                        "sourceAdmission": source,
                        "nativeAdmission": native,
                        "tokenizerEvidence": tokenizer_evidence,
                        "nativeTrainingDigest": training["nativeTrainingDigest"],
                        "publicArtifact": {
                            "public": document["public"],
                            "contexts": document["contexts"],
                        },
                    }
                )
                # Eligibility is a property of the target evidence; model success
                # remains unmeasured until a separate learned-execution study.
                records = tuple(
                    replace(record, provenance="teacher") if record.policy_mask else record
                    for record in episode.records
                )
                episode = replace(
                    episode,
                    records=records,
                    lineage=replace(
                        episode.lineage,
                        family_id=component["id"],
                        ancestor_ids=tuple(
                            "semantic-family:" + name for name in component["members"]
                        ),
                        frozen_split=component["split"],
                    ),
                    supervision=replace(
                        episode.supervision,
                        annotations_json=json.dumps(semantic, separators=(",", ":")),
                    ),
                )
                store.add(episode, codec)
                counts["admitted"] += 1
            except (ValueError, KeyError, TypeError) as error:
                counts["refused"] += 1
                refusals.append({"candidate": index, "reason": str(error)})
        if counts["admitted"] == 0:
            write_json(
                root / "semantic-v8-report.json",
                {
                    "schema": "semantic-corpus-report/8",
                    "counts": dict(counts),
                    "refusals": refusals,
                    "optimizerSteps": 0,
                },
            )
            raise ValueError("No source/native-admitted V8 examples; refusal report retained")
        store.freeze()
        for row in store.accepted_rows():
            if row["split"] != json.loads(row["episode_json"])["lineage"]["frozen_split"]:
                raise ValueError("Deduplication changed an immutable V8 family split")
        pins = {
            "schema": "semantic-preparation/8",
            "representation": "mixed",
            "producerRevision": producer_revision,
            "consumerRevision": consumer_revision,
            "contractDigest": CONTRACT_DIGEST,
            "fixtureDigest": FIXTURE_DIGEST,
            "tokenizerIdentity": codec.fingerprint,
            "tokenizerSha256": LING_TOKENIZER_SHA256,
            "registryDigest": registry["digest"],
            "maxPositions": max_positions,
            "maxHistory": max_history,
            "optimizerSteps": 0,
            "learningClaim": None,
        }
        report = {**pins, "counts": dict(counts), "refusals": refusals}
        manifest = export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=[pins],
            pipeline_fingerprint=stable_algorithm_hash(pins),
            chunking=ChunkingConfig(
                target_tokens=min(512, max_positions),
                soft_max_tokens=min(768, max_positions),
                hard_max_tokens=max_positions,
            ),
            sidecars={"semantic-v8-report.json": report, "semantic-v8-families.json": registry},
        )
    return manifest, report
