"""Fresh source/native admission and immutable-family prepared v7 data."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

from megatoken.generation.codec import LING_TOKENIZER_SHA256
from megatoken.io import atomic_json_array_writer, write_json

from .algorithm_v7 import stable_algorithm_hash
from .chunking import ChunkingConfig
from .dedup import DedupConfig, DedupStore
from .host_operations import CONTRACT_DIGEST
from .semantic_admission_v7 import adapt_semantic_v7, native_admission_v7
from .semantic_family_v7 import verify_split_identity_v7
from .storage import export_prepared


def source_verifier_for_checkout_v7(tracer_root, producer_revision):
    tracer = Path(tracer_root).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tracer, text=True).strip()
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=tracer, text=True
    ).strip()
    extra = subprocess.check_output(
        ["git", "ls-files", "--others", "-z", "--", "src", "tools"], cwd=tracer, text=True
    ).split("\0")
    if (
        actual != producer_revision
        or changed
        or any(Path(name).suffix in {".js", ".mjs", ".cjs", ".json"} for name in extra if name)
    ):
        raise ValueError("V7 source verifier requires its clean exact producer checkout")
    verifier = tracer / "tools/verify-algorithm-admission.mjs"
    if not verifier.is_file():
        raise ValueError("Pinned producer has no v7 source-admission verifier")

    def verify(document):
        with tempfile.TemporaryDirectory(prefix="semantic-source-v7-") as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
            result = subprocess.run(
                ["node", str(verifier), str(path)],
                cwd=tracer,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if result.returncode:
                raise ValueError(
                    "Fresh v7 source replay rejected candidate: " + result.stderr[-4000:]
                )
            return json.loads(result.stdout)

    return verify


def verify_admission_v7(document, source, native, *, producer_revision, consumer_revision, codec):
    from .semantic_contract_v7 import FIXTURE_DIGEST, SCHEMA_DIGEST

    side = document.get("sidecar", {})
    public_digest = stable_algorithm_hash(document["public"])
    contexts_digest = stable_algorithm_hash(document["contexts"])
    if (
        side.get("schema") != "semantic-algorithm-evidence/2"
        or side.get("representation") != "semantic-trace/7"
        or side.get("admission") != "admitted"
        or side.get("referenceStatus") != "agreed"
        or side.get("selectionOrigin") != "owned-interpreter-teacher"
    ):
        raise ValueError(
            "V7 training requires independently admitted owned-source teacher evidence"
        )
    if (
        source.get("schema") != "semantic-source-admission/7"
        or source.get("producerRevision") != producer_revision
        or source.get("episodeDigest") != public_digest
        or source.get("contextsDigest") != contexts_digest
        or source.get("datasetBindingDigest") != stable_algorithm_hash(side.get("datasetBinding"))
        or source.get("sourceArtifactDigest") != side.get("sourceArtifactDigest")
        or source.get("referenceStatus") != "agreed"
        or source.get("optimizerSteps") != 0
        or any(
            source.get(key) is not True
            for key in ("sourceValidated", "journalValidated", "replayValidated")
        )
    ):
        raise ValueError("Fresh v7 source receipt is missing, stale or incompatible")
    expected = {
        "schema": "semantic-native-admission/7",
        "componentGate": "native-consumer-only",
        "traceFormat": "semantic-trace/7",
        "representation": document["public"]["representation"],
        "episodeDigest": public_digest,
        "contextsDigest": contexts_digest,
        "datasetBindingDigest": stable_algorithm_hash(side.get("datasetBinding")),
        "consumerRevision": consumer_revision,
        "schemaDigest": SCHEMA_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "contractDigest": CONTRACT_DIGEST,
        "tokenizerIdentity": codec.fingerprint,
        "tokenizerSha256": LING_TOKENIZER_SHA256,
        "optimizerSteps": 0,
    }
    for receipt in (side.get("nativeAdmission", {}), native):
        if any(receipt.get(key) != value for key, value in expected.items()) or any(
            receipt.get(key) is not True
            for key in (
                "admitted",
                "lossMaskValidated",
                "visibilityValidated",
                "fragmentationValidated",
                "contextValidated",
                "continuationValidated",
            )
        ):
            raise ValueError(
                "V7 native admission must be recollected for exact current bytes and pins"
            )
    if stable_algorithm_hash(native) != stable_algorithm_hash(side["nativeAdmission"]):
        raise ValueError("Saved v7 native admission differs from fresh independent admission")
    return verify_split_identity_v7(document, source_receipt=source)


def semantic_behaviors_v7(episode):
    annotation = json.loads(episode.supervision.annotations_json).get("semantic_execution", {})
    if annotation.get("schema") != "semantic-execution/7":
        raise ValueError("V7 behavior census does not accept old episodes")
    representation = annotation["representation"]
    kinds = []
    for parts in episode.iter_logical_records():
        if not parts[0].policy_mask:
            continue
        raw = b"".join(part.payload for part in parts).decode()
        if representation == "algorithm":
            from .algorithm_v7 import parse_algorithm_record

            kinds.append(parse_algorithm_record(raw, role="generated")["kind"])
        else:
            from .detailed_v7 import parse_record

            kinds.append(parse_record(raw, role="generated")[0])
    result = {"semantic_v7", "semantic_v7_" + representation, "semantic_code_invocation"}
    if "request" in kinds:
        result.add("semantic_tool_feedback")
    if "await" in kinds or "resume" in kinds:
        result.add("semantic_async")
    return result


def prepare_semantic_corpus_v7(
    candidates,
    output,
    *,
    codec,
    producer_revision,
    consumer_revision,
    source_verifier,
    max_positions=8192,
    max_history=1024,
):
    from .semantic_contract_v7 import FIXTURE_DIGEST, SCHEMA_DIGEST

    if not callable(source_verifier) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
        for value in (producer_revision, consumer_revision)
    ):
        raise ValueError(
            "V7 preparation requires independent source replay and exact Git revisions"
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() or (root / "semantic-v7-dedup.sqlite").exists():
        raise ValueError(
            "V7 preparation requires a new output directory; saved data is never migrated"
        )
    counts, refusals = Counter(), []
    registry = None
    episode_splits = {}
    with DedupStore(
        root / "semantic-v7-dedup.sqlite",
        DedupConfig(split_seed="semantic-family-components/1", max_descendants_per_parent=20000),
    ) as store:
        with atomic_json_array_writer(root / "semantic-v7-audit.json") as audit:
            for index, candidate in enumerate(candidates):
                counts["attempted"] += 1
                document = candidate.get("exported", candidate)
                try:
                    identity = verify_split_identity_v7(document)
                    if registry is None:
                        registry = identity["registry"]
                    elif registry != identity["registry"]:
                        raise ValueError(
                            "All prepared rows must use one complete immutable family registry"
                        )
                    source = source_verifier(document)
                    native = native_admission_v7(
                        document,
                        codec=codec,
                        consumer_revision=consumer_revision,
                        max_positions=max_positions,
                        max_history=max_history,
                    )
                    verify_admission_v7(
                        document,
                        source,
                        native,
                        producer_revision=producer_revision,
                        consumer_revision=consumer_revision,
                        codec=codec,
                    )
                    episode, _ = adapt_semantic_v7(
                        document,
                        codec=codec,
                        max_positions=max_positions,
                        max_history=max_history,
                        admission_verifier=lambda _: True,
                    )
                    annotations = json.loads(episode.supervision.annotations_json)
                    annotations["semantic_execution"].update(
                        {
                            "continuation": "teacher_conditioned",
                            "admission": "admitted",
                            "registryDigest": identity["registryDigest"],
                            "componentId": identity["componentId"],
                            "split": identity["split"],
                            "sourceId": identity["sourceId"],
                            "datasetBinding": document["sidecar"]["datasetBinding"],
                            "publicDigest": source["episodeDigest"],
                            "contextDigest": source["contextsDigest"],
                            "sourceAdmission": source,
                            "nativeAdmission": native,
                        }
                    )
                    episode = replace(
                        episode,
                        lineage=replace(
                            episode.lineage,
                            family_id=identity["componentId"],
                            ancestor_ids=tuple(
                                "semantic-family:" + member for member in identity["members"]
                            ),
                            task_id=None,
                            frozen_split=identity["split"],
                        ),
                        supervision=replace(
                            episode.supervision,
                            outcome="success",
                            verification="external_tests",
                            annotations_json=json.dumps(annotations, separators=(",", ":")),
                        ),
                    )
                    disposition = store.add(episode, codec)
                    episode_splits[episode.episode_id] = identity["split"]
                    counts[disposition] += 1
                    counts["admitted"] += 1
                    audit(
                        {
                            "episodeId": episode.episode_id,
                            "source": source,
                            "native": native,
                            "datasetBinding": document["sidecar"]["datasetBinding"],
                        }
                    )
                except (ValueError, KeyError, TypeError) as error:
                    counts["refused"] += 1
                    refusals.append({"candidate": index, "reason": str(error)})
        store.freeze()
        split_counts, profiles = Counter(), Counter()
        for row in store.accepted_rows():
            if row["split"] != episode_splits[row["episode_id"]]:
                raise ValueError("Native deduplication changed a frozen semantic-family split")
            split_counts[row["split"]] += 1
            annotation = json.loads(
                json.loads(row["episode_json"])["supervision"]["annotations_json"]
            )
            profiles[annotation["semantic_execution"]["representation"]] += 1
        if not sum(split_counts.values()):
            write_json(
                root / "semantic-v7-report.json",
                {
                    "schema": "semantic-corpus-report/7",
                    "counts": dict(counts),
                    "refusals": refusals,
                    "optimizerSteps": 0,
                },
            )
            raise ValueError("No admitted native v7 episodes; refusal report retained")
        pins = {
            "schema": "semantic-preparation/7",
            "representation": "algorithm-or-detailed",
            "producerRevision": producer_revision,
            "consumerRevision": consumer_revision,
            "catalogDigest": CONTRACT_DIGEST,
            "schemaDigest": SCHEMA_DIGEST,
            "fixtureDigest": FIXTURE_DIGEST,
            "tokenizerIdentity": codec.fingerprint,
            "registryDigest": registry["digest"],
            "maxPositions": max_positions,
            "maxHistory": max_history,
        }
        report = {
            **pins,
            "counts": dict(counts),
            "splits": dict(split_counts),
            "profiles": dict(profiles),
            "refusals": refusals,
            "optimizerSteps": 0,
            "learningClaim": None,
        }
        manifest = export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=[pins],
            pipeline_fingerprint=stable_algorithm_hash(pins),
            chunking=ChunkingConfig(hard_max_tokens=max_positions),
            sidecars={"semantic-v7-report.json": report, "semantic-v7-families.json": registry},
        )
    return manifest, report
