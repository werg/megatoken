"""Production semantic preparation: independent admission, whole-family splits, native export.

No optimizer is imported or run. Private source/reference evidence is retained
in checksummed sidecars and never put into the model's prediction records.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .chunking import ChunkingConfig
from .dedup import DedupConfig, DedupStore
from .host_operations import CONTRACT_DIGEST, canonical_json
from .semantic_admission import native_admission
from .semantic_contract import FIXTURE_DIGEST, SCHEMA_DIGEST
from .semantic_harness_v5 import adapt_selected_harness_v5
from .storage import export_prepared
from .witness_v5 import parse_record

SOURCE_SCHEMA = "semantic-source-admission/4"


def source_verifier_for_checkout(tracer_root, producer_revision):
    """Invoke the pinned producer's independent source/replay verifier in a fresh process."""
    import subprocess
    import tempfile

    tracer = Path(tracer_root).resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tracer, text=True).strip()
    if actual != producer_revision:
        raise ValueError("Tracer checkout must match the exact producer revision")
    verifier = tracer / "tools/verify-effectful-admission.mjs"
    if not verifier.is_file():
        raise ValueError("Pinned tracer has no independent source-admission verifier")

    def verify(document):
        with tempfile.TemporaryDirectory(prefix="semantic-source-") as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
            result = subprocess.run(
                ["node", str(verifier), str(path)],
                cwd=tracer,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode:
                raise ValueError(
                    "Independent source admission rejected candidate: " + result.stderr[-2000:]
                )
            return json.loads(result.stdout)

    return verify


def public_digest(document):
    return hashlib.sha256(canonical_json(document["public"]).encode()).hexdigest()


def verify_admission(
    document, source_receipt, native_receipt, *, producer_revision, consumer_revision, codec
):
    """Check independently produced receipts against the exact current bytes and pins."""
    sidecar = document["sidecar"]
    digest = public_digest(document)
    if (
        sidecar.get("admission") != "admitted"
        or sidecar.get("selectionOrigin") != "certified_teacher"
        or sidecar.get("referenceStatus") != "agreed"
        or sidecar.get("catalogDigest") != CONTRACT_DIGEST
        or source_receipt.get("schema") != SOURCE_SCHEMA
        or source_receipt.get("episodeDigest") != digest
        or source_receipt.get("producerRevision") != producer_revision
        or source_receipt.get("catalogDigest") != CONTRACT_DIGEST
        or source_receipt.get("referenceStatus") != "agreed"
        or source_receipt.get("optimizerSteps") != 0
        or not all(
            source_receipt.get(key) is True
            for key in ("journalValidated", "replayValidated", "sourceValidated")
        )
    ):
        raise ValueError("Independent source admission is missing, stale, or incompatible")
    previous = sidecar.get("nativeAdmission", {})
    for receipt in (previous, native_receipt):
        if (
            receipt.get("schema") != "semantic-native-admission/4"
            or receipt.get("componentGate") != "native-consumer-only"
            or receipt.get("episodeDigest") != digest
            or receipt.get("consumerRevision") != consumer_revision
            or receipt.get("contractDigest") != CONTRACT_DIGEST
            or receipt.get("schemaDigest") != SCHEMA_DIGEST
            or receipt.get("fixtureDigest") != FIXTURE_DIGEST
            or receipt.get("tokenizerIdentity") != codec.fingerprint
            or receipt.get("optimizerSteps") != 0
            or not all(
                receipt.get(key) is True
                for key in (
                    "admitted",
                    "lossMaskValidated",
                    "visibilityValidated",
                    "fragmentationValidated",
                )
            )
        ):
            raise ValueError("Native admission must be recollected for the pinned consumer")
    provenance = sidecar.get("sourceProvenance", {})
    if (
        not isinstance(sidecar.get("sourceLineage"), str)
        or not sidecar["sourceLineage"]
        or not isinstance(provenance.get("license"), str)
        or not provenance["license"]
        or provenance.get("trustTier") not in {"repository-fixture", "trusted-owned-source"}
    ):
        raise ValueError("Admitted source needs family, license, and supported trust provenance")
    return True


def semantic_behaviors(episode) -> set[str]:
    """Derive supply coverage from actual logical records, not declared dataset labels."""
    behaviors = set()
    records = []
    for parts in episode.iter_logical_records():
        if parts[0].logical_id.startswith("semantic:context:"):
            continue
        payload = b"".join(part.payload for part in parts).decode()
        if parts[0].policy_mask:
            if payload.startswith("{"):
                action = json.loads(payload)
                records.append(("lifecycle", action["name"], action["arguments"]))
            else:
                kind, fields = parse_record(payload)
                records.append(("semantic", kind, fields))
        else:
            item = json.loads(payload)
            record_kind = (
                item["schema"]
                if item.get("schema") == "semantic-completion/4"
                else item.get("kind", item.get("schema", "native-receipt"))
            )
            records.append(("observed", record_kind, item))
    kinds = [kind for _, kind, _ in records]
    invocations = [fields for _, kind, fields in records if kind == "invocation"]
    if "program" in kinds and invocations:
        behaviors.add("semantic_code_invocation")
    if any(count > 1 for count in Counter(i["programId"] for i in invocations).values()):
        behaviors.add("semantic_repeated_invocation")
    if "track_create" in kinds and "track_join" in kinds and "semantic-completion/4" in kinds:
        behaviors.add("semantic_child")
    if any(i.get("activation") == "module-init" for i in invocations):
        behaviors.add("semantic_initialization")
    if "await" in kinds and "resume" in kinds:
        behaviors.add("semantic_async")
    if "event" in kinds and any(i.get("activation") == "callback" for i in invocations):
        behaviors.add("semantic_callback")
    if "source-evict" in kinds and "source-read" in kinds:
        behaviors.add("semantic_source_reread")
    if "reconcile" in kinds or "callback-reattach" in kinds:
        behaviors.add("semantic_recovery")
    if "cancel" in kinds:
        behaviors.add("semantic_cancel")
    for index, (_mode, kind, _) in enumerate(records):
        if (
            kind == "result"
            and "request" in kinds[:index]
            and any(later[0] == "semantic" for later in records[index + 1 :])
        ):
            behaviors.add("semantic_tool_feedback")
    return behaviors


def prepare_semantic_corpus(
    candidates,
    output,
    *,
    codec,
    producer_revision,
    consumer_revision,
    source_verifier,
    max_positions=1024,
    max_history=1024,
    representation="compact",
    split_seed="semantic-family/4",
):
    """Recheck every admitted candidate and write a native training-ready corpus."""
    if not callable(source_verifier):
        raise ValueError("Production preparation requires an independent source verifier")
    if representation != "compact":
        raise ValueError("Only the canonical semantic witness is supported")
    for revision in (producer_revision, consumer_revision):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(
                "Semantic preparation requires exact producer and consumer Git commits"
            )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    counts, reasons, audits = Counter(), [], []
    config = DedupConfig(
        train_fraction=0.8,
        validation_fraction=0.1,
        split_seed=split_seed,
        max_descendants_per_parent=1024,
    )
    with DedupStore(root / "semantic-dedup.sqlite", config) as store:
        for number, candidate in enumerate(candidates):
            counts["attempted"] += 1
            document = candidate.get("exported", candidate)
            status = candidate.get(
                "status", document.get("sidecar", {}).get("admission", "unsupported")
            )
            if status != "admitted":
                counts[status] += 1
                reasons.append(
                    {
                        "candidate": number,
                        "status": status,
                        "reason": candidate.get("reason", "not admitted"),
                    }
                )
                continue
            try:
                source = source_verifier(document)
                native = native_admission(
                    document,
                    codec=codec,
                    consumer_revision=consumer_revision,
                    max_positions=max_positions,
                    max_history=max_history,
                )
                verify_admission(
                    document,
                    source,
                    native,
                    producer_revision=producer_revision,
                    consumer_revision=consumer_revision,
                    codec=codec,
                )
                episode, _ = adapt_selected_harness_v5(
                    document,
                    codec=codec,
                    max_positions=max_positions,
                    max_history=max_history,
                    admission_verifier=lambda _: True,
                )
                if document["sidecar"].get("representation") != "compact":
                    raise ValueError("Only source-grounded semantic traces may enter the corpus")
                disposition = store.add(episode, codec)
                counts[disposition] += 1
                counts["admitted"] += 1
                audits.append(
                    {
                        "episodeId": episode.episode_id,
                        "source": source,
                        "native": native,
                        "supervision": document["sidecar"],
                    }
                )
            except (ValueError, KeyError, TypeError) as error:
                counts["admission_rejected"] += 1
                reasons.append(
                    {"candidate": number, "status": "admission_rejected", "reason": str(error)}
                )
        store.freeze()
        accepted = list(store.accepted_rows())
        if not accepted:
            from megatoken.io import write_json

            report = {
                "schema": "semantic-corpus-report/4",
                "counts": dict(counts),
                "refusals": reasons,
                "optimizerSteps": 0,
            }
            write_json(root / "semantic-report.json", report)
            raise ValueError("No admitted native episodes; refusal report was retained")
        family_splits = {}
        for row in accepted:
            family = row["resolved_family"]
            if family in family_splits and family_splits[family] != row["split"]:
                raise ValueError("Source ancestry leaked across frozen corpus splits")
            family_splits[family] = row["split"]
        pins = {
            "schema": "semantic-preparation/4",
            "producerRevision": producer_revision,
            "consumerRevision": consumer_revision,
            "catalogDigest": CONTRACT_DIGEST,
            "schemaDigest": SCHEMA_DIGEST,
            "fixtureDigest": FIXTURE_DIGEST,
            "tokenizerIdentity": codec.fingerprint,
            "representation": representation,
            "splitSeed": split_seed,
            "maxPositions": max_positions,
            "maxHistory": max_history,
        }
        report = {
            **pins,
            "counts": dict(counts),
            "accepted": len(accepted),
            "familySplits": family_splits,
            "refusals": reasons,
            "optimizerSteps": 0,
        }
        manifest = export_prepared(
            store,
            root,
            codec=codec,
            source_receipts=[pins],
            pipeline_fingerprint=hashlib.sha256(canonical_json(pins).encode()).hexdigest(),
            chunking=ChunkingConfig(hard_max_tokens=max_positions),
            sidecars={"semantic-report.json": report, "semantic-audit.json": audits},
        )
    return manifest, report
