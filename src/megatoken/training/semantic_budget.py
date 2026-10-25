"""Explicit experiment exposure budgets and restart state, independent of optimization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .semantic_evaluation import REPRESENTATIONS, verify_cohort

ENGINEERING_GATES = frozenset(
    {
        "protocol",
        "committedBridge",
        "lifecycle",
        "externalEvents",
        "recovery",
        "hostCapabilities",
        "natlang",
        "teacherAdmission",
        "productionTokenizer",
        "inference",
        "trainingInfrastructure",
        "evaluation",
    }
)


@dataclass(frozen=True)
class SemanticExperimentConfig:
    schema: str
    regime: str
    limit: float
    representation: str
    cohort: str
    engineering_receipt: str
    max_token_overshoot: int = 65535

    def __post_init__(self):
        if (
            self.schema != "semantic-training-experiment/4"
            or self.regime not in {"fixed-examples", "fixed-tokens", "measured-compute"}
            or self.representation not in REPRESENTATIONS
        ):
            raise ValueError("Unknown semantic training experiment contract")
        if type(self.limit) not in (int, float) or not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("Semantic experiment budget must be finite and positive")
        if self.regime != "measured-compute" and self.limit != int(self.limit):
            raise ValueError("Example and token budgets must be integer counts")
        if (
            type(self.max_token_overshoot) is not int
            or self.max_token_overshoot < 0
            or not self.cohort
            or not self.engineering_receipt
        ):
            raise ValueError("Semantic experiment requires frozen cohort and engineering evidence")

    def preflight(self, *, phase=None):
        cohort = json.loads(Path(self.cohort).read_text())
        verify_cohort(cohort)
        if phase is not None and (
            phase not in {"SE", "SC", "SO"}
            or cohort["name"] != ("effectful-tracks-events" if phase == "SE" else "common-support")
            or (phase == "SO") != (self.representation == "outcome-only")
        ):
            raise ValueError("Training phase, frozen cohort and declared representation disagree")
        evidence = json.loads(Path(self.engineering_receipt).read_text())
        if (
            evidence.get("schema") != "semantic-engineering-completion/4"
            or evidence.get("complete") is not True
            or evidence.get("optimizerSteps") != 0
            or set(evidence.get("repositories", {})) != {"tracing-js-self", "megatoken", "natlang"}
            or not ENGINEERING_GATES <= set(evidence.get("gates", {}))
            or not all(value is True for value in evidence["gates"].values())
        ):
            raise ValueError(
                "Training cannot precede complete pinned cross-repository engineering evidence"
            )
        for revision in evidence["repositories"].values():
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(c not in "0123456789abcdef" for c in revision)
            ):
                raise ValueError("Engineering receipt must pin all compatible repository commits")
        return cohort["digest"]


def validate_semantic_episode(episode, *, phase, representation):
    """Bind actual native teaching records to the declared experiment arm."""
    from megatoken.data.witness_v4 import graph, parse_record, value
    from megatoken.io import strict_json_loads

    annotation = json.loads(episode.supervision.annotations_json).get("semantic_execution", {})
    if (
        episode.stream != "semantic_execution"
        or not episode.supervision.verified_success
        or annotation.get("schema") != "semantic-execution/4"
        or annotation.get("representation") != representation
        or annotation.get("continuation") != "teacher_conditioned"
        or (phase == "SO") != (representation == "outcome-only")
        or phase not in {"SE", "SC", "SO"}
    ):
        raise ValueError(
            "Prepared episode differs from its certified training representation/stage"
        )
    invocations = generated = 0
    if phase != "SE" and episode.actions:
        raise ValueError("Common-support training cannot contain interactive actions")
    for parts in episode.iter_logical_records():
        if parts[0].logical_id.startswith("semantic:context:"):
            continue
        raw = b"".join(part.payload for part in parts).decode()
        if parts[0].policy_mask:
            if any(part.provenance != "teacher" for part in parts):
                raise ValueError(
                    "Semantic training requires certified teacher generation provenance"
                )
            generated += 1
            if representation == "outcome-only":
                answer = strict_json_loads(raw)
                if (
                    not isinstance(answer, dict)
                    or set(answer) != {"schema", "completion", "state"}
                    or answer["schema"] != "semantic-outcome/4"
                    or set(answer["completion"]) != {"kind", "value"}
                    or answer["completion"]["kind"] not in {"return", "throw"}
                ):
                    raise ValueError("Outcome-only training has non-outcome generation labels")
                value(answer["completion"]["value"])
                graph(answer["state"])
            elif phase == "SC":
                kind, _ = parse_record(raw)
                if kind in {"request", "promise", "await", "resume", "job"}:
                    raise ValueError("Common-support training contains interactive semantics")
        elif phase != "SE":
            event = strict_json_loads(raw)
            allowed = {"program", "invocation"}
            if phase == "SC":
                allowed.update({"source-read", "source-evict", "model-context"})
            if event.get("kind") not in allowed:
                raise ValueError("Common-support training contains future external observations")
            if event["kind"] == "program" and event.get("capabilities"):
                raise ValueError("Common-support training must declare no external capabilities")
            invocations += event["kind"] == "invocation"
    if not generated or (phase != "SE" and invocations != 1) or (phase == "SO" and generated != 1):
        raise ValueError("Prepared native labels do not match the selected training control")


def validate_semantic_geometry(episode, *, max_history=None, max_positions=None):
    geometry = json.loads(episode.supervision.annotations_json).get("semantic_geometry")
    if (
        not isinstance(geometry, dict)
        or set(geometry) != {"schema", "maxHistory", "maxPositions"}
        or geometry["schema"] != "semantic-native-geometry/4"
        or any(
            type(geometry[key]) is not int or geometry[key] < 1
            for key in ("maxHistory", "maxPositions")
        )
    ):
        raise ValueError("Prepared semantic data lacks its exact physical context geometry")
    if (
        max_history is not None
        and geometry["maxHistory"] > max_history
        or max_positions is not None
        and geometry["maxPositions"] > max_positions
    ):
        raise ValueError(
            "Prepared semantic context exceeds the configured model; reprepare with matching geometry"
        )
    return geometry


def validate_stage_supply(
    roots,
    *,
    phase,
    representation,
    codec,
    producer_revision=None,
    consumer_revision=None,
    max_history=None,
    max_positions=None,
):
    """Validate all prepared metadata, exact teaching labels and both curriculum splits.

    This is also the nonoptimizing preparation/planning entry point. It never
    allocates a model, donor pipeline, gradient, optimizer or training stream.
    """
    from megatoken.data.coverage import require_coverage
    from megatoken.data.host_operations import CONTRACT_DIGEST
    from megatoken.data.recipes import stage_recipe
    from megatoken.data.semantic_admission import validate_native_visibility
    from megatoken.data.semantic_contract import FIXTURE_DIGEST, SCHEMA_DIGEST
    from megatoken.data.storage import PreparedCorpus

    if phase not in {"SE", "SC", "SO"} or (phase == "SO") != (representation == "outcome-only"):
        raise ValueError("Semantic curriculum stage and representation disagree")
    pins = {
        "representation": representation,
        "catalogDigest": CONTRACT_DIGEST,
        "schemaDigest": SCHEMA_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "tokenizerIdentity": codec.fingerprint,
    }
    if producer_revision is not None:
        pins["producerRevision"] = producer_revision
    if consumer_revision is not None:
        pins["consumerRevision"] = consumer_revision
    counts = {}
    with PreparedCorpus(roots, split=None) as corpus:
        if not len(corpus) or corpus.codec_fingerprint != codec.fingerprint:
            raise ValueError("Semantic prepared supply is empty or has a different tokenizer")
        for part in corpus.parts:
            receipts = part.manifest.get("sources", [])
            if not receipts or any(
                receipt.get("schema")
                not in {"semantic-preparation/4", "semantic-lifecycle-admission/4"}
                or any(receipt.get(key) != expected for key, expected in pins.items())
                for receipt in receipts
            ):
                raise ValueError(
                    "Prepared manifest differs from the pinned experiment representation/contracts"
                )
        for episode in corpus.iter_episodes():
            validate_semantic_episode(episode, phase=phase, representation=representation)
            validate_semantic_geometry(
                episode, max_history=max_history, max_positions=max_positions
            )
            validate_native_visibility(episode)
        identity = corpus.identity
    recipe = stage_recipe(phase)
    for split in ("train", "validation"):
        with PreparedCorpus(roots, split=split) as corpus:
            if not len(corpus):
                raise ValueError(f"Semantic curriculum has no {split} episodes")
            require_coverage(
                corpus, recipe.constraints["required_behaviors"], recipe.mixture, split=split
            )
            counts[split] = len(corpus)
    return {
        "schema": "semantic-stage-supply/4",
        "phase": phase,
        **pins,
        "corpusIdentity": identity,
        "episodes": counts,
        "optimizerSteps": 0,
    }


@dataclass
class SemanticExposure:
    examples: int = 0
    generatedTokens: int = 0
    observedTokens: int = 0
    computeSeconds: float = 0.0

    def consume(self, *, examples, generatedTokens, observedTokens, computeSeconds):
        values = locals()
        for key in ("examples", "generatedTokens", "observedTokens"):
            if type(values[key]) is not int or values[key] < 0:
                raise ValueError("Exposure counters must be nonnegative measured integers")
        if (
            type(computeSeconds) not in (int, float)
            or not math.isfinite(computeSeconds)
            or computeSeconds < 0
        ):
            raise ValueError("Compute exposure must be a measured nonnegative duration")
        self.examples += examples
        self.generatedTokens += generatedTokens
        self.observedTokens += observedTokens
        self.computeSeconds += computeSeconds

    def consumed(self, config):
        return getattr(
            self,
            {
                "fixed-examples": "examples",
                "fixed-tokens": "generatedTokens",
                "measured-compute": "computeSeconds",
            }[config.regime],
        )

    def finished(self, config):
        return self.consumed(config) >= config.limit

    def checkpoint(self):
        return {"schema": "semantic-exposure/4", **asdict(self)}

    @classmethod
    def restore(cls, value):
        if (
            not isinstance(value, dict)
            or set(value)
            != {"schema", "examples", "generatedTokens", "observedTokens", "computeSeconds"}
            or value["schema"] != "semantic-exposure/4"
        ):
            raise ValueError("Missing or incompatible semantic experiment restart counters")
        result = cls()
        result.consume(**{key: count for key, count in value.items() if key != "schema"})
        return result

    def receipt(self, config, *, seed, checkpoint_identity, cohort_digest):
        overshoot = max(0, self.consumed(config) - config.limit)
        if config.regime == "fixed-examples" and overshoot:
            raise ValueError("Whole-example budget was exceeded")
        if config.regime == "fixed-tokens" and overshoot > config.max_token_overshoot:
            raise ValueError(
                "Atomic token-window overshoot exceeds its frozen comparison allowance"
            )
        return {
            "schema": "semantic-experiment-budget/4",
            "regime": config.regime,
            "limit": config.limit,
            "overshoot": overshoot,
            "maxTokenOvershoot": config.max_token_overshoot,
            "representation": config.representation,
            "seed": seed,
            "checkpointIdentity": checkpoint_identity,
            "cohortDigest": cohort_digest,
            **asdict(self),
        }


def semantic_training_state(exposure, *, cohort_digest):
    """Persist experiment counters only for an explicitly configured experiment."""
    if exposure is None:
        if cohort_digest is not None:
            raise ValueError("A semantic cohort requires its experiment exposure state")
        return {}
    if (
        not isinstance(cohort_digest, str)
        or len(cohort_digest) != 64
        or any(character not in "0123456789abcdef" for character in cohort_digest)
    ):
        raise ValueError("Semantic exposure requires its frozen cohort digest")
    return {
        "semantic_exposure": exposure.checkpoint(),
        "semantic_cohort_digest": cohort_digest,
    }


def restore_semantic_training_state(payload, *, cohort_digest):
    """Restore the exact configured experiment, without inferring absent fields."""
    fields = {"semantic_exposure", "semantic_cohort_digest"}
    present = fields.intersection(payload)
    if cohort_digest is None:
        if present:
            raise ValueError("A nonsemantic checkpoint cannot contain semantic experiment state")
        return None
    if present != fields or payload["semantic_cohort_digest"] != cohort_digest:
        raise ValueError("Semantic restart requires its exact exposure state and frozen cohort")
    exposure = SemanticExposure.restore(payload["semantic_exposure"])
    semantic_training_state(exposure, cohort_digest=cohort_digest)
    return exposure
