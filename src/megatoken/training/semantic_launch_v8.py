"""Direct V8 production launch authority, corpus and continuation checks.

These checks read the actual admitted rows and ordinary stream identities. They
do not construct a model, start a learner, or turn diagnostic supply into a
production curriculum.
"""

from __future__ import annotations

from megatoken.data.semantic_contract_v8 import (
    CONTRACT_DIGEST,
    FIXTURE_DIGEST,
    INTEROPERABILITY_DIGEST,
)
from megatoken.generation.codec import LING_TOKENIZER_SHA256
from megatoken.training.semantic_v8 import validate_semantic_training_policy_v8


def qualify_policy_v8(config, repositories):
    policy = validate_semantic_training_policy_v8(config.semantic_v8_policy)
    for name, key in (
        ("tracing-js-self", "producerRevision"),
        ("megatoken", "consumerRevision"),
    ):
        identity = repositories.get(name)
        if (
            not isinstance(identity, dict)
            or identity.get("clean") is not True
            or identity.get("revision") != policy[key]
        ):
            raise ValueError(f"V8 launch policy differs from the clean {name} checkout")
    return {
        "schema": "semantic-launch-authority/8",
        "policy": dict(policy),
        "contractDigest": CONTRACT_DIGEST,
        "fixtureDigest": FIXTURE_DIGEST,
        "interoperabilityDigest": INTEROPERABILITY_DIGEST,
        "tokenizerSha256": LING_TOKENIZER_SHA256,
        "cohortDigest": None,
        "scope": "Exact code and contract pins; per-episode source/native proof is checked by the corpus gate",
    }


def qualify_corpus_v8(config, codec, model_config):
    from megatoken.data import PreparedCorpus
    from megatoken.data.chunking import ChunkingConfig
    from megatoken.data.recipes import stage_recipe, validate_stage_supply
    from megatoken.io import strict_json_loads
    from megatoken.semantic_lineage_v8 import build_training_lineage_v8
    from megatoken.training.semantic_v8 import preflight_semantic_training_v8
    from megatoken.training.stream import MixtureWindowStream
    from megatoken.training.windows import window_config

    if (
        config.purpose != "production"
        or config.phase not in {"SC", "SE"}
        or not config.use_stage_mixture
        or config.decoupled is None
    ):
        raise ValueError("V8 launch supply requires the ordinary production semantic recipe")
    supply = preflight_semantic_training_v8(
        config.datasets,
        codec=codec,
        policy=config.semantic_v8_policy,
        max_history=model_config.max_history_chunks,
        max_positions=model_config.max_capacity,
    )
    # Both actual splits must satisfy the unmodified production requirements.
    # In particular, SC data cannot claim SE child/repeated/source-reread supply.
    for split in ("train", "validation"):
        with PreparedCorpus(config.datasets, split=split) as corpus:
            if not len(corpus):
                raise ValueError(f"V8 production curriculum has no {split} episodes")
            validate_stage_supply(config, corpus, split=split)
    with PreparedCorpus(config.datasets, split=None) as corpus:
        corpus_identity = corpus.identity
        # Ordinary training includes every immutable root, including empty
        # shards. ``parts`` omits empty roots even though corpus.identity does
        # not; using it here would give preflight a different lineage.
        manifests = [
            strict_json_loads((root / "manifest.json").read_bytes())["fingerprint"]
            for root in corpus.all_roots
        ]
        chunking = ChunkingConfig.from_identity(corpus.chunking, codec).identity(codec)
    recipe = stage_recipe(config.phase)
    stream = MixtureWindowStream(
        config.datasets,
        codec,
        window_config(config),
        recipe.mixture,
        config.schedule.meter,
        seed=config.seed,
        rank=config.decoupled.resolved_worker_id,
        world_size=config.decoupled.workers,
        reduce_counts=lambda counts: counts,
        independent=True,
    )
    try:
        lineage = build_training_lineage_v8(
            policy=config.semantic_v8_policy,
            tokenizer_identity=codec.fingerprint,
            prepared_manifests=manifests,
            corpus_identity=corpus_identity,
            stream_identity=stream.fingerprint,
            chunking=chunking,
        )
    finally:
        stream.close()
    return {
        **supply,
        "phase": config.phase,
        "corpusIdentity": corpus_identity,
        "trainingLineage": lineage,
        "coverage": {"train": "passed", "validation": "passed"},
        "scope": "Every admitted row, both production splits and the ordinary training stream; no optimizer",
    }


def qualify_checkpoint_lineage_v8(
    payload, metadata, *, config, supply, coordinator_bootstrap=False
):
    from megatoken.semantic_lineage_v8 import checkpoint_lineage_v8

    lineage = checkpoint_lineage_v8(payload, metadata)
    if {"semantic_exposure", "semantic_cohort_digest"}.intersection(payload):
        raise ValueError("V8 launch cannot resume legacy semantic experiment state")
    if coordinator_bootstrap and payload["train_config"]["phase"] != config.phase:
        # A previous curriculum stage supplies weights, not a V8 continuation.
        # Its own lineage, when present, has still been checked above. The
        # coordinator must separately bind the destination's actual V8 supply.
        return lineage
    if (
        lineage is None
        or not isinstance(supply, dict)
        or lineage != supply.get("trainingLineage")
        or lineage["policy"] != config.semantic_v8_policy
    ):
        raise ValueError("V8 checkpoint differs from the freshly qualified launch corpus or stream")
    return lineage


def qualify_coordinator_lineage_v8(manifest, *, config, codec, corpus_identity, lineage):
    from megatoken.semantic_lineage_v8 import validate_training_lineage_v8

    actual = validate_training_lineage_v8(
        manifest.get("semantic_v8_lineage"),
        policy=config.semantic_v8_policy,
        tokenizer_identity=codec.fingerprint,
    )
    if (
        manifest.get("semantic_v8_policy") != config.semantic_v8_policy
        or actual != lineage
        or actual["preparedCorpusDigest"] != corpus_identity
        or manifest.get("chunking_identity", {}).get("fingerprint") != actual["chunkingFingerprint"]
    ):
        raise ValueError(
            "V8 coordinator differs from the freshly qualified launch corpus or stream"
        )
    return actual["digest"]
