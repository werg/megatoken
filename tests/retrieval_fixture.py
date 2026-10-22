"""Minimal valid coherence_embedding retrieval-memory training setup (tiny, CPU, fp32).

Builds grouped native-meridian episodes (a document and a user request observed in
meridian 0, a generated assistant answer in meridian 1), a causal query sidecar,
a document sidecar, and a pinned recipe whose model geometry equals the tiny
model the engine builds.
"""

from __future__ import annotations

import base64
import json
from hashlib import sha256
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.document_embeddings import (
    DocumentEmbeddingExample,
    write_document_sidecar,
)
from megatoken.data.retrieval import RetrievalExample, write_sidecar
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.loading import load_model_config
from megatoken.memory.contracts import Footprint, QueryBoundary, SemanticSpace, SourceRange
from megatoken.teacher import PINNED_TEACHER
from megatoken.training.config import TrainConfig

SEMANTIC_DIM = 16
SEMANTIC_VALUE_DIM = 8
SCOPE = "e2e-scope"
# Model preset for every artifact; scripts may switch it (e.g. "32x512") for GPU runs.
MODEL = "tiny"
MODEL_OVERRIDES = {
    "semantic_dim": SEMANTIC_DIM,
    "semantic_value_dim": SEMANTIC_VALUE_DIM,
}

SPACE = {
    "name": "e2e-space",
    "revision": "v1",
    "teacher": PINNED_TEACHER,
    "teacher_revision": "e2e-teacher-rev-1",
    "dimension": SEMANTIC_DIM,
    "document_recipe": "e2e-document-v1",
    "query_recipe": "e2e-query-v1",
    "context_recipe": "e2e-context-v1",
    "normalized": True,
}


def _vector(seed: int, offset: float = 0.0, dimension: int = SEMANTIC_DIM) -> list[float]:
    # Deterministic, finite, nonzero vectors.
    return [((seed * 31 + i * 7) % 13 + 1 + offset) / 13.0 for i in range(dimension)]


def make_episode(index: int, *, turns: int = 1) -> Episode:
    """Meridian 0 observes doc+request; then answer, [request2, answer2, ...]."""
    doc = f"document {index}: the key {index * 7 + 3} opens the vault".encode()
    request = f"request {index}: which key opens the vault?".encode()
    answer = f"answer {index}: key {index * 7 + 3}".encode()
    names = ["doc", "request", "answer"]
    records = [
        PhysicalRecord("doc", "document", doc),
        PhysicalRecord("request", "user", request),
        PhysicalRecord("answer", "assistant", answer, turn_index=1, policy_mask=True),
    ]
    meridians = [
        Meridian(
            0,
            tuple(
                TrackEvent(n, n, "text", 0, r.role)
                for n, r in zip(names[:2], records[:2], strict=True)
            ),
        ),
        Meridian(1, (TrackEvent("answer", "answer", "text", 1, "assistant"),)),
    ]
    for turn in range(2, turns + 1):
        ask, reply = f"request{turn}", f"answer{turn}"
        records += [
            PhysicalRecord(
                ask,
                "user",
                f"request {index}.{turn}: repeat the key?".encode(),
                turn_index=2 * turn - 2,
            ),
            PhysicalRecord(
                reply,
                "assistant",
                f"answer {index}.{turn}: {index * 7 + 3}".encode(),
                turn_index=2 * turn - 1,
                policy_mask=True,
            ),
        ]
        names += [ask, reply]
        meridians += [
            Meridian(len(meridians), (TrackEvent(ask, ask, "text", 2 * turn - 2, "user"),)),
            Meridian(
                len(meridians) + 1, (TrackEvent(reply, reply, "text", 2 * turn - 1, "assistant"),)
            ),
        ]
    records, meridians = tuple(records), tuple(meridians)
    tracks = tuple(TrackSpec(n, r.role) for n, r in zip(names, records, strict=True))
    return Episode(
        f"episode-{index}",
        records,
        Lineage("e2e", "v1", f"family-{index}"),
        tracks=tracks,
        meridians=meridians,
    )


def _range(episode: Episode, name: str) -> SourceRange:
    payload = next(r.payload for r in episode.records if r.record_id == name)
    return SourceRange(
        f"{episode.episode_id}/{name}",
        "v1",
        0,
        len(payload),
        sha256(payload).hexdigest(),
        SCOPE,
    )


def write_retrieval_artifacts(
    root: Path,
    episodes: list[Episode],
    *,
    seed: int = 0,
    phase: str = "coherence_embedding",
    source_snapshot: str = "snapshot-1",
    doc: str = "doc",
    request: str = "request",
    answer: str = "answer",
    turns: int = 1,
    dimension: int = SEMANTIC_DIM,
) -> dict:
    """Write recipe, query sidecar and document sidecar under ``root``.

    ``doc``/``request``/``answer`` name the episode records they annotate."""
    root.mkdir(parents=True, exist_ok=True)
    model_config = load_model_config(MODEL, **{**MODEL_OVERRIDES, "semantic_dim": dimension})
    recipe = {
        "schema": "megatoken-retrieval-recipe-v1",
        "phase": phase,
        "space": {**SPACE, "dimension": dimension},
        "model": model_config.to_dict(),
        "growth": {
            "value_dim": SEMANTIC_VALUE_DIM,
            "seed": 7,
            "dense_fraction": 0.75,
            "max_points": 100,
            "max_pairs": 30,
            "min_pair_windows": 2,
            "max_neighbors": 4,
            "partition_id": "e2e-partition-1",
        },
        "loss_weights": {"embedding": 1, "document_embedding": 1},
        "max_selected_source": 4,
        "max_selected_memory": 4,
        "max_source_bytes": 256,
        "max_kb_positions": model_config.max_kb_positions,
        "max_readout_groups": 2,
        "embedding_centers": {role: [0.0] * dimension for role in ("query", "document")},
    }
    recipe_path = root / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    space_fp = SemanticSpace(**{**SPACE, "dimension": dimension}).fingerprint
    queries, documents = [], []
    # Turn t's query sits at the causal clock of its answering meridian: 2t - 1.
    asked = [(request, answer, 1)] + [
        (f"request{t}", f"answer{t}", 2 * t - 1) for t in range(2, turns + 1)
    ]
    for i, episode in enumerate(episodes):
        doc_range = _range(episode, doc)
        support = Footprint((doc_range,))
        for ask, reply, clock in asked:
            request_bytes = next(r.payload for r in episode.records if r.record_id == ask)
            request_range = _range(episode, ask)
            footprint = Footprint((request_range,))
            queries.append(
                RetrievalExample(
                    episode_id=episode.episode_id,
                    event_id=reply,
                    query=QueryBoundary(
                        reply,
                        clock,
                        "rag",
                        ask,
                        sha256(request_bytes).hexdigest(),
                        space_fp,
                        source_snapshot,
                        clock,
                    ),
                    query_teacher_inputs=footprint,
                    document_key_context=support,
                    optimization_supervision=support,
                    payload_support=support,
                    positive_content_ids=(doc_range.source_id,),
                    support_weights=(1.0,),
                    annotations_complete=True,
                    source_snapshot=source_snapshot,
                    split_family=f"family-{i}",
                    teacher_target=tuple(_vector(i + seed, dimension=dimension)),
                    main_request_b64=base64.b64encode(request_bytes).decode(),
                )
            )
        doc_bytes = next(r.payload for r in episode.records if r.record_id == doc)
        documents.append(
            DocumentEmbeddingExample(
                episode.episode_id,
                doc,
                sha256(doc_bytes).hexdigest(),
                Footprint((doc_range,)),
                doc_range,
                tuple(_vector(i + 5 + seed, 1.0, dimension)),
                "e2e-preprocess-1",
            )
        )
    query_path, document_path = root / "queries.jsonl", root / "documents.jsonl"
    write_sidecar(query_path, queries)
    write_document_sidecar(document_path, documents)
    return {
        "retrieval_memory_recipe": str(recipe_path),
        "retrieval_sidecar": str(query_path),
        "retrieval_document_sidecar": str(document_path),
    }


def prepare_dataset(root: Path, episodes: list[Episode], *, heldout: bool = False) -> Path:
    """Prepared byte-codec episodes, entirely in the train or the validation split."""
    codec = ByteCodec.byte_fallback()
    store = DedupStore(
        root / "spool.sqlite",
        DedupConfig(train_fraction=0 if heldout else 1, validation_fraction=1 if heldout else 0),
    )
    try:
        with store.transaction():
            for episode in episodes:
                store.add(episode, codec)
        store.freeze()
        export_prepared(
            store,
            root / "data",
            codec=codec,
            source_receipts=[],
            pipeline_fingerprint="retrieval-e2e-v1",
        )
    finally:
        store.close()
    return root / "data"


def retrieval_train_config(root: Path, *, episodes: int = 4, **overrides) -> TrainConfig:
    """A valid coherence_embedding TrainConfig; ``root`` receives data, sidecars and run."""
    root = Path(root)
    items = [make_episode(i) for i in range(episodes)]
    dataset = prepare_dataset(root, items)
    artifacts = write_retrieval_artifacts(root / "retrieval", items)
    values = dict(
        phase="B0",
        dataset=str(dataset),
        output=str(root / "run"),
        model=MODEL,
        model_overrides=dict(MODEL_OVERRIDES),
        codec="byte",
        device="cpu",
        precision="fp32",
        max_steps=1,
        microbatch_episodes=2,
        accumulate_scored=1,
        accumulate_ingested=1000,
        max_accumulation_windows=2,
        generated_capacity=16,
        max_history_tokens=256,
        max_record_bytes=128,
        checkpoint_every=0,
        eval_every=0,
        effort=1,
        gradient_checkpointing=False,
        use_stage_mixture=False,
        use_effort_mixture=False,
        budget=64,
        **artifacts,
    )
    values.update(overrides)
    return TrainConfig(**values)


def live_kb_train_config(
    root: Path,
    *,
    phase: str = "M1",
    episodes: int = 4,
    turns: int = 1,
    **overrides,
) -> TrainConfig:
    """A valid live free-KB TrainConfig (progressive_warmup/M1 or free_kb_pretraining/M2).

    Builds original source assets, a catalog-imported bank whose source rows are keyed
    by the document teacher vectors (so every fixture query can retrieve its document),
    measured-readiness state, a passing phase gate receipt and a matching live recipe.
    """
    from dataclasses import asdict

    from megatoken.io import sha256_file
    from megatoken.memory.catalog_cli import build_source_catalog
    from megatoken.memory.recipe import PHASE_REQUIREMENTS, RetrievalRecipe, open_bank_for_recipe

    root = Path(root)
    recipe_phase = {"M1": "progressive_warmup", "M2": "free_kb_pretraining"}[phase]
    items = [make_episode(i, turns=turns) for i in range(episodes)]
    dataset = prepare_dataset(root, items)
    live = root / "live"
    assets = live / "assets"
    assets.mkdir(parents=True)
    boot = write_retrieval_artifacts(live / "bootstrap", items)
    bootstrap_recipe = RetrievalRecipe.load(boot["retrieval_memory_recipe"])
    entries = []
    for i, episode in enumerate(items):
        payload = next(r.payload for r in episode.records if r.record_id == "doc")
        (assets / f"doc-{i}.bin").write_bytes(payload)
        span = _range(episode, "doc")
        key = _vector(i + 5, 1.0)
        norm = sum(x * x for x in key) ** 0.5
        entries.append(
            {
                "semantic_space_fingerprint": bootstrap_recipe.space.fingerprint,
                "scope": SCOPE,
                "content_id": span.source_id,
                "asset_ref": f"doc-{i}.bin",
                "asset_sha256": sha256(payload).hexdigest(),
                "footprint": [asdict(span)],
                "evidence_span": asdict(span),
                "key": [x / norm for x in key],
                "source_windows": [f"window-{i}-a", f"window-{i}-b"],
            }
        )
    catalog = live / "catalog.jsonl"
    catalog.write_text("".join(json.dumps(e) + "\n" for e in entries))
    bank = live / "kb.sqlite"
    build_source_catalog(bootstrap_recipe, catalog=catalog, assets_root=assets, output=bank)
    artifacts = write_retrieval_artifacts(
        live / "retrieval",
        items,
        phase=recipe_phase,
        source_snapshot=sha256_file(catalog),
        turns=turns,
    )
    recipe = RetrievalRecipe.load(artifacts["retrieval_memory_recipe"])
    with open_bank_for_recipe(recipe, bank) as opened:
        opened.put_growth_state(
            "readiness",
            {
                "coherence": True,
                "embeddings": True,
                "space": recipe.space.fingerprint,
                "recipe": bootstrap_recipe.fingerprint,
            },
        )
    needed = PHASE_REQUIREMENTS[recipe_phase]
    receipt = live / "gate.json"
    receipt.write_text(
        json.dumps(
            {
                "phase": recipe_phase,
                "evidence": {g: f"evidence/{g}.json" for g in needed},
                "model_snapshot": "e2e-model",
                "kb_snapshot": "e2e-kb",
                "corpus_manifest": "e2e-corpus",
                "successful": list(needed),
            }
        )
    )
    values = dict(
        phase=phase,
        dataset=str(dataset),
        output=str(root / "run"),
        model=MODEL,
        model_overrides=dict(MODEL_OVERRIDES),
        codec="byte",
        device="cpu",
        precision="fp32",
        max_steps=2,
        microbatch_episodes=2,
        accumulate_scored=1,
        accumulate_ingested=1000,
        max_accumulation_windows=2,
        generated_capacity=16,
        max_history_tokens=256,
        max_record_bytes=128,
        checkpoint_every=0,
        eval_every=0,
        effort=1,
        gradient_checkpointing=False,
        use_stage_mixture=False,
        use_effort_mixture=False,
        budget=64,
        retrieval_bank=str(bank),
        retrieval_gate_receipt=str(receipt),
        retrieval_source_assets_root=str(assets),
        retrieval_growth_every=1,
        retrieval_growth_budget=2,
        **artifacts,
    )
    values.update(overrides)
    return TrainConfig(**values)


def retrieval_aux(config: TrainConfig, *, supervised: bool = True) -> tuple:
    """The 6-tuple ``forward_meridian_batch`` takes as ``retrieval_aux``, built as ``train()`` does.

    ``supervised=True`` loads the config's real query/document sidecars (use with the
    fixture episodes). ``supervised=False`` keeps the pinned recipe but carries no query
    or document annotations, for tests of grouped/media mechanics on custom episodes.
    """
    from megatoken.data.document_embeddings import read_document_sidecar
    from megatoken.data.retrieval import read_sidecar
    from megatoken.memory.recipe import RetrievalRecipe

    recipe = RetrievalRecipe.load(config.retrieval_memory_recipe)
    if not supervised:
        return {}, recipe, 1, None, {}, 1
    queries = {(e.episode_id, e.event_id): e for e in read_sidecar(config.retrieval_sidecar)}
    documents = {
        (d.episode_id, d.event_id): d
        for d in read_document_sidecar(config.retrieval_document_sidecar)
    }
    return queries, recipe, len(queries), None, documents, len(documents)
