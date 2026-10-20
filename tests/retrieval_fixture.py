"""Minimal valid coherence_embedding retrieval-memory training setup (tiny, CPU, fp32).

Builds grouped native-meridian episodes (a document and a user request observed in
meridian 0, a generated assistant answer in meridian 1), a causal query sidecar,
a document sidecar, and a pinned recipe whose model geometry equals the tiny
model the engine builds.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.document_embeddings import (
    DocumentEmbeddingExample, write_document_sidecar,
)
from megatoken.data.retrieval import RetrievalExample, write_sidecar
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.loading import load_model_config
from megatoken.memory.contracts import Footprint, QueryBoundary, SemanticSpace, SourceRange
from megatoken.training.config import TrainConfig

SEMANTIC_DIM = 16
SEMANTIC_VALUE_DIM = 8
SCOPE = "e2e-scope"
MODEL_OVERRIDES = {
    "semantic_dim": SEMANTIC_DIM, "semantic_value_dim": SEMANTIC_VALUE_DIM,
}

SPACE = {
    "name": "e2e-space", "revision": "v1", "teacher": "e2e-teacher",
    "teacher_revision": "e2e-teacher-rev-1", "dimension": SEMANTIC_DIM,
    "document_recipe": "e2e-document-v1", "query_recipe": "e2e-query-v1",
    "context_recipe": "e2e-context-v1", "normalized": True,
}


def _vector(seed: int, offset: float = 0.0) -> list[float]:
    # Deterministic, finite, nonzero vectors.
    return [((seed * 31 + i * 7) % 13 + 1 + offset) / 13.0 for i in range(SEMANTIC_DIM)]


def make_episode(index: int) -> Episode:
    doc = f"document {index}: the key {index * 7 + 3} opens the vault".encode()
    request = f"request {index}: which key opens the vault?".encode()
    answer = f"answer {index}: key {index * 7 + 3}".encode()
    names = ("doc", "request", "answer")
    records = (
        PhysicalRecord("doc", "document", doc),
        PhysicalRecord("request", "user", request),
        PhysicalRecord("answer", "assistant", answer, turn_index=1, policy_mask=True),
    )
    tracks = tuple(TrackSpec(n, r.role) for n, r in zip(names, records))
    meridians = (
        Meridian(0, tuple(TrackEvent(n, n, "text", 0, r.role)
                          for n, r in zip(names[:2], records[:2]))),
        Meridian(1, (TrackEvent("answer", "answer", "text", 1, "assistant"),)),
    )
    return Episode(f"episode-{index}", records, Lineage("e2e", "v1", f"family-{index}"),
                   tracks=tracks, meridians=meridians)


def _range(episode: Episode, name: str) -> SourceRange:
    payload = next(r.payload for r in episode.records if r.record_id == name)
    return SourceRange(
        f"{episode.episode_id}/{name}", "v1", 0, len(payload),
        sha256(payload).hexdigest(), SCOPE,
    )


def write_retrieval_artifacts(root: Path, episodes: list[Episode], *, seed: int = 0) -> dict:
    """Write recipe, query sidecar and document sidecar under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    model_config = load_model_config("tiny", **MODEL_OVERRIDES)
    recipe = {
        "schema": "megatoken-retrieval-recipe-v1",
        "phase": "coherence_embedding",
        "space": SPACE,
        "model": model_config.to_dict(),
        "growth": {
            "value_dim": SEMANTIC_VALUE_DIM, "seed": 7, "dense_fraction": 0.75,
            "max_points": 100, "max_pairs": 30, "min_pair_windows": 2,
            "max_neighbors": 4, "partition_id": "e2e-partition-1",
        },
        "loss_weights": {"embedding": 1, "document_embedding": 1},
        "max_selected_source": 4, "max_selected_memory": 4, "max_source_bytes": 256,
        "max_kb_positions": model_config.max_kb_positions,
        "max_readout_groups": 2,
    }
    recipe_path = root / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    space_fp = SemanticSpace(**SPACE).fingerprint
    queries, documents = [], []
    for i, episode in enumerate(episodes):
        request = next(r.payload for r in episode.records if r.record_id == "request")
        doc_range = _range(episode, "doc")
        request_range = _range(episode, "request")
        footprint = Footprint((request_range,))
        support = Footprint((doc_range,))
        queries.append(RetrievalExample(
            episode_id=episode.episode_id, event_id="answer",
            query=QueryBoundary(
                "answer", 1, "rag", "request", sha256(request).hexdigest(),
                space_fp, "snapshot-1", 1,
            ),
            query_teacher_inputs=footprint, document_key_context=support,
            optimization_supervision=support, payload_support=support,
            positive_content_ids=(doc_range.source_id,), support_weights=(1.0,),
            annotations_complete=True, source_snapshot="snapshot-1",
            split_family=f"family-{i}", teacher_target=tuple(_vector(i + seed)),
            main_request_b64=base64.b64encode(request).decode(),
        ))
        doc = next(r.payload for r in episode.records if r.record_id == "doc")
        documents.append(DocumentEmbeddingExample(
            episode.episode_id, "doc", sha256(doc).hexdigest(),
            Footprint((doc_range,)), doc_range, tuple(_vector(i + 5 + seed, 1.0)),
            "e2e-preprocess-1",
        ))
    query_path, document_path = root / "queries.jsonl", root / "documents.jsonl"
    write_sidecar(query_path, queries)
    write_document_sidecar(document_path, documents)
    return {
        "retrieval_memory_recipe": str(recipe_path),
        "retrieval_sidecar": str(query_path),
        "retrieval_document_sidecar": str(document_path),
    }


def prepare_dataset(root: Path, episodes: list[Episode]) -> Path:
    codec = ByteCodec.byte_fallback()
    store = DedupStore(root / "spool.sqlite",
                       DedupConfig(train_fraction=1, validation_fraction=0))
    try:
        with store.transaction():
            for episode in episodes:
                store.add(episode, codec)
        store.freeze()
        export_prepared(store, root / "data", codec=codec, source_receipts=[],
                        pipeline_fingerprint="retrieval-e2e-v1")
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
        phase="B0", dataset=str(dataset), output=str(root / "run"), model="tiny",
        model_overrides=dict(MODEL_OVERRIDES), codec="byte", device="cpu",
        precision="fp32", max_steps=1, microbatch_episodes=2,
        accumulate_scored=1, accumulate_ingested=1000, max_accumulation_windows=2,
        generated_capacity=16, max_history_tokens=256, max_record_bytes=128,
        checkpoint_every=0, eval_every=0, effort=1, gradient_checkpointing=False,
        use_stage_mixture=False, use_effort_mixture=False, budget=64, **artifacts,
    )
    values.update(overrides)
    return TrainConfig(**values)
