"""Real free-bank training executes one selected P query and literal RAG E/C admission."""

import numpy as np
import torch

from megatoken.data.retrieval import RetrievalExample
from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.memory.contracts import Footprint, QueryBoundary, SemanticSpace, SourceRange
from megatoken.memory.growth import GrowthPolicy
from megatoken.memory.recipe import RetrievalRecipe
from megatoken.memory.retrieval import RetrievalSession
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.training.bank import BankTrainingOwner
from megatoken.training.source_features import NativeSourceFeatures


def test_selected_query_then_actual_source_bytes_advance_native_clock(tmp_path):
    torch.manual_seed(29)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    raw = b"Original literal evidence for retrieval."
    (source_root / "original.bin").write_bytes(raw)
    source = SourceRange(
        "source-a",
        "v1",
        0,
        len(raw),
        "sha-original-content",
        "team",
    )
    request = SourceRange(
        "request-a",
        "v1",
        0,
        10,
        "sha-request",
        "team",
    )
    space = SemanticSpace(
        "space",
        "v1",
        "teacher",
        "pinned",
        8,
        "doc",
        "query",
        "context",
    )
    config = ModelConfig.tiny(
        semantic_dim=8,
        semantic_value_dim=16,
        max_kb_positions=2,
    )
    recipe = RetrievalRecipe(
        schema="megatoken-retrieval-recipe-v1",
        phase="progressive_warmup",
        space=space,
        model=config,
        growth=GrowthPolicy(value_dim=16, seed=8),
        loss_weights={"embedding": 1.0, "query_ranking": 0.25, "source_value": 0.1},
        max_selected_source=2,
        max_selected_memory=2,
        max_source_bytes=128,
        max_kb_positions=2,
        max_readout_groups=2,
        extra_rules={"source_threshold": "-1.0"},
    )
    model = FacetedGenerator(FacetedCore(config), ByteCodec.byte_fallback())
    multi = MultiTrackGenerator(model)
    with SQLiteKB(tmp_path / "knowledge.sqlite", space, value_dim=16) as bank:
        bank.insert_source(
            scope="team",
            footprint=Footprint((source,)),
            content_id="source-a/chunk0",
            asset_ref="original.bin",
            key=np.array([1.0] + [0.0] * 7),
        )
        bank.put_growth_state(
            "source_catalog",
            {
                "snapshot": "corpus-snapshot",
                "space": space.fingerprint,
                "partition_id": recipe.growth.partition_id,
            },
        )
        feature_reader = NativeSourceFeatures(
            model,
            source_root,
            max_chunk_bytes=128,
        )
        owner = BankTrainingOwner(
            bank,
            recipe,
            device=torch.device("cpu"),
            source_features=feature_reader,
        )
        annotation = RetrievalExample(
            episode_id="ep",
            event_id="query-0",
            query=QueryBoundary(
                "query-0",
                0,
                "source",
                "main",
                "request-digest",
                space.fingerprint,
                "initial",
                4,
            ),
            query_teacher_inputs=Footprint((request,)),
            document_key_context=Footprint((source,)),
            optimization_supervision=Footprint((source,)),
            payload_support=Footprint((source,)),
            positive_content_ids=("source-a/chunk0",),
            support_weights=(1.0,),
            annotations_complete=True,
            source_snapshot="corpus-snapshot",
            split_family="family-a",
            teacher_target=tuple([1.0] + [0.0] * 7),
        )
        start = multi.tracks.initial_state(1)
        embedding, query_state, selected = multi.retrieve_embedding(
            start,
            main_request=b"search source-a",
            purpose="source",
            space_fingerprint=space.fingerprint,
            effort=1,
        )
        assert query_state.history.slots > start.history.slots
        coverage = RetrievalSession()
        result = owner.read(
            query_state,
            embedding,
            annotation,
            global_count=1,
            clock=annotation.query.meridian,
            multi=multi,
            coverage=coverage,
        )
        assert result.source_hits == 1
        assert result.source_committed == 1
        assert result.source_bytes == len(raw)
        assert result.state.meridians_seen.item() == 2
        assert coverage.injected[0].content_id == "source-a"
        assert result.state.history.slots > query_state.history.slots
        future = multi.tracks.predict_group(
            result.state,
            tracks=("assistant",),
            effort=1,
        )
        (future.square().mean() + result.loss).backward()
        assert embedding.grad_fn is not None
        assert any(p.grad is not None for p in model.embedding_realizer.parameters())
