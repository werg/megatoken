"""Native bank-aware grouped training and canonical sparse publication regressions."""
from hashlib import sha256

import numpy as np
import torch

from megatoken.data.retrieval import RetrievalExample
from megatoken.memory.contracts import (
    Collection, Footprint, QueryBoundary, SemanticSpace, SourceRange,
)
from megatoken.memory.growth import GrowthPolicy
from megatoken.memory.recipe import RetrievalRecipe
from megatoken.memory.sqlite import SQLiteKB
from megatoken.model.config import ModelConfig
from megatoken.model.core import FacetedCore
from megatoken.model.multitrack import MultiTrackCore
from megatoken.training.bank import (
    BankTrainingOwner, synchronize_query_demand, synchronize_sparse_rows,
)


class _OneRank:
    is_primary = True

    @staticmethod
    def all_gather_object(value):
        return [value]

    @staticmethod
    def barrier():
        pass


def _fixture(tmp_path, *, early=True):
    config = ModelConfig.tiny(
        semantic_dim=8, semantic_value_dim=16, max_kb_positions=2,
    )
    space = SemanticSpace(
        "shared", "r1", "local", "pinned-teacher", 8,
        "document", "query", "ordered-source",
    )
    recipe = RetrievalRecipe(
        schema="megatoken-retrieval-recipe-v1",
        space=space, model=config,
        growth=GrowthPolicy(value_dim=16, seed=9, max_points=10, max_pairs=5),
        loss_weights={
            "embedding": 1., "query_ranking": 0.5,
            "source_value": 0.1,
        },
        phase="progressive_warmup",
        max_selected_source=2,
        max_selected_memory=2,
        max_kb_positions=2, max_readout_groups=2,
    )
    bank = SQLiteKB(tmp_path / "kb.sqlite", space, value_dim=16)
    available_at = 0 if early else 1
    footprint = Footprint((
        SourceRange(
            "source-a", "v1", 0, 20, "sha-source", "team",
            available_at=available_at, family="family-a",
        ),
    ))
    bank.insert_source(
        scope="team", footprint=footprint,
        content_id="source-a/0", asset_ref="source-asset",
        key=np.array([1.] + [0.] * 7),
    )
    bank.insert_memory(
        scope="team", footprint=footprint, assignment="source-a/0",
        key=np.array([1.] + [0.] * 7),
        value=np.ones(16, dtype=np.float32) * 0.01, birth_step=0,
    )
    bank.put_growth_state("readiness", {"coherence": True, "embeddings": True})
    owner = BankTrainingOwner(bank, recipe, device=torch.device("cpu"))
    boundary = QueryBoundary(
        "event-0", 0, "source", "main", sha256(b"Request").hexdigest(),
        space.fingerprint, "snapshot", 3,
    )
    query_context = Footprint((SourceRange(
        "request", "v1", 0, 7, "sha-request", "team", available_at=0,
    ),))
    annotation = RetrievalExample(
        episode_id="ep-0", event_id="event-0", query=boundary,
        query_teacher_inputs=query_context,
        document_key_context=footprint,
        optimization_supervision=footprint,
        payload_support=footprint,
        positive_content_ids=("source-a/0",), support_weights=(1.,),
        annotations_complete=True, source_snapshot="snapshot",
        split_family="family-a", teacher_target=tuple([1.] + [0.] * 7),
    )
    return config, bank, owner, annotation


def test_live_source_and_memory_leaves_train_through_actual_native_kv(tmp_path):
    config, bank, owner, annotation = _fixture(tmp_path)
    try:
        tracks = MultiTrackCore(FacetedCore(config))
        state = tracks.initial_state(1)
        query = torch.tensor([[0.8, 0.6] + [0.] * 6], requires_grad=True)
        read = owner.read(state, query, annotation, global_count=1)
        assert read.source_hits == read.memory_hits == read.memory_groups == 1
        assert read.state.history.slots == state.history.slots == 0
        assert torch.equal(read.state.meridians_seen, state.meridians_seen)
        future = tracks.predict_group(read.state, tracks=("main",), effort=1)
        loss = read.loss + future.square().mean()
        loss.backward()
        assert query.grad is not None and query.grad.abs().sum() > 0
        assert tracks.core.memory_projector.to_kv.weight.grad is not None
        assert tracks.core.memory_adapters.controller.weight.grad is not None
        grads = {
            (k.collection, k.field): v.grad
            for k, v in owner.sparse.leaves.items()
        }
        assert grads[(Collection.SOURCE, "key")] is not None
        assert grads[(Collection.MEMORY, "key")] is not None
        assert grads[(Collection.MEMORY, "value")] is not None
        assert synchronize_sparse_rows(owner, _OneRank(), optimizer_step=1, max_norm=1.) >= 3
        assert bank.get(Collection.MEMORY, 2).payload_revision > 1
    finally:
        bank.close()


def test_future_source_is_never_selected_by_delivery_deadline(tmp_path):
    config, bank, owner, annotation = _fixture(tmp_path, early=False)
    try:
        tracks = MultiTrackCore(FacetedCore(config))
        state = tracks.initial_state(1)
        query = torch.tensor([[1.] + [0.] * 7], requires_grad=True)
        read = owner.read(state, query, annotation, global_count=1)
        assert read.source_hits == read.memory_hits == read.memory_groups == 0
    finally:
        bank.close()


def test_demand_is_deduplicated_only_at_primary_barrier(tmp_path):
    config, bank, owner, annotation = _fixture(tmp_path)
    try:
        state = MultiTrackCore(FacetedCore(config)).initial_state(1)
        query = torch.tensor([[1.] + [0.] * 7])
        owner.read(state, query, annotation, global_count=1)
        assert bank.db.execute("SELECT count(*) FROM demand_events").fetchone()[0] == 0
        assert synchronize_query_demand(owner, _OneRank()) == 1
        owner.read(state, query, annotation, global_count=1)
        assert synchronize_query_demand(owner, _OneRank()) == 0
        assert bank.db.execute("SELECT count(*) FROM demand_events").fetchone()[0] == 1
    finally:
        bank.close()
