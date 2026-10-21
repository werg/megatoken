"""Query-purpose R must read P's selected event, without training-time state leakage."""

import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator


def test_query_embedding_training_commits_selected_p_draw_without_answer_leakage():
    torch.manual_seed(5)
    model = FacetedGenerator(
        FacetedCore(
            ModelConfig.tiny(
                semantic_dim=8,
                semantic_value_dim=16,
                max_kb_positions=2,
            )
        ),
        ByteCodec.byte_fallback(),
    )
    multi = MultiTrackGenerator(model)
    state = multi.tracks.initial_state(1)
    generator = torch.Generator().manual_seed(19)
    query, updated, raw = multi.retrieve_embedding(
        state,
        main_request=b"Which project change fixed the bug?",
        purpose="source_retrieval",
        space_fingerprint="pinned-space-v1",
        effort=1,
        generator=generator,
    )
    assert updated is not state
    assert state.history.slots == 0
    assert updated.history.slots == model.config.n_facets
    assert updated.meridians_seen.item() == state.meridians_seen.item() + 1
    assert query.shape == (1, 8)
    assert raw.shape == (
        1,
        1,
        model.config.n_facets,
        model.config.facet_dim,
    )
    assert torch.allclose(query.norm(dim=-1), torch.ones(1), atol=1e-5)
    query[:, 0].sum().backward()
    assert model.embedding_realizer.query.weight.grad is not None
    assert model.embedding_realizer.query.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.predictor_parameters()
    )
    with torch.no_grad():
        served, replayed, served_raw = multi.retrieve_embedding(
            state,
            main_request=b"Which project change fixed the bug?",
            purpose="source_retrieval",
            space_fingerprint="pinned-space-v1",
            effort=1,
            generator=torch.Generator().manual_seed(19),
        )
    torch.testing.assert_close(served, query.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(served_raw, raw.detach(), atol=1e-5, rtol=1e-5)
    assert replayed.history.slots == updated.history.slots
    assert replayed.meridians_seen.item() == updated.meridians_seen.item()
