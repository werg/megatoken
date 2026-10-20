"""Writer behavior must train through actual chronological native P/C/R losses."""

import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.memory.writers import AssignmentTargets, SourceWriter
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.training.hindsight import (
    SourceWriterBatch,
    WriterFitSettings,
    fit_source_writer_epoch,
)
from megatoken.training.hindsight_native import (
    ContinuationMeridian,
    WriterContinuation,
    native_source_continuation,
)


def test_permitted_suffix_writer_receives_native_continuation_gradient():
    torch.manual_seed(11)
    generator = MultiTrackGenerator(
        FacetedGenerator(
            FacetedCore(
                ModelConfig.tiny(
                    semantic_dim=8,
                    semantic_value_dim=16,
                    max_kb_positions=2,
                )
            ),
            ByteCodec.byte_fallback(),
        )
    )
    model = generator.generator
    native = generator.tracks.initial_state(1)
    c = model.config
    observation = torch.randn(1, 1, c.n_facets, c.facet_dim)
    plan = model.prepare_target(
        target_bytes=b"native response",
        representation_mode="bootstrap",
    )
    trajectory = WriterContinuation(
        batch_index=0,
        source_slot=0,
        source_meridian=0,
        initial_state=native,
        scope="team",
        identity="source-assignment-a",
        events=(
            ContinuationMeridian(
                meridian=0,
                tracks=("source",),
                roles=(3,),
                raw=observation,
            ),
            ContinuationMeridian(
                meridian=1,
                tracks=("source",),
                roles=(3,),
                raw=observation,
            ),
            ContinuationMeridian(
                meridian=2,
                tracks=("assistant",),
                roles=(2,),
                plans=(plan,),
                controls=(b"",),
                effort=1,
            ),
        ),
    )
    writer = SourceWriter(
        facets=c.n_facets,
        facet_dim=c.facet_dim,
        semantic_dim=8,
        value_dim=16,
        hidden=32,
        heads=4,
        max_blocks=8,
    )
    writer_before = writer.value.weight.detach().clone()
    frozen_before = model.core.memory_projector.to_kv.weight.detach().clone()
    batch = SourceWriterBatch(
        encoded=torch.randn(1, 5, c.n_facets, c.facet_dim),
        valid_blocks=torch.ones(1, 5, dtype=torch.bool),
        assignments=AssignmentTargets(
            slots=torch.tensor([[0]], dtype=torch.long),
            key=torch.randn(1, 1, 8),
            value=torch.randn(1, 1, 16),
            strength=torch.ones(1, 1) * 0.5,
            lifetime_logit=torch.ones(1, 1) * -9,
            distance=torch.zeros(1, 1),
            valid=torch.ones(1, 1, dtype=torch.bool),
        ),
        mode="suffix",
    )
    report = fit_source_writer_epoch(
        writer,
        [batch],
        optimizer=torch.optim.AdamW(writer.parameters(), lr=1e-3),
        settings=WriterFitSettings(
            distillation_weight=0.0,
            behavior_weight=1.0,
        ),
        frozen_owners=(model,),
        behavior=native_source_continuation(generator, (trajectory,)),
    )
    assert report["valid_targets"] == 1
    assert not torch.equal(writer.value.weight, writer_before)
    torch.testing.assert_close(
        model.core.memory_projector.to_kv.weight,
        frozen_before,
    )
    assert model.core.memory_projector.to_kv.weight.grad is None


def test_pair_writer_learns_directional_dependence_through_frozen_native_reader():
    from megatoken.memory.writers import RelationWriter
    from megatoken.training.hindsight import (
        PairWriterBatch,
        fit_relation_writer_epoch,
    )
    from megatoken.training.hindsight_native import (
        PairWriterContinuation,
        native_pair_continuation,
    )

    torch.manual_seed(37)
    generator = MultiTrackGenerator(
        FacetedGenerator(
            FacetedCore(
                ModelConfig.tiny(
                    semantic_dim=8,
                    semantic_value_dim=16,
                    max_kb_positions=2,
                )
            ),
            ByteCodec.byte_fallback(),
        )
    )
    model = generator.generator
    state = generator.tracks.initial_state(1)
    c = model.config
    raw = torch.randn(1, 1, c.n_facets, c.facet_dim)
    plan = model.prepare_target(
        target_bytes=b"answer from independently checked context",
        representation_mode="bootstrap",
    )
    events = (
        ContinuationMeridian(0, ("source-a",), (3,), raw=raw),
        ContinuationMeridian(1, ("source-b",), (3,), raw=raw),
        ContinuationMeridian(
            2,
            ("assistant",),
            (2,),
            plans=(plan,),
            controls=(b"",),
            effort=1,
        ),
    )
    trajectory = PairWriterContinuation(
        batch_index=0,
        scope="team",
        identity="relation-original",
        query=torch.tensor([1.0] + [0.0] * 7),
        source_meridian=0,
        initial_state=state,
        events=events,
        prior=0.8,
        left_support=1.0,
        right_support=1.0,
        left_checked=True,
        right_checked=True,
        distinct_support=True,
    )
    writer = RelationWriter(
        key_dim=8,
        value_dim=16,
        statistics_dim=3,
        hidden=32,
    )
    before = writer.dependence.weight.detach().clone()
    batch = PairWriterBatch(
        left_key=torch.tensor([[1.0] + [0.0] * 7]),
        right_key=torch.tensor([[1.0] + [0.0] * 7]),
        left_value=torch.randn(1, 16),
        right_value=torch.randn(1, 16),
        statistics=torch.randn(1, 3),
        target_value=torch.randn(1, 16),
        target_strength=torch.ones(1) * 0.5,
        target_lifetime=torch.ones(1) * -8,
        target_dependence=torch.tensor([[0.1, 0.5]]),
        valid=torch.ones(1, dtype=torch.bool),
        distance=torch.zeros(1),
    )
    result = fit_relation_writer_epoch(
        writer,
        [batch],
        optimizer=torch.optim.AdamW(writer.parameters(), lr=1e-3),
        settings=WriterFitSettings(distillation_weight=0.0, behavior_weight=1.0),
        frozen_owners=(model,),
        behavior=native_pair_continuation(generator, (trajectory,)),
    )
    assert result["valid_targets"] == 1
    assert not torch.equal(writer.dependence.weight, before)
    assert model.core.memory_projector.to_kv.weight.grad is None
