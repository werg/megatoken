"""Writer behavior must train through actual chronological native P/C/R losses."""
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.policy import FacetedGenerator
from megatoken.memory.writers import AssignmentTargets, SourceWriter
from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.training.hindsight import (
    SourceWriterBatch, WriterFitSettings, fit_source_writer_epoch,
)
from megatoken.training.hindsight_native import (
    ContinuationMeridian, WriterContinuation, native_source_continuation,
)


def test_permitted_suffix_writer_receives_native_continuation_gradient():
    torch.manual_seed(11)
    generator = MultiTrackGenerator(FacetedGenerator(
        FacetedCore(ModelConfig.tiny(
            semantic_dim=8, semantic_value_dim=16, max_kb_positions=2,
        )), ByteCodec.byte_fallback(),
    ))
    model = generator.generator
    native = generator.tracks.initial_state(1)
    c = model.config
    observation = torch.randn(1, 1, c.n_facets, c.facet_dim)
    plan = model.prepare_target(
        target_bytes=b"native response", representation_mode="bootstrap",
    )
    trajectory = WriterContinuation(
        batch_index=0, source_slot=0, source_meridian=0,
        initial_state=native, scope="team", identity="source-assignment-a",
        events=(
            ContinuationMeridian(
                meridian=0, tracks=("source",), roles=(3,), raw=observation,
            ),
            ContinuationMeridian(
                meridian=1, tracks=("source",), roles=(3,), raw=observation,
            ),
            ContinuationMeridian(
                meridian=2, tracks=("assistant",), roles=(2,), plans=(plan,),
                controls=(b"",), effort=1,
            ),
        ),
    )
    writer = SourceWriter(
        facets=c.n_facets, facet_dim=c.facet_dim,
        semantic_dim=8, value_dim=16,
        hidden=32, heads=4, max_blocks=8,
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
        writer, [batch], optimizer=torch.optim.AdamW(writer.parameters(), lr=1e-3),
        settings=WriterFitSettings(
            distillation_weight=0., behavior_weight=1.,
        ),
        frozen_owners=(model,),
        behavior=native_source_continuation(generator, (trajectory,)),
    )
    assert report["valid_targets"] == 1
    assert not torch.equal(writer.value.weight, writer_before)
    torch.testing.assert_close(
        model.core.memory_projector.to_kv.weight, frozen_before,
    )
    assert model.core.memory_projector.to_kv.weight.grad is None
