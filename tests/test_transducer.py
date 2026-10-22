"""Executable CPU correctness gates for the v10 continuous-state transducer."""

from __future__ import annotations

import pytest
import torch

from megatoken.memory.transducer import TransducerBank, TransducerGeometry
from megatoken.memory.transducer_growth import TransducerGrowth
from megatoken.memory.transducer_optimizer import BlockScaleDescent
from megatoken.memory.transducer_packing import pack_fp4, pack_fp8, unit
from megatoken.model.attention import AttentionFeatures, AttentionRead
from megatoken.model.config import ModelConfig
from megatoken.model.core import FacetedCore
from megatoken.model.transducer import (
    HindsightRealizer,
    NativeTransducer,
    event_attention_mask,
)
from megatoken.training.transducer import BoundaryScheduler, HindsightSchedule
from megatoken.training.transducer_diloco import TransducerCoordinator, WorkerRound


def small_bank(**overrides: int) -> TransducerBank:
    values = dict(
        location_dim=4,
        query_dim=8,
        value_dim=128,
        neighbors=2,
        selected=2,
        exploration=0,
        max_edges=100,
        byte_budget=100000,
    )
    values.update(overrides)
    return TransducerBank(TransducerGeometry(**values))


def edge(
    bank: TransducerBank,
    *,
    departure: tuple[float, ...],
    key: tuple[float, ...] | None = None,
    value: float = 1.0,
    destination: tuple[float, ...] | None = None,
) -> int:
    return bank.add_edge(
        departure=torch.tensor(departure),
        key=torch.tensor(key if key is not None else (1, 0, 0, 0, 0, 0, 0, 0)),
        value=torch.full((bank.geometry.value_dim,), value),
        destination=torch.tensor(destination if destination is not None else departure),
    )


def test_packed_fp4_and_fp8_store_actual_codes_and_reject_nonfinite():
    x = torch.linspace(-6, 6, 129)
    value = pack_fp4(x)
    address = pack_fp8(x)
    assert value.format == "e2m1" and len(value.codes) == 65
    assert len(value.scales) == 9
    assert address.format == "e4m3" and len(address.codes) == 129
    assert len(address.scales) == 8
    assert (value.decode() - x).abs().mean() < 0.5
    assert (address.decode() - x).abs().max() < 0.25
    assert torch.equal(value.decode(), type(value).from_dict(value.to_dict()).decode())
    with pytest.raises(ValueError):
        pack_fp4(torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        pack_fp8(torch.tensor([float("inf")]))


def test_agenda_only_shortlists_sink_and_no_duplicate_landing():
    bank = small_bank(neighbors=1)
    a = edge(bank, departure=(1, 0, 0, 0), value=2)
    b = edge(bank, departure=(-1, 0, 0, 0), value=3)
    agenda = bank.agenda_from_location(torch.tensor([1.0, 0, 0, 0]))
    lookup = bank.lookup(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), agenda)
    assert lookup is not None
    assert lookup.candidates == (a,)
    assert b not in lookup.candidates
    assert lookup.weights.numel() == 2  # one pair and an unconditional sink
    assert torch.allclose(lookup.weights.sum(), torch.tensor(1.0))
    assert lookup.sink_weight > 0
    assert not bank.commit(agenda, lookup)
    assert len(agenda.locations) == 1 and agenda.lookups == 1


def test_all_shortlisted_pairs_scored_before_selection():
    bank = small_bank(neighbors=3, selected=1)
    first = edge(bank, departure=(1, 0, 0, 0), key=(0, 1, 0, 0, 0, 0, 0, 0))
    last = edge(bank, departure=(1, 0, 0, 0), key=(1, 0, 0, 0, 0, 0, 0, 0))
    edge(bank, departure=(1, 0, 0, 0), key=(0, 0, 1, 0, 0, 0, 0, 0))
    agenda = bank.agenda_from_location(torch.tensor([1.0, 0, 0, 0]))
    result = bank.lookup(torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]), agenda)
    assert result is not None
    assert result.candidates == (last,)
    assert first not in result.candidates


def test_refresh_uses_only_new_departure_table_and_rejects_v9_snapshot():
    bank = small_bank(neighbors=1)
    edge(bank, departure=(1, 0, 0, 0))
    agenda = bank.agenda_from_location(torch.tensor([1.0, 0, 0, 0]))
    bank.refresh(agenda)
    old = agenda.neighborhoods
    edge(bank, departure=(1, 0, 0, 0), value=9)
    bank.refresh(agenda)
    assert agenda.version == bank.version
    assert agenda.neighborhoods[0] == bank.nearest(agenda.current)
    assert old == [(1,)]
    saved = bank.state_snapshot()
    bank.load_snapshot(saved)
    with pytest.raises(ValueError, match="v10"):
        bank.load_snapshot({**saved, "identity": "v9-point-pair"})


def test_location_modulation_gradient_and_hindsight_steering():
    bank = small_bank(neighbors=2)
    edge(bank, departure=(1, 0, 0, 0), destination=(0, 1, 0, 0))
    agenda = bank.agenda_from_location(torch.tensor([1.0, 0, 0, 0]))
    q = torch.ones(8, requires_grad=True)
    hindsight = torch.randn(128)
    result = bank.lookup(q, agenda, hindsight=hindsight, overlay=1)
    assert result is not None
    assert torch.allclose(result.value, hindsight, atol=1e-5)
    assert torch.allclose(result.landing, bank._content_location(hindsight))
    no_overlay = bank.lookup(q, agenda, hindsight=hindsight, overlay=0)
    assert no_overlay is not None
    assert not torch.allclose(no_overlay.landing, result.landing)
    no_overlay.bank_value.sum().backward()
    assert bank.modulation.grad is not None
    assert q.grad is not None


def test_sparse_optimizer_addresses_written_values_and_master_retention():
    bank = small_bank()
    i = edge(bank, departure=(1, 0, 0, 0))
    optimizer = BlockScaleDescent(bank, cache_edges=1)
    leaf = bank.field(i, "key", train=True)
    leaf.sum().backward()
    result = optimizer.step(1)
    assert result["updated_fields"] == 1
    assert i in optimizer.lru
    assert optimizer.fields[i, "key"].count == pytest.approx(1)
    optimizer.add_write(i, torch.full((128,), 4.0), 3.0)
    result = optimizer.step(2)
    assert result["written_values"] == 1
    assert bank.rows[i].write_mass == pytest.approx(3.0)
    assert (bank.rows[i].value.decode() - 4).abs().mean() < 1.0
    optimizer.switch_to_learned()
    with pytest.raises(RuntimeError, match="stop"):
        optimizer.add_write(i, torch.ones(128), 1.0)


def test_diloco_merges_with_untouched_workers_in_denominator_and_deduplicates():
    bank = small_bank()
    i = edge(bank, departure=(1, 0, 0, 0))
    optimizer = BlockScaleDescent(bank)
    coordinator = TransducerCoordinator(bank, optimizer, TransducerGrowth(bank))
    a = WorkerRound(0, 0, 0, 10, deltas={(i, "bias"): torch.tensor([2.0])})
    b = WorkerRound(1, 0, 0, 10)  # This worker contributes zero to this row.
    assert coordinator.submit(a)
    assert coordinator.submit(b)
    assert not coordinator.submit(a)
    counts = coordinator.merge(step=32)
    assert counts["submitted"] == 2
    assert bank.rows[i].bias == pytest.approx(1.0)
    assert not coordinator.submit(a)


def test_structural_writeout_has_content_destination_and_sink_only_exclusion():
    bank = small_bank()
    growth = TransducerGrowth(bank)
    agenda = bank.agenda_from_location(torch.tensor([1.0, 0, 0, 0]))
    result = bank.lookup(torch.ones(8), agenda)
    assert result is not None
    h = torch.randn(128)
    gradient = -h
    assert growth.write_out(result, h, gradient, step=9, sink_only=True) is None
    admitted = growth.write_out(result, h, gradient, step=9)
    assert admitted == 1
    row = bank.rows[1]
    assert row.write_mass == 1
    assert float(unit(row.destination.decode()) @ bank._content_location(h)) > 0.98


def test_causal_full_and_group_inverse_masks_and_sink_survive_empty_keys():
    groups = torch.tensor([[0, 0, 1, 1]])
    meridians = torch.tensor([[0, 1, 2, 3]])
    tracks = torch.tensor([[0, 1, 0, 1]])
    inverse = event_attention_mask(
        groups=groups, meridians=meridians, tracks=tracks, mode="group-inverse"
    )
    assert inverse[0, 0].tolist() == [True, True, True, True]
    assert inverse[0, 3].tolist() == [False, False, True, True]
    causal = event_attention_mask(groups=groups, meridians=meridians, tracks=tracks)
    assert causal[0, 0].tolist() == [True, False, False, False]
    assert event_attention_mask(
        groups=groups, meridians=meridians, tracks=tracks, mode="full"
    ).all()
    att = AttentionRead(8, 2, 1, 4)
    q = torch.randn(1, 3, 8, requires_grad=True)
    empty = AttentionFeatures(torch.empty(1, 1, 0, 4), torch.empty(1, 1, 0, 4))
    out = att.read(q, empty)
    assert out.shape == q.shape and torch.isfinite(out).all()
    out.sum().backward()
    assert att.sink_value.grad is not None
    with pytest.raises(ValueError):
        event_attention_mask(groups=groups, meridians=meridians, tracks=tracks, mode="unknown")


def test_hindsight_realizer_sink_only_roundtrips_in_fp4():
    core = FacetedCore(ModelConfig.tiny())
    bank = small_bank()
    realizer = HindsightRealizer(core, bank)
    h = realizer(None)
    assert h.shape == (128,)
    assert torch.allclose(h, pack_fp4(h.detach()).decode())
    assert realizer(torch.randn(8, 32)).shape == (128,)


def test_training_schedule_and_no_recursive_internal_retrieval():
    schedule = HindsightSchedule()
    assert [schedule.stage(s) for s in (0, 4096, 8192)] == ["full", "gap1", "gap2"]
    assert schedule.overlay(0, 3, learned=False) == 1
    assert schedule.overlay(16384, 3, learned=False) == 1
    assert schedule.overlay(81920, 1, learned=True) == 0
    assert schedule.overlay(81920, 8, learned=True) == 1
    scheduler = BoundaryScheduler()
    assert scheduler.first_metadata()
    scheduler.internal_commit()
    assert not scheduler.begin()
    scheduler.finish(committed=True)
    assert scheduler.lookups == 1
    for _ in range(7):
        assert not scheduler.ordinary_commit()
    assert scheduler.ordinary_commit()


def test_native_metadata_query_and_knowledge_are_committed_once():
    core = FacetedCore(ModelConfig.tiny())
    bank = small_bank()
    native = NativeTransducer(core, bank)
    native.eval()
    state, agenda, lookup = native.start({"request": "what is the answer?"})
    assert agenda.lookups == 1
    assert lookup.weights.shape == (1,)
    assert state.meridians_seen.item() == 3
    assert len(state.track_registry) == 1  # metadata; query/knowledge reserved
    assert state.frontier_tracks == ("__transducer_knowledge__",)


def test_transducer_allocator_rejects_reuse_after_retirement_and_restore():
    bank = small_bank()
    first = edge(bank, departure=(1, 0, 0, 0))
    bank.retire(first)
    second = edge(bank, departure=(1, 0, 0, 0))
    assert second == first + 1
    snapshot = bank.state_snapshot()
    restored = small_bank()
    restored.load_snapshot(snapshot)
    restored.retire(second)
    assert edge(restored, departure=(1, 0, 0, 0)) == second + 1
    with pytest.raises(ValueError, match="high-water"):
        restored.load_snapshot({**snapshot, "next_edge_id": second})


def test_bank_rejects_stale_lookup_without_mutating_agenda():
    bank = small_bank()
    agenda = bank.agenda_from_location(torch.tensor([1.0, 0, 0, 0]))
    lookup = bank.lookup(torch.ones(8), agenda)
    assert lookup is not None
    edge(bank, departure=(1, 0, 0, 0))
    with pytest.raises(RuntimeError, match="Stale"):
        bank.commit(agenda, lookup)
    assert agenda.lookups == 0 and len(agenda.locations) == 1


def test_coordinator_accepts_late_unduplicated_rounds_and_new_statistics():
    from megatoken.memory.transducer_growth import EdgeEvidence

    bank = small_bank()
    row = edge(bank, departure=(1, 0, 0, 0))
    optimizer = BlockScaleDescent(bank)
    growth = TransducerGrowth(bank)
    coordinator = TransducerCoordinator(bank, optimizer, growth)
    incoming = WorkerRound(
        0,
        2,
        0,
        10,
        optimizer_increments={(row, "key"): (torch.tensor([4.0]), 2.0)},
        structural_increments={row: EdgeEvidence(utility=2.0, visits=4.0)},
    )
    assert coordinator.submit(incoming)
    coordinator.merge(step=32)
    assert optimizer.fields[row, "key"].count == pytest.approx(2.0)
    assert optimizer.fields[row, "key"].squared_sums.item() == pytest.approx(4.0)
    assert growth.evidence[row].utility == pytest.approx(2.0)
    late = WorkerRound(0, 1, 0, 10)
    assert coordinator.submit(late)
    coordinator.merge(step=64)
    assert not coordinator.submit(late)
    assert not coordinator.submit(incoming)
    state = coordinator.state_dict()
    restored = TransducerCoordinator(bank, optimizer, growth)
    restored.load_state_dict(state)
    assert not restored.submit(late)


def test_growth_split_carries_optimizer_summaries_and_retirement_cleans_them():
    from megatoken.memory.transducer_growth import EdgeEvidence
    from megatoken.memory.transducer_optimizer import FieldStatistics

    bank = small_bank()
    row = bank.add_edge(
        departure=torch.tensor([1.0, 0, 0, 0]),
        key=torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0]),
        value=torch.ones(128),
        destination=torch.tensor([0.0, 1, 0, 0]),
        write_mass=8.0,
    )
    optimizer = BlockScaleDescent(bank)
    growth = TransducerGrowth(bank, optimizer=optimizer)
    optimizer.fields[row, "key"] = FieldStatistics(torch.tensor([6.0]), 8.0, 1)
    optimizer.last_write[row] = 1
    growth.evidence[row] = EdgeEvidence(utility=4.0, visits=64.0, dispersion=1000.0)
    assert growth._split(row, 2) == (row, row + 1)
    for edge_id in (row, row + 1):
        assert optimizer.fields[edge_id, "key"].count == pytest.approx(4.0)
        assert optimizer.fields[edge_id, "key"].squared_sums.item() == pytest.approx(3.0)
        assert optimizer.last_write[edge_id] == 1
    assert bank.next_edge_id == row + 2
    growth._retire(row)
    assert (row, "key") not in optimizer.fields
    assert row not in optimizer.last_write and row not in bank.rows
    assert (row + 1, "key") in optimizer.fields
