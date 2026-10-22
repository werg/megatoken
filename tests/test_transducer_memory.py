"""Packed v10 bank: quantization, agenda search, sparse optimization and growth."""

from __future__ import annotations

import math

import pytest
import torch

from megatoken.memory.transducer import BankFull, TransducerBank, TransducerGeometry
from megatoken.memory.transducer_growth import PROVISIONAL_BASE, TransducerGrowth
from megatoken.memory.transducer_optimizer import WRITE_HALF_LIFE, BlockScaleDescent
from megatoken.memory.transducer_packing import (
    fp4_nbytes,
    fp8_nbytes,
    pack_fp4,
    pack_fp8,
    row_seeds,
    unit,
)

SMALL = TransducerGeometry(
    location_dim=8,
    query_dim=16,
    value_dim=64,
    neighbors=4,
    selected=4,
    exploration=1,
    max_lookups=6,
    max_edges=64,
    byte_budget=64 * 10_000,
)


def bank_with(count: int, *, seed: int = 0, geometry: TransducerGeometry = SMALL):
    torch.manual_seed(seed)
    bank = TransducerBank(geometry)
    g = geometry
    if count:
        bank.add_edges(
            departure=torch.randn(count, g.location_dim),
            key=torch.randn(count, g.query_dim),
            value=torch.randn(count, g.value_dim),
            destination=torch.randn(count, g.location_dim),
            bias=torch.zeros(count),
            write_mass=torch.ones(count),
            step=0,
        )
    return bank


def test_fp4_round_trip_scales_and_unbiased_rounding():
    x = torch.randn(3, 70) * torch.tensor([[1.0], [1e-3], [50.0]])
    rows = pack_fp4(x)
    assert rows.codes.shape == (3, 35) and rows.group_scales.shape == (3, 5)
    decoded = rows.decode(70)
    # Group scales round up, so nothing is clipped: error stays within one FP4 step.
    assert torch.all((decoded - x).abs() <= x.abs().amax(1, keepdim=True) / 6 * 1.01)
    seeds = row_seeds(torch.arange(4000), "value", 1, 0)
    sample = torch.full((4000, 16), 0.3)
    mean = pack_fp4(sample, seeds).decode(16).mean()
    assert abs(float(mean) - 0.3) < 0.01
    assert fp4_nbytes(32768) == 16384 + 2048 + 4


def test_fp8_rows_and_seeded_rounding_is_reproducible():
    x = unit(torch.randn(5, 300))
    a = pack_fp8(x, row_seeds(torch.arange(5), "key", 3, 1))
    b = pack_fp8(x, row_seeds(torch.arange(5), "key", 3, 1))
    assert torch.equal(a.codes.view(torch.uint8), b.codes.view(torch.uint8))
    assert a.scales.shape == (5, 3)
    torch.testing.assert_close(a.decode(), x, atol=0.01, rtol=0.1)
    assert fp8_nbytes(1024) == 1024 + 32


def test_agenda_lookup_scores_all_shortlisted_pairs_and_commits():
    bank = bank_with(20)
    agenda = bank.initial_agenda(torch.randn(16))
    lookup = bank.lookup(torch.randn(16), agenda)
    assert lookup is not None
    assert len(lookup.pairs) == SMALL.selected  # no exploration outside training
    assert torch.isclose(lookup.weights.sum(), torch.tensor(1.0))
    assert float(lookup.edge_weights.sum() + lookup.sink_weight) == pytest.approx(1.0, abs=1e-5)
    # Every pair comes from the cached departure shortlist of its location.
    shortlist = set(bank.table.ids_of(agenda.shortlists[0]).tolist())
    assert {edge for _, edge in lookup.pairs} <= shortlist
    appended = bank.commit(agenda, lookup)
    assert agenda.lookups == 1 and agenda.state is lookup.landing
    assert len(agenda.locations) == 1 + appended


def test_landing_close_to_a_known_location_is_not_appended():
    bank = bank_with(6)
    agenda = bank.initial_agenda(torch.randn(16))
    with torch.no_grad():
        bank.sink_bias.fill_(50.0)  # the sink holds the search still
    lookup = bank.lookup(torch.randn(16), agenda)
    assert not bank.commit(agenda, lookup)
    assert len(agenda.locations) == 1 and agenda.lookups == 1


def test_empty_bank_lookup_is_the_sink_alone():
    bank = TransducerBank(SMALL)
    agenda = bank.initial_agenda(torch.randn(16))
    lookup = bank.lookup(torch.randn(16), agenda, training=True)
    assert lookup.pairs == () and float(lookup.sink_weight) == 1.0
    torch.testing.assert_close(lookup.landing, agenda.state)


def test_hindsight_steering_and_overlay():
    bank = bank_with(10)
    agenda = bank.initial_agenda(torch.randn(16))
    h = torch.randn(64)
    lookup = bank.lookup(torch.randn(16), agenda, hindsight=h, overlay=1.0)
    torch.testing.assert_close(lookup.value, h)
    torch.testing.assert_close(lookup.landing, bank.content_location(h))


def test_lookup_cap_and_stale_commit():
    bank = bank_with(10)
    agenda = bank.initial_agenda(torch.randn(16))
    lookup = bank.lookup(torch.randn(16), agenda)
    bank.publish_bias(bank.edge_ids()[:1], torch.ones(1))
    bank._changed()
    with pytest.raises(RuntimeError, match="Stale"):
        bank.commit(agenda, lookup)
    agenda.lookups = SMALL.max_lookups
    assert bank.lookup(torch.randn(16), agenda) is None


def test_budget_and_ceiling_raise_bank_full():
    tight = TransducerGeometry(**{**vars(SMALL), "max_edges": 3})
    bank = bank_with(3, geometry=tight)
    with pytest.raises(BankFull):
        bank_with(0, geometry=tight)
        bank.add_edges(
            departure=torch.randn(1, 8),
            key=torch.randn(1, 16),
            value=torch.randn(1, 64),
            destination=torch.randn(1, 8),
            bias=torch.zeros(1),
            write_mass=torch.zeros(1),
            step=1,
        )
    assert bank.packed_bytes == 3 * tight.row_bytes


def test_retire_reuses_slots_and_never_reuses_ids():
    bank = bank_with(5)
    bank.retire([2, 3])
    ids = bank.add_edges(
        departure=torch.randn(2, 8),
        key=torch.randn(2, 16),
        value=torch.randn(2, 64),
        destination=torch.randn(2, 8),
        bias=torch.zeros(2),
        write_mass=torch.zeros(2),
        step=1,
    )
    assert ids.tolist() == [6, 7] and bank.edge_count == 5
    assert bank.table.high_water == 5


def master_row(bank, name, edge):
    """The cached FP32 master row of one edge's trained field."""
    store = bank.masters.stores[name]
    return store.data[store.find([edge])][0]


def test_optimizer_steps_selected_rows_only_and_respects_angle_cap():
    bank = bank_with(12)
    optimizer = BlockScaleDescent(bank, address_lr=1.0)
    agenda = bank.initial_agenda(torch.randn(16))
    lookup = bank.lookup(torch.randn(16), agenda, training=True)
    before = {e: bank.decode("key", [e])[0].clone() for e in bank.edge_ids().tolist()}
    (lookup.bank_value.square().sum() + lookup.bank_landing_raw.sum()).backward()
    rows, _ = bank.masters.collect()["key"]
    trained = set(bank.masters.stores["key"].edge[rows].tolist())
    assert trained and trained <= set(lookup.edge_ids.tolist())
    masters = {e: master_row(bank, "key", e).clone() for e in trained}
    result = optimizer.step(1)
    assert result["updated_fields"] > 0
    for edge, old in before.items():
        new = bank.decode("key", [edge])[0]
        if edge in trained:
            moved = master_row(bank, "key", edge)
            angle = math.acos(min(1.0, float(unit(masters[edge]) @ unit(moved))))
            assert 0 < angle <= 0.0101  # the 0.01 rad cap on the FP32 master
        else:
            assert torch.equal(new, old)
    assert not bank.masters.collect()  # every window gradient was consumed


def test_values_are_written_as_decayed_weighted_means():
    bank = bank_with(1)
    optimizer = BlockScaleDescent(bank)
    edge = bank.edge_ids()
    old = bank.decode("value", edge)[0]
    h = torch.randn(64)
    optimizer.add_write(edge, torch.tensor([3.0]), h)
    optimizer.step(16384)
    expected = (old * 0.5 + 3.0 * h) / 3.5  # old mass 1, halved by one half-life
    step = float(expected.abs().max()) / 3  # FP4 holds about one significant step
    assert float((bank.decode("value", edge)[0] - expected).abs().max()) <= step
    assert float(bank.table.write_mass[0]) == pytest.approx(3.5)
    with pytest.raises(RuntimeError):
        bank.learned_values = True
        optimizer.add_write(edge, torch.tensor([1.0]), h)


def test_master_cache_evicts_least_recently_used_edges():
    bank = bank_with(10)
    optimizer = BlockScaleDescent(bank, cache_edges=2)
    for edge in bank.edge_ids().tolist()[:5]:
        bank.masters.leaf("key", [edge], bank.decode("key", [edge]))
    optimizer.step(1)
    assert bank.masters.stores["key"].ids() == [4, 5]


def test_write_out_removal_utility_and_budget_rules():
    bank = bank_with(8)
    optimizer = BlockScaleDescent(bank)
    growth = TransducerGrowth(bank, optimizer=optimizer)
    agenda = bank.initial_agenda(torch.randn(16))
    h = torch.randn(64)
    lookup = bank.lookup(torch.randn(16), agenda, training=True)
    gradient = -h  # hindsight would have reduced the loss
    entry = growth.candidate(lookup, h, gradient)
    assert entry is not None
    assert growth.candidate(lookup, h, h) is None  # it would not have helped
    added = growth.write_out([entry], step=1)
    assert added.tolist() == [9]
    torch.testing.assert_close(
        bank.decode("destination", added)[0], bank.content_location(h), atol=0.02, rtol=0.1
    )
    growth.observe(
        lookup,
        gradient_value=torch.randn(64),
        gradient_location=torch.randn(8),
        step=1,
        hindsight=h,
    )
    rows = growth.evidence.rows(lookup.edge_ids.tolist(), 1)
    assert torch.all(growth.evidence.visits[rows] == 1)
    # Remove: old edges whose utility is not positive, including the never-visited
    # ones. The write-out is younger than 4,096 steps and stays.
    growth.evidence.utility[rows] = -1.0
    result = growth.evolve(4096)
    assert result["removed"] == 8
    assert bank.edge_ids().tolist() == [9]


def test_split_duplicates_and_halves():
    bank = bank_with(2)
    growth = TransducerGrowth(bank, optimizer=BlockScaleDescent(bank))
    rows = growth.evidence.rows([1], 0)
    growth.evidence.visits[rows] = 100
    growth.evidence.utility[rows] = 1.0
    growth.evidence.dispersion[rows] = 10 * float(bank.table.value_norm2[0])
    result = growth.evolve(10)
    assert result["split"] == 1 and bank.edge_count == 3
    child = 3
    assert float(bank.decode("bias", [child])[0]) == pytest.approx(-math.log(2))
    # Halved as aged to the split step, whose clock it then restarts from.
    aged = 0.5 * 2 ** (-10 / WRITE_HALF_LIFE)
    assert float(bank.table.write_mass[bank.table.slots([1])][0]) == pytest.approx(aged)
    # The child copies the parent's value and destination codes, not a requantization.
    parent, child_slot = bank.table.slots([1, child]).tolist()
    for column in ("value_codes", "value_groups", "value_rows", "destination_codes"):
        assert torch.equal(
            getattr(bank.table, column)[child_slot].float(),
            getattr(bank.table, column)[parent].float(),
        )


def test_provisional_write_outs_keep_canonical_ids():
    bank = bank_with(3)
    growth = TransducerGrowth(bank, provisional=True)
    agenda = bank.initial_agenda(torch.randn(16))
    h = torch.randn(64)
    lookup = bank.lookup(torch.randn(16), agenda, training=True)
    entry = growth.candidate(lookup, h, -h)
    added = growth.write_out([entry], step=1)
    assert added.tolist() == [PROVISIONAL_BASE]
    assert bank.table.next_id == 4


def test_snapshot_round_trip():
    bank = bank_with(7)
    bank.retire([3])
    copy = TransducerBank(SMALL)
    copy.load_snapshot(bank.state_snapshot())
    assert copy.edge_ids().tolist() == bank.edge_ids().tolist()
    torch.testing.assert_close(copy.decode("value", [5]), bank.decode("value", [5]))
    assert copy.table.next_id == 8


def test_pair_scores_match_an_unfused_per_location_reference(monkeypatch):
    import megatoken.memory.transducer as module

    monkeypatch.setattr(module, "PAIR_CHUNK", 3)  # several chunks
    bank = bank_with(20)
    with torch.no_grad():
        bank.modulation.normal_()
        bank.log_beta_query.fill_(0.3)
    locations = unit(torch.randn(3, 8))
    query = unit(torch.randn(16))
    modulated = bank.modulated(query, locations)
    slots = torch.tensor([0, 4, 7, 4, 19, 0, 2])
    where = torch.tensor([0, 0, 1, 1, 2, 2, 2])
    beta_q, beta_s = bank.log_beta_query.exp(), bank.log_beta_location.exp()
    with torch.no_grad():
        fused = bank._pair_scores(slots, where, modulated, locations, beta_q, beta_s)
        reference = torch.stack(
            [
                beta_q * unit(bank.table.decode("key", slot[None]))[0] @ modulated[j]
                + beta_s * unit(bank.table.decode("departure", slot[None]))[0] @ locations[j]
                + bank.table.bias[slot]
                for slot, j in zip(slots, where, strict=True)
            ]
        )
    # Selection scores read BF16 unit keys (8 significant bits) against FP32 here.
    torch.testing.assert_close(fused, reference, atol=0.02, rtol=0)


def _assert_caches_match_packed_rows(bank):
    from megatoken.memory.transducer_packing import packed_unit

    table, slots = bank.table, bank.table.alive_slots()
    departure = packed_unit(table.decode("departure", slots))
    key = packed_unit(table.decode("key", slots)).to(torch.bfloat16)
    assert torch.equal(table.departure_unit[slots], departure)
    assert torch.equal(table.key_unit[slots], key)
    torch.testing.assert_close(
        table.value_norm2[slots], table.decode("value", slots).square().sum(1)
    )


def test_cached_unit_rows_follow_every_write_path():
    bank = bank_with(20)
    growth = TransducerGrowth(bank, optimizer=BlockScaleDescent(bank))
    ids = torch.tensor([1, 2, 3])
    bank.publish_addresses("departure", ids, torch.randn(3, 8), step=1, worker=0)
    bank.publish_addresses("key", ids, torch.randn(3, 16), step=1, worker=0)
    bank.publish_values(ids, torch.randn(3, 64), step=1, worker=0)
    _assert_caches_match_packed_rows(bank)
    growth._split([1, 2, 3], 5)
    _assert_caches_match_packed_rows(bank)
    bank.retire([4, 5])  # their slots are reused by the import below
    fresh = bank_with(2, seed=3)
    rows = fresh.export_rows([1, 2])
    rows["ids"] = torch.tensor([100, 101])
    bank.import_rows(rows, retired=[], next_id=102)
    _assert_caches_match_packed_rows(bank)
    copy = TransducerBank(SMALL)
    copy.load_snapshot(bank.state_snapshot())
    _assert_caches_match_packed_rows(copy)


def test_a_failed_add_leaves_the_bank_unchanged():
    bank = bank_with(3)
    before = (bank.edge_ids().tolist(), bank.table.next_id, bank.version)
    departure = torch.randn(2, 8)
    departure[1] = 0  # a zero address cannot be packed
    with pytest.raises(ValueError, match="Zero"):
        bank.add_edges(
            departure=departure,
            key=torch.randn(2, 16),
            value=torch.randn(2, 64),
            destination=torch.randn(2, 8),
            bias=torch.zeros(2),
            write_mass=torch.ones(2),
            step=1,
        )
    assert (bank.edge_ids().tolist(), bank.table.next_id, bank.version) == before
    agenda = bank.initial_agenda(torch.randn(16))
    lookup = bank.lookup(torch.randn(16), agenda)
    assert {edge for _, edge in lookup.pairs} <= {1, 2, 3}


def test_loading_a_snapshot_invalidates_cached_shortlists():
    first, second = bank_with(30, seed=1), bank_with(30, seed=2)
    assert first.version == second.version
    agenda = first.initial_agenda(torch.randn(16))
    first.refresh(agenda)
    first.load_snapshot(second.state_snapshot())
    first.refresh(agenda)
    expected = second.initial_agenda(agenda.locations[0].new_zeros(16))
    expected.locations, expected.state = agenda.locations, agenda.state
    second.refresh(expected)
    assert torch.equal(agenda.shortlists[0], expected.shortlists[0])
