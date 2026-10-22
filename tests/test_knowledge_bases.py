"""Several knowledge bases searched as one, and the ingested bases themselves."""

from __future__ import annotations

import math

import pytest
import torch

from megatoken.generation.codec import ByteCodec
from megatoken.kb import add_items, describe, model_identity, new_base, open_base
from megatoken.loading import build_model
from megatoken.memory.knowledge import KnowledgeBase, check_identity
from megatoken.memory.transducer import TRAINED, BankFull, TransducerBank, TransducerGeometry
from megatoken.model import ModelConfig
from megatoken.model.ingest import IngestItem
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.model.transducer import metadata_payload

GEOMETRY = TransducerGeometry(
    location_dim=8,
    query_dim=16,
    value_dim=64,
    neighbors=4,
    selected=16,
    exploration=1,
    max_lookups=6,
    max_edges=64,
    byte_budget=64 * 10_000,
)
IDENTITY = {"format": "x", "geometry": {}, "teacher": "t", "architecture": "a", "weights": "w"}


def fields(count: int, seed: int, geometry=GEOMETRY):
    generator = torch.Generator().manual_seed(seed)
    return dict(
        departure=torch.randn(count, geometry.location_dim, generator=generator),
        key=torch.randn(count, geometry.query_dim, generator=generator),
        value=torch.randn(count, geometry.value_dim, generator=generator),
        destination=torch.randn(count, geometry.location_dim, generator=generator),
        bias=torch.zeros(count),
        write_mass=torch.ones(count),
        step=0,
    )


def entries(count: int, seed: int):
    """What ``KnowledgeBase.add`` takes: the four address and value fields."""
    keep = ("departure", "key", "value", "destination")
    return {name: value for name, value in fields(count, seed).items() if name in keep}


def bank_with(count: int, seed: int = 0) -> TransducerBank:
    torch.manual_seed(seed)
    bank = TransducerBank(GEOMETRY)
    if count:
        bank.add_edges(**fields(count, seed))
    return bank


def base_with(count: int, seed: int = 0, *, writable: bool = False) -> KnowledgeBase:
    """An ingested base holding the same rows ``bank_with(count, seed)`` has."""
    base = KnowledgeBase(GEOMETRY, IDENTITY, name="b", writable=True)
    if count:
        base.add_edges(**fields(count, seed))
    base.writable = writable
    return base


def agenda_for(bank: TransducerBank, mounted=None, seed: int = 1):
    torch.manual_seed(seed)
    return bank.initial_agenda(torch.randn(GEOMETRY.query_dim), mounted)


def test_one_mounted_base_searches_exactly_like_the_trained_bank_alone():
    bank = bank_with(12)
    twin = base_with(12)  # identical packed rows, held as an ingested base
    query = torch.randn(GEOMETRY.query_dim)
    alone = bank.lookup(query, agenda_for(bank))
    mounted = bank.mount({"twin": (twin, 1.0)}, trained=0)
    assert [m.name for m in mounted] == ["twin"]
    other = bank.lookup(query, agenda_for(bank, mounted))
    assert {edge for _, _, edge in other.pairs} == {edge for _, _, edge in alone.pairs}
    assert {name for _, name, _ in other.pairs} == {"twin"}
    for field in ("bank_value", "bank_landing_raw", "weights", "landing"):
        torch.testing.assert_close(getattr(other, field), getattr(alone, field))
    # Nothing of an ingested base is a trained edge: it is never written or evolved.
    assert other.edge_ids.numel() == 0 and alone.edge_ids.numel() > 0


def test_weight_zero_unmounts_and_a_bank_with_nothing_mounted_is_the_sink_alone():
    bank = bank_with(8)
    twin = base_with(8, seed=5)
    mounted = bank.mount({"twin": (twin, 0.0)})
    assert [m.name for m in mounted] == [TRAINED]
    same = bank.lookup(torch.ones(16), agenda_for(bank, mounted))
    alone = bank.lookup(torch.ones(16), agenda_for(bank))
    torch.testing.assert_close(same.bank_value, alone.bank_value)
    nothing = bank.mount({"twin": (twin, 0.0)}, trained=0)
    assert nothing == ()
    empty = bank.lookup(torch.ones(16), agenda_for(bank, nothing))
    assert empty.pairs == () and float(empty.sink_weight) == 1.0


def test_a_blend_weight_is_a_log_prior_on_the_bases_pair_scores():
    bank = bank_with(0)
    first, second = base_with(6, seed=3), base_with(6, seed=3)  # the same rows twice
    query = torch.randn(16)
    even = bank.lookup(query, agenda_for(bank, bank.mount({"a": (first, 1.0), "b": (second, 1.0)})))
    skewed = bank.lookup(
        query, agenda_for(bank, bank.mount({"a": (first, 1.0), "b": (second, 0.25)}))
    )

    def logits(lookup):
        return {
            (loc, name, edge): float(x)
            for (loc, name, edge), x in zip(lookup.pairs, lookup.pair_logits, strict=True)
        }

    assert len(even.pairs) == len(skewed.pairs) == 8  # 4 neighbours x 2 bases: all fit the budget
    before, after = logits(even), logits(skewed)
    for key, value in before.items():
        shift = math.log(0.25) if key[1] == "b" else 0.0
        assert after[key] == pytest.approx(value + shift, abs=1e-5)
    with pytest.raises(ValueError, match="nonnegative"):
        bank.mount({"a": (first, -1.0)})


def test_pairs_from_every_base_share_one_softmax_with_the_one_sink():
    bank = bank_with(5, seed=1)
    other = base_with(5, seed=2)
    lookup = bank.lookup(torch.randn(16), agenda_for(bank, bank.mount({"other": (other, 1.0)})))
    assert {name for _, name, _ in lookup.pairs} == {TRAINED, "other"}
    assert float(lookup.weights.sum()) == pytest.approx(1.0, abs=1e-5)
    assert len(lookup.weights) == len(lookup.pairs) + 1
    # The blend is one tensor over everything mounted: pair values plus the sink's.
    values = torch.cat(
        [
            (bank if name == TRAINED else other).decode("value", [edge])
            for _, name, edge in lookup.pairs
        ]
    )
    expected = lookup.weights[:-1] @ values + lookup.weights[-1] * bank.sink_value
    torch.testing.assert_close(lookup.bank_value, expected)


def test_ingested_bases_never_change_and_nothing_of_them_trains():
    bank = bank_with(0)
    base = base_with(10)
    before = base.table.state()
    version = base.version
    agenda = agenda_for(bank, bank.mount({"b": (base, 1.0)}))
    bank.log_beta_query.requires_grad_(True)
    lookup = bank.lookup(torch.randn(16), agenda, training=True)
    assert lookup.pairs and lookup.edge_ids.numel() == 0
    (lookup.bank_value.sum() + lookup.bank_landing_raw.sum()).backward()
    bank.commit(agenda, lookup)
    after = base.table.state()
    assert base.version == version
    assert all(torch.equal(before[name], after[name]) for name in before)
    # No gradient masters were made for rows the bank does not own.
    assert all(len(store) == 0 for store in bank.masters.stores.values())
    with pytest.raises(RuntimeError, match="read-only"):
        base.add_edges(**fields(1, 9))


def test_each_bases_version_refreshes_only_its_own_cached_shortlists():
    bank = bank_with(6)
    user = base_with(3, seed=4, writable=True)
    agenda = agenda_for(bank, bank.mount({"user": (user, 1.0)}))
    bank.refresh(agenda)
    trained_lists, user_lists = agenda.shortlists[0], agenda.shortlists[1]
    assert agenda.versions == [bank.version, user.version]
    first = user.add(**entries(2, 8), sources=["a", "b"])
    assert len(first) == 2 and user.version == agenda.versions[1] + 1
    bank.refresh(agenda)
    assert agenda.shortlists[0] is trained_lists  # the trained base did not move: no rescan
    assert agenda.shortlists[1] is not user_lists
    assert user.table.ids_of(agenda.shortlists[1][0]).numel() == 4  # neighbours = 4 of 5 edges
    assert agenda.versions == [bank.version, user.version]
    # A lookup computed before a base changed cannot commit after it.
    lookup = bank.lookup(torch.randn(16), agenda)
    user.add(**entries(1, 12), sources=["c"])
    with pytest.raises(RuntimeError, match="Stale"):
        bank.commit(agenda, lookup)


def test_commit_lets_the_search_move_from_one_bases_region_into_anothers():
    bank = bank_with(0)
    with torch.no_grad():
        bank.log_beta_location.fill_(math.log(20.0))  # location decides ...
        bank.log_beta_query.fill_(math.log(0.01))  # ... not the keys
        bank.sink_bias.fill_(-50.0)
    e0, e1, e2 = torch.eye(8)[:3]

    def region(departs_at, leads_to, seed):
        base = KnowledgeBase(GEOMETRY, IDENTITY, writable=True)
        rows = fields(3, seed)
        rows["departure"], rows["destination"] = departs_at.repeat(3, 1), leads_to.repeat(3, 1)
        base.add_edges(**rows)
        base.writable = False
        return base

    first, second = region(e0, e1, 1), region(e1, e2, 2)
    agenda = agenda_for(bank, bank.mount({"first": (first, 1.0), "second": (second, 1.0)}))
    agenda.locations[0] = agenda.state = e0
    one = bank.lookup(torch.randn(16), agenda)

    def share(lookup, name, at=None):
        pairs = zip(lookup.pairs, lookup.weights[:-1], strict=True)
        return sum(float(w) for (loc, who, _), w in pairs if who == name and at in (None, loc))

    assert share(one, "first") > 0.99  # the search starts in the first base's region
    assert bank.commit(agenda, one)
    assert len(agenda.shortlists[0]) == len(agenda.shortlists[1]) == 2
    # It landed where the second base departs: now that region is searched from there.
    two = bank.lookup(torch.randn(16), agenda)
    assert share(two, "second", at=1) > 0.4 and share(two, "first", at=0) > 0.4
    assert share(two, "second", at=0) < 0.01


def test_geometry_mismatch_and_foreign_devices_refuse_to_mount():
    bank = bank_with(2)
    wide = TransducerGeometry(**{**vars(GEOMETRY), "value_dim": 128})
    other = KnowledgeBase(wide, IDENTITY)
    with pytest.raises(ValueError, match="different geometry"):
        bank.mount({"wide": (other, 1.0)})
    with pytest.raises(ValueError, match="trained"):
        bank.mount({TRAINED: (base_with(1), 1.0)})


# ---------------------------------------------------------------- files


def test_a_base_file_round_trips_and_refuses_another_models_identity(tmp_path):
    base = base_with(7, writable=True)
    base.name = "manuals"
    base.sources = {1: "a.txt", 2: "a.txt"}
    base.save(tmp_path / "manuals.kb")
    loaded = KnowledgeBase.load(tmp_path / "manuals.kb", geometry=GEOMETRY, identity=IDENTITY)
    assert not loaded.writable and loaded.name == "manuals"
    assert loaded.sources == base.sources and loaded.table.next_id == base.table.next_id
    before, after = base.table.state(), loaded.table.state()
    assert all(torch.equal(before[name], after[name]) for name in before)
    for key in IDENTITY:
        with pytest.raises(ValueError, match=f"different {key}"):
            KnowledgeBase.load(
                tmp_path / "manuals.kb", geometry=GEOMETRY, identity={**IDENTITY, key: "other"}
            )
    with pytest.raises(ValueError, match="different weights"):
        check_identity(IDENTITY, {**IDENTITY, "weights": "newer"})
    torch.save({"format": "something else"}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="Not a megatoken knowledge base"):
        KnowledgeBase.load(tmp_path / "other.pt", geometry=GEOMETRY, identity=IDENTITY)


def test_a_base_respects_the_edge_ceiling():
    base = KnowledgeBase(GEOMETRY, IDENTITY, writable=True)
    base.add_edges(**fields(64, 0))
    with pytest.raises(BankFull):
        base.add_edges(**fields(1, 1))


# ------------------------------------------------------- model-level ingest


def tiny_model():
    torch.manual_seed(2)
    return build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).eval()


def native_items(model, count, *, metadata=None, document=""):
    c = model.config
    payload = metadata_payload(metadata or {"task": "manuals"})
    return [
        IngestItem(torch.randn(c.n_facets, c.facet_dim), "compressed_text", payload, document, part)
        for part in range(count)
    ]


def test_an_item_is_read_alone_so_nothing_else_can_reach_it():
    model = tiny_model()
    heads, core = model.transducer.ingest, model.core
    a, b, c = (torch.randn(1, *native_items(model, 1)[0].raw.shape) for _ in range(3))
    kinds = ["compressed_text", "compressed_text"]
    with torch.no_grad():
        alone = heads.states(core, a, kinds[:1])
        beside_b = heads.states(core, torch.cat((a, b)), kinds)
        beside_c = heads.states(core, torch.cat((a, c)), kinds)
        values_alone, keys_alone = heads(core, a, kinds[:1])
        values_beside, keys_beside = heads(core, torch.cat((a, c)), kinds)
    # Another item (before, after or instead) changes nothing about this one's states.
    torch.testing.assert_close(beside_b[0], alone[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(beside_c[0], alone[0], atol=1e-4, rtol=1e-4)
    # Values are rounded to 4 bits, so a rounding step may flip: compare by direction.
    similar = torch.nn.functional.cosine_similarity(values_beside[0], values_alone[0], dim=0)
    assert similar > 0.99
    torch.testing.assert_close(keys_beside[0], keys_alone[0], atol=1e-4, rtol=1e-4)
    # The item attends to the whole item (the mask is full), not to a causal prefix of it.
    changed = a.clone()
    changed[0, -1] += 1.0
    with torch.no_grad():
        later = heads.states(core, changed, kinds[:1])
    assert not torch.allclose(later[0, 0], alone[0, 0], atol=1e-6)
    assert torch.allclose(keys_alone.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_entries_depart_where_metadata_starts_a_session_and_parts_chain():
    model = tiny_model()
    multi = MultiTrackGenerator(model)
    native = model.transducer
    parts = native_items(model, 3, document="manual")
    single = native_items(model, 1, metadata={"task": "photos", "modality": ["image"]})
    rows = native.ingest_rows(multi, [*parts, *single])
    bank = native.bank
    # Part 0 and a standalone item start from what their metadata maps to, the same map a
    # session's prefix takes (metadata -> prefix head -> initial location).
    state = multi.tracks.initial_state(1, device=bank.device)
    with torch.no_grad():
        _, agenda = native.begin(multi, state, parts[0].metadata)
        _, other = native.begin(multi, state, single[0].metadata)
    torch.testing.assert_close(rows.departure[0], agenda.locations[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rows.departure[3], other.locations[0], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(rows.departure[0], rows.departure[3], atol=1e-3)
    # Part i departs where part i - 1 leads; a destination is the content location of the value.
    torch.testing.assert_close(rows.departure[1], rows.destination[0])
    torch.testing.assert_close(rows.departure[2], rows.destination[1])
    torch.testing.assert_close(rows.destination, bank.content_location(rows.value))
    with pytest.raises(ValueError, match="not in this call"):
        native.ingest_rows(multi, parts[1:2])


def test_a_base_built_by_the_model_mounts_and_is_found_by_its_own_key():
    model = tiny_model()
    multi = MultiTrackGenerator(model)
    identity = model_identity(model)
    base = new_base(model, "notes", identity=identity)
    items = []
    for _ in range(4):
        raw = torch.randn(model.config.n_facets, model.config.facet_dim)
        items.append(IngestItem(raw, "compressed_text", metadata_payload({"task": "notes"})))
    rows = model.transducer.ingest_rows(multi, items)
    ids = base.add(**vars(rows), sources=[f"note-{i}" for i in range(4)])
    assert ids.tolist() == [1, 2, 3, 4] and base.sources[3] == "note-2"
    bank = model.transducer.bank
    # Ask with an entry's own key, from where it departs: that entry wins the lookup.
    agenda = bank.initial_agenda(torch.zeros(16), bank.mount({"notes": (base, 1.0)}))
    agenda.locations[0] = agenda.state = rows.departure[2]
    with torch.no_grad():
        lookup = bank.lookup(rows.key[2], agenda)
    best = int(lookup.weights[:-1].argmax())
    assert lookup.pairs[best] == (0, "notes", 3)


def test_files_round_trip_through_the_model_identity(tmp_path):
    model = tiny_model()
    base = new_base(model, "notes")
    multi = MultiTrackGenerator(model)
    rows = model.transducer.ingest_rows(multi, native_items(model, 2, document="d"))
    base.add(**vars(rows), sources=["d.txt", "d.txt"])
    path = tmp_path / "notes.kb"
    base.save(path)
    summary = describe(path)
    assert summary["edges"] == 2 and summary["sources"] == 1
    assert summary["identity"] == model_identity(model)
    reopened = open_base(path, model)
    assert reopened.edge_count == 2 and not reopened.writable
    # A different model (other weights) refuses the base.
    torch.manual_seed(99)
    other = build_model(ModelConfig.tiny(), ByteCodec.byte_fallback()).eval()
    with pytest.raises(ValueError, match="different weights"):
        open_base(path, other)
    assert add_items.__doc__ and "Read each item alone" in add_items.__doc__
