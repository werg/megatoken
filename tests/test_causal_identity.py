from dataclasses import replace

from megatoken.data import DedupStore
from megatoken.data.causal_identity import training_contract_hash
from megatoken.data.dedup import reconcile
from megatoken.data.schema import (
    Episode,
    Lineage,
    Meridian,
    PhysicalRecord,
    TrackEvent,
    TrackSpec,
    make_records,
)
from megatoken.generation.codec import ByteCodec


def example(*, serial=False, prefix="first", family="source", split="train"):
    ids = [prefix + str(i) for i in range(3)]
    records = (
        PhysicalRecord(ids[0], "user", b"Give both facts."),
        PhysicalRecord(ids[1], "assistant", b"fact one", policy_mask=True),
        PhysicalRecord(ids[2], "assistant", b"fact two", policy_mask=True),
    )
    source, reply = prefix + "input", prefix + "reply"
    groups = [Meridian(0, (TrackEvent(ids[0], source, "text", 0, "user"),))]
    groups.append(
        Meridian(
            1,
            tuple(
                TrackEvent(ids[i], reply, "text", 1, "assistant")
                for i in ((1,) if serial else (1, 2))
            ),
        )
    )
    if serial:
        groups.append(Meridian(2, (TrackEvent(ids[2], reply, "text", 2, "assistant"),)))
    return Episode(
        prefix,
        records,
        Lineage("source", "v1", family, frozen_split=split),
        tracks=(TrackSpec(source, "user"), TrackSpec(reply, "assistant")),
        meridians=tuple(groups),
    )


def test_dedup_preserves_causal_variants_and_normalizes_opaque_addresses(tmp_path):
    peers, sequential, renamed = (
        example(),
        example(serial=True, prefix="sequential"),
        example(prefix="renamed"),
    )
    assert peers.content_sha256 == sequential.content_sha256 == renamed.content_sha256
    assert training_contract_hash(peers) == training_contract_hash(renamed)
    assert training_contract_hash(peers) != training_contract_hash(sequential)
    codec = ByteCodec.byte_fallback()
    with DedupStore(tmp_path / "data.sqlite") as store:
        assert store.add(peers, codec) == "accepted_candidate"
        assert store.add(sequential, codec) == "accepted_candidate"
        assert store.add(renamed, codec) == "duplicate"
        store.freeze()
        assert len(list(store.accepted_rows())) == 2


def test_different_causal_contracts_still_union_train_validation_families(tmp_path):
    with DedupStore(tmp_path / "data.sqlite") as store:
        store.add(example(), ByteCodec.byte_fallback())
        store.add(
            example(serial=True, prefix="later", family="other", split="validation"),
            ByteCodec.byte_fallback(),
        )
        store.freeze()
        assert store.split_of("source") == store.split_of("other") == "quarantine"
        assert not list(store.accepted_rows())


def test_shard_reconciliation_keeps_distinct_causal_views(tmp_path):
    paths = []
    for index, episode in enumerate((example(), example(serial=True, prefix="later"))):
        with DedupStore(tmp_path / f"store-{index}.sqlite") as store:
            store.add(episode, ByteCodec.byte_fallback())
            path = tmp_path / f"digest-{index}.sqlite"
            store.write_digest(path)
            paths.append(path)
    result = reconcile(paths, tmp_path / "reconciled")
    assert result["accepted"] == 2 and result["cross_shard_duplicates"] == 0


def test_physical_cuts_are_not_additional_training_tasks():
    def cut(size):
        return Episode(
            str(size),
            make_records(
                b"the same complete logical text",
                logical_id="logical",
                role="assistant",
                max_bytes=size,
                policy_mask=True,
            ),
            Lineage("source", "v1", "family"),
        )

    assert training_contract_hash(cut(4)) == training_contract_hash(cut(8))
    changed = replace(cut(8), control_seed=b"a different requested behavior")
    assert training_contract_hash(changed) != training_contract_hash(cut(8))
