"""Causal selected point/pair readout into complete native-facet KV groups.

All record/pair semantic values are independent FP32 optimizer leaves. Relation
source support, ordered endpoint dependence and the checked-context mask must
be validated BEFORE a learned pair changes either point or pair evidence.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Protocol, Sequence

import torch
from torch import Tensor

from megatoken.data.retrieval import RetrievalExample
from megatoken.memory.base import Record
from megatoken.memory.contracts import Collection, Visibility
from megatoken.memory.reader import EvidenceScores, selected_evidence
from megatoken.memory.relations import pair_evidence
from megatoken.memory.semantic import AdmittedGroup, Constituent
from megatoken.memory.sqlite import SQLiteKB



class SemanticFieldReader(Protocol):
    def field(self, collection: Collection, kind: str, record_id: int,
              name: str) -> Tensor: ...


class FrozenMemoryFields:
    """Serve the SAME point/pair semantic values without optimizer leaves."""

    def __init__(self, bank: SQLiteKB, device: torch.device):
        self.bank, self.device = bank, device

    def field(self, collection: Collection, kind: str,
              record_id: int, name: str) -> Tensor:
        if kind == "record":
            row = self.bank.get(collection, record_id)
            fields = {
                "key": row.key, "value": row.value,
                "strength": row.strength,
                "lifetime_logit": row.lifetime_logit,
            }
        elif kind == "relation" and collection is Collection.MEMORY:
            row = self.bank.get_relation(record_id)
            fields = {
                "value": row.value,
                "strength": row.strength,
                "lifetime_logit": row.lifetime_logit,
                "dependence_logits": row.dependence_logits,
            }
        else:
            raise ValueError("Unknown semantic storage kind")
        return torch.as_tensor(fields[name], dtype=torch.float32, device=self.device)


def _causal_support(record: Record, units: set[str]) -> float:
    return float(bool(
        {span.source_id for span in record.footprint.ranges}.intersection(units)
    ))


def selected_joint_groups(
    *, bank: SQLiteKB, selected: SemanticFieldReader, query: Tensor,
    records: Sequence[Record], visibility: Visibility,
    anchor: int, max_groups: int, threshold: float = 0.25,
    annotation: RetrievalExample | None = None,
    pair_support_units: tuple[str, ...] = (),
    pair_context_complete: bool = False,
) -> tuple[tuple[AdmittedGroup, ...], EvidenceScores]:
    """Materialize live point/pair mixtures with exact null and causal support."""
    ordered = tuple(sorted(records, key=lambda row: row.id))
    if any(
        row.collection is not Collection.MEMORY or row.scope != visibility.scope
        or not visibility.permits(row.footprint) or not row.active
        for row in ordered
    ):
        raise ValueError("Selected free-memory points require complete visible support")
    if not ordered or max_groups == 0:
        empty = query.new_empty((query.shape[0], 0, query.shape[-1]))
        return (), selected_evidence(query, empty)
    if query.shape[0] != 1:
        raise ValueError("Reference pair reader handles one independent causal episode")
    if max_groups < 0 or not -1 <= threshold <= 1:
        raise ValueError("Invalid memory group or retrieval threshold")
    ids = [record.id for record in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("Sparse memory selection must not repeat a record ID")
    keys = torch.stack([
        selected.field(Collection.MEMORY, "record", i, "key")
        for i in ids
    ])[None].to(query.device)
    values = torch.stack([
        selected.field(Collection.MEMORY, "record", i, "value")
        for i in ids
    ])[None].to(query.device)
    strengths = torch.stack([
        selected.field(Collection.MEMORY, "record", i, "strength").reshape(())
        for i in ids
    ])[None].to(query.device)
    lifetimes = torch.stack([
        selected.field(Collection.MEMORY, "record", i, "lifetime_logit").reshape(())
        for i in ids
    ])[None].to(query.device)
    scores = selected_evidence(
        query, keys, strengths=strengths, threshold=threshold,
    )
    selected_ids = {value: i for i, value in enumerate(ids)}
    valid_pairs = [
        edge for edge in bank.relations(visibility.scope)
        if edge.left in selected_ids and edge.right in selected_ids
        and visibility.permits(edge.footprint)
    ]
    if annotation is not None:
        pair_support_units = annotation.pair_support_units
        pair_context_complete = annotation.pair_context_complete
        if not set(pair_support_units).issubset({
            r.source_id for r in annotation.query_teacher_inputs.ranges
        }):
            raise ValueError("Unobserved pair evidence must never enter the native reader")
    support_units = set(pair_support_units)
    depends: list[Tensor] = []
    priors: list[float] = []
    pair_strengths: list[Tensor] = []
    pair_lifetimes: list[Tensor] = []
    pair_values: list[Tensor] = []
    left = []
    right = []
    left_partner = []
    right_partner = []
    checked = []
    distinct = []
    for edge in valid_pairs:
        li, ri = selected_ids[edge.left], selected_ids[edge.right]
        left.append(li)
        right.append(ri)
        depends.append(
            selected.field(Collection.MEMORY, "relation", edge.id, "dependence_logits")
        )
        pair_values.append(
            selected.field(Collection.MEMORY, "relation", edge.id, "value")
        )
        pair_strengths.append(
            selected.field(Collection.MEMORY, "relation", edge.id, "strength").reshape(())
        )
        pair_lifetimes.append(
            selected.field(
                Collection.MEMORY, "relation", edge.id, "lifetime_logit",
            ).reshape(())
        )
        priors.append(edge.prior)
        # left_partner is PREVIOUS causal support for right, and vice versa.
        prior_right = _causal_support(ordered[ri], support_units)
        prior_left = _causal_support(ordered[li], support_units)
        left_partner.append(prior_right)
        right_partner.append(prior_left)
        # A fully checked absent endpoint may penalize the point only if the
        # caller has explicitly audited the whole eligible context.
        checked.append(float(pair_context_complete))
        distinct.append(float(bool(prior_right or prior_left)))
    point_coeff = scores.coefficient[0]
    pair_coeff = query.new_empty(0)
    if valid_pairs:
        dep = torch.stack(depends).to(query.device)
        pair_weights = torch.stack(pair_strengths).to(query.device)
        left_indices = torch.tensor(left, dtype=torch.long, device=query.device)
        right_indices = torch.tensor(right, dtype=torch.long, device=query.device)
        pair_evidence_out = pair_evidence(
            point_coeff, left=left_indices, right=right_indices,
            dependence_logits=dep,
            prior=torch.tensor(priors, dtype=torch.float32, device=query.device),
            pair_strength=pair_weights,
            left_partner=torch.tensor(left_partner, device=query.device),
            right_partner=torch.tensor(right_partner, device=query.device),
            left_checked=torch.tensor(checked, device=query.device),
            right_checked=torch.tensor(checked, device=query.device),
            distinct_support=torch.tensor(distinct, device=query.device),
        )
        point_coeff = pair_evidence_out.point
        pair_coeff = pair_evidence_out.pair
    buckets: dict[str, list[Constituent]] = defaultdict(list)
    for i, row in enumerate(ordered):
        group = row.group or f"point:{row.id}"
        buckets[f"{row.scope}/{group}"].append(Constituent(
            identity=f"memory:{row.id}", semantic=values[:, i],
            coefficient=point_coeff[i:i+1],
            lifetime_logit=lifetimes[:, i].reshape(1),
            admitted_at=anchor, version=f"{row.key_revision}:{row.payload_revision}",
        ))
    for i, edge in enumerate(valid_pairs):
        if not bool(distinct[i]):
            continue  # no second independent support leg
        li, ri = selected_ids[edge.left], selected_ids[edge.right]
        left_group = ordered[li].group or f"point:{edge.left}"
        right_group = ordered[ri].group or f"point:{edge.right}"
        # Only explicitly compatible source groups are mixed with points;
        # otherwise the learned pair occupies a bounded separate group.
        relation_group = (
            left_group if left_group == right_group else f"pair:{edge.id}"
        )
        buckets[f"{edge.scope}/{relation_group}"].append(Constituent(
            identity=f"pair:{edge.id}", semantic=pair_values[i][None].to(query.device),
            coefficient=pair_coeff[i:i+1],
            lifetime_logit=pair_lifetimes[i].reshape(1).to(query.device),
            admitted_at=anchor, version=f"relation:{edge.payload_revision}",
        ))
    ranked = sorted(
        buckets.items(),
        key=lambda pair: (
            -sum(float(c.coefficient.detach().sum()) for c in pair[1]),
            pair[0],
        ),
    )[:max_groups]
    revision = bank.revisions()
    groups = tuple(AdmittedGroup(
        group_id=f"{name}/{bank.space.revision}",
        constituents=tuple(items), anchor=anchor,
        revision=(
            f"{bank.space.revision}:k{revision.key}:p{revision.payload}:g{revision.graph}"
        ),
        compatible_group=name,
    ) for name, items in ranked)
    return groups, scores
