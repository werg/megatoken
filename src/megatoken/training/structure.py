"""Bounded, training-only transition teaching; no future rows enter normal search."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from megatoken.data.relations import local_relations
from megatoken.model.ingest import IngestItem
from megatoken.model.transducer import metadata_payload


class StructuralTeaching:
    def __init__(self, native, pairs, keys, departures, destinations, *, negatives=32):
        self.native = native
        self.pairs = pairs  # source record -> (positive edge index, weight)
        from megatoken.memory.transducer_packing import packed_unit

        self.keys = packed_unit(keys)
        self.departures = departures
        self.destinations = destinations
        self.negatives = negatives
        self.losses = []
        self.recalled = 0
        self.examples = 0

    @classmethod
    @torch.no_grad()
    def prepare(cls, model, multi, episode, steps, *, max_examples=8, negatives=32):
        """Privileged target rows are held here only, never mounted or checkpointed."""
        available = {e.record.record_id: e for step in steps for e in step.records}
        records = {r.record_id: r for r in episode.records}
        relations = [
            r
            for r in local_relations(episode)
            if r["left"] in available
            and r["right"] in available
            and records[r["left"]].media_type.startswith("text/")
            and records[r["right"]].media_type.startswith("text/")
        ][:max_examples]
        if not relations:
            return None
        names = list(dict.fromkeys(x for r in relations for x in (r["left"], r["right"])))
        # More unrelated records provide bounded negatives; no extra loss examples.
        names += [
            name
            for name, r in records.items()
            if name not in names and name in available and r.media_type.startswith("text/")
        ][:negatives]
        items = []
        for name in names:
            ids = model.codec.encode(available[name].payload)
            if not ids or len(ids) > model.config.max_capacity:
                raise ValueError("Structural target does not fit one model chunk")
            raw = model.core.encode_chunk(torch.tensor([ids], device=model.transducer.bank.device))[
                0
            ]
            items.append(IngestItem(raw, "text", metadata_payload({})))
        rows = model.transducer.ingest_rows(multi, items)
        index = {name: i for i, name in enumerate(names)}
        # Each supervised pair owns an edge, even when destinations repeat.
        targets = [index[r["right"]] for r in relations]
        sources = [index[r["left"]] for r in relations]
        # Standalone negative item edges depart from their own content locations.
        keys = torch.cat((rows.key[targets], rows.key))
        deps = torch.cat((rows.destination[sources], rows.destination))
        dst = torch.cat((rows.destination[targets], rows.destination))
        pairs = {}
        for i, row in enumerate(relations):
            pairs.setdefault(row["left"], []).append((i, row["weight"]))
        return cls(model.transducer, pairs, keys, deps, dst, negatives=negatives)

    def after(self, multi, state, record_ids, *, effort):
        positive = [p for name in record_ids for p in self.pairs.get(name, ())]
        if not positive or not torch.is_grad_enabled():
            return
        _, query = self.native.causal_query(multi, state, training=True, effort=effort)
        bank = self.native.bank
        for edge, weight in positive:
            location = self.departures[edge : edge + 1]
            modulated = bank.modulated(query, location)[0]
            scores = bank.log_beta_query.float().exp() * (
                self.keys @ modulated
            ) + bank.log_beta_location.float().exp() * (self.departures @ location[0])
            # Measure ordinary selection before inserting the correct pair.
            with torch.no_grad():
                near = (
                    (self.departures @ location[0])
                    .topk(min(bank.geometry.neighbors, len(scores)))
                    .indices
                )
                selected = near[scores[near].topk(min(bank.geometry.selected, len(near))).indices]
                self.recalled += int(bool((selected == edge).any()))
                candidates = scores.detach().topk(min(self.negatives + 1, len(scores))).indices
                candidates = torch.unique(torch.cat((candidates, candidates.new_tensor([edge]))))
            sink = bank.sink_bias.float() + bank.sink_direction.float() @ F.normalize(
                query.float(), dim=0
            )
            logits = torch.cat((scores[candidates], sink[None]))
            loss = torch.logsumexp(logits, 0) - scores[edge]
            self.losses.append(loss * weight)
            self.examples += 1

    def terms(self):
        if not self.losses:
            return None, {}
        return torch.stack(self.losses).mean(), {
            "navigation_examples": self.examples,
            "navigation_shortlist_recall": self.recalled / self.examples,
        }
