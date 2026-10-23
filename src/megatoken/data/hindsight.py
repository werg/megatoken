"""Immutable, window-aligned teacher targets. Never mounted as searchable references."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from megatoken.io import content_hash, sha256_file

from .media_assets import TensorAsset, TensorAssetStore

FORMAT = "hindsight-targets-v1"


def window_settings(config):
    return {
        name: getattr(config, name)
        for name in ("phase", "max_history_tokens", "scored_decisions", "max_window_records")
    }


def window_key(episode, start, end):
    return content_hash({"episode": episode.to_dict(), "start": start, "end": end})


def layout(episode, end, max_lookups):
    """Same causal group boundaries used by meridian training, with interval jitter off."""
    from megatoken.contracts import ROLE_TO_ID
    from megatoken.model.transducer import completed_turn, real_boundary
    from megatoken.training.transducer import Meridian, plan_boundaries

    positions = {r.record_id: i for i, r in enumerate(episode.records)}
    records = {r.record_id: r for r in episode.records}
    groups, meridians = [], []
    for group in episode.meridians:
        offsets = [positions[e.record_id] for e in group.events]
        if max(offsets) >= end:
            if min(offsets) < end:
                raise ValueError("Teacher target window splits a meridian")
            break
        group_records = [records[e.record_id] for e in group.events]
        if any(e.modality != "text" for e in group.events):
            raise ValueError("Stored teacher targets currently support text-only windows")
        roles = tuple(ROLE_TO_ID[r.role] for r in group_records)
        complete = [r.part_index + 1 == r.part_count for r in group_records]
        meridians.append(
            Meridian(
                tuple(e.track_id for e in group.events),
                roles,
                ("text",) * len(roles),
                real_boundary(roles, complete),
                completed_turn(roles, complete, "user"),
            )
        )
        groups.append(group_records)
    before = plan_boundaries(meridians, max_lookups=max_lookups, generator=None)
    return groups, before


def suffixes(groups, before):
    """Gap-2 read views: lookup t sees groups owned by lookup t+1 and later."""
    owners, current = [], -1
    for count in before:
        current += count
        owners.append(max(0, current))
    result = []
    for t in range(sum(before)):
        rows = [
            r for group, owner in zip(groups, owners, strict=True) if owner >= t + 1 for r in group
        ]
        if not rows:
            result.append(None)
            continue
        texts = [r.payload.decode("utf-8") for r in rows]
        text = "[D] " + "\n".join(texts)
        spans, offset = [], 4
        for record in rows:
            spans.append(
                {"record": record.record_id, "start": offset, "end": offset + len(record.payload)}
            )
            offset += len(record.payload) + 1
        result.append((text, spans))
    return result


def pool_bytes(text, offsets, hidden, rows=64):
    """Assign each retained token by its UTF-8 starting byte; zero rows stay identifiable."""
    byte_at = [0]
    for char in text:
        byte_at.append(byte_at[-1] + len(char.encode("utf-8")))
    retained = [(i, a, b) for i, (a, b) in enumerate(offsets) if b > a and b > 4 and a >= 4]
    if not retained:
        raise ValueError("Teacher view has no content tokens")
    extent = max(byte_at[b] for _, _, b in retained)
    length = max(1, extent - 4)
    pooled = hidden.new_zeros(rows, hidden.shape[-1], dtype=torch.float32)
    counts = hidden.new_zeros(rows, dtype=torch.float32)
    for i, a, _ in retained:
        row = min(rows - 1, (byte_at[a] - 4) * rows // length)
        pooled[row] += hidden[i].float()
        counts[row] += 1
    return pooled / counts.clamp_min(1)[:, None], counts > 0, extent


class TargetEncoding:
    """Train-fitted channel standardization plus one fixed PCA basis."""

    def __init__(self, mean, scale, projection, *, rows=64, clip=5.0):
        self.mean, self.scale, self.projection = mean, scale, projection
        self.rows, self.clip = rows, clip

    @classmethod
    def fit(cls, samples, *, width=512, rows=64):
        x = torch.cat([values[mask].float().cpu() for values, mask in samples])
        if len(x) < 2 or width > x.shape[1]:
            raise ValueError("Teacher fitting needs two nonempty rows and enough hidden channels")
        mean = x.mean(0)
        scale = x.std(0).clamp_min(1e-5)
        x = ((x - mean) / scale).clamp(-5, 5)
        if width == x.shape[1]:
            projection = torch.eye(width)
        else:
            _, vectors = torch.linalg.eigh(x.T @ x / max(1, len(x) - 1))
            projection = vectors[:, -width:].flip(1).contiguous()
            # Remove the arbitrary eigenvector sign from the encoding identity.
            largest = projection.abs().argmax(0)
            projection *= projection[largest, torch.arange(width)].sign()[None]
        return cls(mean, scale, projection, rows=rows)

    def encode(self, pooled, mask):
        value = ((pooled.cpu() - self.mean) / self.scale).clamp(
            -self.clip, self.clip
        ) @ self.projection
        value[~mask.cpu()] = 0
        if not bool(torch.isfinite(value).all()) or not bool(value.square().sum() > 0):
            raise ValueError("Teacher encoding is nonfinite or collapsed")
        return value.flatten()

    def state_dict(self):
        return dict(
            mean=self.mean,
            scale=self.scale,
            projection=self.projection,
            rows=self.rows,
            clip=self.clip,
        )


class HindsightTargets:
    def __init__(self, root, *, config=None, codec=None, value_dim=None, metadata_only=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] != FORMAT:
            raise ValueError("Unsupported hindsight target format")
        self.identity = sha256_file(self.root / "manifest.json")
        self._index = None
        self.metadata_only = metadata_only
        if self.manifest.get("scope") != "text_windows":
            raise ValueError("Obsolete teacher artifact; prepare targets for the selected corpus")
        index = self.manifest["index"]
        if index["path"] != "targets.sqlite":
            raise ValueError("Unsupported teacher index path")
        if not metadata_only:
            import sqlite3

            path = self.root / index["path"]
            if sha256_file(path) != index["sha256"]:
                raise ValueError("Teacher target index checksum changed")
            self._index = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        from .relations import EXTRACTOR

        if self.manifest.get("navigation_extractor") != EXTRACTOR:
            raise ValueError("Prepared navigation extractor identity differs from training")
        if config is not None and config.purpose == "production":
            coverage = self.manifest["coverage"]
            if config.trainable == "all":
                if not coverage["target_values"]:
                    raise ValueError("Default recipe has no actual teacher target values")
                if not coverage["navigation_examples"]:
                    raise ValueError("Default recipe has no eligible structural links")
        if config is not None and self.manifest["window_settings"] != window_settings(config):
            raise ValueError("Teacher target window settings differ from training")
        if codec is not None and self.manifest["codec"] != codec.fingerprint:
            raise ValueError("Teacher target codec differs from training")
        if value_dim is not None and self.manifest["value_dim"] != value_dim:
            raise ValueError("Teacher target value width differs from model")
        self.assets = TensorAssetStore(self.root)

    def close(self):
        if self._index is not None:
            self._index.close()
            self._index = None

    def validate_corpus(self, identity):
        if self.manifest["corpus_identity"] != identity:
            raise ValueError("Teacher targets were prepared for a different training corpus")

    def load(self, episode, start, end, before, *, device):
        key = window_key(episode, start, end)
        if self.metadata_only:
            raise RuntimeError("Coordinator metadata cannot be used to read teacher values")
        if self._index is not None:
            found = self._index.execute(
                "SELECT row_json FROM windows WHERE key=?", (key,)
            ).fetchone()
            row = None if found is None else json.loads(found[0])
        else:
            raise RuntimeError("Teacher target index is closed")
        if row is None:
            raise ValueError(
                "Training window has no stored teacher coverage; prepare this tranche first"
            )
        if row.get("unsupported"):
            from .hindsight_corpus import supported_text_window

            if supported_text_window(episode, end):
                raise ValueError("Unsupported teacher coverage does not match the actual window")
            return None, None
        if row["plan"] != before:
            raise ValueError("Stored teacher boundary plan differs from actual training window")
        values, sink = [], []
        for index, target in enumerate(row["targets"]):
            if (target is None) != (index + 1 >= sum(before)):
                raise ValueError("Teacher target sink flags disagree with gap-2 read groups")
            sink.append(target is None)
            if target is None:
                values.append(torch.zeros(self.manifest["value_dim"], device=device))
            else:
                asset = TensorAsset.from_dict(target["asset"])
                if (
                    asset.kind != "hindsight"
                    or asset.shape != (self.manifest["value_dim"],)
                    or asset.extractor != "hindsight:" + self.manifest["encoding_identity"]
                ):
                    raise ValueError("Teacher target asset has a different encoding identity")
                values.append(torch.from_numpy(self.assets.load(asset)).to(device))
        if len(values) != sum(before):
            raise ValueError("Stored teacher target count differs from its boundary plan")
        if not values:
            return torch.empty(0, self.manifest["value_dim"], device=device), torch.empty(
                0, dtype=torch.bool, device=device
            )
        return torch.stack(values), torch.tensor(sink, device=device, dtype=torch.bool)


def target_annotation(asset, spans, extent):
    return {"asset": asdict(asset), "source_spans": spans, "rendered_byte_end": extent}
