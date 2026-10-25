"""Ordered native bytes plus exact/hashed n-grams; no donor dependency at serve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.tensors import is_host

from .config import ModelConfig
from .layers import RMSNorm, masked_mean, position_features


@dataclass(frozen=True)
class SourceFeatures:
    features: Tensor
    mask: Tensor
    byte_lengths: Tensor


@dataclass(frozen=True)
class BakeReport:
    baked_rows: int
    removed_parameters: int
    max_lookup_error: float
    optimizer_rows_reset: bool


def validate_import_tensors(config, rows, descriptors, coverage, support, *, allow_hashed=False):
    """Shared donor validation without allocating a model or a learned map."""
    if rows.ndim != 1 or descriptors.shape != (rows.numel(), 128):
        raise ValueError("Expected rows[n] and fixed descriptors[n,128]")
    if coverage.shape != rows.shape or support.shape != rows.shape:
        raise ValueError("Coverage and support must each have one value per row")
    if rows.numel() == 0 or rows.unique().numel() != rows.numel():
        raise ValueError("Imported rows must be nonempty and unique")
    upper = config.ngram_rows if allow_hashed else config.exact_rows
    if bool(((rows < 0) | (rows >= upper)).any()):
        kind = "native" if allow_hashed else "exact native"
        raise ValueError(f"Imported rows must fit the {kind} key space")
    if allow_hashed and bool((rows < config.exact_rows).any()):
        raise ValueError("Native hashed imports must not target exact byte rows")
    if not bool(torch.isfinite(descriptors).all()):
        raise ValueError("Descriptors must be finite")
    if not bool(((coverage >= 0) & (coverage <= 1) & (support >= 0) & (support <= 1)).all()):
        raise ValueError("Coverage and support must lie in [0,1]")


class _NGramImport(nn.Module):
    """One donor's independent projected descriptors and learned native map."""

    def __init__(self, table: NativeNGramTable | None = None, *, device=None, dtype=None):
        super().__init__()
        self.import_map = nn.Linear(128, table.config.ngram_dim if table else 16, bias=False)
        if table is not None:
            self.import_map = self.import_map.to(
                device=table.embedding.weight.device, dtype=table.embedding.weight.dtype
            )
        elif device is not None:
            self.import_map = self.import_map.to(device=device, dtype=dtype)
        # Checkpoint loading creates additional donors after the table has moved.
        # Their buffers must start beside the map, including on CUDA.
        buffer_device = self.import_map.weight.device
        self.register_buffer("import_rows", torch.empty(0, dtype=torch.long, device=buffer_device))
        self.register_buffer("descriptors", torch.empty(0, 128, device=buffer_device))
        self.register_buffer("import_weights", torch.empty(0, device=buffer_device))

    def contribution(self, indices: Tensor, dtype: torch.dtype) -> Tensor:
        positions = torch.searchsorted(self.import_rows, indices)
        safe_positions = positions.clamp_max(self.import_rows.numel() - 1)
        covered = (positions < self.import_rows.numel()) & (
            self.import_rows[safe_positions] == indices
        )
        with torch.autocast(device_type=indices.device.type, enabled=False):
            contribution = F.linear(
                self.descriptors[safe_positions].float(), self.import_map.weight.float()
            )
            contribution = contribution * self.import_weights[safe_positions, None].float()
        return contribution.to(dtype) * covered.unsqueeze(-1)

    def _apply(self, fn, recurse=True):
        descriptors, weights = self.descriptors, self.import_weights
        result = super()._apply(fn, recurse=recurse)
        self.descriptors = descriptors.to(device=self.import_map.weight.device, dtype=torch.float32)
        self.import_weights = weights.to(device=self.import_map.weight.device, dtype=torch.float32)
        return result

    def install(self, rows: Tensor, descriptors: Tensor, coverage: Tensor, support: Tensor):
        permutation = torch.argsort(rows)
        self.import_rows = rows[permutation].detach()
        self.descriptors = descriptors.to(rows.device, torch.float32)[permutation].detach()
        self.import_weights = (
            (coverage * support).to(rows.device, torch.float32)[permutation].detach()
        )
        nn.init.normal_(self.import_map.weight, std=0.01)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name in ("import_rows", "descriptors", "import_weights"):
            key = prefix + name
            if key in state_dict:
                current = getattr(self, name)
                setattr(self, name, torch.empty_like(state_dict[key], device=current.device))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class NativeNGramTable(nn.Module):
    """Exact unigrams/bigrams and collision-labelled longer n-gram fallback.

    Every byte retains its ordered embedding path. Hashed n-grams are explicitly
    a fallback, never presented as exact strings. In the production configuration the
    entire table has exactly 2,500,000 x 16 = 40,000,000 trainable values.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.ngram_rows, config.ngram_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.import_map: nn.Linear | None = None
        self.register_buffer("import_rows", torch.empty(0, dtype=torch.long))
        self.register_buffer("descriptors", torch.empty(0, 128))
        self.register_buffer("import_weights", torch.empty(0))
        self.import_branches = nn.ModuleList()

    def _apply(self, fn, recurse=True):
        descriptors, weights = self.descriptors, self.import_weights
        result = super()._apply(fn, recurse=recurse)
        # Frozen extraction descriptors and support coefficients are reference
        # assets; moving the model to BF16 must not silently quantize them.
        self.descriptors = descriptors.to(device=self.embedding.weight.device, dtype=torch.float32)
        self.import_weights = weights.to(device=self.embedding.weight.device, dtype=torch.float32)
        return result

    def indices(self, byte_ids: Tensor) -> tuple[Tensor, Tensor]:
        if byte_ids.ndim != 2 or byte_ids.dtype not in (torch.int32, torch.int64, torch.uint8):
            raise ValueError("Native inputs must have shape [batch, bytes] and integer dtype")
        byte_ids = byte_ids.long()
        length = byte_ids.shape[1]
        order_indices: list[Tensor] = []
        validity: list[Tensor] = []
        hash_rows = self.config.ngram_rows - self.config.exact_rows
        positions = torch.arange(length, device=byte_ids.device)
        for order in self.config.ngram_orders:
            if order == 1:
                ids = byte_ids
            elif order == 2 and self.config.exact_bigrams:
                previous = F.pad(byte_ids, (1, 0))[:, :length]
                ids = 256 + previous * 256 + byte_ids
            else:
                windows = F.pad(byte_ids, (order - 1, 0)).unfold(-1, order, 1)
                hashed = torch.full_like(byte_ids, order)
                for i in range(order):
                    hashed = (hashed * 257 + windows[..., i] + 1).remainder(hash_rows)
                ids = self.config.exact_rows + hashed
            order_indices.append(ids)
            validity.append((positions >= order - 1).expand_as(byte_ids))
        return torch.stack(order_indices, -1), torch.stack(validity, -1)

    def _static_contribution(self, positions: Tensor, dtype: torch.dtype) -> Tensor:
        assert self.import_map is not None
        # Frozen descriptor and sensitive reduction are always FP32. Round the
        # static term before addition, identically in lookup and bake paths.
        with torch.autocast(device_type=positions.device.type, enabled=False):
            contribution = F.linear(
                self.descriptors[positions].float(), self.import_map.weight.float()
            )
            contribution = contribution * self.import_weights[positions, None].float()
        return contribution.to(dtype)

    @property
    def has_import(self) -> bool:
        return self.import_map is not None or bool(self.import_branches)

    def lookup(self, indices: Tensor) -> Tensor:
        base = self.embedding(indices)
        result = base
        if self.import_map is not None and self.import_rows.numel():
            positions = torch.searchsorted(self.import_rows, indices)
            safe_positions = positions.clamp_max(self.import_rows.numel() - 1)
            covered = (positions < self.import_rows.numel()) & (
                self.import_rows[safe_positions] == indices
            )
            result = result + self._static_contribution(
                safe_positions, base.dtype
            ) * covered.unsqueeze(-1)
        for branch in self.import_branches:
            result = result + branch.contribution(indices, base.dtype)
        return result

    def forward(self, byte_ids: Tensor) -> Tensor:
        indices, valid = self.indices(byte_ids)
        return self.lookup(indices) * valid.unsqueeze(-1)

    def install_import(
        self,
        rows: Tensor,
        descriptors: Tensor,
        coverage: Tensor,
        support: Tensor,
        *,
        allow_hashed: bool = False,
    ) -> None:
        """Attach one donor's immutable 128D descriptors to exact rows.

        Donor normalization/extraction belongs to the offline asset pipeline.
        Each donor gets an independent learned map; mapped contributions are added
        in asset order. Coverage and support are fixed, not learned gates.
        Install before constructing the optimizer (or explicitly add its group).
        """
        rows = rows.to(device=self.embedding.weight.device, dtype=torch.long)
        validate_import_tensors(
            self.config, rows, descriptors, coverage, support, allow_hashed=allow_hashed
        )
        if self.import_map is None:
            permutation = torch.argsort(rows)
            self.import_rows = rows[permutation].detach()
            self.descriptors = descriptors.to(rows.device, torch.float32)[permutation].detach()
            self.import_weights = (
                (coverage * support).to(rows.device, torch.float32)[permutation].detach()
            )
            self.import_map = nn.Linear(128, self.config.ngram_dim, bias=False).to(
                device=rows.device,
                dtype=self.embedding.weight.dtype,
            )
            nn.init.normal_(self.import_map.weight, std=0.01)
        else:
            branch = _NGramImport(self).to(device=rows.device)
            branch.install(rows, descriptors, coverage, support)
            self.import_branches.append(branch)

    @torch.no_grad()
    def bake(
        self,
        optimizer: Any | None = None,
        chunk_rows: int = 8192,
        master_weights: Tensor | None = None,
    ) -> BakeReport:
        """Fold static contributions, remove map/assets, and reset affected moments.

        Optimizers with FP32 shadow weights may implement ``on_engram_bake``;
        ordinary torch optimizers are handled directly. A before/after snapshot
        is the caller's responsibility; this operation intentionally is explicit.
        """
        if self.import_map is None and not self.import_branches:
            raise RuntimeError("No temporary engram import to bake")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        if master_weights is not None and (
            master_weights.dtype != torch.float32
            or master_weights.shape != self.embedding.weight.shape
        ):
            raise ValueError("Global master weights must be FP32 and match the native table")
        row_sets = ([self.import_rows] if self.import_map is not None else []) + [
            branch.import_rows for branch in self.import_branches
        ]
        rows = torch.unique(torch.cat(row_sets), sorted=True)
        before = self.lookup(rows).clone()
        removed = tuple(
            ([*self.import_map.parameters()] if self.import_map is not None else [])
            + [parameter for branch in self.import_branches for parameter in branch.parameters()]
        )
        if self.import_map is not None:
            for start in range(0, self.import_rows.numel(), chunk_rows):
                positions = torch.arange(
                    start, min(self.import_rows.numel(), start + chunk_rows), device=rows.device
                )
                indices = self.import_rows[positions]
                self.embedding.weight[indices] += self._static_contribution(
                    positions, self.embedding.weight.dtype
                )
        for branch in self.import_branches:
            for start in range(0, branch.import_rows.numel(), chunk_rows):
                indices = branch.import_rows[start : start + chunk_rows]
                self.embedding.weight[indices] += branch.contribution(
                    indices, self.embedding.weight.dtype
                )
        if master_weights is not None:
            # Match regular optimizer baking: after sequential model-dtype additions,
            # rebase the union's masters to the effective table value. Other masters
            # retain all global FP32 precision.
            master_weights.index_copy_(0, rows, self.embedding.weight[rows].float())
        reset = optimizer is not None
        if optimizer is not None:
            if hasattr(optimizer, "on_engram_bake"):
                optimizer.on_engram_bake(self.embedding.weight, rows, removed)
            else:
                removed_ids = {id(parameter) for parameter in removed}
                for group in optimizer.param_groups:
                    group["params"] = [p for p in group["params"] if id(p) not in removed_ids]
                for parameter in removed:
                    optimizer.state.pop(parameter, None)
                for value in optimizer.state.get(self.embedding.weight, {}).values():
                    if isinstance(value, Tensor) and value.shape == self.embedding.weight.shape:
                        value[rows] = 0
        self.import_map = None
        self.import_branches = nn.ModuleList()
        self.import_rows = rows.new_empty(0)
        self.descriptors = self.descriptors.new_empty(0, 128)
        self.import_weights = self.import_weights.new_empty(0)
        after = self.embedding(rows)
        error = float((before.float() - after.float()).abs().max())
        if error != 0:
            raise RuntimeError(f"Engram bake lookup parity failed: max error {error}")
        count = rows.numel()
        return BakeReport(count, sum(p.numel() for p in removed), error, reset)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        map_key = prefix + "import_map.weight"
        if map_key in state_dict and self.import_map is None:
            self.import_map = nn.Linear(128, self.config.ngram_dim, bias=False).to(
                self.embedding.weight
            )
        elif map_key not in state_dict:
            self.import_map = None
        branch_prefix = prefix + "import_branches."
        branch_indices = sorted(
            {
                int(key[len(branch_prefix) :].split(".", 1)[0])
                for key in state_dict
                if key.startswith(branch_prefix) and key.endswith(".import_map.weight")
            }
        )
        if branch_indices != list(range(len(branch_indices))):
            error_msgs.append(f"{prefix}engram branch indices are not contiguous")
        # Bootstrap can load after the optimizer has registered these parameters.
        # Preserve existing branches so its gradient hooks and masters stay attached.
        while len(self.import_branches) < len(branch_indices):
            self.import_branches.append(_NGramImport(self))
        if len(self.import_branches) > len(branch_indices):
            self.import_branches = self.import_branches[: len(branch_indices)]
        for name in ("import_rows", "descriptors", "import_weights"):
            key = prefix + name
            if key in state_dict:
                current = getattr(self, name)
                setattr(self, name, torch.empty_like(state_dict[key], device=current.device))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class ByteFrontend(nn.Module):
    """Causal stride-sized byte features, shared by ingestion and seed control.

    Every byte has an ordered embedding path. A learned, content-dependent
    mixture adds the available n-gram orders without treating prefix padding
    as native source bytes. Hierarchical reduction belongs to ChunkEncoder.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.bytes = nn.Embedding(256, config.byte_dim)
        self.ngrams = NativeNGramTable(config)
        self.ordered_projection = nn.Linear(
            config.source_stride * config.byte_dim, config.source_dim, bias=False
        )
        self.ngram_projection = nn.Linear(
            len(config.ngram_orders) * config.ngram_dim, config.source_dim, bias=False
        )
        self.ngram_mixture = nn.Linear(config.byte_dim, len(config.ngram_orders), bias=False)
        nn.init.zeros_(self.ngram_mixture.weight)
        self.convolution = nn.Conv1d(config.source_dim, 2 * config.source_dim, 3, bias=False)
        self.norm = RMSNorm(config.source_dim, config.norm_eps)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, byte_ids: Tensor, lengths: Tensor | None = None) -> SourceFeatures:
        if byte_ids.ndim != 2:
            raise ValueError("A physical record batch must have shape [batch,bytes]")
        if byte_ids.dtype not in (torch.uint8, torch.int32, torch.int64):
            raise TypeError("byte_ids must be integer native bytes")
        # Value checks inspect host tensors only; device tensors come from byte_tensor.
        if is_host(byte_ids) and bool(((byte_ids < 0) | (byte_ids > 255)).any()):
            raise ValueError("Input byte values must lie in [0,255]")
        batch, length = byte_ids.shape
        if batch < 1:
            raise ValueError("A physical record batch must contain at least one record")
        if lengths is None:
            lengths = torch.full((batch,), length, device=byte_ids.device, dtype=torch.long)
        if lengths.dtype not in (torch.int32, torch.int64, torch.uint8):
            raise TypeError("Record lengths must be integer byte counts")
        if lengths.shape != (batch,) or (
            is_host(lengths) and bool(((lengths < 0) | (lengths > length)).any())
        ):
            raise ValueError("Every record length must be in [0, padded byte length]")
        if length == 0:
            byte_ids = byte_ids.new_zeros((batch, 1))
            length = 1
        lengths = lengths.to(device=byte_ids.device, dtype=torch.long)
        mask = torch.arange(length, device=byte_ids.device)[None] < lengths[:, None]
        stride = self.config.source_stride
        padding = (-length) % stride
        ids = byte_ids.long()
        byte_features = self.bytes(ids) * mask[..., None]
        ordered = F.pad(byte_features, (0, 0, 0, padding)).reshape(
            batch, -1, stride * self.config.byte_dim
        )
        indices, available = self.ngrams.indices(ids)
        mixture_logits = self.ngram_mixture(byte_features).float()
        mixture = mixture_logits.masked_fill(~available, torch.finfo(torch.float32).min).softmax(-1)
        mixture = mixture * available
        ngrams = self.ngrams.lookup(indices)
        ngrams = (ngrams * mixture.to(ngrams.dtype).unsqueeze(-1)).flatten(-2)
        ngrams = ngrams * mask[..., None]
        ngrams = (
            F.pad(ngrams, (0, 0, 0, padding)).reshape(batch, -1, stride, ngrams.shape[-1]).mean(-2)
        )
        x = self.ordered_projection(ordered) + self.ngram_projection(ngrams)
        source_mask = torch.arange(x.shape[1], device=x.device)[None] * stride < lengths[:, None]
        x = x + position_features(x.shape[1], x.shape[-1], x.device).to(x.dtype)
        value, gate = (
            self.convolution(F.pad(x.transpose(1, 2), (2, 0))).transpose(1, 2).chunk(2, -1)
        )
        x = self.norm(x + self.residual_gate.sigmoid() * value * gate.sigmoid())
        return SourceFeatures(x * source_mask[..., None], source_mask, lengths)


class SeedControlEncoder(nn.Module):
    """Encode explicit causal seed/reference features without touching memory."""

    # The optimizer gives this module half the learning rate (training.optimizer).
    slow_learning = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.source_dim, config.norm_eps)
        self.projection = nn.Linear(config.source_dim, 2 * config.control_dim, bias=False)
        self.output = nn.Linear(config.control_dim, config.control_dim, bias=False)

    def forward(self, source: SourceFeatures) -> Tensor:
        value, gate = self.projection(masked_mean(self.norm(source.features), source.mask)).chunk(
            2, -1
        )
        return self.output(value * gate.sigmoid())
