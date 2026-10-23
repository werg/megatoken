#!/usr/bin/env python
"""Bootstrap probe: is the byte codec alive, is the predictor alive, are the experts alive?

THE QUESTIONS
-------------
Megatoken turns each chunk of bytes into a small block of learned vectors (the "latent",
32 vectors in the real model), predicts the next chunk's latent from the earlier ones, and
decodes a latent back to bytes with a deterministic decoder. The first training stage
(phase B0, "bootstrap") teaches all of that on plain text. Before spending cloud credits,
this probe trains a short bootstrap run on a small text mix and asks three things.

1. ROUND TRIP. Take held-out text the model never saw, cut it into chunks, encode each chunk,
   add Gaussian noise to the latent, decode, and compare with the original. Noise matters
   because at serving time the latent is a random draw, not an exact encoding. We sweep the
   noise level and (by training one model per setting) the maximum chunk size.
2. PREDICTOR ALIVE. For each held-out chunk (after the first of its document) the model
   predicts that chunk's latent from the earlier chunks. We compare its squared error with
   two trivial guesses: "same as the previous chunk's latent" and "the average of the
   latents seen so far in this document". The predictor is only worth keeping if it beats
   both.
3. EXPERT HEALTH. The model shares banks of "experts" (small sub-networks) and has routers
   that pick among them. We report how evenly each bank and router is used, and list any
   trainable parameter that never received a gradient (or received exactly zero) during
   the run. It reuses the trainer's own routing statistics, not a re-implementation.

HOW TO RUN (always with /home/werg/megatoken-runs/env/dev/bin/python)
-------------------------------------------------------------------
Smoke test (CPU, well under 90 seconds on an idle machine; shows the plumbing, not quality):

    python scripts/probes/probe_bootstrap.py --smoke --output /tmp/bootstrap_smoke.json

Cheap codec tuning (trains only the encoder and decoder on the noisy reconstruction
objective, in batches; far cheaper per step than the real trainer. It cannot say anything
about the predictor, so the predictor section is skipped). Observed on the tiny preset:
3000 steps of 32 chunks took about 5 minutes on the shared GPU (it should take 25+ minutes on
a busy CPU) and reached byte accuracy ~0.9 on held-out 32-byte chunks; 300 steps reached
nothing (it only guessed the most common byte):

    python scripts/probes/probe_bootstrap.py --device cuda --train-mode codec --steps 3000 \
        --codec-batch 32 --chunk-caps 16,32,64 --train-noise 0.05,0.1,0.2 --output /tmp/codec.json

Full probe with the real B0 trainer (all three questions). It is slow: about 50 seconds per
optimizer step for ONE window on a busy 20-core CPU (25 steps took 20 minutes in total,
including evaluation), so budget hours for the few hundred steps a predictor signal needs.
25 steps showed the plumbing works but, as expected, nothing was learned yet:

    python scripts/probes/probe_bootstrap.py --train-mode full --steps 300 --windows-per-step 2 \
        --chunk-caps 32 --output /tmp/full.json
    python scripts/probes/probe_bootstrap.py --model 32x256 --device cuda --precision bf16 \
        --train-mode full --steps 500 --chunk-caps 128 --windows-per-step 4 \
        --text-dir /path/to/real/text --output /tmp/full_gpu.json   (untested at this size)

Text: by default a built-in mix of Python/JavaScript/C code, JSON tool-call-like lines and
prose is generated (no downloads). It is deliberately LOW ENTROPY (small vocabularies,
templates, a few random numbers), so success on it is necessary but not sufficient; point
--text-dir at real files (.py/.js/.c/... = code, .json/.jsonl = json, .md/.txt = prose)
for a harder test. Held-out text is separate documents (separate files with --text-dir).

WHAT THE NUMBERS MEAN
---------------------
Per run (one chunk cap and one training noise level), under "round_trip":
  exact_match_rate   fraction of held-out chunks decoded byte-for-byte identical. 1.0 = perfect.
  byte_accuracy      fraction of original byte positions where the decoded byte is the same
                     (an insertion or deletion shifts everything after it, so this is harsh).
  similarity         difflib similarity ratio between decoded and original, 0..1; forgiving of
                     shifts. Reported per content type ("by_type") and per chunk length bucket
                     ("by_length"), and for every noise level.
  baseline_byte_accuracy  accuracy of always guessing the single most common training byte;
                     byte_accuracy must be well above it for the codec to be doing anything.
  latent_rms / snr   typical size of a latent coordinate (after subtracting the mean over
                     held-out chunks) and that size divided by the training noise std. When the
                     SNR is below about 1 the noise drowns the information and no decoder
                     can recover the bytes.
  Noise levels are the --noise-levels values plus the training noise std (tag
  "training_tau") and the average learned prior std measured on held-out data (tag
  "serving_prior_std", the level a served sample actually has).
Under "predictor" (full mode only):
  mse_model / mse_repeat / mse_running_mean   mean squared error per latent coordinate against
                     the encoder's latent for the true next chunk.
  ratio_vs_repeat, ratio_vs_running_mean   mse_model divided by each baseline. Below 1 means
                     the predictor beats the trivial guess; 0.5 means half the error.
  ratio_vs_constant  the same against an oracle constant (the held-out mean latent).
  kl_model           the actual B0 prediction loss (Gaussian KL per coordinate) of the model.
  generation_from_prediction  exact-match rate when decoding the predicted latent itself.
Under "expert_health" (per bank and per router):
  normalized_entropy usage entropy divided by log(number of experts); 1.0 = perfectly even,
                     0 = one expert gets everything.
  frac_below_1pct_of_avg  fraction of experts used less than 1% as much as the average expert.
  zero_gradient      parameter names that had no gradient at all, or exactly zero gradient,
                     in every optimizer step, grouped by category (see below);
                     "dead_experts" lists experts inside a shared bank whose weights never
                     received any gradient.

HEALTHY VERSUS UNHEALTHY (all thresholds are GUESSES from the design, not measured on a
trained model; the "assessment" field applies them mechanically)
------------------------------------------------------------------------------------
  Round trip at the training noise: exact_match_rate >= 0.5 healthy, 0.1-0.5 weak, < 0.1
    unhealthy. byte_accuracy >= 0.9 healthy. A healthy codec loses at most about half of its
    noise-free exact-match rate when the noise doubles.
    If unhealthy and noise-free (sigma 0) is also bad: the codec is under-trained or the chunk
    is too long for the latent; train longer, raise --lr, or lower the chunk cap (chunking
    hard limit / max_bytes). If good at sigma 0 but bad at the training noise: lower the model's
    codec_noise_std (--set codec_noise_std=0.05) or train longer so latents grow. If good at the
    training noise but bad at the serving std: the learned prior scale is too wide; raise the
    facet_prediction loss weight or lower prior_noise_std.
  Predictor: ratio_vs_repeat <= 0.9 and ratio_vs_running_mean <= 0.95 healthy; >= 1.0 on
    either means the predictor is no better than a trivial guess (dead or untrained). Fix:
    more steps, a higher --lr, a higher facet_prediction weight (default 0.1), check C/P
    gradients in the zero_gradient list. With only a few hundred steps on the tiny preset a
    ratio near 1 is expected; the probe then shows the signal is not yet visible, not that it
    is absent.
  Experts: normalized_entropy >= 0.8 and frac_below_1pct_of_avg <= 0.1 healthy;
    entropy < 0.5 or more than 0.3 of experts starved means routing is collapsing; raise the
    router_balance loss weight (default 0.005) or the router z-loss. Zero-gradient
    parameters: the output sorts them into categories (by design for text bootstrap, memory
    path that needs lookups, shared pool never routed to, outside the codec-only path, and
    "unexpected"). Only the "unexpected" ones are surprises; healthy is zero of them after a
    run of at least ~50 steps (very short runs can leave rarely used parameters untouched).
    A shared pool that is never routed to is also worth asking about, although pools that only
    media or long chunks use stay idle on short text by design.

SETTINGS TO CHANGE: --lr, --steps, --chunk-caps, --train-noise / --set codec_noise_std=...,
--set prior_noise_std=..., the loss weights through --loss-weight NAME=VALUE (router_balance,
facet_prediction, reconstruction), --model preset, --effort.

LIMITS: the tiny preset (4 facets x 32 dims) holds roughly 100-200 numbers per chunk, so it
can only show that the plumbing and learning signal exist; the 32x256 preset on GPU for a
few hundred steps is the smallest setting that says anything about real exact-match rates.
In full mode the evaluation replays documents through the model without the memory-bank
lookups that training included, and runs one document at a time.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import fcntl
import hashlib
import io
import json
import math
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

GPU_LOCK = "/tmp/megatoken-gpu.lock"
KINDS = ("code", "json", "prose")
LENGTH_EDGES = (8, 16, 32, 64, 128, 256, 512, 1024)
# Parameter families that bootstrap training on plain text cannot reach by design. They are
# listed apart from surprises. (Guesses from reading the code; the output names every one.)
BY_DESIGN_UNUSED = (
    "core.media_encoder",  # media inputs
    "embedding_realizer",  # the embedding readout used by media and long-document stages
    "transducer.hindsight_realizer",  # memory write path
    "transducer.ingest",
    "core.reference_projection",  # reference and seed controls: plain documents carry none
    "core.control_injection",
    "core.control_encoder",
    "core.modality_head",  # only one output kind (text) is ever legal
    "core.expert_pools.router_semantic_projection",  # no track purposes in plain documents
    "core.adapters.owners.e.semantic_preference",
    "decoder.schema_codes",  # tool-call schemas, copy sources and identifiers
    "decoder.layout_codes",
    "decoder.identifier_slots",
    "decoder.binding_query",
    "decoder.free_binding",
    "decoder.public_projection",
)
# Encoder, predictor and memory pieces that codec-only training never touches.
OUTSIDE_CODEC_PATH = (
    "core.stages",
    "core.central",
    "core.remix",
    "core.p_router",
    "core.effort_embedding",
    "core.stage_codes",
    "core.initial_facets",
    "core.adapters",
    "facet_distribution",
)


# --------------------------------------------------------------------------------------
# Text corpus
# --------------------------------------------------------------------------------------

_PARTS = (
    "parse load user token buffer index value node cache queue item path count total name key "
    "size limit batch state"
).split()
_NOUNS = (
    "river garden market engine window letter signal bridge forest ledger harbor kitchen mirror "
    "ticket valley report lantern compass orchard server"
).split()
_ADJS = (
    "quiet broken early distant careful hidden golden narrow patient rapid ancient simple bright "
    "crooked gentle"
).split()
_VERBS = (
    "crosses repairs follows measures describes borrows watches carries answers builds forgets "
    "reaches"
).split()
_PREPS = "near behind beside beyond under across along inside".split()
_CONNECTIVES = ("However,", "Meanwhile,", "In fact,", "Later,", "As a result,", "Still,", "Then")
_TOOLS = ("search_files", "read_file", "write_file", "run_tests", "list_dir", "http_get")
_STATUS = ("ok", "error", "pending", "done")
_PATHS = ("src/main", "src/util", "tests/unit", "docs/guide", "lib/core", "app/models")
_EXT = (".py", ".js", ".md", ".json", ".txt")


def _ident(rng: random.Random, k: int = 2) -> str:
    return "_".join(rng.sample(_PARTS, k))


def _code_block(rng: random.Random) -> str:
    style = rng.choices(("python", "javascript", "c"), weights=(6, 2, 2))[0]
    name, args = _ident(rng), [_ident(rng, 1) for _ in range(rng.randint(1, 3))]
    body: list[str] = []
    for _ in range(rng.randint(2, 5)):
        a, b = rng.choice(args), _ident(rng, 1)
        n = rng.randint(0, 99)
        kind = rng.choice(("assign", "if", "loop", "call"))
        if style == "python":
            body += {
                "assign": [f"    {b} = {a} + {n}"],
                "if": [f"    if {a} > {n}:", f"        return {b}"],
                "loop": [f"    for {b} in range({n}):", f"        {a} += {b}"],
                "call": [f"    print({a}, {n})"],
            }[kind]
        elif style == "javascript":
            body += {
                "assign": [f"  const {b} = {a} + {n};"],
                "if": [f"  if ({a} > {n}) {{", f"    return {b};", "  }"],
                "loop": [f"  for (let {b} = 0; {b} < {n}; {b}++) {{", f"    {a} += {b};", "  }"],
                "call": [f"  console.log({a}, {n});"],
            }[kind]
        else:
            body += {
                "assign": [f"    int {b} = {a} + {n};"],
                "if": [f"    if ({a} > {n}) {{", f"        return {b};", "    }"],
                "loop": [
                    f"    for (int {b} = 0; {b} < {n}; {b}++) {{",
                    f"        {a} += {b};",
                    "    }",
                ],
                "call": [f'    printf("%d\\n", {a});'],
            }[kind]
    ret = rng.choice(args)
    if style == "python":
        head = f"def {name}({', '.join(args)}):"
        doc = f'    """{_ident(rng).replace("_", " ").capitalize()} for {args[0]}."""'
        return "\n".join([head, doc, *body, f"    return {ret}", "", ""])
    if style == "javascript":
        return "\n".join(
            [f"function {name}({', '.join(args)}) {{", *body, f"  return {ret};", "}", "", ""]
        )
    sig = ", ".join(f"int {x}" for x in args)
    return "\n".join([f"int {name}({sig}) {{", *body, f"    return {ret};", "}", "", ""])


def _json_line(rng: random.Random) -> str:
    kind = rng.choice(("call", "result", "record"))
    if kind == "call":
        obj: dict[str, Any] = {
            "name": rng.choice(_TOOLS),
            "arguments": {
                "path": rng.choice(_PATHS) + rng.choice(_EXT),
                "limit": rng.randint(1, 50),
                "recursive": rng.choice((True, False)),
            },
        }
    elif kind == "result":
        obj = {
            "status": rng.choice(_STATUS),
            "count": rng.randint(0, 99),
            "items": [{"id": rng.randint(1, 999), "name": _ident(rng, 1)} for _ in range(2)],
        }
    else:
        obj = {
            "user": _ident(rng, 1),
            "score": round(rng.random() * 10, 2),
            "tags": rng.sample(_NOUNS, 2),
        }
    return json.dumps(obj) + "\n"


def _sentence(rng: random.Random) -> str:
    def noun_phrase() -> str:
        return f"the {rng.choice(_ADJS)} {rng.choice(_NOUNS)}"

    text = f"{noun_phrase().capitalize()} {rng.choice(_VERBS)} {noun_phrase()}"
    text += f" {rng.choice(_PREPS)} the {rng.choice(_NOUNS)}."
    if rng.random() < 0.4:
        text = f"{rng.choice(_CONNECTIVES)} {text[0].lower()}{text[1:]}"
    return text


def make_document(kind: str, rng: random.Random, size: int) -> bytes:
    """One synthetic document of roughly ``size`` bytes."""
    pieces: list[str] = []
    total = 0
    while total < size:
        if kind == "code":
            piece = _code_block(rng)
        elif kind == "json":
            piece = _json_line(rng)
        else:
            piece = " ".join(_sentence(rng) for _ in range(rng.randint(2, 4))) + "\n"
        pieces.append(piece)
        total += len(piece)
    return "".join(pieces).encode()


@dataclass(frozen=True)
class Doc:
    kind: str
    name: str
    payload: bytes


def builtin_corpus(seed: int, per_kind: int, size: int, *, offset: int) -> list[Doc]:
    rng = random.Random(f"bootstrap-corpus:{seed}:{offset}")
    return [
        Doc(kind, f"{kind}-{offset}-{i}", make_document(kind, rng, size))
        for kind in KINDS
        for i in range(per_kind)
    ]


_CODE_SUFFIX = {".py", ".js", ".ts", ".c", ".h", ".cc", ".cpp", ".rs", ".go", ".java", ".sh"}
_JSON_SUFFIX = {".json", ".jsonl"}
_PROSE_SUFFIX = {".md", ".txt", ".rst"}


def directory_corpus(
    root: Path, seed: int, heldout_fraction: float, max_files: int, max_bytes: int
) -> tuple[list[Doc], list[Doc]]:
    """Real files; split train/held-out by a hash of the path (by position if very few files)."""
    docs: list[Doc] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        kind = (
            "code"
            if suffix in _CODE_SUFFIX
            else "json"
            if suffix in _JSON_SUFFIX
            else "prose"
            if suffix in _PROSE_SUFFIX
            else None
        )
        if kind is None:
            continue
        data = path.read_bytes()[:max_bytes]
        if data.strip():
            docs.append(Doc(kind, str(path.relative_to(root)), data))
        if len(docs) >= max_files:
            break
    if not docs:
        raise SystemExit(f"--text-dir {root} holds no usable text files")
    train, held = [], []
    if len(docs) < 5:
        for doc in docs:
            cut = int(len(doc.payload) * (1 - heldout_fraction))
            train.append(replace(doc, payload=doc.payload[:cut]))
            held.append(replace(doc, name=doc.name + "#tail", payload=doc.payload[cut:]))
        return train, held
    for doc in docs:
        bucket = int(hashlib.sha256(f"{seed}:{doc.name}".encode()).hexdigest(), 16) % 1000
        (held if bucket < heldout_fraction * 1000 else train).append(doc)
    if not held:
        held.append(train.pop())
    return train, held


# --------------------------------------------------------------------------------------
# Chunking, prepared data, configuration
# --------------------------------------------------------------------------------------


def chunk_document(doc: Doc, cap: int) -> list[bytes]:
    """The project's own record splitter (line/word aware, never above ``cap`` bytes)."""
    from megatoken.data.schema import make_records

    records = make_records(
        doc.payload,
        logical_id=doc.name,
        role="document",
        policy_mask=True,
        provenance="natural",
        max_bytes=cap,
    )
    return [r.payload for r in records if r.payload]


def make_episode(doc: Doc, index: int, cap: int):
    from megatoken.data import Episode, Lineage
    from megatoken.data.schema import Meridian, TrackEvent, TrackSpec, make_records

    records = make_records(
        doc.payload,
        logical_id=f"doc{index}",
        role="document",
        policy_mask=True,
        provenance="natural",
        max_bytes=cap,
    )
    meridians = tuple(
        Meridian(k, (TrackEvent(r.record_id, "doc", "text", k, "document"),))
        for k, r in enumerate(records)
    )
    return Episode(
        f"probe-bootstrap-{index}",
        records,
        Lineage("probe-bootstrap", "v1", f"family-{index}"),
        stream="bootstrap",
        tracks=(TrackSpec("doc", "document"),),
        meridians=meridians,
    )


def prepare_training_data(root: Path, docs: list[Doc], cap: int) -> tuple[Path, dict[str, int]]:
    from megatoken.data import DedupConfig, DedupStore
    from megatoken.generation.codec import ByteCodec

    codec = ByteCodec.byte_fallback()
    store = DedupStore(root / "spool.sqlite", DedupConfig(train_fraction=1, validation_fraction=0))
    statuses: Counter[str] = Counter()
    try:
        with store.transaction():
            for index, doc in enumerate(docs):
                statuses[str(store.add(make_episode(doc, index, cap), codec))] += 1
        store.freeze()
        from megatoken.data.storage import export_prepared

        export_prepared(
            store,
            root / "data",
            codec=codec,
            source_receipts=[],
            pipeline_fingerprint="probe-bootstrap-v1",
        )
    finally:
        store.close()
    return root / "data", dict(statuses)


def parse_overrides(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def model_overrides(args: argparse.Namespace, tau: float) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if args.model != "tiny":
        values["vocab_size"] = 256  # the byte codec has 256 tokens
    values.update(parse_overrides(args.set))
    values["codec_noise_std"] = tau
    return values


def lr_at(step: int, steps: int, peak: float, warmup_fraction: float) -> float:
    """Linear warm-up from 0.1x to 1x peak over the first fraction of steps, then flat."""
    warm = max(1, int(steps * warmup_fraction))
    return peak * (0.1 + 0.9 * step / warm) if step < warm else peak


# --------------------------------------------------------------------------------------
# Instrumentation: reuse the trainer's own routing statistics
# --------------------------------------------------------------------------------------


class RunMonitor:
    """Accumulates router assignment counts, gradient liveness and loss history over a run."""

    def __init__(self) -> None:
        self.steps = 0
        self.model: Any = None
        self.router_counts: dict[str, np.ndarray] = {}
        self.router_experts: dict[str, int] = {}
        self.grad_max: dict[str, float] = {}
        self.grad_present_steps: dict[str, int] = {}
        self.banks: dict[str, Any] = {}
        self.bank_of_param: dict[str, str] = {}
        self.param_bank: dict[str, str] = {}
        self.expert_grad_max: dict[str, np.ndarray] = {}
        self.bank_rows: dict[str, np.ndarray] = {}
        self.lr_history: list[float] = []

    # called once the model exists (from the patched build_model or the codec loop)
    def attach(self, model: Any) -> None:
        self.model = model
        pools = model.core.expert_pools
        self.banks = {f"pool:{name}": bank for name, bank in pools.pools.items()}
        self.banks["router_bank"] = pools.router_bank

    def on_router_window(self, window: Any, routers: dict[str, Any]) -> None:
        for name, router in routers.items():
            state = window.states.get(id(router))
            self.router_experts[name] = int(router.experts)
            counts = self.router_counts.setdefault(name, np.zeros(int(router.experts)))
            if state is not None:
                counts += state.counts.detach().cpu().numpy()

    def on_dispatch_reset(self) -> None:
        """The trainer zeroes the P and U bank counters every step: bank them first."""
        for label, bank in self.banks.items():
            rows = bank.routed_rows.detach().cpu().numpy().astype(np.float64)
            self.bank_rows[label] = self.bank_rows.get(label, 0.0) + rows

    def bank_usage(self, label: str) -> np.ndarray:
        current = self.banks[label].routed_rows.detach().cpu().numpy().astype(np.float64)
        return current + self.bank_rows.get(label, 0.0)

    def on_step(self, optimizer: Any) -> None:
        self.steps += 1
        if not self.bank_of_param and self.model is not None:
            by_id = {id(p): n for n, p in optimizer.model_parameters.items()}
            for label, bank in self.banks.items():
                for p in bank.parameters():
                    name = by_id.get(id(p))
                    if name is not None:
                        self.param_bank[name] = label
                        if "adapter" not in name:
                            self.bank_of_param[name] = label
        names = list(optimizer.masters)
        present = [(n, optimizer.masters[n].grad) for n in names]
        live = [(n, g) for n, g in present if g is not None]
        if live:
            values = torch.stack([g.detach().abs().max().float() for _, g in live]).cpu().tolist()
        else:
            values = []
        for (name, _), value in zip(live, values, strict=True):
            self.grad_max[name] = max(self.grad_max.get(name, 0.0), float(value))
            self.grad_present_steps[name] = self.grad_present_steps.get(name, 0) + 1
        for name in names:
            self.grad_max.setdefault(name, 0.0)
            self.grad_present_steps.setdefault(name, 0)
        for name in self.bank_of_param:
            grad = optimizer.masters[name].grad
            if grad is None:
                continue
            per_expert = grad.detach().abs().flatten(1).amax(1).float().cpu().numpy()
            previous = self.expert_grad_max.get(name)
            self.expert_grad_max[name] = (
                per_expert if previous is None else np.maximum(previous, per_expert)
            )


@contextlib.contextmanager
def instrument(monitor: RunMonitor, lr_fn=None, capture_model: bool = False):
    """Temporarily wrap the optimizer step, router replay and model construction."""
    from megatoken.model.core import FacetedCore
    from megatoken.training import engine
    from megatoken.training.optimizer import MasterAdamW
    from megatoken.training.router_balance import RouterBalanceWindow

    original_dispatch = FacetedCore.dispatch_metrics
    original_step, original_set_lr = MasterAdamW.step, MasterAdamW.set_lr
    original_backward, original_build = RouterBalanceWindow.backward, engine.build_model

    def step(self, max_norm: float = 1.0):
        monitor.on_step(self)
        return original_step(self, max_norm)

    def set_lr(self, lr: float):
        applied = lr if lr_fn is None else lr_fn(monitor.steps)
        monitor.lr_history.append(applied)
        return original_set_lr(self, applied)

    def backward(self, distributed, *, routers, coefficient):
        monitor.on_router_window(self, routers)
        return original_backward(self, distributed, routers=routers, coefficient=coefficient)

    def dispatch(self, *, reset: bool = False):
        if reset and monitor.banks:
            monitor.on_dispatch_reset()
        return original_dispatch(self, reset=reset)

    def build(*a, **k):
        model = original_build(*a, **k)
        monitor.attach(model)
        return model

    MasterAdamW.step, MasterAdamW.set_lr = step, set_lr
    FacetedCore.dispatch_metrics = dispatch
    RouterBalanceWindow.backward = backward
    if capture_model:
        engine.build_model = build
    try:
        yield
    finally:
        MasterAdamW.step, MasterAdamW.set_lr = original_step, original_set_lr
        FacetedCore.dispatch_metrics = original_dispatch
        RouterBalanceWindow.backward = original_backward
        engine.build_model = original_build


@contextlib.contextmanager
def gpu_lock(device: str):
    if device != "cuda":
        yield
        return
    with open(GPU_LOCK, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def train_full(args, tau: float, cap: int, train_docs: list[Doc], work: Path, monitor: RunMonitor):
    """The real B0 trainer. Returns (model, summary dict)."""
    from megatoken.contracts import DEFAULT_LOSS_WEIGHTS
    from megatoken.training.config import TrainConfig
    from megatoken.training.engine import train

    dataset, statuses = prepare_training_data(work / "prepared", train_docs, cap)
    weights = {**DEFAULT_LOSS_WEIGHTS, **parse_loss_weights(args.loss_weight)}
    window_records = 8
    config = TrainConfig(
        purpose="diagnostic",
        phase="B0",
        datasets=[str(dataset)],
        output=str(work / "run"),
        model=args.model,
        model_overrides=model_overrides(args, tau),
        codec="byte",
        allow_byte_codec=True,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
        max_steps=args.steps,
        microbatch_episodes=args.windows_per_step,
        accumulate_scored=1,
        accumulate_ingested=10**9,
        max_accumulation_windows=1,
        generated_capacity=min(cap, 128 if args.model == "tiny" else 1024),
        max_history_tokens=max(256, cap * window_records),
        max_window_records=window_records,
        scored_decisions=args.scored_decisions,
        max_record_bytes=cap,
        checkpoint_every=0,
        eval_every=0,
        keep_checkpoints=1,
        effort=args.effort,
        gradient_checkpointing=args.gradient_checkpointing,
        use_stage_mixture=False,
        use_effort_mixture=False,
        loss_weights=weights,
        grad_clip=args.grad_clip,
        log_every=1,
    )
    lr_fn = lambda step: lr_at(step, args.steps, args.lr, args.warmup_fraction)  # noqa: E731
    started = time.time()
    sink = io.StringIO()
    with instrument(monitor, lr_fn=lr_fn, capture_model=True), contextlib.redirect_stdout(sink):
        result = train(config)
    curve = read_training_curve(work / "run" / "metrics.jsonl")
    for row in curve:  # the trainer logs its own schedule value; report what was applied
        if 0 < row["step"] <= len(monitor.lr_history):
            row["lr"] = monitor.lr_history[row["step"] - 1]
    return monitor.model, {
        "train_seconds": time.time() - started,
        "optimizer_steps": result["global_step"],
        "ingested_bytes": result["counters"]["I"],
        "training_docs": statuses,
        "curve": curve,
    }


def read_training_curve(path: Path) -> list[dict[str, float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("event") == "train":
            loss = record.get("loss", {})
            rows.append(
                {
                    "step": record["global_step"],
                    "lr": record.get("lr"),
                    **{k: v for k, v in loss.items() if k in LOGGED_LOSSES},
                }
            )
    return rows


LOGGED_LOSSES = ("lexical", "reconstruction", "facet_prediction", "structure", "router_balance")


def parse_loss_weights(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"--loss-weight expects NAME=VALUE, got {item!r}")
        out[key] = float(value)
    return out


class _SingleProcess:
    """The one thing the router replay needs from a distributed context."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def sum_tensors(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.detach().to(self.device, torch.float64) for k, v in values.items()}


def train_codec(
    args, tau: float, cap: int, train_chunks: list[bytes], monitor: RunMonitor
) -> tuple[Any, dict]:
    """Encoder + decoder only: noisy reconstruction through R, batched (no C, no P)."""
    from megatoken.contracts import DEFAULT_LOSS_WEIGHTS, ROLE_TO_ID
    from megatoken.generation.codec import ByteCodec
    from megatoken.loading import build_model, load_model_config
    from megatoken.model.multitrack import _GroupedTeacherReadout
    from megatoken.tensors import byte_tensor
    from megatoken.training.optimizer import MasterAdamW
    from megatoken.training.router_balance import RouterBalanceWindow, balance_routers

    device = torch.device(args.device)
    config = load_model_config(args.model, **model_overrides(args, tau))
    model = build_model(config, ByteCodec.byte_fallback(), device=device, precision=args.precision)
    weights = {**DEFAULT_LOSS_WEIGHTS, **parse_loss_weights(args.loss_weight)}
    monitor.attach(model)
    model.train()
    core = model.core
    readout = _GroupedTeacherReadout(core, model.decoder, model.media_realizers)
    routers = balance_routers(model)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    context = _SingleProcess(device)
    roles = ROLE_TO_ID["document"]
    started = time.time()
    curve: list[dict[str, float]] = []
    with instrument(monitor):
        optimizer = MasterAdamW(model, args.lr * 0.1, weight_decay=0.1)
        for step in range(args.steps):
            lr = lr_at(step, args.steps, args.lr, args.warmup_fraction)
            optimizer.set_lr(lr)
            optimizer.zero_grad()
            payloads = [rng.choice(train_chunks) for _ in range(args.codec_batch)]
            window = RouterBalanceWindow()
            for router in routers.values():
                router.balance_recorder = window
            try:
                core.routing_aux_loss(clear=True)
                values, lengths = byte_tensor(payloads, device)
                track_ids = torch.zeros(len(payloads), dtype=torch.long, device=device)
                encoded = core.encode_chunk(values, lengths, track_ids=track_ids)
                noise = torch.randn(encoded.shape, device=device, generator=generator)
                sample = encoded.float() + tau * noise
                plan = model.prepare_target(
                    target_bytes=tuple(payloads),
                    representation_mode="bootstrap",
                    role_ids=[roles] * len(payloads),
                )
                terms = readout.losses_for_selected(
                    sample[:, None], (plan,), None, None, track_ids[:, None]
                )
                _, z_loss = core.routing_aux_loss(clear=True)
                loss = z_loss * weights["router_z"]
                logged: dict[str, float] = {}
                for name in ("lexical", "structure", "binding", "placement", "terminal"):
                    total, count = terms[name]
                    if count:
                        key = "reconstruction" if name == "lexical" else name
                        term = total / count
                        loss = loss + weights[key] * term
                        logged[key] = float(term.detach())
                loss.backward()
                window.backward(context, routers=routers, coefficient=weights["router_balance"])
            finally:
                for router in routers.values():
                    router.balance_recorder = None
            optimizer.step(args.grad_clip)
            curve.append({"step": step + 1, "lr": lr, **logged})
        optimizer.close()
    return model, {
        "train_seconds": time.time() - started,
        "optimizer_steps": args.steps,
        "ingested_bytes": int(
            args.steps * args.codec_batch * np.mean([len(c) for c in train_chunks])
        ),
        "curve": curve,
    }


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def length_bucket(n: int) -> str:
    low = 0
    for edge in LENGTH_EDGES:
        if n <= edge:
            return f"{low + 1}-{edge}"
        low = edge
    return f">{LENGTH_EDGES[-1]}"


class Scores:
    """Exact-match, byte accuracy and similarity, overall and by key."""

    def __init__(self) -> None:
        self.groups: dict[tuple[str, str], list[float]] = {}

    def add(self, kind: str, original: bytes, decoded: bytes) -> None:
        correct = sum(1 for a, b in zip(original, decoded, strict=False) if a == b)
        values = [
            1.0,
            float(original == decoded),
            correct / max(1, len(original)),
            float(len(original) == len(decoded)),
            difflib.SequenceMatcher(None, original, decoded, autojunk=False).ratio()
            if original != decoded
            else 1.0,
        ]
        for key in (("all", ""), ("type", kind), ("length", length_bucket(len(original)))):
            bucket = self.groups.setdefault(key, [0.0] * len(values))
            for index, value in enumerate(values):
                bucket[index] += value

    @staticmethod
    def _summary(v: list[float]) -> dict[str, float]:
        n = v[0]
        return {
            "chunks": int(n),
            "exact_match_rate": v[1] / n,
            "byte_accuracy": v[2] / n,
            "length_match_rate": v[3] / n,
            "similarity": v[4] / n,
        }

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self._summary(self.groups[("all", "")]))
        out["by_type"] = {
            k: self._summary(v) for (g, k), v in sorted(self.groups.items()) if g == "type"
        }
        out["by_length"] = {
            k: self._summary(v)
            for (g, k), v in sorted(
                ((key, val) for key, val in self.groups.items() if key[0] == "length"),
                key=lambda item: _bucket_order(item[0][1]),
            )
        }
        return out


def _bucket_order(label: str) -> int:
    return 10**6 if label.startswith(">") else int(label.split("-")[0])


def decode_latent(mt, model, raw: torch.Tensor, limit: int, device) -> tuple[bytes, bool]:
    """Serving-path decode of one latent block [1, F, D]; (bytes, ok)."""
    from megatoken.generation.structure import ActionSpec

    spec = ActionSpec(max_tokens=limit)
    track_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    try:
        payloads, *_ = mt._realize(raw[:, None], (spec,), None, mixture=None, track_ids=track_ids)
    except ValueError:
        return b"", False
    return payloads[0], True


@torch.no_grad()
def encode_payloads(model, payloads: list[bytes], device, batch: int = 32) -> torch.Tensor:
    """Encoder output for each chunk under the document track, [N, F, D] float32 on CPU."""
    from megatoken.tensors import byte_tensor

    outputs = []
    for start in range(0, len(payloads), batch):
        part = payloads[start : start + batch]
        values, lengths = byte_tensor(part, device)
        ids = torch.zeros(len(part), dtype=torch.long, device=device)
        outputs.append(model.core.encode_chunk(values, lengths, track_ids=ids).float().cpu())
    return torch.cat(outputs)


@torch.no_grad()
def evaluate_round_trip(
    model,
    chunks: list[tuple[str, bytes]],
    serving: tuple[float, str],
    args,
    device,
    baseline_byte: int,
    tau: float,
) -> dict[str, Any]:
    from megatoken.model.multitrack import MultiTrackGenerator

    model.eval()
    mt = MultiTrackGenerator(model)
    limit = model.config.max_capacity
    latents = encode_payloads(model, [p for _, p in chunks], device)
    generator = torch.Generator().manual_seed(args.seed + 7)
    noise = torch.randn((args.noise_draws, *latents.shape), generator=generator)
    centred = latents - latents.mean(0, keepdim=True)
    rms = float(centred.square().mean().sqrt())
    levels = make_levels(args, tau, serving[0], serving[1], rms)
    total_bytes = sum(len(p) for _, p in chunks)
    baseline = sum(p.count(bytes([baseline_byte])) for _, p in chunks) / max(1, total_bytes)
    by_noise = []
    for sigma, tag in levels:
        scores, failures = Scores(), 0
        for draw in range(args.noise_draws):
            for index, (kind, payload) in enumerate(chunks):
                z = (latents[index : index + 1] + sigma * noise[draw, index : index + 1]).to(device)
                decoded, ok = decode_latent(mt, model, z, limit, device)
                failures += int(not ok)
                scores.add(kind, payload, decoded)
        by_noise.append(
            {"sigma": sigma, "tag": tag, "decode_failures": failures, **scores.report()}
        )
    return {
        "chunks": len(chunks),
        "latent_rms": rms,
        "snr_vs_training_noise": rms / tau,
        "baseline_byte_accuracy": baseline,
        "by_noise": by_noise,
    }


@torch.no_grad()
def evaluate_predictor(
    model, docs: list[tuple[str, list[bytes]]], args, device, tau: float
) -> dict[str, Any]:
    """Held-out next-chunk latent prediction against two trivial predictors."""
    from megatoken.model.core import ROLE_IDS
    from megatoken.model.expert_pools import expert_scope
    from megatoken.model.multitrack import MultiTrackGenerator
    from megatoken.tensors import byte_tensor

    model.eval()
    mt, core, names = MultiTrackGenerator(model), model.core, ("doc",)
    limit = model.config.max_capacity
    rows: list[dict[str, Any]] = []
    started = time.time()
    for d, (kind, chunks) in enumerate(docs):
        state = mt.tracks.initial_state(1, device=device)
        for i, payload in enumerate(chunks):
            ids = (f"held{d}:{i}",)
            control = mt._controls((b"",), state, names)
            mean = mt.tracks.predict_group(
                state,
                tracks=names,
                event_ids=ids,
                control=control,
                effort=args.effort,
                modalities=("text",),
            )
            prior = mt._prior(mean, state, names, control)
            track_ids, _ = mt.tracks._track_indices(state, names)
            semantics = mt.tracks.description_controls(state, names)
            plan = core.adapters.plan("e", state.adapters, track_ids, semantics)
            weights = None if plan is None else plan.weights[0, :, 0]
            values, lengths = byte_tensor([payload], device)
            with expert_scope(weights):
                encoded = core.encode_chunk(
                    values,
                    lengths,
                    track_ids=track_ids[:, 0],
                    track_semantics=None if semantics is None else semantics[:, 0],
                )
            guess, _ = decode_latent(mt, model, mean[:, 0].float(), limit, device)
            state = mt.tracks.advance_group(
                state,
                encoded[:, None],
                track_semantics=control,
                tracks=names,
                event_ids=ids,
                role_ids=torch.tensor([[ROLE_IDS["document"]]], device=device),
                modalities=("text",),
            )
            rows.append(
                {
                    "kind": kind,
                    "doc": d,
                    "pos": i,
                    "payload": payload,
                    "guess": guess,
                    "e": encoded.float().flatten().cpu().numpy(),
                    "mu": mean.float().flatten().cpu().numpy(),
                    "log_sigma": prior.log_scale.float().flatten().cpu().numpy(),
                }
            )
    return predictor_report(rows, tau, time.time() - started)


def predictor_report(rows: list[dict[str, Any]], tau: float, seconds: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("no held-out chunks for the predictor pass")
    everything = np.stack([r["e"] for r in rows])
    constant = everything.mean(0)
    per_kind: dict[str, dict[str, list[float]]] = {}
    generation = Scores()
    sigma_values = []
    running, previous = None, None
    for r in rows:
        sigma = np.exp(r["log_sigma"])
        sigma_values.append(float(sigma.mean()))
        if r["pos"] == 0:
            running, count, previous = r["e"].copy(), 1, r["e"]
            continue
        target = r["e"]
        guesses = {
            "model": r["mu"],
            "repeat": previous,
            "running_mean": running / count,
            "constant": constant,
            "zero": np.zeros_like(target),
        }
        entry = per_kind.setdefault(r["kind"], {})
        for name, guess in guesses.items():
            err = (target - guess) ** 2
            entry.setdefault("mse_" + name, []).append(float(err.mean()))
            # KL(N(e, tau^2) || N(guess, sigma^2)) per coordinate, with the model's own sigma
            kl = np.log(sigma / tau) + (tau**2 + err) / (2 * sigma**2) - 0.5
            entry.setdefault("kl_" + name, []).append(float(kl.mean()))
        generation.add(r["kind"], r["payload"], r["guess"])
        running, count, previous = running + target, count + 1, target

    def summarize(entries: dict[str, list[float]]) -> dict[str, Any]:
        mean = {k: float(np.mean(v)) for k, v in entries.items()}
        out = {"chunks": len(entries["mse_model"]), **mean}
        for name in ("repeat", "running_mean", "constant", "zero"):
            out[f"ratio_vs_{name}"] = mean["mse_model"] / max(mean["mse_" + name], 1e-30)
        return out

    pooled: dict[str, list[float]] = {}
    for entry in per_kind.values():
        for key, values in entry.items():
            pooled.setdefault(key, []).extend(values)
    if not pooled:
        raise ValueError("every held-out document had a single chunk; no next-chunk targets")
    return {
        **summarize(pooled),
        "kl_model_per_coordinate": float(np.mean(pooled["kl_model"])),
        "by_type": {k: summarize(v) for k, v in sorted(per_kind.items())},
        "prior_sigma_mean": float(np.mean(sigma_values)),
        "generation_from_prediction": generation.report(),
        "documents": len({r["doc"] for r in rows}),
        "eval_seconds": seconds,
    }


# --------------------------------------------------------------------------------------
# Expert health
# --------------------------------------------------------------------------------------


def usage_stats(counts: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=np.float64)
    n, total = len(counts), float(counts.sum())
    if total <= 0 or n < 1:
        return {
            "experts": n,
            "assignments": 0,
            "normalized_entropy": 0.0,
            "frac_below_1pct_of_avg": 1.0,
            "used_experts": 0,
            "max_share": 0.0,
            "min_share": 0.0,
            "exercised": False,
        }
    share = counts / total
    entropy = float(-(share[share > 0] * np.log(share[share > 0])).sum())
    return {
        "experts": n,
        "assignments": int(total),
        "normalized_entropy": entropy / math.log(n) if n > 1 else 1.0,
        "frac_below_1pct_of_avg": float((counts < 0.01 * counts.mean()).mean()),
        "used_experts": int((counts > 0).sum()),
        "max_share": float(share.max()),
        "min_share": float(share.min()),
        "exercised": True,
    }


def categorize_unused(name: str, monitor: RunMonitor, mode: str, short_chunks: bool) -> str:
    """Why a parameter without gradient is, or is not, a surprise."""
    if name.startswith(BY_DESIGN_UNUSED):
        return "by_design_for_text_bootstrap"
    if short_chunks and name.startswith("core.frontend.reduction"):
        return "by_design_for_text_bootstrap"  # chunk fits one readout level: no halving needed
    if name.startswith("transducer."):
        return "memory_path_needs_lookups"
    label = monitor.param_bank.get(name)
    if label is None and name.startswith("core.expert_pools.pool_codes."):
        label = "pool:" + name.rsplit(".", 1)[1].removesuffix("__codes")
    if label is not None and monitor.bank_usage(label).sum() == 0:
        return "pool_never_routed_to"
    if mode == "codec" and (
        name.startswith(OUTSIDE_CODEC_PATH) or ("adapter_" in name and label is not None)
    ):
        return "outside_codec_only_path"
    return "unexpected"


def expert_health(monitor: RunMonitor, mode: str, short_chunks: bool) -> dict[str, Any]:
    banks = {}
    for label in monitor.banks:
        stats = usage_stats(monitor.bank_usage(label))
        dead: dict[str, list[int]] = {}
        for name, owner in monitor.bank_of_param.items():
            if owner == label and name in monitor.expert_grad_max:
                zero = np.flatnonzero(monitor.expert_grad_max[name] == 0.0).tolist()
                if zero:
                    dead[name] = zero
        banks[label] = {**stats, "dead_experts": dead}
    routers = {n: {**usage_stats(c)} for n, c in sorted(monitor.router_counts.items())}
    no_grad = sorted(n for n, k in monitor.grad_present_steps.items() if k == 0)
    zero_grad = sorted(
        n for n, k in monitor.grad_present_steps.items() if k > 0 and monitor.grad_max[n] == 0.0
    )

    def split(names: list[str]) -> dict[str, Any]:
        categories: dict[str, list[str]] = {}
        for name in names:
            categories.setdefault(categorize_unused(name, monitor, mode, short_chunks), []).append(
                name
            )
        return {
            "count": len(names),
            "unexpected_count": len(categories.get("unexpected", [])),
            "by_category": categories,
        }

    return {
        "banks": banks,
        "routers": routers,
        "trainable_parameters": len(monitor.grad_max),
        "zero_gradient": {
            "no_gradient_at_all": split(no_grad),
            "exactly_zero_gradient": split(zero_grad),
        },
        "optimizer_steps_observed": monitor.steps,
    }


# --------------------------------------------------------------------------------------
# Assessment, orchestration, CLI
# --------------------------------------------------------------------------------------


def assess(result: dict[str, Any], tau: float) -> dict[str, str]:
    """Mechanical application of the guessed thresholds in the module docstring."""
    out: dict[str, str] = {}
    rt = result["round_trip"]
    at_tau = next((x for x in rt["by_noise"] if x["tag"] and "training" in x["tag"]), None)
    if at_tau:
        exact, acc = at_tau["exact_match_rate"], at_tau["byte_accuracy"]
        out["round_trip"] = (
            "healthy" if exact >= 0.5 and acc >= 0.9 else "weak" if exact >= 0.1 else "unhealthy"
        )
        clean = next((x for x in rt["by_noise"] if x["sigma"] == 0.0), None)
        if clean and clean["exact_match_rate"] > 0.05:
            out["noise_robustness"] = (
                "healthy" if exact >= 0.5 * clean["exact_match_rate"] else "fragile"
            )
        out["above_trivial_bytes"] = "yes" if acc > rt["baseline_byte_accuracy"] + 0.05 else "no"
    if "predictor" in result:
        p = result["predictor"]
        a, b = p["ratio_vs_repeat"], p["ratio_vs_running_mean"]
        out["predictor"] = (
            "healthy"
            if a <= 0.9 and b <= 0.95
            else "dead_or_untrained"
            if max(a, b) >= 1.0
            else "marginal"
        )
    health = result["expert_health"]
    flagged = []
    for scope in ("banks", "routers"):
        for name, s in health[scope].items():
            if s["exercised"] and (
                s["normalized_entropy"] < 0.5 or s["frac_below_1pct_of_avg"] > 0.3
            ):
                flagged.append(name)
    out["experts"] = "healthy" if not flagged else "uneven: " + ", ".join(sorted(flagged)[:6])
    zg = health["zero_gradient"]
    surprise = zg["no_gradient_at_all"]["unexpected_count"]
    surprise += zg["exactly_zero_gradient"]["unexpected_count"]
    unrouted = {
        n.split(".")[3]
        for n in zg["no_gradient_at_all"]["by_category"].get("pool_never_routed_to", [])
        if n.startswith("core.expert_pools.pools.")
    }
    if unrouted:
        out["unrouted_pools"] = f"{len(unrouted)} shared pool(s) never routed to"
    out["gradients"] = "all trainable parameters live" if not surprise else f"{surprise} unexpected"
    return out


def make_levels(
    args, tau: float, serving_sigma: float, serving_note: str, rms: float
) -> list[tuple[float, str]]:
    """Noise std values to test: the CLI list, tau, the serving std, and fractions of latent size."""
    levels: dict[float, list[str]] = {}
    for sigma in [0.0, *args.noise_levels]:
        levels.setdefault(round(sigma, 6), [])
    levels.setdefault(round(tau, 6), []).append("training_tau")
    levels.setdefault(round(serving_sigma, 6), []).append(serving_note)
    for fraction in args.noise_rms_fractions:
        levels.setdefault(round(fraction * rms, 6), []).append(f"{fraction:g}x_latent_rms")
    return [(sigma, "+".join(tags)) for sigma, tags in sorted(levels.items())]


def run_one(args, cap: int, tau: float, corpus, work: Path) -> dict[str, Any]:
    train_docs, held_docs = corpus
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    monitor = RunMonitor()
    rng = random.Random(args.seed + 11)
    held_chunks = [(d.kind, c) for d in held_docs for c in chunk_document(d, cap)]
    per_kind = max(1, args.roundtrip_chunks // len(KINDS))
    selected: list[tuple[str, bytes]] = []
    for kind in KINDS:
        pool = [x for x in held_chunks if x[0] == kind]
        rng.shuffle(pool)
        selected += pool[:per_kind]
    if not selected:
        raise SystemExit("no held-out chunks produced; give more or longer documents")
    train_chunks = [c for d in train_docs for c in chunk_document(d, cap)]
    if args.train_mode == "full":
        model, training = train_full(args, tau, cap, train_docs, work, monitor)
    else:
        model, training = train_codec(args, tau, cap, train_chunks, monitor)
    health = expert_health(
        monitor,
        args.train_mode,
        short_chunks=cap <= model.config.n_facets * model.config.source_stride,
    )  # before evaluation touches the usage counters
    counts = Counter(b for c in train_chunks for b in c)
    baseline_byte = counts.most_common(1)[0][0]
    result: dict[str, Any] = {
        "chunk_cap": cap,
        "training_noise_std": tau,
        "training": {
            **training,
            "mode": args.train_mode,
            "train_chunks": len(train_chunks),
            "heldout_chunks": len(held_chunks),
            "mean_chunk_bytes": float(np.mean([len(c) for c in train_chunks])),
        },
    }
    serving, note = float(model.config.prior_noise_std), "serving_prior_std(untrained_default)"
    if args.train_mode == "full":
        seqs: list[tuple[str, list[bytes]]] = []
        for kind in KINDS:
            docs = [d for d in held_docs if d.kind == kind][
                : max(1, args.predictor_docs // len(KINDS))
            ]
            seqs += [(kind, chunk_document(d, cap)[: args.max_chunks_per_doc]) for d in docs]
        seqs = [s for s in seqs if len(s[1]) >= 2]
        result["predictor"] = evaluate_predictor(model, seqs, args, device, tau)
        serving, note = result["predictor"]["prior_sigma_mean"], "serving_prior_std"
    else:
        result["predictor"] = None
        result["predictor_note"] = "skipped: codec-only training does not train the predictor"
    result["round_trip"] = evaluate_round_trip(
        model, selected, (serving, note), args, device, baseline_byte, tau
    )
    result["expert_health"] = health
    if args.train_mode == "codec":
        health["note"] = "codec-only mode exercises just the encoder and decoder paths"
    result["assessment"] = assess({k: v for k, v in result.items() if v is not None}, tau)
    return result


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--smoke", action="store_true", help="tiny, fast configuration (CPU, < 90 s)")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", type=Path, help="write the full result as JSON here")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--train-mode", choices=("full", "codec"), default="full")
    p.add_argument("--model", default="tiny", help="tiny, 32x256 or 32x512")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="model config override, e.g. codec_noise_std=0.05",
    )
    p.add_argument("--loss-weight", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument(
        "--steps", type=int, default=None, help="optimizer steps (default codec 2000, full 200)"
    )
    p.add_argument("--lr", type=float, default=None, help="peak LR (default 5e-3 tiny, else 2e-4)")
    p.add_argument("--warmup-fraction", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--effort", type=int, choices=(1, 2, 3, 6), default=None)
    p.add_argument("--windows-per-step", type=int, default=2, help="full mode: windows per step")
    p.add_argument(
        "--scored-decisions", type=int, default=4, help="full mode: scored chunks/window"
    )
    p.add_argument("--codec-batch", type=int, default=16, help="codec mode: chunks per step")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument(
        "--chunk-caps", default=None, help="comma list of max chunk bytes (default 48; smoke 32)"
    )
    p.add_argument("--train-noise", default=None, help="comma list of training noise std values")
    p.add_argument("--noise-levels", default="0.1,0.2,0.4", help="evaluation noise std list")
    p.add_argument(
        "--noise-rms-fractions",
        default="0.1,0.3,1.0",
        help="also test noise std equal to these fractions of the typical latent size",
    )
    p.add_argument("--noise-draws", type=int, default=1)
    p.add_argument("--text-dir", type=Path, default=None)
    p.add_argument("--docs-per-type", type=int, default=40)
    p.add_argument("--heldout-docs-per-type", type=int, default=8)
    p.add_argument("--doc-bytes", type=int, default=1200)
    p.add_argument("--heldout-fraction", type=float, default=0.2, help="for --text-dir")
    p.add_argument("--roundtrip-chunks", type=int, default=150)
    p.add_argument("--predictor-docs", type=int, default=12)
    p.add_argument("--max-chunks-per-doc", type=int, default=24)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--keep-work", action="store_true")
    return p


def apply_defaults(args: argparse.Namespace) -> None:
    if args.smoke:
        args.steps, args.windows_per_step, args.scored_decisions = 2, 1, 2
        args.docs_per_type, args.heldout_docs_per_type = 3, 2
        args.doc_bytes, args.roundtrip_chunks, args.predictor_docs = 300, 6, 3
        args.max_chunks_per_doc, args.noise_levels, args.codec_batch = 4, "0.1", 8
        args.noise_rms_fractions = "0.3"
    if args.steps is None:
        args.steps = 2000 if args.train_mode == "codec" else 200
    args.lr = args.lr if args.lr is not None else (5e-3 if args.model == "tiny" else 2e-4)
    args.effort = args.effort if args.effort is not None else (1 if args.model == "tiny" else 6)
    default_cap = "32" if args.smoke else "48"
    args.chunk_caps = [int(x) for x in str(args.chunk_caps or default_cap).split(",")]
    args.noise_levels = parse_floats(args.noise_levels)
    args.noise_rms_fractions = parse_floats(args.noise_rms_fractions)
    args.train_noise = parse_floats(args.train_noise) if args.train_noise else None


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summary_lines(runs: dict[str, Any]) -> list[str]:
    lines = []
    for name, run in runs.items():
        t = run["training"]
        lines.append(
            f"== {name}: {t['mode']} mode, {t['optimizer_steps']} steps, "
            f"{t['train_seconds']:.0f}s, last losses "
            f"{ {k: round(v, 3) for k, v in (t['curve'][-1] if t['curve'] else {}).items() if k not in ('step', 'lr')} }"
        )
        rt = run["round_trip"]
        lines.append(
            f"   round trip ({rt['chunks']} held-out chunks; latent SNR "
            f"{rt['snr_vs_training_noise']:.1f}; most-common-byte baseline "
            f"{rt['baseline_byte_accuracy']:.2f}):"
        )
        for level in rt["by_noise"]:
            lines.append(
                f"     sigma {level['sigma']:<7.4g} exact {level['exact_match_rate']:.3f}  "
                f"byte-acc {level['byte_accuracy']:.3f}  similarity "
                f"{level['similarity']:.3f}  {level['tag']}"
            )
        if run["predictor"]:
            p = run["predictor"]
            lines.append(
                f"   predictor: mse model {p['mse_model']:.4g} | repeat {p['mse_repeat']:.4g}"
                f" | running mean {p['mse_running_mean']:.4g} -> ratios "
                f"{p['ratio_vs_repeat']:.3f} / {p['ratio_vs_running_mean']:.3f} "
                f"(vs constant {p['ratio_vs_constant']:.3f})"
            )
        h = run["expert_health"]
        for scope in ("banks", "routers"):
            worst = sorted(
                ((s["normalized_entropy"], n) for n, s in h[scope].items() if s["exercised"])
            )[:3]
            lines.append(
                f"   {scope}: lowest normalized entropy "
                + ", ".join(f"{n} {e:.2f}" for e, n in worst)
            )
        zg = h["zero_gradient"]
        lines.append(
            "   zero-gradient params: "
            f"{zg['no_gradient_at_all']['count']} none-at-all "
            f"({zg['no_gradient_at_all']['unexpected_count']} unexpected), "
            f"{zg['exactly_zero_gradient']['count']} exactly-zero "
            f"({zg['exactly_zero_gradient']['unexpected_count']} unexpected) of "
            f"{h['trainable_parameters']}"
        )
        lines.append(f"   assessment: {run['assessment']}")
    return lines


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    apply_defaults(args)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        return run(args)
    finally:
        torch.set_num_threads(previous_threads)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model == "tiny" and max(args.chunk_caps) > 128:
        raise SystemExit("the tiny preset decodes at most 128 bytes per chunk")
    if args.text_dir:
        corpus = directory_corpus(args.text_dir, args.seed, args.heldout_fraction, 500, 1 << 16)
    else:
        corpus = (
            builtin_corpus(args.seed, args.docs_per_type, args.doc_bytes, offset=0),
            builtin_corpus(args.seed, args.heldout_docs_per_type, args.doc_bytes, offset=1),
        )
    taus = args.train_noise or [None]
    own_work = args.work_dir is None
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="probe-bootstrap-"))
    runs: dict[str, Any] = {}
    started = time.time()
    try:
        for cap in args.chunk_caps:
            for tau in taus:
                from megatoken.loading import load_model_config

                resolved = (
                    tau
                    if tau is not None
                    else float(
                        load_model_config(
                            args.model,
                            **{
                                k: v
                                for k, v in model_overrides(args, 0.1).items()
                                if k != "codec_noise_std"
                            },
                        ).codec_noise_std
                    )
                )
                name = f"cap{cap}_tau{resolved:g}"
                work = work_root / name
                work.mkdir(parents=True, exist_ok=True)
                with gpu_lock(args.device):
                    runs[name] = run_one(args, cap, resolved, corpus, work)
    finally:
        if own_work and not args.keep_work:
            shutil.rmtree(work_root, ignore_errors=True)
    result = {
        "probe": "bootstrap",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "corpus": {
            "source": str(args.text_dir) if args.text_dir else "builtin",
            "train_docs": Counter(d.kind for d in corpus[0]),
            "heldout_docs": Counter(d.kind for d in corpus[1]),
        },
        "runs": runs,
        "total_seconds": time.time() - started,
    }
    result = clean(result)
    print("\n".join(summary_lines(result["runs"])))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1, allow_nan=False))
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
