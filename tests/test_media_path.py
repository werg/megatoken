"""The single media path: pinned teacher identity, hidden-state tap, one input
projection, text anchors, media-bridge training scope and name-keyed optimizer state."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from megatoken.data.media_assets import TensorAssetStore
from megatoken.generation.codec import ByteCodec
from megatoken.loading import build_model
from megatoken.model.config import ModelConfig
from megatoken.omni.embeddinggemma2 import EmbeddingGemma2
from megatoken.omni.loader import read_locked_manifest
from megatoken.teacher import (
    HIDDEN_TAP,
    HIDDEN_WIDTH,
    MAX_TOKENS,
    PINNED_TEACHER,
    PREPROCESSING,
    extractor_id,
    locked_teacher,
    validate_extractor,
)
from megatoken.training.config import TrainConfig
from megatoken.training.media_dependencies import (
    required_media_modalities,
    validate_media_realizers,
)
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.text_anchors import TextAnchors, anchor_loss
from tests.media_support import LATENT, features, install_fake_realizers, teacher_embedding

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def other_extractor() -> str:
    return f"embeddinggemma2:{'a' * 40}:{'b' * 64}:{HIDDEN_TAP}:{PREPROCESSING}"


# --- teacher identity -------------------------------------------------------------------


def test_extractor_identity_is_independent_of_where_the_snapshot_lives():
    files = {"teacher/config.json": sha(b"c"), "teacher/model.safetensors": sha(b"w")}
    moved = {"packs/elsewhere/teacher/config.json": sha(b"c")}
    moved["packs/elsewhere/teacher/model.safetensors"] = sha(b"w")
    a = extractor_id(REVISION, files, "teacher")
    assert a == extractor_id(REVISION, moved, "packs/elsewhere/teacher")
    assert a == extractor_id(REVISION, files, "teacher/")
    assert validate_extractor(a) == a


def test_extractor_identity_binds_revision_and_every_file():
    files = {"t/a": sha(b"a"), "t/b": sha(b"b")}
    base = extractor_id(REVISION, files, "t")
    assert extractor_id("0" * 40, files, "t") != base
    assert extractor_id(REVISION, {**files, "t/b": sha(b"B")}, "t") != base
    assert extractor_id(REVISION, {**files, "t/c": sha(b"c")}, "t") != base
    assert extractor_id(REVISION, {"t/a": files["t/a"], "t/renamed": files["t/b"]}, "t") != base
    assert extractor_id(REVISION, dict(reversed(files.items())), "t") == base


@pytest.mark.parametrize(
    ("revision", "files", "snapshot"),
    [
        ("main", {"t/a": "0" * 64}, "t"),
        (REVISION.upper(), {"t/a": "0" * 64}, "t"),
        (REVISION, {}, "t"),
        (REVISION, {"other/a": "0" * 64}, "t"),
        (REVISION, {"t/a": "0" * 64, "x/b": "0" * 64}, "t"),
        (REVISION, {"tt/a": "0" * 64}, "t"),
    ],
)
def test_extractor_identity_needs_a_revision_and_files_inside_the_snapshot(
    revision, files, snapshot
):
    with pytest.raises(ValueError, match="Teacher identity"):
        extractor_id(revision, files, snapshot)


def test_validate_extractor_accepts_only_the_pinned_family_of_ids():
    assert validate_extractor(PINNED_TEACHER) == PINNED_TEACHER
    assert validate_extractor(other_extractor())
    for bad in (
        "",
        None,
        "siglip2:" + "0" * 40,
        PINNED_TEACHER + ":extra",
        PINNED_TEACHER.replace(HIDDEN_TAP, "language_model.embedding_projection:output"),
        PINNED_TEACHER.replace(PREPROCESSING, "chat-fp16-v0"),
        PINNED_TEACHER.replace(REVISION, "main"),
        PINNED_TEACHER.upper(),
        "x" + PINNED_TEACHER,
    ):
        with pytest.raises(ValueError, match="pinned EmbeddingGemma2"):
            validate_extractor(bad)


def test_model_config_pins_the_teacher():
    assert ModelConfig.tiny().media_extractor == PINNED_TEACHER
    assert ModelConfig().media_extractor == PINNED_TEACHER
    with pytest.raises(ValueError, match="pinned EmbeddingGemma2"):
        ModelConfig.tiny(media_extractor="whistle-v3")
    # A different (still well-formed) teacher is a different model geometry.
    other = ModelConfig.tiny(media_extractor=other_extractor())
    assert other.state_geometry != ModelConfig.tiny().state_geometry


def write_pack(root, *, tamper=None):
    """A locked pack manifest whose teacher snapshot lives under ``root/teacher``."""
    snapshot = root / "teacher"
    snapshot.mkdir(parents=True)
    contents = {
        "config.json": b'{"model_type": "embeddinggemma2"}',
        "model.safetensors": b"\x00weights\x01",
    }
    for name, data in contents.items():
        (snapshot / name).write_bytes(data)
    files = {f"teacher/{name}": sha(data) for name, data in contents.items()}
    manifest = {
        "version": 1,
        "packs": {"embeddinggemma2": {"revision": REVISION, "snapshot": "teacher", "files": files}},
    }
    if tamper:
        tamper(manifest)
    (root / "pack-lock.json").write_text(json.dumps(manifest))
    return root / "pack-lock.json"


def test_locked_teacher_identity_comes_from_the_verified_manifest(tmp_path):
    first = write_pack(tmp_path / "machine-a")
    second = write_pack(tmp_path / "deep" / "machine-b")
    base, packs = read_locked_manifest(str(first))
    snapshot, identity = locked_teacher(base, packs)
    assert snapshot == base / "teacher" and validate_extractor(identity) == identity
    base_b, packs_b = read_locked_manifest(str(second))
    assert locked_teacher(base_b, packs_b)[1] == identity  # layout independent
    assert identity != PINNED_TEACHER  # these are test bytes, not the real checkpoint


def test_locked_teacher_requires_the_teacher_and_verified_bytes(tmp_path):
    manifest = write_pack(tmp_path / "a")
    base, packs = read_locked_manifest(str(manifest))
    with pytest.raises(ValueError, match="does not lock"):
        locked_teacher(base, {})
    (tmp_path / "a" / "teacher" / "config.json").write_bytes(b"swapped")
    with pytest.raises(ValueError, match="checksum"):
        read_locked_manifest(str(manifest))
    (tmp_path / "a" / "teacher" / "config.json").write_bytes(b'{"model_type": "embeddinggemma2"}')
    (tmp_path / "a" / "teacher" / "extra.bin").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="Unhashed"):
        read_locked_manifest(str(manifest))


def test_hub_download_bookkeeping_never_binds_the_identity(tmp_path):
    manifest = write_pack(tmp_path / "a")
    cache = tmp_path / "a" / "teacher" / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "config.json.metadata").write_text("etag per machine")
    base, packs = read_locked_manifest(str(manifest))  # unlisted .cache entries are tolerated
    clean = locked_teacher(base, packs)[1]
    assert clean == locked_teacher(*read_locked_manifest(str(write_pack(tmp_path / "b"))))[1]
    locked = {**packs["embeddinggemma2"]["files"], "teacher/.cache/x": "0" * 64}
    with pytest.raises(ValueError, match="model file hashes"):
        extractor_id(REVISION, locked, "teacher")


def test_manifest_accepts_only_realizer_and_teacher_packs(tmp_path):
    def rename(manifest):
        manifest["packs"]["whistle"] = manifest["packs"].pop("embeddinggemma2")

    with pytest.raises(ValueError, match="Supported packs"):
        read_locked_manifest(str(write_pack(tmp_path / "a", tamper=rename)))


def test_install_realizers_registers_every_locked_realizer_but_not_the_teacher(
    tmp_path, monkeypatch
):
    from megatoken.omni.loader import install_realizers
    from megatoken.omni.realizers import SanaRealizer
    from tests.media_support import FakeRealizer

    manifest = write_pack(tmp_path)
    sana = tmp_path / "sana"
    sana.mkdir()
    (sana / "model_index.json").write_text("{}")
    lock = json.loads(manifest.read_text())
    files = {"sana/model_index.json": sha(b"{}")}
    lock["packs"]["sana"] = {"revision": "b" * 40, "snapshot": "sana", "files": files}
    manifest.write_text(json.dumps(lock))
    built = []

    def fake(**kwargs):
        built.append(kwargs)
        return FakeRealizer("image", kwargs["facet_count"], kwargs["facet_width"])

    monkeypatch.setattr(SanaRealizer, "from_checkpoint", staticmethod(fake))
    model = tiny_model()
    install_realizers(model, str(manifest), device="cpu")
    assert list(model.media_realizers.packs) == ["sana"]
    config = model.config
    assert (built[0]["facet_count"], built[0]["facet_width"]) == (config.n_facets, config.facet_dim)
    expected = sha(
        json.dumps(
            {"revision": "b" * 40, "files": files}, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    assert model.media_realizers.packs["sana"].identity == f"sana:{expected}"


# --- the hidden-state tap ---------------------------------------------------------------


class FakeLanguageModel(nn.Module):
    def __init__(self, out_features=768):
        super().__init__()
        self.embedding_projection = nn.Linear(HIDDEN_WIDTH, out_features)


class FakeTeacher(nn.Module):
    """Calls its projection positionally or by keyword, like the real wrapper's variants."""

    def __init__(self, *, keyword=False, calls=1, out_features=768):
        super().__init__()
        self.embed = nn.Embedding(32, HIDDEN_WIDTH)
        self.language_model = FakeLanguageModel(out_features)
        self.keyword, self.calls = keyword, calls
        self.received = None

    def forward(self, input_ids, attention_mask):
        self.received = (input_ids, attention_mask)
        hidden = self.embed(input_ids)
        for _ in range(self.calls):
            projection = self.language_model.embedding_projection
            out = projection(input=hidden) if self.keyword else projection(hidden)
        return SimpleNamespace(last_hidden_state=out)


class FakeProcessor:
    """Right-pads a batch of conversations to ``tokens`` (or to ``lengths`` per row)."""

    def __init__(self, tokens=7, lengths=None):
        self.tokens, self.lengths, self.conversations = tokens, lengths, []

    def apply_chat_template(self, conversations, **kwargs):
        assert kwargs["tokenize"] and kwargs["return_dict"] and kwargs["return_tensors"] == "pt"
        assert kwargs["processor_kwargs"] == {"padding": True}
        self.conversations.extend(conversations)
        lengths = self.lengths or [self.tokens] * len(conversations)
        width = max(lengths)
        mask = torch.tensor([[1] * n + [0] * (width - n) for n in lengths])
        return {
            "input_ids": torch.arange(width).repeat(len(lengths), 1) % 32,
            "attention_mask": mask,
        }


@pytest.mark.parametrize("keyword", [False, True])
def test_one_pass_taps_hidden_states_and_pools_the_projection(keyword):
    model = FakeTeacher(keyword=keyword)
    teacher = EmbeddingGemma2(model, FakeProcessor(7), extractor=PINNED_TEACHER)
    observation = teacher.encode("video", "clip.mp4")
    hidden = model.embed(torch.arange(7)[None] % 32)[0]
    assert observation.features.shape == (7, HIDDEN_WIDTH) != (7, 768)
    assert observation.features.dtype == torch.float32 and not observation.features.requires_grad
    torch.testing.assert_close(observation.features, hidden)
    pooled = model.language_model.embedding_projection(hidden).mean(0)
    torch.testing.assert_close(observation.embedding, torch.nn.functional.normalize(pooled, dim=0))
    assert (observation.modality, observation.extractor) == ("video", PINNED_TEACHER)
    assert not model.language_model.embedding_projection._forward_hooks
    assert all(not p.requires_grad for p in model.parameters()) and not model.training


def test_hook_survives_a_bf16_teacher_and_keeps_fp32_features():
    model = FakeTeacher().to(torch.bfloat16)
    result = EmbeddingGemma2(model, FakeProcessor(5), extractor=PINNED_TEACHER).encode(
        "audio", "a.wav"
    )
    assert result.features.dtype == torch.float32 and result.features.shape == (5, HIDDEN_WIDTH)


def test_sources_and_prompts_reach_the_processor_by_modality():
    processor = FakeProcessor()
    teacher = EmbeddingGemma2(FakeTeacher(), processor, extractor=PINNED_TEACHER)
    teacher.encode("text", "hello", prompt="title: none | text: ")
    teacher.encode("image", "pic.png")
    teacher.encode("image", "pic.png", prompt="task: search result | query: ")
    text, image, prompted = (c[0]["content"] for c in processor.conversations)
    assert text == [{"type": "text", "text": "title: none | text: hello"}]
    assert image == [{"type": "image", "path": "pic.png"}]
    assert prompted[0] == {"type": "text", "text": "task: search result | query: "}
    with pytest.raises(ValueError, match="text, audio, image or video"):
        teacher.encode("document", "x")


def test_padded_batches_keep_only_real_positions():
    model = FakeTeacher()
    teacher = EmbeddingGemma2(model, FakeProcessor(lengths=[3, 5]), extractor=PINNED_TEACHER)
    short, long = teacher.encode_batch([("text", "a"), ("text", "bbbbb")])
    assert short.features.shape == (3, HIDDEN_WIDTH) and long.features.shape == (5, HIDDEN_WIDTH)
    alone = EmbeddingGemma2(model, FakeProcessor(3), extractor=PINNED_TEACHER).encode("text", "a")
    torch.testing.assert_close(short.embedding, alone.embedding)


def test_teacher_rejects_other_heads_overlong_input_and_untapped_forward():
    with pytest.raises(ValueError, match="512->768"):
        EmbeddingGemma2(FakeTeacher(out_features=512), FakeProcessor(), extractor=PINNED_TEACHER)
    with pytest.raises(ValueError, match="512->768"):
        EmbeddingGemma2(nn.Linear(2, 2), FakeProcessor(), extractor=PINNED_TEACHER)
    too_long = EmbeddingGemma2(
        FakeTeacher(), FakeProcessor(MAX_TOKENS + 1), extractor=PINNED_TEACHER
    )
    with pytest.raises(ValueError, match="8192"):
        too_long.encode("text", "long")
    twice = FakeTeacher(calls=2)
    with pytest.raises(RuntimeError, match="exactly once"):
        EmbeddingGemma2(twice, FakeProcessor(), extractor=PINNED_TEACHER).encode("text", "x")
    # A failed tap must not leave the hook installed.
    assert not twice.language_model.embedding_projection._forward_hooks


# --- the MediaEncoder: one projection into E --------------------------------------------


def tiny_model():
    return build_model(ModelConfig.tiny(), ByteCodec.byte_fallback(), precision="fp32")


def test_media_encoder_reduces_any_length_to_the_facet_block():
    core = tiny_model().core
    c = core.config
    for tokens in (1, 3, c.n_facets, 9, 40):
        out = core.encode_media(features(tokens), extractor=PINNED_TEACHER)
        assert out.shape == (1, c.n_facets, c.facet_dim) and torch.isfinite(out).all()
    batch = torch.cat((features(9, seed=1), features(9, seed=2)))
    assert core.encode_media(batch, extractor=PINNED_TEACHER).shape[0] == 2


def test_media_encoder_rejects_another_teacher_and_bad_features():
    core = tiny_model().core
    good = features(6)
    with pytest.raises(ValueError, match="different teacher"):
        core.encode_media(good, extractor=other_extractor())
    for bad in (
        torch.randn(6, HIDDEN_WIDTH),
        torch.randn(1, 6, 768),
        torch.randn(1, 0, HIDDEN_WIDTH),
        torch.ones(1, 6, HIDDEN_WIDTH, dtype=torch.long),
    ):
        with pytest.raises(ValueError, match="device-local"):
            core.encode_media(bad, extractor=PINNED_TEACHER)
    for mask in (
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 6),  # not bool
        torch.zeros(1, 6, dtype=torch.bool),
    ):
        with pytest.raises(ValueError, match="mask"):
            core.encode_media(good, mask, extractor=PINNED_TEACHER)


def test_masked_teacher_positions_cannot_reach_the_facets():
    core = tiny_model().core
    x = features(12)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[:, :5] = True
    reference = core.encode_media(x, mask, extractor=PINNED_TEACHER)
    changed = x.clone()
    changed[:, 5:] = 100.0
    torch.testing.assert_close(
        core.encode_media(changed, mask, extractor=PINNED_TEACHER), reference
    )
    unmasked = core.encode_media(changed, extractor=PINNED_TEACHER)
    assert not torch.allclose(unmasked, reference)


def test_gradient_reaches_the_single_input_projection():
    model = tiny_model()
    core = model.core
    core.encode_media(features(12), extractor=PINNED_TEACHER).square().sum().backward()
    projection = core.media_encoder.projection.weight
    assert projection.shape == (core.config.source_dim, HIDDEN_WIDTH)
    assert core.media_encoder.projection.output_gain.grad is not None
    assert core.media_encoder.projection.output_gain.grad.abs() > 0
    shared = list(core.shared_parameters())
    assert shared and any(p.grad is not None and p.grad.abs().sum() > 0 for p in shared)
    # The reduction and facet readout are the text path's own weights.
    assert any(p.grad is not None for p in core.frontend.parameters())


# --- text anchors -----------------------------------------------------------------------


def write_anchors(tmp_path, store, rows):
    path = tmp_path / "anchors.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def anchor_row(store, payload, *, extractor=PINNED_TEACHER, family=None, modality="text"):
    key = TextAnchors.key(payload)
    asset = store.put(
        features(9, seed=3)[0].numpy(),
        modality=modality,
        kind="features",
        extractor=extractor,
        source_family=family or key,
    )
    embedding = teacher_embedding(store, modality, family or key, extractor=extractor)
    return {
        "text_sha256": key,
        "asset": dataclasses.asdict(asset),
        "embedding": dataclasses.asdict(embedding),
    }


def test_text_anchor_sidecar_loads_features_for_exact_text(tmp_path):
    store = TensorAssetStore(tmp_path)
    anchors = TextAnchors(
        write_anchors(tmp_path, store, [anchor_row(store, b"hello")]),
        [store],
        extractor=PINNED_TEACHER,
    )
    key = TextAnchors.key(b"hello")
    assert key == sha(b"hello") and set(anchors.assets) == {key}
    assert anchors.features(key, torch.device("cpu")).shape == (1, 9, HIDDEN_WIDTH)
    assert TextAnchors.key(b"hello!") not in anchors.assets


def test_text_anchor_features_are_found_in_whichever_root_holds_them(tmp_path):
    empty, holder = TensorAssetStore(tmp_path / "empty"), TensorAssetStore(tmp_path / "holder")
    (tmp_path / "empty").mkdir()
    sidecar = write_anchors(tmp_path, holder, [anchor_row(holder, b"hello")])
    key = TextAnchors.key(b"hello")
    anchors = TextAnchors(sidecar, [empty, holder], extractor=PINNED_TEACHER)
    assert anchors.features(key, torch.device("cpu")).shape == (1, 9, HIDDEN_WIDTH)
    with pytest.raises(ValueError, match="none of the prepared roots"):
        TextAnchors(sidecar, [empty], extractor=PINNED_TEACHER).features(key, torch.device("cpu"))


def test_text_anchor_sidecar_rejects_foreign_teacher_and_mismatched_text(tmp_path):
    store = TensorAssetStore(tmp_path)

    def load(*rows, extractor=PINNED_TEACHER):
        return TextAnchors(write_anchors(tmp_path, store, rows), [store], extractor=extractor)

    with pytest.raises(ValueError, match="different teacher"):
        load(anchor_row(store, b"x", extractor=other_extractor()))
    with pytest.raises(ValueError, match="different teacher"):
        load(anchor_row(store, b"x"), extractor=other_extractor())
    with pytest.raises(ValueError, match="exactly the text"):
        load(anchor_row(store, b"x", family=sha(b"another text")))
    with pytest.raises(ValueError, match="exactly the text"):
        load(anchor_row(store, b"x"), anchor_row(store, b"x"))  # duplicated key
    with pytest.raises(ValueError, match="text features"):
        load(anchor_row(store, b"x", modality="audio"))
    with pytest.raises(ValueError, match="Empty"):
        load()
    # The sidecar names the bytes: a row filed under another text's hash is refused.
    row = anchor_row(store, b"x")
    row["text_sha256"] = sha(b"y")
    with pytest.raises(ValueError, match="exactly the text"):
        load(row)


def test_anchor_loss_backpropagates_to_byte_e_and_the_media_projection():
    model = tiny_model()
    loss = anchor_loss(model, b"The quick brown fox", features(9, seed=3))
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    core = model.core
    assert core.media_encoder.projection.router.weight.grad.abs().sum() > 0
    assert core.media_encoder.projection._bank.weight.grad.abs().sum() > 0
    byte_side = [p for p in core.frontend.parameters() if p.grad is not None]
    assert byte_side and any(p.grad.abs().sum() > 0 for p in byte_side)
    byte_table = [
        p for name, p in core.frontend.named_parameters() if "byte" in name.lower() and p.ndim == 2
    ]
    assert byte_table and any(p.grad is not None and p.grad.abs().sum() > 0 for p in byte_table)


def test_anchor_loss_is_zero_only_when_both_paths_agree():
    model = tiny_model()
    payload, x = b"same text", features(9, seed=3)
    value = anchor_loss(model, payload, x)
    ids = torch.tensor([list(payload)])
    student = model.core.encode_chunk(ids, torch.tensor([len(payload)]))
    teacher = model.core.encode_media(x, extractor=PINNED_TEACHER)
    torch.testing.assert_close(value, (student - teacher).square().mean())


# --- media_bridge training scope --------------------------------------------------------


def bridge_model():
    return install_fake_realizers(tiny_model())


def test_bridge_parameters_are_the_projection_modality_embedding_and_conditions():
    model = bridge_model()
    owners = {id(p): name for name, p in model.named_parameters()}
    names = sorted(owners[id(p)] for p in model.media_bridge_parameters())
    assert names
    assert all(
        name.startswith(("core.media_encoder.", "media_realizers.packs."))
        or name == "core.central.modality_embedding.weight"
        for name in names
    )
    assert "core.central.modality_embedding.weight" in names
    assert any(".condition." in name for name in names)
    assert any(name.startswith("core.media_encoder.projection") for name in names)
    assert not any("donor" in name for name in names)
    assert len(names) == len(set(names))


def test_media_bridge_step_changes_only_the_bridge():
    model = bridge_model()
    optimizer = MasterAdamW(model, 1e-2, weight_decay=0.0)
    initial_masters = set(optimizer.masters)
    bridge = {id(p) for p in model.media_bridge_parameters()}
    optimizer.freeze([n for n, p in optimizer.model_parameters.items() if id(p) not in bridge])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x = features(9, seed=3)
    realizer = model.media_realizers.resident("audio")
    target = torch.randn(1, *LATENT["audio"], generator=torch.Generator().manual_seed(5))
    loss = realizer.loss(model.core.encode_media(x, extractor=PINNED_TEACHER), target, None)
    loss = loss + anchor_loss(model, b"anchored text", x)
    optimizer.zero_grad()
    loss.backward()
    # The frozen owners (E, C, P, R, ...) get no gradient even though the loss crosses E.
    assert all(p.grad is None for n, p in model.named_parameters() if id(p) not in bridge)
    used = [*model.core.media_encoder.parameters(), *realizer.parameters()]
    assert all(p.grad is not None for p in used)  # the other modalities' realizers are unused
    before_features = model.core.encode_media(x, extractor=PINNED_TEACHER).detach()
    optimizer.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and {id(p) for n, p in model.named_parameters() if n in changed} <= bridge
    # Normalization removes scalar gain; useful updates change feature direction.
    assert not torch.allclose(before_features, model.core.encode_media(x, extractor=PINNED_TEACHER))
    assert set(optimizer.masters) == initial_masters  # DiLoCo owner set is unchanged
    assert optimizer.frozen_names and optimizer.frozen_names.isdisjoint(changed)


def test_frozen_donor_state_never_enters_the_optimizer_or_checkpoint_view():
    model = bridge_model()
    optimizer = MasterAdamW(model, 1e-2)
    assert not any("donor" in name for name in optimizer.masters)
    assert not any("donor" in key for key in model.state_dict())
    realizer = model.media_realizers.resident("image")
    model.train()
    assert realizer.training and not realizer.donor.training


# --- name-keyed optimizer state ---------------------------------------------------------


class Net(nn.Module):
    def __init__(self, *, realizer=False, extra=False, reverse=False):
        super().__init__()
        parts = {"first": nn.Linear(3, 3), "second": nn.Linear(3, 2, bias=False)}
        for name in reversed(list(parts)) if reverse else parts:
            self.add_module(name, parts[name])
        if realizer:
            self.media_realizers = nn.Module()
            self.media_realizers.packs = nn.ModuleDict({"pocket": nn.Linear(2, 2)})
        if extra:
            self.stray = nn.Linear(2, 2)

    def forward(self, x):
        out = self.second(self.first(x))
        if hasattr(self, "media_realizers"):
            out = self.media_realizers.packs["pocket"](out)
        if hasattr(self, "stray"):
            out = self.stray(out)
        return out


def train_steps(net, optimizer, steps):
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        optimizer.zero_grad()
        x = torch.randn(4, 3, generator=generator).to(next(net.parameters()).dtype)
        net(x).square().mean().backward()
        optimizer.step()


def test_optimizer_state_round_trips_by_name_not_by_position():
    torch.manual_seed(0)
    source = Net()
    a = MasterAdamW(source, 1e-2)
    train_steps(source, a, 3)
    state = copy.deepcopy(a.state_dict())  # as a serialized checkpoint would be
    assert state["format_version"] == 3
    assert set(state["master_weights"]) == {n for n, _ in source.named_parameters()}
    assert set(state["moments"]) == set(state["master_weights"])
    torch.manual_seed(1)
    target = Net(reverse=True)  # same owners, different registration order
    b = MasterAdamW(target, 1e-2)
    b.load_state_dict(state)
    for name, _ in source.named_parameters():
        torch.testing.assert_close(b.masters[name], a.masters[name])
        assert torch.equal(
            b.optimizer.state[b.masters[name]]["exp_avg"],
            a.optimizer.state[a.masters[name]]["exp_avg"],
        )
    for n, p in target.named_parameters():  # masters are the fp32 params themselves here
        torch.testing.assert_close(p, dict(source.named_parameters())[n])
    # Resumed training continues the same trajectory.
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(4, 3, generator=generator)
    for net, optimizer in ((source, a), (target, b)):
        optimizer.zero_grad()
        net(x).square().mean().backward()
        optimizer.step()
    for name, parameter in source.named_parameters():
        torch.testing.assert_close(dict(target.named_parameters())[name], parameter)


def test_bf16_params_keep_fp32_masters_across_a_round_trip():
    net = Net().to(torch.bfloat16)
    a = MasterAdamW(net, 1e-2)
    train_steps(net, a, 2)
    state = a.state_dict()
    assert all(v.dtype == torch.float32 for v in state["master_weights"].values())
    fresh = Net().to(torch.bfloat16)
    b = MasterAdamW(fresh, 1e-2)
    b.load_state_dict(state)
    for name in state["master_weights"]:
        torch.testing.assert_close(b.masters[name], a.masters[name])


def test_new_prefixes_admit_only_newly_installed_realizer_conditions():
    torch.manual_seed(0)
    old = Net()
    a = MasterAdamW(old, 1e-2)
    train_steps(old, a, 2)
    state = a.state_dict()
    grown = Net(realizer=True)
    b = MasterAdamW(grown, 1e-2)
    with pytest.raises(ValueError, match="ownership"):
        b.load_state_dict(state)
    fresh = {n: p.detach().clone() for n, p in grown.named_parameters() if "media_realizers" in n}
    b.load_state_dict(state, new=("media_realizers.packs.",))
    for name in state["master_weights"]:
        torch.testing.assert_close(b.masters[name], a.masters[name])
    for name, value in fresh.items():  # installed owners keep their initialization
        assert torch.equal(b.masters[name], value)
        assert b.masters[name] not in b.optimizer.state
    train_steps(grown, b, 1)
    old_step = b.optimizer.state[b.masters["first.weight"]]["step"]
    new_step = b.optimizer.state[b.masters["media_realizers.packs.pocket.weight"]]["step"]
    assert (int(old_step), int(new_step)) == (3, 1)


def test_new_prefixes_do_not_admit_other_owners_or_dropped_owners():
    old = Net()
    a = MasterAdamW(old, 1e-2)
    train_steps(old, a, 1)
    state = a.state_dict()
    with pytest.raises(ValueError, match="ownership"):  # a stray owner outside the prefix
        MasterAdamW(Net(realizer=True, extra=True), 1e-2).load_state_dict(
            state, new=("media_realizers.packs.",)
        )
    big = Net(realizer=True)
    big_state = MasterAdamW(big, 1e-2).state_dict()
    with pytest.raises(ValueError, match="ownership"):  # a checkpointed owner disappeared
        MasterAdamW(Net(), 1e-2).load_state_dict(big_state, new=("media_realizers.packs.",))
    with pytest.raises(ValueError, match="inner optimizer"):
        MasterAdamW(Net(), 1e-2).load_state_dict({**state, "inner_optimizer": "muon"})
    with pytest.raises(ValueError, match="format"):
        MasterAdamW(Net(), 1e-2).load_state_dict({**state, "format_version": 2})


def test_frozen_owners_keep_their_masters_through_a_checkpoint():
    net = Net(realizer=True)
    a = MasterAdamW(net, 1e-2)
    a.freeze(["first.weight", "first.bias"])
    train_steps(net, a, 2)
    state = a.state_dict()
    assert "first.weight" in state["master_weights"] and "first.weight" not in state["moments"]
    twin = Net(realizer=True)
    b = MasterAdamW(twin, 1e-2)
    b.load_state_dict(state)
    torch.testing.assert_close(b.masters["first.weight"], a.masters["first.weight"])


# --- TrainConfig ------------------------------------------------------------------------


def config(**overrides):
    settings = dict(
        purpose="diagnostic",
        model="tiny",
        phase="B0",
        budget=64,
    )
    settings.update(overrides)
    return TrainConfig(**settings)


def test_the_stage_plan_decides_what_trains():
    assert config().trainable == "all"
    bridge = config(phase="MB", trainable="media_bridge", packs="pack-lock.json")
    assert bridge.trainable == "media_bridge"
    with pytest.raises(ValueError, match="Stage MB trains"):
        config(phase="MB", packs="pack-lock.json")
    with pytest.raises(ValueError, match="Stage B0 trains"):
        config(trainable="media_bridge", packs="pack-lock.json")


def test_trainable_rejects_unknown_scopes():
    assert config(trainable=None).trainable == "all"  # Omission inherits the shared policy.
    for value in ("text", "bridge", ""):
        with pytest.raises(ValueError, match="all, media_bridge, or latent_policy"):
            config(trainable=value)


def test_media_bridge_requires_packs():
    with pytest.raises(ValueError, match="realizer packs"):
        config(phase="MB", trainable="media_bridge")


def test_startup_media_requirements_follow_outputs_and_active_autoencode_inputs():
    dataset = SimpleNamespace(
        feature_count=lambda split, modality: {"audio": 3, "image": 2}.get(modality, 0)
    )
    # A media observation has no output modality. Only decoded generated
    # media targets are included in the prepared summary.
    assert required_media_modalities(dataset, (), autoencode_items=0) == ()
    # Creation and edit tasks score generated media; audiovisual creation needs
    # both decoders, while encode-and-reconstruct also samples present media.
    assert required_media_modalities(dataset, ("audio", "video", "image"), autoencode_items=0) == (
        "audio",
        "image",
        "video",
    )
    assert required_media_modalities(dataset, (), autoencode_items=16) == ("audio", "image")


def test_startup_media_requirements_report_missing_realizer_modalities():
    model = SimpleNamespace(
        media_realizers=SimpleNamespace(
            resident=lambda modality: (_ for _ in ()).throw(ValueError("not installed"))
        )
    )
    with pytest.raises(ValueError, match="audio, video"):
        validate_media_realizers(model, ("audio", "video"))


def test_text_anchor_weight_is_a_fraction_in_the_half_open_unit_interval():
    assert config(text_anchor_weight=1.0).text_anchor_weight == 1.0
    assert config(text_anchor_weight=0.01).text_anchor_weight == 0.01
    assert config().text_anchor_weight == 0.1
    for value in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError, match="anchor weight"):
            config(text_anchor_weight=value)


def test_pack_manifest_path_is_nonempty_and_legacy_fields_are_gone():
    with pytest.raises(ValueError, match="nonempty"):
        config(packs="")
    for legacy in ("pretrained_packs", "embeddinggemma2_warmup", "native_pack_manifest"):
        with pytest.raises(TypeError):
            config(**{legacy: None})


def _payload(model, packs_fingerprint):
    return {"model": model.state_dict(), "packs_fingerprint": packs_fingerprint}


def test_phase_transition_may_install_realizers_and_nothing_else():
    from megatoken.loading import NEW_REALIZER_OWNERS, restore_weights

    text_phase = tiny_model()
    media = bridge_model()
    payload = _payload(text_phase, None)
    with pytest.raises(ValueError, match="pack manifest differs"):
        restore_weights(media, payload, "f" * 64, phase_transition=False)
    assert restore_weights(media, payload, "f" * 64, phase_transition=True) == NEW_REALIZER_OWNERS
    for name, value in text_phase.state_dict().items():
        torch.testing.assert_close(media.state_dict()[name], value)
    # Packs cannot be swapped or removed, even at a phase transition.
    with pytest.raises(ValueError, match="pack manifest differs"):
        restore_weights(media, _payload(media, "a" * 64), "f" * 64, phase_transition=True)
    with pytest.raises(ValueError, match="pack manifest differs"):
        restore_weights(text_phase, _payload(media, "f" * 64), None, phase_transition=True)
    # Same packs: an exact, strict restore.
    assert restore_weights(media, _payload(media, "f" * 64), "f" * 64, phase_transition=False) == ()
    broken = _payload(text_phase, None)
    broken["model"] = {k: v for k, v in broken["model"].items() if "media_encoder" not in k}
    with pytest.raises(ValueError, match="beyond newly installed realizers"):
        restore_weights(media, broken, "f" * 64, phase_transition=True)
