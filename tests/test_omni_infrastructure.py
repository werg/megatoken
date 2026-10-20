import hashlib
from fractions import Fraction

import pytest
import torch

from megatoken.controls import effective_event_payloads
from megatoken.generation.latents import DiagonalGaussian
from megatoken.omni.bundles import AlignedBundle, Artifact, Relation, TaskView, compile_views
from megatoken.omni.clocks import TimeWindow, available_windows
from megatoken.omni.flow import FacetFlow, flow_matching_loss
from megatoken.omni.packs import PackRegistry, PackSpec
from megatoken.omni.recipe import FamilySampler
from megatoken.omni.relations import coordinate_kl, endpoint_targets


def test_distinct_event_requests_are_causal_and_repeatable():
    prompts = effective_event_payloads(b"global", ("speech", "caption"), (("Speak English", 2), ("Describe visuals", 7)))
    assert prompts[0] != prompts[1]
    assert prompts == effective_event_payloads(b"global", ("speech", "caption"), (("Speak English", 2), ("Describe visuals", 7)))
    assert b"Describe visuals" not in prompts[0]
    with pytest.raises(ValueError):
        effective_event_payloads(b"", ("same", "same"), (("a", 0), ("b", 1)))

def test_bundle_compiler_keeps_joint_targets_out_of_prefix():
    a = (Artifact("source", "text", b"hello"), Artifact("speech", "text", b"hello spoken"), Artifact("translation", "text", b"hallo"))
    relation = Relation("translation_of", "translation", "source")
    bundle = AlignedBundle("bundle1", "speaker1", "train", a, (relation,))
    view = TaskView("speech_translation", ("source",), ("speech", "translation"),
                    {x.artifact_id: x.artifact_id + " instruction" for x in a}, relations=(relation,))
    c = compile_views(bundle, (view,))[0]
    assert len(c.episode.meridians) == 2
    assert len(c.episode.meridians[1].events) == 2
    assert all(not r.policy_mask for r in c.episode.records[:1])
    assert all(r.policy_mask for r in c.episode.records[1:])
    assert c.episode.lineage.frozen_split == "train"

def test_codec_pairing_never_silently_swaps():
    digest = hashlib.sha256(b"weights").hexdigest()
    g = PackSpec("sana", "generator", "provider/sana", "rev1", digest, "dcae", "rev2")
    d = PackSpec("dcae", "codec", "provider/dcae", "rev2", digest)
    assert PackRegistry((g,d)).verify_pair("sana", "dcae")
    with pytest.raises(ValueError):
        PackRegistry((g, PackSpec("dcae", "codec", "provider/dcae", "rev3", digest))).verify_pair("sana", "dcae")

def test_latent_flow_free_rollout_and_masked_counts():
    flow = FacetFlow(8, 4, 16)
    raw = torch.randn(2, 32, 8)
    target = torch.randn(2, 6, 4)
    numerator, count = flow.training_loss(target, raw)
    assert count.item() == 48 and numerator.isfinite()
    sample = flow.sample(raw, target.shape, steps=2)
    assert sample.shape == target.shape and sample.isfinite().all()
    n, c = flow_matching_loss(torch.zeros_like(target), target, torch.ones(2,6,dtype=torch.bool))
    assert c.item() == 48 and n.item() > 0

def test_analytic_kl_count_and_duplicate_invariance():
    q = DiagonalGaussian(torch.ones(1,32,2), torch.zeros(1,32,2))
    p = DiagonalGaussian(torch.zeros(1,32,2), torch.zeros(1,32,2))
    n, c = coordinate_kl(q,p,q.mean.numel())
    assert torch.allclose(n/c, torch.tensor(0.5))
    q2 = DiagonalGaussian(q.mean.repeat(3,1,1),q.log_scale.repeat(3,1,1))
    p2 = DiagonalGaussian(p.mean.repeat(3,1,1),p.log_scale.repeat(3,1,1))
    n2,c2=coordinate_kl(q2,p2,q2.mean.numel())
    assert torch.allclose(n2/c2,n/c)

def test_endpoints_and_prefix_limits():
    video = torch.randn(1,5,3,16,16)
    first,last=endpoint_targets(video,1,3)
    assert torch.equal(first,video[:,1]) and torch.equal(last,video[:,3])
    with pytest.raises(ValueError):
        available_windows((TimeWindow(Fraction(0),Fraction(2),Fraction(3)),),Fraction(2))
    assert FamilySampler(seed=42).choose(("replay","speech_translation")) in ("replay","speech_translation")
