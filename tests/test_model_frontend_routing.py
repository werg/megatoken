import pytest
import torch

from megatoken.model import FacetedCore, ModelConfig
from megatoken.model.frontend import ByteFrontend, NativeNGramTable
from megatoken.model.routing import EpochRouter


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(23)


def test_ordered_bytes_preserve_case_order_zero_and_padding():
    frontend = ByteFrontend(ModelConfig.tiny()).eval()
    a = frontend(torch.tensor([[65, 0, 66, 67], [65, 0, 66, 67]])).features
    b = frontend(torch.tensor([[65, 66, 0, 67], [97, 0, 66, 67]])).features
    assert not torch.equal(a[0], b[0])
    assert not torch.equal(a[1], b[1])
    padded = frontend(torch.tensor([[65, 0, 66, 67, 255, 255]]), torch.tensor([4])).features
    torch.testing.assert_close(a[:1], padded)


def test_import_bake_exact_parity_and_optimizer_transition():
    table = NativeNGramTable(ModelConfig.tiny()).bfloat16()
    rows = torch.tensor([65, 0, 66])
    table.install_import(
        rows, torch.randn(3, 128), torch.tensor([1.0, 0.5, 1.0]), torch.tensor([0.5, 1.0, 1.0])
    )
    optimizer = torch.optim.AdamW(table.parameters(), lr=0.01)
    table.lookup(torch.tensor([0, 1, 65, 66, 65])).float().square().sum().backward()
    optimizer.step()
    old_map = table.import_map.weight
    before = table.lookup(torch.arange(300)).clone()
    clone = NativeNGramTable(ModelConfig.tiny()).bfloat16()
    clone.load_state_dict(table.state_dict())
    torch.testing.assert_close(clone.lookup(torch.arange(300)), before, rtol=0, atol=0)
    report = table.bake(optimizer, chunk_rows=2)
    assert report.max_lookup_error == 0 and report.baked_rows == 3
    assert report.removed_parameters == 128 * 8
    torch.testing.assert_close(table.lookup(torch.arange(300)), before, rtol=0, atol=0)
    assert table.import_map is None and table.descriptors.numel() == 0
    assert old_map not in optimizer.state
    assert not any(p is old_map for group in optimizer.param_groups for p in group["params"])
    assert optimizer.state[table.embedding.weight]["exp_avg"][rows].count_nonzero() == 0
    assert not any("import_map" in key for key in table.state_dict())


def test_selected_palette_and_row_confidence_receive_task_gradients():
    router = EpochRouter(8, 8, 16, 4)
    route = router.route(torch.randn(2, 8), 7, 17)
    route.confidence.sum().backward()
    assert router.palette_codes.grad.norm() > 0
    assert router.row_codes.grad.norm() > 0
    assert route.eligible.shape == (2, 4)
    assert (route.expert_ids[:, :, None] == route.eligible[:, None]).any(-1).all()


def test_routing_deterministic_replay_and_batch_membership_invariance():
    router = EpochRouter(8, 8, 16, 4, exploration=0.2)
    context = torch.randn(3, 8)
    ids = torch.tensor([51, 72, 93])
    together = router.route(context, 9, 31, record_ids=ids)
    for index in range(3):
        alone = router.route(context[index : index + 1], 9, 31, record_ids=ids[index : index + 1])
        torch.testing.assert_close(
            alone.expert_ids, together.expert_ids[index : index + 1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            alone.confidence, together.confidence[index : index + 1], rtol=1e-6, atol=1e-6
        )
    other = router.route(context, 9, 31, record_ids=ids)
    torch.testing.assert_close(other.expert_ids, together.expert_ids, rtol=0, atol=0)
    # Policy replay overrides training exploration without mutating module mode.
    deterministic = router.route(context, 9, 31, allow_exploration=False)
    router.eval()
    evaluated = router.route(context, 9, 31)
    torch.testing.assert_close(deterministic.expert_ids, evaluated.expert_ids, rtol=0, atol=0)


def test_record_role_is_causal_model_input():
    core = FacetedCore(ModelConfig.tiny()).eval()
    state = core.initial_state(1)
    ids = torch.tensor([[111, 107]])
    assistant = core.observe(ids, state, role_ids=torch.tensor([2]))
    tool = core.observe(ids, state, role_ids=torch.tensor([3]))
    torch.testing.assert_close(assistant.facets, tool.facets, rtol=0, atol=0)
    assert not torch.equal(core.process(assistant).facets, core.process(tool).facets)


def test_empty_physical_record_is_a_role_boundary_not_a_fake_byte():
    core = FacetedCore(ModelConfig.tiny())
    state = core.initial_state(2)
    observed = core.observe(
        torch.tensor([[17], [255]]),
        state,
        lengths=torch.tensor([0, 0]),
        role_ids=torch.tensor([2, 2]),
    )
    torch.testing.assert_close(observed.facets[0], observed.facets[1])
    updated = core.process(observed)
    assert updated.records_seen.tolist() == [1, 1]
    torch.testing.assert_close(updated.facets[0], updated.facets[1])
    updated.facets.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in core.parameters() if p.grad is not None)
    assert core.frontend.bytes.weight.grad.count_nonzero() == 0


def test_bake_updates_fp32_masters_and_removes_map_ownership():
    from megatoken.training.optimizer import MasterAdamW

    table = NativeNGramTable(ModelConfig.tiny()).bfloat16()
    rows = torch.tensor([3, 9, 15])
    table.install_import(rows, torch.randn(3, 128), torch.ones(3), torch.ones(3))
    optimizer = MasterAdamW(table, lr=0.01)
    table.lookup(torch.tensor([1, 3, 9, 15])).float().square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    untouched_master = optimizer.masters["embedding.weight"][1].clone()
    untouched_moment = optimizer.optimizer.state[optimizer.masters["embedding.weight"]]["exp_avg"][
        1
    ].clone()
    before = table.lookup(rows).clone()
    table.bake(optimizer)
    assert set(optimizer.masters) == {"embedding.weight"}
    assert set(optimizer.model_parameters) == {"embedding.weight"}
    assert set(optimizer._hooks) == {"embedding.weight"}
    master = optimizer.masters["embedding.weight"]
    torch.testing.assert_close(master[rows], before.float(), rtol=0, atol=0)
    torch.testing.assert_close(master[1], untouched_master, rtol=0, atol=0)
    state = optimizer.optimizer.state[master]
    assert state["exp_avg"][rows].count_nonzero() == 0
    torch.testing.assert_close(state["exp_avg"][1], untouched_moment, rtol=0, atol=0)
    # The next optimizer step cannot restore stale pre-bake master values.
    table.lookup(rows).float().square().sum().backward()
    optimizer.step()
    assert torch.isfinite(table.lookup(rows)).all()
    checkpoint = optimizer.state_dict()
    replacement = MasterAdamW(table, lr=0.01)
    replacement.load_state_dict(checkpoint)
    optimizer.close()
    replacement.close()
