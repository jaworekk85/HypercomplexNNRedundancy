from itertools import permutations
import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from src.cnn_training import first_update_audit
from src.models.channel_routing import FixedChannelPermutation, LearnableChannelPermutation, RoutedCifarCNN, log_sinkhorn, maximum_assignment
from src.models.cifar_cnn_torch import CifarCNN
from src.compression.quaternion_channels import compact_block, component_indices


def test_assignment_solver_matches_exhaustive_optimum_and_handles_ties():
    rng = np.random.default_rng(471)
    for n in range(2, 6):
        for _ in range(3):
            scores = rng.normal(size=(n, n))
            indices = maximum_assignment(scores)
            expected = max(sum(scores[row, col] for row,col in enumerate(order)) for order in permutations(range(n)))
            assert sorted(indices) == list(range(n))
            assert scores[np.arange(n), indices].sum() == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(maximum_assignment(np.zeros((4, 4))), maximum_assignment(np.zeros((4, 4))))
    with pytest.raises(ValueError):
        maximum_assignment([[np.nan]])


def test_soft_assignment_gradcheck_and_balanced_marginals():
    torch.manual_seed(47)
    logits = (.2*torch.randn(4, 4, dtype=torch.float64)).requires_grad_()
    inputs = torch.randn(2, 4, 2, 2, dtype=torch.float64, requires_grad=True)
    def route(weights, features):
        return torch.einsum("ij,njhw->nihw", log_sinkhorn(weights, .7, 30), features)
    assert torch.autograd.gradcheck(route, (logits, inputs), atol=1e-5)
    matrix = log_sinkhorn(logits, .7, 30)
    torch.testing.assert_close(matrix.sum(0), torch.ones(4, dtype=torch.float64), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(matrix.sum(1), torch.ones(4, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_synthetic_permutation_is_learned_and_exported_exactly():
    torch.manual_seed(3)
    router = LearnableChannelPermutation(4, 9, initial_strength=.2).double()
    inputs = torch.randn(12, 4, 3, 3, dtype=torch.float64)
    target_indices = torch.tensor([2, 0, 3, 1])
    target = inputs[:, target_indices]
    optimizer = torch.optim.Adam(router.parameters(), lr=.2)
    initial_loss = float((router(inputs)-target).square().mean().detach())
    original = router.logits.detach().clone()
    for _ in range(120):
        optimizer.zero_grad()
        loss = (router(inputs)-target).square().mean()
        loss.backward()
        assert bool(torch.isfinite(router.logits.grad).all())
        optimizer.step()
    assert not torch.equal(original, router.logits)
    assert float(loss.detach()) < initial_loss*.1
    router.harden()
    assert router.indices.tolist() == target_indices.tolist()
    torch.testing.assert_close(router(inputs), target, atol=0, rtol=0)
    exported = router.export()
    assert sum(p.numel() for p in exported.parameters()) == 0
    torch.testing.assert_close(exported(inputs), router(inputs), atol=0, rtol=0)
    stream = io.BytesIO()
    torch.save(router.state_dict(), stream)
    stream.seek(0)
    restored = LearnableChannelPermutation(4, 9, initial_strength=.2).double()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert not restored.logits.requires_grad
    torch.testing.assert_close(restored(inputs), target, atol=0, rtol=0)


def test_identity_matches_existing_cnn_and_random_start_is_paired():
    torch.manual_seed(2026)
    original = CifarCNN(q_widths=(2, 3, 4)).eval()
    torch.manual_seed(2026)
    identity = RoutedCifarCNN("identity", q_widths=(2, 3, 4)).eval()
    inputs = torch.randn(3, 3, 32, 32)
    torch.testing.assert_close(original(inputs), identity(inputs), atol=0, rtol=0)
    random = RoutedCifarCNN("random", q_widths=(2, 3, 4))
    learned = RoutedCifarCNN("learned", q_widths=(2, 3, 4))
    torch.testing.assert_close(random.router.indices, learned.router.initial_indices)


def test_cnn_audit_updates_routing_and_all_quaternion_components():
    torch.manual_seed(313)
    model = RoutedCifarCNN("learned", q_widths=(2, 3, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    result = first_update_audit(model, torch.randn(8, 3, 32, 32), torch.arange(8)%10, optimizer)
    assert all(result["checks"].values())
    assert next(row for row in result["parameters"] if row["name"] == "router.logits")["max_abs_update"] > 0
    model.eval()
    model.router.harden()
    exported = model.export_hard()
    assert isinstance(exported.router, FixedChannelPermutation)
    inputs = torch.randn(3, 3, 32, 32)
    torch.testing.assert_close(exported(inputs), model(inputs), atol=0, rtol=0)
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in exported.parameters()) == 8*8


def test_compact_last_block_preserves_the_exported_router():
    torch.manual_seed(71)
    model = RoutedCifarCNN("learned", q_widths=(2, 3, 4)).eval()
    model.router.harden()
    original = model.export_hard()
    kept = [0, 2]
    compact = compact_block(original, 2, kept)
    torch.testing.assert_close(compact.router.indices, original.router.indices, atol=0, rtol=0)
    def mask(module, args, output):
        result = output.clone()
        result[:, component_indices([1, 3], 4)] = 0
        return result
    handle = original.blocks[2].register_forward_hook(mask)
    inputs = torch.randn(3, 3, 32, 32)
    expected = original(inputs)
    handle.remove()
    torch.testing.assert_close(compact(inputs), expected, atol=1e-6, rtol=1e-5)
