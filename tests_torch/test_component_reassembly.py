import torch
from torch import nn

from src.cnn_training import first_update_audit
from src.compression.component_reassembly import (TinyQuaternionPair, compiled_pair, decoded_producer_rows,
    hamilton_basis, packing_indices, parameter_counts, project_hamilton_kernel, quaternion_design, refit_consumer)
from src.models.quaternion_conv_torch import hamilton_kernel


def test_projection_is_orthogonal_and_recovers_hamilton_kernels():
    torch.set_num_threads(2)
    torch.manual_seed(37)
    kernel = torch.randn(4,3,2,3,3,dtype=torch.float64)
    dense = hamilton_kernel(kernel)
    torch.testing.assert_close(project_hamilton_kernel(dense),kernel,atol=1e-14,rtol=1e-14)
    arbitrary = torch.randn_like(dense)
    projected = project_hamilton_kernel(arbitrary)
    residual = arbitrary-hamilton_kernel(projected)
    torch.testing.assert_close(project_hamilton_kernel(residual),torch.zeros_like(projected),atol=1e-14,rtol=0)


def test_quartet_cost_equals_decoded_variance_and_pairwise_distance():
    torch.manual_seed(5)
    rows = torch.randn(4,8,3,3,dtype=torch.float64)
    decoded = decoded_producer_rows(rows)
    center = decoded.mean(0)
    projected = project_hamilton_kernel(rows)[:,0]
    torch.testing.assert_close(center,projected,atol=1e-14,rtol=1e-14)
    cost = (rows-hamilton_kernel(projected[:,None])).square().sum()
    variance = (decoded-center).square().sum()
    pairwise = sum((decoded[a]-decoded[b]).square().sum() for a in range(4) for b in range(a+1,4))/4
    torch.testing.assert_close(cost,variance,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(cost,pairwise,atol=1e-12,rtol=1e-12)


def test_original_quartet_is_exact_but_mixing_need_not_be():
    torch.manual_seed(9)
    teacher = TinyQuaternionPair().double().eval()
    x = torch.randn(16,4,1,1,dtype=torch.float64)
    original,metrics = compiled_pair(teacher,[(0,2,4,6)])
    assert metrics['producer_kernel_squared_error'] == 0
    torch.testing.assert_close(original.features(x),teacher.features(x)[:,[0,2,4,6]],atol=0,rtol=0)
    mixed,metrics = compiled_pair(teacher,[(0,3,4,7)])
    assert metrics['producer_kernel_squared_error'] > 1e-6
    assert parameter_counts(teacher)['real_parameters'] == 32
    assert parameter_counts(mixed)['real_parameters'] == 16
    assert list(dict(mixed.named_children())) == ['producer','bn','consumer']
    assert mixed.producer.out_channels == mixed.consumer.in_channels == 1


def test_packing_respects_component_blocked_layout_and_full_identity():
    torch.manual_seed(2)
    teacher = TinyQuaternionPair().double().eval()
    groups = [(0,2,4,6),(1,3,5,7)]
    assert packing_indices(groups,2) == list(range(8))
    replica,metrics = compiled_pair(teacher,groups)
    with torch.no_grad():
        replica.consumer.kernel.copy_(teacher.consumer.kernel)
    x = torch.randn(6,4,1,1,dtype=torch.float64)
    torch.testing.assert_close(replica(x),teacher(x),atol=0,rtol=0)
    assert metrics['producer_kernel_squared_error'] == 0


def test_quaternion_design_and_output_regression_match_forward():
    torch.manual_seed(12)
    teacher = TinyQuaternionPair(2,2,2).double().eval()
    x = torch.randn(50,8,1,1,dtype=torch.float64)
    h = teacher.features(x)
    design = quaternion_design(h[:,:,0,0])
    theta = teacher.consumer.kernel[:,:,:,0,0].permute(0,2,1).reshape(8,2)
    y = teacher(x)
    expected = y[:,:,0,0].reshape(50,4,2).reshape(200,2)
    torch.testing.assert_close(design@theta,expected,atol=1e-12,rtol=1e-12)
    clone = TinyQuaternionPair(2,2,2).double().eval()
    error = refit_consumer(clone,h,y,1e-10)
    assert error < 1e-14


def test_compact_pair_tracks_gradients_and_updates_all_q_components():
    torch.set_num_threads(2)
    torch.manual_seed(16)
    model = nn.Sequential(TinyQuaternionPair(1,1,1).double(),nn.Flatten())
    x = torch.randn(16,4,1,1,dtype=torch.float64)
    audit = first_update_audit(model,x,torch.arange(16)%4,torch.optim.SGD(model.parameters(),lr=.01))
    assert all(audit['checks'].values())


def test_invalid_packing_rejects_duplicate_or_outside_components():
    import pytest
    for groups in ([(0,0,1,2)],[(0,1,2,8)],[(0,1,2)]):
        with pytest.raises(ValueError):
            packing_indices(groups,2)
