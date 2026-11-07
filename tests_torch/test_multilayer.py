import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from src.compression.internal_merging import expand_right
from src.compression.multilayer import (SiteMoments, compress_site, consumer_energy, greedy_victims, kept_width,
    magnitude_victims, site_gram, site_maps, sites, substituted_logits, units)
from src.compression.multisource import fit_full_multi, fit_quaternion_multi, fit_real_multi
from src.models.cifar_cnn_torch import CifarCNN


def randomized(model):
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.weight.copy_(torch.rand(module.num_features, generator=generator)+.5)
                module.bias.copy_(torch.randn(module.num_features, generator=generator)*.1)
                module.running_mean.copy_(torch.randn(module.num_features, generator=generator)*.1)
                module.running_var.copy_(torch.rand(module.num_features, generator=generator)+.5)
    return model.eval()


def quaternion_moments(values):
    channels = values.transpose(0, 1, 3, 2).reshape(-1, 4*values.shape[2])
    return SiteMoments(channels.T@channels, len(channels), True)


def test_moment_fits_match_raw_value_fits():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(30, 7, 5, 4))
    moments, kept = quaternion_moments(values), [0, 1, 3, 4]
    matrices, mse = moments.fit(2, kept, 1e-3, "quaternion_right")
    np.testing.assert_allclose(matrices, expand_right(fit_quaternion_multi(values, 2, kept, 1e-3)), atol=1e-10)
    x = values[..., kept, :].reshape(-1, len(kept), 4)
    residual = values[..., 2, :].reshape(-1, 4)-np.einsum("koi,nki->no", matrices, x)
    assert abs(mse-np.square(residual).mean()) < 1e-10
    full, full_mse = moments.fit(2, kept, 1e-3, "full_real")
    np.testing.assert_allclose(full, fit_full_multi(values, 2, kept, 1e-3), atol=1e-10)
    assert full_mse <= mse+1e-12
    scalar = rng.normal(size=(20, 6, 5)).reshape(-1, 5)
    real = SiteMoments(scalar.T@scalar, len(scalar), False)
    coefficients, real_mse = real.fit(4, [0, 1, 2, 3], 1e-3, "real")
    np.testing.assert_allclose(coefficients, fit_real_multi(scalar, 4, [0, 1, 2, 3], 1e-3), atol=1e-10)
    assert abs(real_mse-np.square(scalar[:, 4]-scalar[:, :4]@coefficients).mean()) < 1e-10


def test_greedy_removes_a_planted_redundant_unit_first():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(40, 9, 6, 4))
    values[..., 3, :] = np.einsum("koi,npki->npo", expand_right(rng.normal(size=(2, 4))), values[..., [0, 5], :])
    # Quaternions are invertible, so units 0 and 5 are exactly reconstructible as well.
    victims = greedy_victims(quaternion_moments(values), 1, np.ones(6), 0., "quaternion_right")
    assert victims[0] in (0, 3, 5)
    assert quaternion_moments(values).fit(victims[0], [u for u in range(6) if u != victims[0]], 0., "quaternion_right")[1] < 1e-20
    assert len(greedy_victims(quaternion_moments(values), 5, np.ones(6), 1e-3, "quaternion_right")) == 5


@pytest.mark.parametrize("kind", ["quaternion", "real"])
def test_every_site_folds_exactly(kind):
    torch.manual_seed(3)
    model = randomized(CifarCNN(kind, q_widths=(3, 4, 5), real_widths=(5, 6, 4)))
    inputs = torch.randn(24, 3, 16, 16)
    mode = "quaternion_right" if kind == "quaternion" else "real"
    for site in sites(model):
        moments = SiteMoments(*site_gram(model, site, inputs[:16]), kind == "quaternion")
        victims = greedy_victims(moments, 2, consumer_energy(model, site), 1e-3, mode)
        maps = site_maps(moments, victims, 1e-3, mode)
        compact, kept, lost = compress_site(model, site, victims, maps)
        with torch.no_grad():
            torch.testing.assert_close(compact(inputs[16:]), substituted_logits(model, site, maps, inputs[16:]), atol=1e-5, rtol=1e-5)
        assert lost < 1e-20 and units(compact, site) == units(model, site)-2
        assert sum(p.numel() for p in compact.parameters()) < sum(p.numel() for p in model.parameters())


def test_unconstrained_maps_are_lossy_only_at_quaternion_convolutions():
    torch.manual_seed(4)
    model = randomized(CifarCNN(q_widths=(3, 4, 5)))
    inputs = torch.randn(16, 3, 16, 16)
    for site in ((1, 1), (2, 1)):
        moments = SiteMoments(*site_gram(model, site, inputs), True)
        maps = site_maps(moments, [0, 2], 1e-3, "full_real")
        compact, _, lost = compress_site(model, site, [0, 2], maps)
        if site == (2, 1):
            assert lost == 0.
            with torch.no_grad():
                torch.testing.assert_close(compact(inputs), substituted_logits(model, site, maps, inputs), atol=1e-5, rtol=1e-5)
        else:
            assert lost > 0.


@pytest.mark.parametrize("kind", ["quaternion", "real"])
def test_uniform_chain_is_the_narrower_standard_architecture(kind):
    torch.manual_seed(5)
    q_widths, real_widths = (4, 8, 8), (8, 8, 16)
    model = randomized(CifarCNN(kind, q_widths=q_widths, real_widths=real_widths))
    inputs = torch.randn(12, 3, 16, 16)
    current = model
    for site in sites(model):
        width = units(current, site)
        current = compress_site(current, site, magnitude_victims(current, site, width-kept_width(width, .75)))[0]
    narrow = CifarCNN(kind, [kept_width(w, .75) for w in q_widths], [kept_width(w, .75) for w in real_widths]).eval()
    narrow.load_state_dict(current.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(narrow(inputs), current(inputs), atol=0, rtol=0)
    assert sum(p.numel() for p in narrow.parameters()) == sum(p.numel() for p in current.parameters())
