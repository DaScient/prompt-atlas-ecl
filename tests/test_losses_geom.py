"""Tests for Phase 2 distribution-geometry losses."""
import torch

from src.losses import kl_sym
from src.losses_geom import gaussian_kl_sym, mmd2, sinkhorn_wasserstein


torch.manual_seed(0)


def test_mmd_identical_distributions_is_near_zero():
    g = torch.Generator().manual_seed(123)
    x = torch.randn(64, 8, generator=g)
    # Same distribution, fresh sample.
    g2 = torch.Generator().manual_seed(456)
    y = torch.randn(64, 8, generator=g2)
    val = mmd2(x, y)
    assert val.item() >= 0.0
    assert val.item() < 0.1  # small for matched distributions


def test_mmd_distinguishes_shifted_distributions():
    g = torch.Generator().manual_seed(123)
    x = torch.randn(64, 8, generator=g)
    y = torch.randn(64, 8, generator=g) + 5.0
    val_match = mmd2(x, x.clone())
    val_diff = mmd2(x, y)
    assert val_diff.item() > val_match.item()


def test_mmd_handles_degenerate_batch():
    x = torch.randn(1, 4)
    y = torch.randn(8, 4)
    val = mmd2(x, y)
    assert val.item() == 0.0  # falls through degenerate guard


def test_sinkhorn_nonnegative_and_zero_on_identical():
    x = torch.randn(16, 4)
    val_same = sinkhorn_wasserstein(x, x.clone(), epsilon=0.05, n_iters=50)
    # With identical clouds, transport cost should be (near) zero.
    assert val_same.item() >= 0.0
    assert val_same.item() < 1e-2

    y = x + 3.0
    val_shift = sinkhorn_wasserstein(x, y, epsilon=0.05, n_iters=50)
    assert val_shift.item() > val_same.item()


def test_sinkhorn_is_differentiable():
    x = torch.randn(8, 4, requires_grad=True)
    y = torch.randn(8, 4)
    loss = sinkhorn_wasserstein(x, y, epsilon=0.1, n_iters=20)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gaussian_kl_sym_is_nonneg_and_symmetric():
    x = torch.randn(32, 6)
    y = torch.randn(32, 6) + 2.0
    a = gaussian_kl_sym(x, y)
    b = gaussian_kl_sym(y, x)
    assert a.item() >= 0.0
    assert torch.allclose(a, b, atol=1e-5)


def test_kl_sym_uses_geometry_backend():
    # The Phase 1 stub returned exactly 0; Phase 2 should produce a
    # positive value when the two batches differ in mean.
    x = torch.randn(32, 6)
    y = torch.randn(32, 6) + 5.0
    val = kl_sym(x, y)
    assert val.item() > 0.0
