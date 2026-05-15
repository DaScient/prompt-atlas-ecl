"""Phase 2 — distribution-geometry losses.

These complement the InfoNCE objective in :mod:`src.losses` by giving the
optimizer signals about the *distributional* alignment between Writer and
Tester latents, not just their per-sample pairing.

All functions are differentiable, batched, and return a scalar
``torch.Tensor`` so they can be added directly into a training loss.

The implementations are intentionally lightweight (no external optimal-
transport dependency) so they stay in sync with the rest of the
Phase 1 "lazy / fallback" philosophy: real geometry when torch is
available, no extra services required.
"""
from __future__ import annotations

from typing import Iterable

import torch


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between rows of ``x`` and ``y``.

    Shapes: ``x: (n, d)``, ``y: (m, d)`` → ``(n, m)``.

    Uses the ``|x-y|^2 = |x|^2 + |y|^2 - 2 x·y`` identity and clamps to
    avoid small negative values from floating-point error.
    """
    if x.dim() != 2 or y.dim() != 2 or x.size(-1) != y.size(-1):
        raise ValueError(
            f"expected 2-D tensors with matching feature dim, got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    x2 = (x * x).sum(dim=-1, keepdim=True)            # (n, 1)
    y2 = (y * y).sum(dim=-1, keepdim=True).t()        # (1, m)
    return (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)


def mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: Iterable[float] = (0.5, 1.0, 2.0, 4.0, 8.0),
    *,
    unbiased: bool = True,
) -> torch.Tensor:
    """Multi-bandwidth Gaussian MMD² between empirical distributions.

    Following Gretton et al. 2012, with a sum of RBF kernels at several
    bandwidths so the estimator is sensitive across scales — this avoids
    the standard "single sigma" pitfall when the latent geometry is
    not yet calibrated.

    Returns a non-negative scalar (it can be tiny-negative under the
    unbiased estimator when ``x`` and ``y`` are nearly identical; we
    clamp at zero for numerical safety in a loss).
    """
    if x.size(0) < 2 or y.size(0) < 2:
        # MMD is not well-defined with one sample; return 0 so training
        # doesn't explode for degenerate batches.
        return x.new_zeros(())

    dxx = _pairwise_sq_dists(x, x)
    dyy = _pairwise_sq_dists(y, y)
    dxy = _pairwise_sq_dists(x, y)

    n, m = x.size(0), y.size(0)
    total = x.new_zeros(())

    for sigma in bandwidths:
        gamma = 1.0 / (2.0 * float(sigma) ** 2)
        kxx = torch.exp(-gamma * dxx)
        kyy = torch.exp(-gamma * dyy)
        kxy = torch.exp(-gamma * dxy)

        if unbiased:
            # Drop diagonal — see Gretton 2012, eq. (3).
            kxx_sum = (kxx.sum() - torch.diagonal(kxx).sum()) / (n * (n - 1))
            kyy_sum = (kyy.sum() - torch.diagonal(kyy).sum()) / (m * (m - 1))
        else:
            kxx_sum = kxx.mean()
            kyy_sum = kyy.mean()
        kxy_sum = kxy.mean()

        total = total + kxx_sum + kyy_sum - 2.0 * kxy_sum

    return total.clamp_min(0.0)


def sinkhorn_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 50,
    p: int = 2,
) -> torch.Tensor:
    """Entropy-regularized Wasserstein-``p`` distance via Sinkhorn iterations.

    Uniform marginals on the empirical samples ``x`` and ``y`` (Cuturi 2013).
    Runs in log-space for numerical stability.

    Returns a non-negative scalar — the optimal transport cost
    ``<P, C>`` where ``C_ij = ||x_i - y_j||^p``. For ``p=2`` this is the
    (squared) Wasserstein-2 distance up to the entropy regularizer.
    """
    n, m = x.size(0), y.size(0)
    if n == 0 or m == 0:
        return x.new_zeros(())

    # Cost matrix.
    if p == 2:
        cost = _pairwise_sq_dists(x, y)
    else:
        cost = _pairwise_sq_dists(x, y).clamp_min(0.0).pow(p / 2.0)

    # log-marginals (uniform).
    log_a = torch.full((n,), -torch.log(torch.tensor(float(n))), device=x.device, dtype=x.dtype)
    log_b = torch.full((m,), -torch.log(torch.tensor(float(m))), device=x.device, dtype=x.dtype)

    log_K = -cost / epsilon                     # (n, m)
    log_u = torch.zeros(n, device=x.device, dtype=x.dtype)
    log_v = torch.zeros(m, device=x.device, dtype=x.dtype)

    for _ in range(n_iters):
        # log_u = log_a - logsumexp(log_K + log_v, dim=1)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)

    # Transport plan in log space: log_P = log_u[:,None] + log_K + log_v[None,:]
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    P = torch.exp(log_P)
    return (P * cost).sum().clamp_min(0.0)


def gaussian_kl_sym(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Symmetric KL between Gaussian moment-matches of two latent batches.

    Each batch is summarized by its diagonal-covariance Gaussian, then
    we return ``0.5 * (KL(p||q) + KL(q||p))``. This is the proper
    replacement for the previous ``kl_sym`` stub that returned 0.

    Non-negative; zero iff the two empirical means *and* per-dim variances
    coincide.
    """
    if x.size(0) < 2 or y.size(0) < 2:
        return x.new_zeros(())

    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    var_x = x.var(dim=0, unbiased=True).clamp_min(eps)
    var_y = y.var(dim=0, unbiased=True).clamp_min(eps)

    # KL(N(mu_x, var_x) || N(mu_y, var_y)) per dim, summed.
    def _kl(mu_a, var_a, mu_b, var_b):
        return 0.5 * (
            (var_a / var_b)
            + ((mu_b - mu_a).pow(2) / var_b)
            - 1.0
            + (var_b.log() - var_a.log())
        ).sum()

    return 0.5 * (_kl(mu_x, var_x, mu_y, var_y) + _kl(mu_y, var_y, mu_x, var_x))


__all__ = ["mmd2", "sinkhorn_wasserstein", "gaussian_kl_sym"]
