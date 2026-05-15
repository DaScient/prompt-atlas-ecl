import torch, torch.nn.functional as F

# Phase 2 — distribution-geometry losses are kept in a separate module so
# this file stays focused on the original contrastive objective. They are
# re-exported here for backwards compatibility with callers that did
# ``from src.losses import kl_sym``.
from src.losses_geom import gaussian_kl_sym, mmd2, sinkhorn_wasserstein


def info_nce(hW, hT, tau=0.1):
    hW = F.normalize(hW, dim=-1); hT = F.normalize(hT, dim=-1)
    logits = hW @ hT.t() / tau
    labels = torch.arange(hW.size(0), device=hW.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return 0.5*loss


def kl_sym(q1, q2):
    """Symmetric KL between two empirical latent batches.

    Phase 2: this used to be a stub returning ``0``. It now delegates to
    :func:`src.losses_geom.gaussian_kl_sym`, which fits a diagonal
    Gaussian to each batch and returns the symmetrized divergence.
    Callers may still receive a zero tensor if the batches are too
    small (n < 2) to estimate variance.
    """
    return gaussian_kl_sym(q1, q2)


__all__ = ["info_nce", "kl_sym", "gaussian_kl_sym", "mmd2", "sinkhorn_wasserstein"]
