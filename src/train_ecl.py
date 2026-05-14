import torch, yaml
from torch import nn
from torch.optim import AdamW

from src.losses import info_nce, kl_sym, mmd2, sinkhorn_wasserstein
from src.state_bus import EntanglementBus
from src.tracking import MLflowTracker

def get_device(pref=None):
    if isinstance(pref, str):
        pref = pref.strip().lower()
    if pref in {"cpu","cuda","mps"}:
        if pref == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA requested but not available; falling back to auto.")
        elif pref == "mps" and not torch.backends.mps.is_available():
            print("[warn] MPS requested but not available; falling back to auto.")
        else:
            return pref
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def load_cfg(p="configs/ecl_llm_llm.yaml"):
    with open(p, "r") as fh:
        return yaml.safe_load(fh)

def train():
    cfg = load_cfg()
    device = get_device(cfg.get("device"))
    print(f"→ Using device: {device}")

    B          = int(cfg.get("batch_size", 4))
    state_dim  = int(cfg.get("state_dim", 64))
    tau        = float(cfg.get("infoNCE_tau", 0.1))
    max_steps  = int(cfg.get("max_steps", 1000))
    eval_every = int(cfg.get("eval_every", 100))
    weights = cfg["loss_weights"]
    a, b, g, d, e = [weights[k] for k in ["lat", "mi", "u", "con", "div"]]
    # Phase 2 weights default to 0 so legacy configs still train identically.
    w_wass = float(weights.get("wass", 0.0))
    w_mmd  = float(weights.get("mmd", 0.0))

    sinkhorn_cfg = cfg.get("sinkhorn", {}) or {}
    sk_eps   = float(sinkhorn_cfg.get("epsilon", 0.05))
    sk_iters = int(sinkhorn_cfg.get("n_iters", 30))

    bus   = EntanglementBus(state_dim, in_dim=256*2).to(device)
    projW = nn.Linear(256, 256, bias=False).to(device)
    projT = nn.Linear(256, 256, bias=False).to(device)

    params = list(bus.parameters()) + list(projW.parameters()) + list(projT.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=0.01)

    S = torch.zeros(B, state_dim, device=device)

    # Phase 2 — MLflow experiment tracker. Honors PAE_TRACKING env var; the
    # config can also force-enable via tracking.enabled. Silent no-op when
    # MLflow isn't installed.
    tracking_cfg = cfg.get("tracking", {}) or {}
    tracker = MLflowTracker(
        experiment=tracking_cfg.get("experiment", "prompt-atlas-ecl"),
        run_name=tracking_cfg.get("run_name"),
        enabled=tracking_cfg.get("enabled") if tracking_cfg.get("enabled") is not None else None,
    )
    tracker.start_run()
    tracker.log_params({
        "device": device,
        "batch_size": B,
        "state_dim": state_dim,
        "infoNCE_tau": tau,
        "max_steps": max_steps,
        **{f"w_{k}": v for k, v in weights.items()},
        "sinkhorn_epsilon": sk_eps,
        "sinkhorn_iters": sk_iters,
    })

    try:
        for step in range(max_steps):
            S = S.detach()

            baseW = torch.randn(B, 256, device=device)
            baseT = torch.randn(B, 256, device=device)
            hW = projW(baseW)
            hT = projT(baseT)

            S = bus(S, hW, hT)

            # Latent symmetric-KL between Writer/Tester batches (replaces 0-stub).
            L_lat = kl_sym(hW, hT)
            L_mi  = info_nce(hW, hT, tau=tau)
            L_u   = torch.tensor(0.0, device=device)
            L_con = torch.tensor(0.7, device=device)
            L_div = torch.tensor(0.1, device=device)
            L_state = 1e-3 * (S.pow(2).mean())

            # Phase 2 — only pay the geometry cost when the weight is non-zero.
            L_wass = (
                sinkhorn_wasserstein(hW, hT, epsilon=sk_eps, n_iters=sk_iters)
                if w_wass > 0.0
                else torch.tensor(0.0, device=device)
            )
            L_mmd = (
                mmd2(hW, hT)
                if w_mmd > 0.0
                else torch.tensor(0.0, device=device)
            )

            loss = (
                a * L_lat + b * L_mi + g * L_u + d * L_con + e * L_div
                + w_wass * L_wass + w_mmd * L_mmd
                + L_state
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if step % eval_every == 0:
                print(
                    f"step {step} | loss {loss.item():.3f} | L_mi {L_mi.item():.3f} "
                    f"| L_lat {L_lat.item():.4f} | L_wass {L_wass.item():.4f} "
                    f"| L_mmd {L_mmd.item():.4f} | L_state {L_state.item():.6f}"
                )
                tracker.log_metrics(
                    {
                        "loss": loss.item(),
                        "L_lat": L_lat.item(),
                        "L_mi": L_mi.item(),
                        "L_wass": L_wass.item(),
                        "L_mmd": L_mmd.item(),
                        "L_state": L_state.item(),
                    },
                    step=step,
                )
    finally:
        tracker.end_run()

if __name__ == "__main__":
    train()
