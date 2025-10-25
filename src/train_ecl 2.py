import torch, yaml
from torch import nn
from torch.optim import AdamW

from src.losses import info_nce, kl_sym   # kl_sym stays stubbed for now
from src.state_bus import EntanglementBus

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
    if torch.backends.mps.is_available(): return "mps"
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
    a,b,g,d,e  = [cfg["loss_weights"][k] for k in ["lat","mi","u","con","div"]]

    # --- Models ---
    bus  = EntanglementBus(state_dim, in_dim=256*2).to(device)
    projW = nn.Linear(256, 256, bias=False).to(device)   # tiny trainable heads
    projT = nn.Linear(256, 256, bias=False).to(device)

    # Optional: init close to identity-ish for stability
    nn.init.eye_(projW.weight.data[:256,:256]) if projW.weight.shape[0]==projW.weight.shape[1] else None
    nn.init.eye_(projT.weight.data[:256,:256]) if projT.weight.shape[0]==projT.weight.shape[1] else None

    # Optimizer
    params = list(bus.parameters()) + list(projW.parameters()) + list(projT.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=0.01)

    # State
    S = torch.zeros(B, state_dim, device=device)

    for step in range(max_steps):
        # Pretend these are pooled LLM hiddens (random now, trainable via proj heads)
        baseW = torch.randn(B, 256, device=device)
        baseT = torch.randn(B, 256, device=device)

        hW = projW(baseW)  # requires grad
        hT = projT(baseT)  # requires grad

        # Update shared state (depends on GRU weights)
        S = bus(S, hW, hT)

        # Losses
        L_lat = torch.tensor(0.0, device=device)            # placeholder for later latent posteriors
        L_mi  = info_nce(hW, hT, tau=tau)                   # drives projW/projT to align
        L_u   = torch.tensor(0.0, device=device)            # placeholder
        L_con = torch.tensor(0.7, device=device)            # placeholder
        L_div = torch.tensor(0.1, device=device)            # placeholder

        # Tie loss to trainable GRU via S so gradients flow through the bus now
        L_state = 1e-3 * (S.pow(2).mean())

        loss = a*L_lat + b*L_mi + g*L_u + d*L_con + e*L_div + L_state

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        if step % eval_every == 0:
            with torch.no_grad():
                # quick proxy grad norm check
                gn = 0.0
                for p in params:
                    if p.grad is not None:
                        gn += (p.grad.detach().norm().item())
                print(f"step {step} | loss {loss.item():.3f} | L_mi {L_mi.item():.3f} | L_state {L_state.item():.6f} | grad_sum {gn:.3f}")

if __name__ == "__main__":
    train()
