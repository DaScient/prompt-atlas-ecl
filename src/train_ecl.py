import torch, yaml
from torch import nn
from torch.optim import AdamW

from src.losses import info_nce, kl_sym
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
    a,b,g,d,e  = [cfg["loss_weights"][k] for k in ["lat","mi","u","con","div"]]

    bus   = EntanglementBus(state_dim, in_dim=256*2).to(device)
    projW = nn.Linear(256, 256, bias=False).to(device)
    projT = nn.Linear(256, 256, bias=False).to(device)

    params = list(bus.parameters()) + list(projW.parameters()) + list(projT.parameters())
    opt = AdamW(params, lr=3e-4, weight_decay=0.01)

    S = torch.zeros(B, state_dim, device=device)

    for step in range(max_steps):
        S = S.detach()

        baseW = torch.randn(B, 256, device=device)
        baseT = torch.randn(B, 256, device=device)
        hW = projW(baseW)
        hT = projT(baseT)

        S = bus(S, hW, hT)

        L_lat = torch.tensor(0.0, device=device)
        L_mi  = info_nce(hW, hT, tau=tau)
        L_u   = torch.tensor(0.0, device=device)
        L_con = torch.tensor(0.7, device=device)
        L_div = torch.tensor(0.1, device=device)
        L_state = 1e-3 * (S.pow(2).mean())

        loss = a*L_lat + b*L_mi + g*L_u + d*L_con + e*L_div + L_state

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % eval_every == 0:
            print(f"step {step} | loss {loss.item():.3f} | L_mi {L_mi.item():.3f} | L_state {L_state.item():.6f}")

if __name__ == "__main__":
    train()
