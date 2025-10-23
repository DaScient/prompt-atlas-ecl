import torch, yaml
from losses import kl_sym, info_nce
from state_bus import EntanglementBus

def load_cfg(p="configs/ecl_llm_llm.yaml"):
    return yaml.safe_load(open(p))

def train():
    cfg = load_cfg()
    device = cfg.get("device", "cuda")
    B = cfg.get("batch_size", 4)

    bus = EntanglementBus(cfg["state_dim"], in_dim=256*2).to(device)
    S = torch.zeros(B, cfg["state_dim"], device=device)

    for step in range(cfg["max_steps"]):
        hW = torch.randn(B, 256, device=device)
        hT = torch.randn(B, 256, device=device)

        S = bus(S, hW, hT)

        L_lat = torch.tensor(0.0, device=device)  # placeholder if posteriors wired later
        L_mi  = info_nce(hW, hT, tau=cfg["infoNCE_tau"])
        L_u   = torch.tensor(0.0, device=device)
        L_con = torch.tensor(0.7, device=device)
        L_div = torch.tensor(0.1, device=device)

        a,b,g,d,e = [cfg["loss_weights"][k] for k in ["lat","mi","u","con","div"]]
        loss = a*L_lat + b*L_mi + g*L_u + d*L_con + e*L_div

        loss.backward()
        # opt.step(); opt.zero_grad()

        if step % cfg["eval_every"] == 0:
            print(f"step {step} | loss {loss.item():.3f}")

if __name__ == "__main__":
    train()
