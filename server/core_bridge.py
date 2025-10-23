# server/core_bridge.py
from typing import Dict, Any, List, Optional
import torch
from torch import nn

from src.state_bus import EntanglementBus
from src.losses import info_nce

class Core(nn.Module):
    """
    Thin stepper for the API.
    Uses tiny projection heads + GRU bus to advance a shared state and compute an E* proxy.
    """
    def __init__(self, device: str = "cpu", state_dim: int = 64):
        super().__init__()
        self.device = device
        self.state_dim = state_dim

        # small projections (stand-ins for pooled LLM hiddens)
        self.projW = nn.Linear(256, 256, bias=False)
        self.projT = nn.Linear(256, 256, bias=False)

        # shared state bus
        self.bus = EntanglementBus(state_dim, in_dim=256 * 2)

        self.to(self.device)

    @torch.inference_mode()
    def step(self, S_list: Optional[List[float]] = None) -> Dict[str, Any]:
        B = 1
        if S_list is None:
            S = torch.zeros(B, self.state_dim, device=self.device)
        else:
            S = torch.tensor(S_list, device=self.device).view(B, -1)

        # pseudo base features (swap with pooled LLM hiddens later)
        baseW = torch.randn(B, 256, device=self.device)
        baseT = torch.randn(B, 256, device=self.device)
        hW = self.projW(baseW)
        hT = self.projT(baseT)

        # advance shared state
        S_next = self.bus(S, hW, hT)

        # E* proxy: invert InfoNCE (lower NCE ⇒ higher coherence)
        L_mi = info_nce(hW, hT, tau=0.1).detach().item()
        e_star = float(max(0.0, 2.0 - L_mi))

        # shaped JSONs
        spec = {
            "assumptions": ["models co-learn via shared state"],
            "data": {"sources": ["synthetic"]},
            "steps": ["writer: draft spec", "tester: draft tests", "update: shared state"],
            "interfaces": ["api:/runs/{id}/step"],
            "acceptance": ["spec+tests present", "E* reported"],
            "risks": ["stub dynamics", "no LLM plugged yet"],
        }
        tests = [
            {"name": "spec_has_acceptance", "checks": ["acceptance length > 0"]},
            {"name": "has_risks", "checks": ["risks length > 0"]},
        ]

        return {
            "spec": spec,
            "tests": tests,
            "e_star": e_star,
            "state": S_next.detach().cpu().view(-1).tolist(),
        }
