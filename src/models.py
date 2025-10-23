from dataclasses import dataclass
import torch, torch.nn as nn

@dataclass
class LLMOut:
    tokens: list
    hidden: torch.Tensor  # [B, T, D]
    logits: torch.Tensor  # optional rubric logits

class Summarizer(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, H):
        pooled = H.mean(dim=1)
        return self.proj(pooled)

class LatentHead(nn.Module):
    def __init__(self, in_dim, z=16):
        super().__init__()
        self.mu = nn.Linear(in_dim, z)
        self.lv = nn.Linear(in_dim, z)
    def forward(self, H):
        pooled = H.mean(dim=1)
        return self.mu(pooled), self.lv(pooled)
