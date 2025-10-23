import torch, torch.nn as nn

class EntanglementBus(nn.Module):
    def __init__(self, state_dim=64, in_dim=512):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=state_dim, batch_first=True)

    def forward(self, S_t, hW, hT):
        x = torch.cat([hW, hT], dim=-1).unsqueeze(1)
        _, h = self.gru(x, S_t.unsqueeze(0))
        return h.squeeze(0)
