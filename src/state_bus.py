import torch, torch.nn as nn

class EntanglementBus(nn.Module):
    def __init__(self, state_dim=64, in_dim=512):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=state_dim, batch_first=True)
        self.to_tokens = nn.Linear(state_dim, 8 * 64)  # 8 tokens x 64 dims

    def forward(self, S_t, hW, hT):
        x = torch.cat([hW, hT], dim=-1).unsqueeze(1)   # [B,1,in_dim]
        out, h = self.gru(x, S_t.unsqueeze(0))         # h: [1,B,state_dim]
        S_next = h.squeeze(0)
        return S_next

    def tokens(self, S):
        t = self.to_tokens(S)                          # [B, 512]
        return t.view(S.size(0), 8, 64)                # [B, T_tokens, D]
