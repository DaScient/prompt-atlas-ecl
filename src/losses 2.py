import torch, torch.nn.functional as F

def info_nce(hW,hT,tau=0.1):
    hW = F.normalize(hW, dim=-1); hT = F.normalize(hT, dim=-1)
    logits = hW @ hT.t() / tau
    import torch
    labels = torch.arange(hW.size(0), device=hW.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return 0.5*loss

def kl_sym(q1,q2):
    return torch.tensor(0.0)
