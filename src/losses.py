import torch, torch.nn.functional as F

def kl_sym(q1, q2):
    mu1, lv1 = q1
    mu2, lv2 = q2
    def kl(mu_p, lv_p, mu_q, lv_q):
        v_p, v_q = lv_p.exp(), lv_q.exp()
        return 0.5 * ( (v_p / v_q) + (mu_q - mu_p).pow(2)/v_q - 1 + (lv_q - lv_p) ).sum(dim=-1)
    return (kl(mu1, lv1, mu2, lv2) + kl(mu2, lv2, mu1, lv1)).mean()

def info_nce(hW, hT, tau=0.1):
    hW = F.normalize(hW, dim=-1); hT = F.normalize(hT, dim=-1)
    logits = hW @ hT.t() / tau
    labels = torch.arange(hW.size(0), device=hW.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    return loss * 0.5

def entropy_from_logits(logits):
    p = logits.log_softmax(dim=-1).exp()
    return -(p * p.log()).sum(dim=-1).mean()
