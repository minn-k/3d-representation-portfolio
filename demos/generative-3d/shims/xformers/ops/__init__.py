import torch
import torch.nn.functional as F
from . import fmha


def _sdpa(q, k, v):
    # q,k,v: [B, N, H, C] -> SDPA 는 [B, H, N, C]
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def memory_efficient_attention(q, k, v, attn_bias=None):
    if attn_bias is None:
        return _sdpa(q, k, v)
    assert isinstance(attn_bias, fmha.BlockDiagonalMask) and q.shape[0] == 1
    qs, ks = attn_bias.q_seqlen, attn_bias.kv_seqlen
    out = torch.empty(q.shape[:-1] + (v.shape[-1],), dtype=q.dtype, device=q.device)
    # 길이가 같은 블록끼리 묶어 한 번에 계산 (창 주의(windowed)는 블록이 많다)
    qo = [0]
    for n in qs:
        qo.append(qo[-1] + n)
    ko = [0]
    for n in ks:
        ko.append(ko[-1] + n)
    groups = {}
    for i, (a, b) in enumerate(zip(qs, ks)):
        groups.setdefault((a, b), []).append(i)
    for (a, b), idx in groups.items():
        if a == 0:
            continue
        qi = torch.stack([q[0, qo[i]:qo[i] + a] for i in idx])
        ki = torch.stack([k[0, ko[i]:ko[i] + b] for i in idx])
        vi = torch.stack([v[0, ko[i]:ko[i] + b] for i in idx])
        oi = _sdpa(qi, ki, vi)
        for j, i in enumerate(idx):
            out[0, qo[i]:qo[i] + a] = oi[j]
    return out
