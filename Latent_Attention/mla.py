import math
import torch
from torch import nn

def freqs(seq_len, dim, theta=10000.00):
    tn = 1.0 / (theta ** torch.arange(0, dim, 2).float() / dim)
    m = torch.arange(0, seq_len)
    frs = torch.outer(m, tn).float()
    fc = torch.polar(torch.ones_like(frs), frs)
    return fc

def rope(x, fc):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    fc = fc.unsqueeze(0)
    xr = xc * fc
    xo = torch.view_as_real(xr)
    xo = xo.reshape(*x.shape)
    return xo.type_as(x)

class attn(nn.Module):
    def __init__(self, d_model, num_heads, d_q, d_kv) -> None:
        super().__init__()

        self.hd = d_model // num_heads
        self.nh = num_heads
        self.d_model = d_model

        self.w_down_q = nn.Linear(d_model, d_q)
        self.w_up_q = nn.Linear(d_q, d_model)

        self.w_down_kv = nn.Linear(d_model, d_kv)
        self.w_up_k = nn.Linear(d_kv, d_model)
        self.w_up_v = nn.Linear(d_kv, d_model)

        self.wq_r = nn.Linear(d_q, d_model)
        self.wk_r = nn.Linear(d_kv, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x):
        bs, sl, dim = x.shape

        ct_q = self.w_down_q(x)
        q_ct = self.w_up_q(ct_q)
        q_ct = q_ct.view(bs, sl, self.nh, self.hd).transpose(1, 2)

        frs = freqs(sl, self.d_model)
        rope_q = rope(self.wq_r(ct_q), frs).view(bs, sl, self.nh, self.hd).transpose(1, 2)
        q = q_ct + rope_q

        ct_kv = self.w_down_kv(x)
        k_ct = self.w_up_k(ct_kv).view(bs, sl, self.nh, self.hd).transpose(1, 2)

        rope_k = rope(self.wk_r(ct_kv), frs).view(bs, sl, self.nh, self.hd).transpose(1, 2)

        k = k_ct + rope_k
        v = self.w_up_v(ct_kv).view(bs, sl, self.nh, self.hd).transpose(1, 2)

        a = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hd)
        a = a.softmax(dim=-1)
        a = torch.matmul(a, v)
        a = a.transpose(1, 2).contiguous().view(bs, sl, self.nh * self.hd)
        a = self.wo(a)
        return a

x = torch.randn(5, 55, 768*8)
y = attn(768*8, 16, 3, 3)
print(y(x).shape)



