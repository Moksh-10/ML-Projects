import torch
from torch import nn
import os
import json
import requests
from pathlib import Path
import re

QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}



class feed_for(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = nn.functional.silu(x1) * x2
        return self.fc3(x)


class rms_norm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_comp=True):
        super().__init__()
        self.eps = eps
        self.qc = qwen3_comp
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        idt = x.dtype

        if self.qc:
            x = x.to(torch.float32)

        nx = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        nx = nx * self.scale

        if self.shift is not None:
            nx = nx + self.shift

        return nx.to(idt)


def calc_freq(head_dim, theta_base=10_000, context_len=4096, dtype=torch.float32):
    assert head_dim % 2 == 0
    inv_fr = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    pos = torch.arange(context_len, dtype=dtype)
    ang = pos.unsqueeze(1) * inv_fr.unsqueeze(0)
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    return sin, cos

class gqattn(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0

        self.nh = num_heads
        self.nkv = num_kv_groups
        self.ratio = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.hd = head_dim
        self.dout = num_heads * head_dim

        self.wq = nn.Linear(d_in, self.dout, bias=False, dtype=dtype)
        self.wk = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.wv = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.wo = nn.Linear(self.dout, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = rms_norm(head_dim, eps=1e-6)
            self.k_norm = rms_norm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, sl, dd = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(b, sl, self.nh, self.hd).transpose(1, 2)
        k = k.view(b, sl, self.nkv, self.hd).transpose(1, 2)
        q = q.view(b, sl, self.nkv, self.hd).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        k = k

