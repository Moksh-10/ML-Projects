import einops
import torch
from torch import nn


class sa(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.d = dim
        self.hd = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x):
        assert x.dim() == 3
        qkv = self.to_qkv(x)
        q, k, v = tuple(einops.rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.h))
        sc = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = torch.softmax(sc, dim=-1)
        attn = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = einops.rearrange(attn, 'b h t d -> b t (h d)')
        return self.wo(out)

x = torch.randn(2, 8, 768)
a = sa(768, 6)
print(a(x).shape)