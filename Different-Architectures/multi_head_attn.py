import math
import torch
import torch.nn as nn
from time import time

batch_size = 50
seq_len = 256
num_heads = 16
head_dim = 128
total_dim = 16 * 128

class mha(nn.Module):
    def __init__(self, total_dim: int, n_heads: int):
        super().__init__()
        self.nh = n_heads
        self.dim = total_dim
        assert self.dim % self.nh == 0, "must be divisible"
        self.head_dim = self.dim // self.nh

        self.wq = nn.Linear(self.dim, self.dim)
        self.wk = nn.Linear(self.dim, self.dim)
        self.wv = nn.Linear(self.dim, self.dim)
        self.wo = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor):
        bs, sl, dims = x.shape

        # (bs, sl, dim) --> (bs, sl, dim)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # (bs, sl, dim) --> (bs, sl, nh, hd) --> (bs, nh, sl, hd)
        q = q.view(bs, sl, self.nh, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bs, sl, self.nh, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bs, sl, self.nh, self.head_dim).permute(0, 2, 1, 3)

        # (bs, nh, sl, hd) @ (bs, nh, hd, sl) --> (bs, nh, sl, sl)
        attn = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
        att = nn.functional.softmax(attn, dim=-1)
        # (bs, nh, sl, sl) @ (bs, nh, sl, hd) --> (bs, nh, sl, hd)
        atts = att @ v

        # (bs, nh, sl, hd) --> (bs, sl, nh, hd) --> (bs, sl, nh * hd)
        x = atts.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.dim)
        # (bs, sl, nh * hd) --> (bs, sl, dim)
        x = self.wo(x)
        return x

x = torch.randn(batch_size, seq_len, total_dim)
a = mha(total_dim=total_dim, n_heads=num_heads)
start = time()
b = a(x)
finished = time()
print(b.shape)
print(f'time: {finished - start}s')

#  time: 1.8468880653381348s