import math
import torch
import torch.nn as nn
from time import time

batch_size = 50
seq_len = 256
num_q_heads = 16
num_kv_heads = 8
head_dim = 128
total_dim = 16 * 128

def repetition(x: torch.Tensor, factor: int):
    bs, sl, nh, hd = x.shape
    y = x[:, :, :, None, :].expand(bs, sl, nh, factor, hd).contiguous().view(bs, sl, nh * factor, hd)
    return y


class gqa(nn.Module):
    def __init__(self, q_heads: int, kv_heads: int, total_dim: int):
        super().__init__()
        self.q_h = q_heads
        self.kv_h = kv_heads
        self.dim = total_dim
        self.head_dim = self.dim // self.q_h
        self.factor = self.q_h // self.kv_h

        self.wq = nn.Linear(self.dim, self.q_h * self.head_dim)
        self.wk = nn.Linear(self.dim, self.kv_h * self.head_dim)
        self.wv = nn.Linear(self.dim, self.kv_h * self.head_dim)
        self.wo = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor):
        bs, sl, dims = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bs, sl, self.q_h, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bs, sl, self.kv_h, self.head_dim)
        v = v.view(bs, sl, self.kv_h, self.head_dim)

        k = repetition(k, self.factor).permute(0, 2, 1, 3)
        v = repetition(v, self.factor).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
        att = nn.functional.softmax(attn, dim=-1)
        a = att @ v
        x = a.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.dim)
        x = self.wo(x)
        return x


x = torch.randn(batch_size, seq_len, total_dim)
y = gqa(q_heads=num_q_heads, kv_heads=num_kv_heads, total_dim=total_dim)
start = time()
dd = y(x)
done = time()
print(dd.shape)
print(f'time: {done - start}')
