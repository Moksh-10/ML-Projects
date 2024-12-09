import math
import torch
import torch.nn as nn
from time import time

batch_size = 50
seq_len = 50
num_heads = 16
head_dim = 128
total_dim = 16 * 128

class mqa(nn.Module):
    def __init__(self, num_h, dim):
        super().__init__()
        self.dim = dim
        self.h_dim = dim // num_h
        self.nh = num_h

        self.wq = nn.Linear(self.dim, self.dim)
        self.wk = nn.Linear(self.dim, self.h_dim)
        self.wv = nn.Linear(self.dim, self.h_dim)
        self.wo = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor):
        bs, sl, dim = x.shape
        q = self.wq(x) # (bs, sl, dim)
        k = self.wk(x) # (bs, sl, hd)
        v = self.wv(x)
        print(k.shape, q.shape)

        q = q.view(bs, -1, self.nh, self.h_dim).permute(0, 2, 1, 3) # (bs, sl, dim) -> (bs, sl, nh, hd) -> (bs, nh, sl, hd)
        k = k.unsqueeze(1) # (bs, sl, hd) -> (bs, 1, sl, hd)
        v = v.unsqueeze(1)
        print(v.shape, q.shape)
        print("attn")
        print(q.shape)
        print(k.transpose(2, 3).shape)
        attn = (q @ k.transpose(2, 3)) / math.sqrt(self.h_dim)
        print(attn.shape)
        att = nn.functional.softmax(attn, dim=-1)
        x = torch.matmul(att, v)
        print(x.shape)
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.dim)
        print(x.shape)
        x = self.wo(x)
        return x


x = torch.randn(batch_size, seq_len, total_dim)
a = mqa(num_h=num_heads, dim=total_dim)
s = time()
b = a(x)
d = time()
print(b.shape)
print(f'time: {d - s}s')

# time: 1.2304139137268066s