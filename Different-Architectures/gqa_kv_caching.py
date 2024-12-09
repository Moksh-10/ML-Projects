import math
import torch
import torch.nn as nn

batch_size = 5
seq_len = 12
num_q_heads = 8
num_kv_heads = 4
head_dim = 64
total_dim = 8 * 64

def rkv(x: torch.Tensor, rep: int):
    bs, sl, nh, d = x.shape
    return x[:, :, :, None, :].expand(bs, sl, nh, rep, d).contiguous().view(bs, sl, nh * rep, d)



class gqa_with_kv_cache(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = total_dim
        self.head_dim = head_dim
        self.n_kv_heads = num_kv_heads
        self.n_q_heads = num_q_heads
        self.rep = self.n_q_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.head_dim * self.n_q_heads)
        self.wk = nn.Linear(self.dim, self.head_dim * self.n_kv_heads)
        self.wv = nn.Linear(self.dim, self.head_dim * self.n_kv_heads)
        self.wo = nn.Linear(self.head_dim * self.n_q_heads, self.dim)

        self.k_cache = torch.zeros((batch_size, seq_len, num_kv_heads, head_dim))
        self.v_cache = torch.zeros((batch_size, seq_len, num_kv_heads, head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, seq_len_for_kv: int):
        bs, sl, d = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(bs, sl, self.n_q_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bs, sl, self.n_kv_heads, self.head_dim)
        v = v.view(bs, sl, self.n_kv_heads, self.head_dim)

        self.k_cache[:bs, start_pos: start_pos + seq_len_for_kv] = k
        self.v_cache[:bs, start_pos: start_pos + seq_len_for_kv] = v

        keys = self.k_cache[:bs, : start_pos + seq_len_for_kv]
        values = self.v_cache[:bs, : start_pos + seq_len_for_kv]

        keys = rkv(keys, self.rep).permute(0, 2, 1, 3)
        values = rkv(values, self.rep).permute(0, 2, 1, 3)

        attn = (q @ keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        att = nn.functional.softmax(attn, dim=-1)
        a = att @ values
        x = a.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.dim)
        x = self.wo(x)
        return x


x = torch.randn(batch_size, seq_len, total_dim)
y = gqa_with_kv_cache()
g = y(x, 1, seq_len)
print(g.shape)



