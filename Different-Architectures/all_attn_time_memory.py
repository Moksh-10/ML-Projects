import math
import torch
import torch.nn as nn
from time import time
import torch.profiler


class multi_head_atnn(nn.Module):
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


def repetition(x: torch.Tensor, factor: int):
    bs, sl, nh, hd = x.shape
    y = x[:, :, :, None, :].expand(bs, sl, nh, factor, hd).contiguous().view(bs, sl, nh * factor, hd)
    return y


class multi_query_atnn(nn.Module):
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

        q = q.view(bs, -1, self.nh, self.h_dim).permute(0, 2, 1, 3) # (bs, sl, dim) -> (bs, sl, nh, hd) -> (bs, nh, sl, hd)
        k = k.unsqueeze(1) # (bs, sl, hd) -> (bs, 1, sl, hd)
        v = v.unsqueeze(1)

        attn = (q @ k.transpose(2, 3)) / math.sqrt(self.h_dim)
        att = nn.functional.softmax(attn, dim=-1)
        x = torch.matmul(att, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, -1, self.dim)
        x = self.wo(x)
        return x


class group_query_attn(nn.Module):
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


def time_memory_stats():
    batch_size = 100
    seq_len = 256
    num_q_heads = 64
    num_heads = 64
    num_kv_heads = 32
    head_dim = 256
    total_dim = 64 * 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(batch_size, seq_len, total_dim, device=device)
    model = multi_head_atnn(total_dim=total_dim, n_heads=num_heads).to(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        profile_memory=True, with_stack=True, record_shapes=True
    ) as pro:
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        start = time()

        u = model(x)

        end = time()
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        print(f'output_shape: {u.shape}')
        print(f'time: {end - start:.4f} s')
        print(f'memory before start: {start_mem / 1e6:.2f} MB')
        print(f'memory after: {end_mem / 1e6:.2f} MB')
        print(f'peak memory: {peak_mem / 1e6:.2f} MB')

        print("stats:")
        print(pro.key_averages())

if __name__=="__main__":
    time_memory_stats()

