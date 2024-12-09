import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttentionBlock(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (bs, seq_len, dim)
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

        # (bs, seq_len, dim) --> (bs, seq_len, dim*3) --> 3 different tensor of (bs, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (bs, seq_len, dim) -->  # (bs, seq_len, h, dim/h) -->  # (bs, h, seq_len, dim/h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (bs, h, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # mask where the upper triangle (above principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)

        weight = F.softmax(weight, dim=-1)

        # (bs, h, seq_len, seq_len) @ # (bs, h, seq_len, dim/h) --> # (bs, h, seq_len, seq_len, dim/h)
        output = weight @ v

        #  # (bs, h, seq_len, dim/h) @ # (bs, seq_len, h, dim/h)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (bs, seq_len, dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent) : (bs, seq_len_Q, dim_Q)
        # y: context: (bs, seq_len_KV, dim_KV) = (bs, 77, 768)
        input_shape = x.shape
        bs, seq_len, d_embed = input_shape

        interim_shape = (bs, -1, self.n_heads, self.d_head)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).tranpose(1, 2)

        wei = q @ k.transpose(-1, -2)
        wei /= math.sqrt(self.d_head)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(input_shape)
        out = self.out_proj(out)
        return out
