import math
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms


def attn(q, k, v, mask):
    d_k = q.shape[-1]

    attn_scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill_(mask == 0, 1e-9)
    attn_scores = attn_scores.softmax(dim=-1)
    return attn_scores @ v


class GELU(nn.Module):
    def forward(self, x):
        return (0.5 * x) * (1 + torch.tanh((math.sqrt(2/math.pi))* (x + 0.044715 * (x**3))))


class multi_head_attn(nn.Module):
    def __init__(self, d_model: int, h: int):
        super().__init__()
        self.d_model = d_model
        self.h = h

        self.d_k = d_model // h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        # (batch, seq_len, dim) --> (batch, seq_len, d_model)
        query = self.wq(q)
        value = self.wv(v)
        key = self.wk(k)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)

        # (batch, h, seq_len, d_k)
        x = attn(query, key, value, mask)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.wo(x)


class feed_for(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.act = GELU()

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.l2(self.act(self.l1(x)))


class layer_norm(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class res_conn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = layer_norm()

    def forward(self, x, sublayer):
        return x + sublayer(self.layer_norm(x))


class encoder_block(nn.Module):
    def __init__(self, feedfor: feed_for, self_attn: multi_head_attn) -> None:
        super().__init__()
        self.feed_for = feedfor
        self.self_attn = self_attn
        self.res_conn = nn.ModuleList([res_conn() for _ in range(2)])

    def forward(self, x, mask):
        x = self.res_conn[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.res_conn[1](x, self.feed_for)
        return x


class enc(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.layer_norm = layer_norm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class patch_emb(nn.Module):
    def __init__(self, emb_dim, patch_size, num_patches, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            # (batch, c, h, w) --> (batch, emb_dim, h, w)
            nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size),
            # (batch, emb_dim, h, w) --> (batch, emb_dim, h*w)
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, emb_dim)), requires_grad=True)
        self.pos_emb = nn. Parameter(torch.randn(size=(1, num_patches+1, emb_dim)), requires_grad=True)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)  # (b, h*w, emb_dim)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_emb + x
        return x


class vit(nn.Module):
    def __init__(self, emb_dim, patch_size, num_patches, in_channels, num_enc, num_classes, d_model, h, d_ff):
        super().__init__()
        self.emb_block = patch_emb(emb_dim, patch_size, num_patches, in_channels)
        self.enc_blocks = nn.ModuleList()
        for _ in range(num_enc):
            enc_self_att = multi_head_attn(emb_dim, h)
            enc_feed_for = feed_for(emb_dim, d_ff)
            enc_block = encoder_block(enc_feed_for, enc_self_att)
            self.enc_blocks.append(enc_block)
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        x = self.emb_block(x)
        for block in self.enc_blocks:
            x = block(x, None)
        x = self.mlp(x[:, 0, :])
        return x
