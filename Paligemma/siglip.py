import torch
from torch import nn
from typing import Optional, Tuple

class SigipVisionConfig:
    def __init__(self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class vision_embed(nn.Module):
    def __init__(self, config: SigipVisionConfig):
        super().__init__()
        self.config = config
        self.emb_dim = config.hidden_size
        self.img_size = config.image_size
        self.ps = config.patch_size

        self.patch_emb = nn.Conv2d(config.num_channels, self.emb_dim,
                                   kernel_size=self.ps,
                                   stride=self.ps,
                                   padding="valid" # no padding
                                   )

        self.num_ps = (self.img_size // self.ps) ** 2
        self.num_pos = self.num_ps
        self.pos_emb = nn.Embedding(self.num_pos, self.emb_dim)
        self.register_buffer("position_ids",
                             torch.arange(self.num_pos).expand((1, -1)),
                             persistent=False # means register_buffer won't be a part of model parameters
                             )

    def forward(self, pixel_vals: torch.FloatTensor) -> torch.Tensor:
        _, _, h, w = pixel_vals.shape
        patch_emb = self.patch_emb(pixel_vals)
        emb = patch_emb.flatten(2)
        emb = emb.transpose(1, 2)
        emb += self.pos_emb(self.position_ids)
        return emb


class attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.emb_dim // self.num_heads
        self.dr = config.attention_dropout

        self.wk = nn.Linear(self.emb_dim, self.emb_dim)
        self.wq = nn.Linear(self.emb_dim, self.emb_dim)
        self.wv = nn.Linear(self.emb_dim, self.emb_dim)
        self.wo = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, hid_st: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bs, sl, _ = hid_st.size()
        q = self.wq(hid_st)
        k = self.wk(hid_st)
        v = self.wv(hid_st)
        q = q.view(bs, sl, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, sl, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, sl, self.num_heads, self.head_dim).transpose(1, 2)
        att = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5)
        if att.size() != (bs, self.num_heads, sl, sl):
            raise ValueError(
                f"attn weights must be of size {(bs, self.num_heads, sl, sl)}"
                f"current size: {att.size()}"
            )
        att = nn.functional.softmax(att, dim=-1)
        att = nn.functional.dropout(att, self.dr, training=self.training)
        att_out = torch.matmul(att, v)
        if att_out.size() != (bs, self.num_heads, sl, self.head_dim):
            raise ValueError(
                f"attn_out must be of size: {(bs, self.num_heads, sl, self.head_dim)}"
                f"current size: {att_out.size()}"
            )
        att_out = att_out.transpose(1, 2).contiguous()
        att_out = att_out.reshape(bs, sl, self.emb_dim)
        att_out = self.wo(att_out)
        return att_out, att


class mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.l1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.l2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hid_st: torch.Tensor) -> torch.Tensor:
        x = self.l1(hid_st)
        x = nn.functional.gelu(x, approximate="tanh")
        x = self.l2(x)
        return x


class enc(nn.Module):
    def __init__(self, config: SigipVisionConfig):
        super().__init__()
        self.emb_dim = config.hidden_size
        self.attn = attention(config)
        self.ln1 = nn.LayerNorm(self.emb_dim, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(self.emb_dim, eps=config.layer_norm_eps)
        self.mlp = mlp(config)

    def forward(self, hid_st: torch.Tensor) -> torch.Tensor:
        res = hid_st
        hid_st = self.ln1(hid_st)
        hid_st, _ = self.attn(hid_st)
        hid_st = res + hid_st
        res = hid_st
        hid_st = self.ln2(hid_st)
        hid_st = self.mlp(hid_st)
        hid_st = res + hid_st
        return hid_st


class encoder(nn.Module):
    def __init__(self, config: SigipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [enc(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_emb: torch.Tensor) -> torch.Tensor:
        hid_st = input_emb
        for l in self.layers:
            hid_st = l(hid_st)
        return hid_st


class vit(nn.Module):
    def __init__(self, config: SigipVisionConfig):
        super().__init__()
        self.config = config
        emb_dim = config.hidden_size

        self.emb = vision_embed(config)
        self.enc = encoder(config)
        self.pln = nn.LayerNorm(emb_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_vals: torch.Tensor) -> torch.Tensor:
        hid_st = self.emb(pixel_vals)
        last_hs = self.enc(hid_st)
        last_hs = self.pln(last_hs)
        return last_hs


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SigipVisionConfig):
        super().__init__()
        self.config = config
        self.vm = vit(config)

    def forward(self, pixel_vlas) -> Tuple:
        return self.vm(pixel_vlas)
