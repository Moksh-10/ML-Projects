import math
from typing import List, Tuple, Optional
import torch
from torch import nn


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class kv_cache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self):
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]  # bs, n_kv_h, sl, hd

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )  # concat along sl bs, n_kv_h, sl, hd
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class rms_norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.wei = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float())
        out = out * (1.0 + self.wei.float())
        return out.type_as(x)


class mlp(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.inter_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False)
        self.down_proj = nn.Linear(self.inter_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


def repeat_kv(x: torch.Tensor, n_rep: int):
    bs, n_kv_h, sl, hd = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bs, n_kv_h, n_rep, sl, hd)
    return x.reshape(bs, n_kv_h * n_rep, sl, hd)


class rope(nn.Module):
    def __init__(self, dim, max_pe=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_pe = max_pe
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, pos_ids, sl=None):
        # pos_ids = bs, sl
        self.inv_freq.to(x.device)
        inv_freq_exp = (
            self.inv_freq[None, :, None].float().expand(pos_ids.shape[0], -1, 1)
        )  # bs, hd//2 , 1
        pos_ids_exp = pos_ids[:, None, :].float()  # bs, 1, sl
        dt = x.device.type
        dt = dt if isinstance(dt, str) and dt != "mps" else "cpu"
        with torch.autocast(device_type=dt, enabled=False):
            freqs = (inv_freq_exp.float() @ pos_ids_exp.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x1.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)  # adding head
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb


class gemma_attn(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_dr = config.attention_dropout
        self.hid_size = config.hidden_size
        self.nh = config.num_attention_heads
        self.hd = config.head_dim
        self.n_kv_h = config.num_key_value_heads
        self.n_kv_r = self.nh // self.n_kv_h
        self.max_pe = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True

        assert self.hid_size % self.nh == 0

        self.q_poj = nn.Linear(self.hid_size, self.nh * self.hd, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hid_size, self.n_kv_h * self.hd, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hid_size, self.n_kv_h * self.hd, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.nh * self.hd, self.hid_size, bias=config.attention_bias)
        self.rotary_emb = rope(self.hd, self.max_pe, self.rope_theta)

    def forward(self, x: torch.Tensor, attn_mask:Optional[torch.Tensor]=None, pos_ids:Optional[torch.LongTensor]=None, kv_cache: Optional[kv_cache]=None):
        bs, sl, dim = x.size()
        q = self.q_poj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(bs, sl, self.nh, self.hd).transpose(1, 2)
        k = k.view(bs, sl, self.n_kv_h, self.hd).transpose(1, 2)
        v = v.view(bs, sl, self.n_kv_h, self.hd).transpose(1, 2)

        cos, sin = self.rotary_emb(v, pos_ids, sl=None)
        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.n_kv_r)
        v = repeat_kv(v, self.n_kv_r)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hd)

        assert attn_mask is not None
        attn = attn + attn_mask

        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = nn.functional.dropout(attn, p=self.attn_dr, training=self.training)
        attn_out = torch.matmul(attn, v)

        if attn_out.size() != (bs, self.nh, sl, self.hd):
            raise ValueError(f"expected : {bs, self.nh, sl, self.hd}, and got : {attn_out.size()}")

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(bs, sl, -1)
        attn_out = self.o_proj(attn_out)

        return attn_out, attn


class gemma_dec(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hid_size = config.hidden_size
        self.att = gemma_attn(config, layer_idx)
        self.mlp = mlp(config)
        self.in_norm = rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_ln = rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, pos_ids: Optional[torch.LongTensor] = None, kv_cache: Optional[kv_cache]=None):
        res = x
        x = self.in_norm(x)
        x, _, = self.att(x, attn_mask, pos_ids, kv_cache)
        x = res + x
        res = x
        x = self.post_attn_ln(x)
        x = self.mlp(x)
        x = res + x
        return x


class gemma_model(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.pad_idx = config.pad_token_id
        self.vs = config.vocab_size
        self.emb_token = nn.Embedding(config.vocab_size, config.hidden_size, self.pad_idx)
        self.ls = nn.ModuleList([gemma_dec(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeds(self):
        return self.emb_token

    def forward(self, attn_mask: Optional[torch.Tensor]=None, pos_ids: Optional[torch.LongTensor]=None, input_embds: Optional[torch.FloatTensor]=None, kv_cache: Optional[kv_cache]=None):
        x = input_embds
        norms = torch.tensor(self.config.hidden_size**0.5, dtype=x.dtype)
        x = x * norms
        for l in self.ls:
            x = l(x, a, po, kv_cache)
        x = self.norm(x)
        return x


class gemma_casualLM(nn.Module):
    def __init__(self, confif: GemmaConfig):
        super().__init__()
        self.config = confif
        self.model = gemma_model(confif)
        self.vs = confif.vocab_size
        self.lm_head = nn.Linear(confif.hidden_size, confif.vocab_size, bias=False)

    def get_input_embeds(self):
        return self.model.emb_token

    def tie_wei(self):
        self.lm_head.weight = self.model.emb_token.weight

    def forward(self, attn_mask: Optional[torch.Tensor]=None, pos_ids:Optional[torch.LongTensor]=None, input_embds:Optional[torch.FloatTensor]=None):
        x = self.model(attn_mask, pos_ids, input_embds, kv_cache)
        hs = x
        logits = self.lm_head(hs)
        logits = logits.float()
        rd = {"logits": logits,}
        if kv_cache is not None:
            rd["kv_cache"] = kv_cache
        return rd


class pg_mm_proj(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.l = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=True)

    def forwars(self, x):
        return self.l(x)
