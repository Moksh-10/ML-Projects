# import torch
# import torch.nn as nn
#
#
# class hyper_parameters:
#     def __init__(self,
#                  embed_size=768,
#                  hidden_size=768 * 4,
#                  num_hidden_layers=12,
#                  num_channels=3,
#                  image_size=224,
#                  patch_size=16,
#                  num_attn_heads=12,
#                  ln_eps=1e-6,
#                  attn_dropout=0.0,
#                  num_img_tokens: int = None,
#                  **kwargs):
#         super().__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.num_channels = num_channels
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.ln_eps = ln_eps
#         self.attn_dropout = attn_dropout
#         self.num_img_tokens = num_img_tokens
#         self.num_attn_heads = num_attn_heads
#         self.num_hidden_layers = num_hidden_layers
#
#
# class attn(nn.Module):
#     def __init__(self, config: hyper_parameters):
#         super().__init__()
#         self.config = config
#
#         self.attn_heads = config.num_attn_heads
#         self.emb_dim = config.embed_size
#         self.head_dim = self.emb_dim // self.attn_heads
#         self.d = config.attn_dropout
#
#         self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
#         self.k_proj = nn.Linear(self.emb_dim, self.emb_dim)
#         self.v_proj = nn.Linear(self.emb_dim, self.emb_dim)
#         self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         bs, seq_len, tt = x.size()
#
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)
#
#         q = q.view(bs, seq_len, self.attn_heads, self.head_dim).transpose(1, 2)
#         k = k.view(bs, seq_len, self.attn_heads, self.head_dim).transpose(1, 2)
#         v = v.view(bs, seq_len, self.attn_heads, self.head_dim).transpose(1, 2)
#
#         attn_wei = (torch.matmul(q, k.transpose(2, 3)) / self.head_dim ** 0.5)
#         attn_wei = torch.nn.functional.softmax(attn_wei, dim=-1, dtype=torch.float32).to(q.dtype)
#         attn_wei = torch.nn.functional.dropout(attn_wei, p=self.d, training=self.training)
#         attn_wei = torch.matmul(attn_wei, v)
#         attn_wei = attn_wei.transpose(1, 2).contiguous()
#         attn_wei = attn_wei.reshape(bs, seq_len, self.emb_dim)
#         attn_wei = self.out_proj(attn_wei)
#
#         return attn_wei
#
#
# class mlp(nn.Module):
#     def __init__(self, config: hyper_parameters):
#         super().__init__()
#         self.config = config
#         self.fc1 = nn.Linear(config.embed_size, config.hidden_size)
#         self.fc2 = nn.Linear(config.hidden_size, config.embed_size)
#
#     def forward(self, x):
#         y = self.fc1(x)
#         x = self.fc2(nn.functional.gelu(y, approximate="tanh"))
#         return x
#
#
# class enc_layer(nn.Module):
#     def __init__(self, config: hyper_parameters):
#         super().__init__()
#         self.config = config
#         self.mlp = mlp(config)
#         self.attn = attn(config)
#         self.emb_dim = config.embed_size
#         self.norm1 = nn.LayerNorm(self.emb_dim, eps=config.ln_eps)
#         self.norm2 = nn.LayerNorm(self.emb_dim, eps=config.ln_eps)
#
#     def forward(self, x):
#         res = x
#         x = self.norm1(x)
#         x = self.attn(x)
#         x += res
#         res = x
#         x = self.norm2(x)
#         x = self.mlp(x)
#         x += res
#         return x
#
#
# class enc(nn.Module):
#     def __init__(self, config: hyper_parameters):
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([
#             enc_layer(config) for _ in range(config.num_hidden_layers)
#         ])
#
#     def forward(self, x):
#         y = x
#         for l in self.layers:
#             y = l(x)
#         return y
#
#
# class embeddings(nn.Module):
#     def __init__(self, config: hyper_parameters):
#         super().__init__()
#         self.config = config
#         self.embed_size = config.embed_size
#         self.patch_size = config.patch_size
#         self.img_size = config.image_size
#         self.nc = config.num_channels
#
#         self.patch_emb = nn.Conv2d(in_channels=self.nc,
#                                    out_channels=self.embed_size,
#                                    kernel_size=self.patch_size,
#                                    stride=self.patch_size,
#                                    padding="valid")
#
#         self.num_patches = (
#                                        self.img_size // self.patch_size) ** 2  # squared because its across height and width like n * n
#         self.num_pos_emb = self.num_patches
#         self.pos_emb = nn.Embedding(self.num_pos_emb, self.embed_size)
#         self.register_buffer(
#             "position_ids",
#             torch.arange(self.num_pos_emb).expand((1, -1)),
#             persistent=False  # tells whether it should be a part of meodel_state_dict() while laoding and saving
#         )
#
#     def forward(self, x):
#         bs, nc, h, w = x.size()
#
#         # (bs, c, h, w) --> (bs, embed_dim, new_h, new_w)
#         patch_emb = self.patch_emb(x)
#
#         # (bs, embed_dim, new_h, new_w) --> (bs, embed_dim, new_h * new_w)
#         emb = patch_emb.flatten(2)
#
#         # (bs, embed_dim, new_h * new_w) --> (bs, new_h * new_w, embed_dim,)
#         emb = emb.transpose(1, 2)
#
#         emb += self.pos_emb(self.position_ids)
#
#         # (bs, num_patches, embed_dim)
#         return emb
#
#
# class vit(nn.Module):
#     def __init__(self, config: hyper_parameters):
#         super().__init__()
#         self.config = config
#         self.embed_size = config.embed_size
#         self.ln = nn.LayerNorm(self.embed_size, eps=config.ln_eps)
#         self.enc = enc(config)
#         self.emb = embeddings(config)
#
#     def forward(self, x):
#         x = self.emb(x)
#         y = self.enc(x=x)
#         y = self.ln(x)
#         return y
#
#
# config = hyper_parameters()
# ed = vit(config)
# c = torch.randn((5, 3, 224, 224))
# ff = ed(c)
# print(ff.shape)
