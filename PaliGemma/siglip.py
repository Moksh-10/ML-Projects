import torch
import torch.nn as nn

class siglip_vision_config:

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


class siglip_vision_embeddings(nn.Module):
    def __init__(self, config : siglip_vision_config):
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_emb = nn.Conv2d(in_channels=config.num_channels,
                                   out_channels=self.embed_dim,
                                   kernel_size=self.patch_size,
                                   stride=self.patch_size,
                                   padding="valid", #this indicated no padding is added
                                  )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_pos = self.num_patches
        self.pos_emb = nn.Embedding(self.num_pos, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_pos).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # (bs, c, h, w)
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_emb(pixel_values)
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.pos_emb(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class siglip_attn(nn.Module):
    def __init__(self, config: siglip_vision_config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bs, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_wei = (torch.matmul(q, k.transpose(2, 3)) * self.scale)

        if attn_wei.size() != (bs, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"attn weights should be of size {(bs, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_wei.size()}"
            )

        attn_wei = nn.functional.softmax(attn_wei, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_wei = nn.functional.dropout(attn_wei, p=self.dropout, training=self.training)
        attn_out = torch.matmul(attn_wei, v)

        if attn_out.size() != (bs, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"attn_out should be of size {(bs, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_out.size()}"
            )

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(bs, seq_len, self.embed_dim)
        attn_out = self.out_proj(attn_out)

        return attn_out, attn_wei



class siglip_mlp(nn.Module):
    def __init__(self, config: siglip_vision_config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (bs, num_patches, embed_dim) --> (bs, num_patches, intermediate_size)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # (bs, num_patches, intermediate_size) --> (bs, num_patches, embed_dim)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class siglip_encoder_layer(nn.Module):
    def __init__(self, config: siglip_vision_config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = siglip_attn(config)
        self.layern_norm_1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = siglip_mlp(config)
        self.layern_norm_2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # res: (bs, num_patches, emb_dim)
        res = hidden_states
        # (bs, np, ed) --> (bs, np, ed)
        hidden_states = self.layern_norm_1(hidden_states)
        # (bs, np, ed) --> (bs, np, ed)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # (bs, np, ed) --> (bs, np, ed)
        hidden_states += res
        # (bs, np, ed) --> (bs, np, ed)
        res = hidden_states
        # (bs, np, ed) --> (bs, np, ed)
        hidden_states = self.layern_norm_2(hidden_states)
        # (bs, np, ed) --> (bs, np, ed)
        hidden_states = self.mlp(hidden_states)
        # (bs, np, ed) --> (bs, np, ed)
        hidden_states += res
        # (bs, np, ed)
        return hidden_states


class siglip_encoder(nn.Module):
    def __init__(self, config: siglip_vision_config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            siglip_encoder_layer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for e in self.layers:
            hidden_states = e(hidden_states)
        return hidden_states


class siglip_vision_transformer(nn.Module):
    def __init__(self, config: siglip_vision_config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = siglip_vision_embeddings(config)
        self.encoder = siglip_encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (bs, c, h, w) --> (bs, num_patches, embed_dim)
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class siglip_vision_model(nn.Module):
    def __init__(self, config: siglip_vision_config):
        super().__init__()
        self.config = config
        self.vision_model = siglip_vision_transformer(config)

    def forward(self, pixel_value) -> Tuple:
        #(bs, c, h, w) --> (bs, num_pathces, embed_dim)
        return self.vision_model(pixel_values=pixel_value)

