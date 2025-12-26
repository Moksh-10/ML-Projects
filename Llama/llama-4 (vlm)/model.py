import torch
from torch import nn 
import torch.nn.functional as F 
from dataclasses import dataclass

@dataclass
class text_config:
    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 4096 * 32
    rms_norm_eps: float = 1e-5
    pad_token_id: int = 200018
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 500000
    attention_dropout: float = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    use_qk_norm: bool = True
    no_rope_layer_interval: int = 4
    attention_chunk_size: int = 8192
    attn_temperature_tuning: float = 4
    floor_scale: int = 8192
    attn_scale: float = 0.1

@dataclass
class vision_config:
    hidden_size: int = 768
    num_hidden_layers: int = 34
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5632
    vision_output_dim: int = 7680
    image_size: int = 448
    patch_size: int = 14
    norm_eps: float = 1e-5
    pixel_shuffle_ratio: float = 0.5
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096
    projector_dropout: float = 0.0
    attention_dropout: float = 0.0
    rope_theta: int = 10000


class text_experts(nn.Module):
    def __init__(self, config: text_config) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.inter_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2*self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))

        nn.init.normal_(self.gate_up_proj)
        nn.init.normal_(self.down_proj)

    def forward(self, x):
        pass 


class text_moe(nn.Module):
    def __init__(self, config: text_config) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_exp = config.num_local_experts
        self.experts = text_experts(config)
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
    
    def forward(self, x):
        bs, sl, dim = x.shape
        x = x.view(-1, dim)
        router_logits = self.router(x) # (-1, dim) --> (-1, num_local_experts) basically each token is mapped to a router

        tokens_per_exp = bs * sl 

        

if __name__ == "__main__":
    config = text_config(
            hidden_size=768,
            intermediate_size=768*2, 
            intermediate_size_mlp=768*2
            )
    x = torch.randn(2, 8, 768)
    tmoe = text_moe(config)
    tmoe(x) 

