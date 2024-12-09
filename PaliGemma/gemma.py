import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
from siglip import siglip_vision_config, siglip_vision_model


class paligemma_for_condn_generation(nn.Module):
    def __init__(self, config: paligemma_config):
        super().__init__()
        self.config = config
        self.vision_tower = siglip_vision_model(config.vision_config)
        self.multi_modal_proj = paligemma_multi_modal_projector(config)
        self.vocab_size = config.vocab_size

        lang_model = gemma_for_casualLM(config.text_config)
        self.lang_model = lang_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.lang_model.tie_weights()

    def forward(self,
                input_ids: torch.LongTensor = None,
                pixel_values: torch.FloatTensor = None,
                attn_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None) -> Tuple:





