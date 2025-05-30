import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SigipVisionConfig, SiglipVisionModel


class PaliGemmaConfig():
    def __init__(
        self,
        vision_config=None,  # vision enc
        text_config=None,  # text dec
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

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2  # no. of patches for each img
        self.vision_config.projection_dim = projection_dim


class GemmaConfig():
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


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)  # vision encoder
        self.multi_model_projector = PaliGemmaMultiModalProjector(config) # kind of linear proj, size of embd of vision enc --> to size of each token for lang model
        self.vocab_size = config.vocab_size

        language_model = GemmaForCasualLLM(config.text_config) # gemma decoder
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()  # token id --> embd and embd --> token id as both are inverse , hence we can reduce no. of params

    def _merge_input_ids_with_image_features(self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None):
        _, _, embed_dim = image_features.shape
        bs, seq_len = inputs_embeds.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # (bs, sl, hidden_dim)
        scaled_img_feat = image_features / (self.config.hidden_size**0.5)

        # final tensor that will hold all of combined
        final_embed = torch.zeros(bs, seq_len, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # just differentiation betn padding, img and place holder token
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id) # (bs, sl)
        img_mask = input_ids == self.config.image_token_index # (bs, sl)
        pad_mask = input_ids == self.pad_token_id # (bs, sl)

        text_mask_ex = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        img_mask_ex = img_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_ex = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embed = torch.where(text_mask_ex, inputs_embeds, final_embed) # when text mask is 1, copy from input embds otherwise final embds
        final_embed = final_embed.masked_scatter(img_mask_ex, scaled_img_feat) # where img mask is true, take img feat, We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embed = torch.where(pad_mask_ex, torch.zeros_like(final_embed), final_embed)

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_type = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        # pre filling
        if kv_cache is None or kv_cache.num_items() == 0:
            casual_mask = torch.full((bs, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            casual_mask = torch.full((bs, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        casual_mask = casual_mask.unsqueeze(1)


    def forward(self, input_ids: torch.LongTensor=None, # input ids -- extracted from paligemma processor
                pixel_vaues: torch.FloatTensor=None,
                attention_mask: Optional[torch.Tensor]=None,
                kv_cache: Optional[KVCache] = None) -> Tuple:
        # input is right padded
        assert torch.all(attention_mask == 1), "input can't be padded"

        # input embd of size (bs, seq_len, hidden_size)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # merge text and imgs (bs, c, h, w) --> (bs, num_patches, emb_dim)
        selected_img_features = self.vision_tower(pixel_vaues.to(input_embed.dtype))

        # (bs, num_patches, emb_dim) --> (bs, num_patches, hidden_size)
        image_features = selected_img_features.multi_modal_projector(selected_img_features)

        # merge embd of text and img tokens
        input_embed, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, input_embed, input_ids, attention_mask, kv_cache)

        outputs = self.language_model(
            attention_mask, position_ids, input_embed, kv_cache
        )

        return outputs


